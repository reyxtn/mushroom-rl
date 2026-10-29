"""
Generators of the velocity command a quadruped is asked to follow, ported from
``isaacsim_envs/quadruped_commands.py`` (branch ``new_isaac``) to plain torch.

The only thing the Isaac version read from the simulation was the base
orientation, to turn the heading target into a yaw rate; here the environment
hands the current heading (yaw angle of every environment) to ``reset`` and
``step`` instead, so the generator has no simulator dependency at all.

"""

import math

import torch


def _rand(lower, upper, shape, device):
    return torch.rand(shape, device=device) * (upper - lower) + lower


def _wrap_to_pi(angles):
    return torch.atan2(torch.sin(angles), torch.cos(angles))


class CommandGenerator:
    """
    Base class for the generators of the command a quadruped is asked to follow.

    """

    def __init__(self):
        self._n_envs = None
        self._dt = None
        self._device = None

    def initialize(self, n_envs, dt, device):
        """
        Allocates the command of every environment.

        Args:
            n_envs (int): the number of parallel environments;
            dt (float): the duration of a control step;
            device: the torch device the commands live on.

        """
        self._n_envs = n_envs
        self._dt = dt
        self._device = device

    def reset(self, env_ids, heading):
        """
        Draws the command of the given environments for the episode they start.

        Args:
            env_ids (torch.Tensor): the environments being reset, (k,) long;
            heading (torch.Tensor): the yaw angle of those environments, (k,).

        """
        raise NotImplementedError

    def step(self, env_ids, heading):
        """
        Updates the command of the given environments after a control step.

        Args:
            env_ids (torch.Tensor): the environments that took the step, (k,) long;
            heading (torch.Tensor): the yaw angle of those environments, (k,).

        """
        raise NotImplementedError

    @property
    def commands(self):
        """
        The body-frame linear velocity along x and y and the yaw rate every
        environment is commanded, (n_envs, 3).

        """
        raise NotImplementedError

    @property
    def command_bounds(self):
        """
        The largest absolute value each of the three commands can take, (3,).

        """
        raise NotImplementedError


class VelocityCommandGenerator(CommandGenerator):
    """
    Base class for the generators of velocity commands drawn from ranges. The
    yaw rate of a fraction of the environments tracks a heading target instead,
    and the drawn commands can be skewed towards standing still, turning on the
    spot and walking slowly.

    Args:
        max_command_ranges (dict, None): the widest ranges the generator will
            ever sample from, keyed ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z``
            and ``heading``. They bound the command and every range
            :attr:`command_ranges` can be set to;
        command_ranges (dict, None): the ranges to start sampling from, keyed
            like ``max_command_ranges`` and bounded by them;
        command_dead_zone (float): linear velocity commands whose norm falls
            below this are set to zero;
        command_resampling_time_range (tuple, None): the range, in seconds, the
            time until an environment resamples its command is drawn from.
            ``None`` resamples with a fixed per-step probability (1/500 after
            50 steps);
        heading_control_stiffness (float): the gain turning the heading error
            into the yaw rate command;
        rel_heading_envs (float): the fraction of environments whose yaw rate
            tracks a heading target;
        rel_standing_envs (float): the fraction of environments commanded to
            stand still;
        frac_rotating_envs (float): the fraction of the moving environments
            commanded to rotate in place;
        frac_low_speed_envs (float): the fraction of the moving environments
            commanded to move below ``low_speed_threshold``;
        low_speed_threshold (float): the velocity below which a command counts
            as a low speed one.

    """

    def __init__(self, max_command_ranges=None, command_ranges=None, command_dead_zone=0.2,
                 command_resampling_time_range=None, heading_control_stiffness=0.5, rel_heading_envs=1.,
                 rel_standing_envs=0., frac_rotating_envs=0., frac_low_speed_envs=0., low_speed_threshold=0.5):
        super().__init__()

        self._max_command_ranges = dict(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_z=(-math.pi, math.pi),
                                        heading=(-3.14, 3.14))
        self._max_command_ranges |= max_command_ranges or {}

        self._command_ranges = dict(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_z=(-1., 1.),
                                    heading=(-3.14, 3.14))
        self._command_ranges |= command_ranges or {}
        self._check_command_ranges(self._command_ranges)

        self._command_dead_zone = command_dead_zone
        self._command_resampling_time_range = command_resampling_time_range
        self._heading_control_stiffness = heading_control_stiffness
        self._rel_heading_envs = rel_heading_envs
        self._rel_standing_envs = rel_standing_envs
        self._frac_rotating_envs = frac_rotating_envs
        self._frac_low_speed_envs = frac_low_speed_envs
        self._low_speed_threshold = low_speed_threshold

        self._commands = None
        self._is_heading_env = None
        self._is_standing_env = None
        self._time_to_resample = None
        self._steps = None

    def initialize(self, n_envs, dt, device):
        super().initialize(n_envs, dt, device)

        # [vx, vy, yaw rate, heading target]
        self._commands = torch.zeros(n_envs, 4, device=device)
        self._is_heading_env = torch.ones(n_envs, dtype=torch.bool, device=device)
        self._is_standing_env = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self._time_to_resample = torch.zeros(n_envs, device=device)
        self._steps = torch.zeros(n_envs, dtype=torch.long, device=device)

    def reset(self, env_ids, heading):
        self._steps[env_ids] = 0
        self._resample(env_ids)
        self._track_heading(env_ids, heading)

    def step(self, env_ids, heading):
        self._steps[env_ids] += 1
        self._resample(self._environments_to_resample(env_ids))
        self._track_heading(env_ids, heading)

    def set_commands(self, env_ids, lin_vel_x, lin_vel_y, ang_vel_z):
        """
        Overrides the command of the given environments and stops the generator
        from touching it (no heading tracking, no standing): for teleoperation.

        """
        self._commands[env_ids, 0] = lin_vel_x
        self._commands[env_ids, 1] = lin_vel_y
        self._commands[env_ids, 2] = ang_vel_z
        self._is_heading_env[env_ids] = False
        self._is_standing_env[env_ids] = False

    @property
    def commands(self):
        return self._commands[:, :3]

    @property
    def command_bounds(self):
        ranges = self._max_command_ranges
        return torch.tensor([max(abs(bound) for bound in ranges[name])
                             for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")], device=self._device)

    @property
    def command_ranges(self):
        """
        The ranges currently sampled from, keyed ``lin_vel_x``, ``lin_vel_y``,
        ``ang_vel_z`` and ``heading``. Assigning to this overrides only the
        given keys, which have to stay within the maximum ranges.

        """
        return dict(self._command_ranges)

    @command_ranges.setter
    def command_ranges(self, ranges):
        updated = dict(self._command_ranges)
        updated.update(ranges)
        self._check_command_ranges(updated)
        self._command_ranges = updated

    # ------------------------------------------------------------------

    def _resample(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        dev = self._device

        self._sample(env_ids)
        self._bias_commands(env_ids)

        # set small commands to zero
        self._commands[env_ids, :2] *= \
            (torch.norm(self._commands[env_ids, :2], dim=1) > self._command_dead_zone).unsqueeze(1)

        if self._command_resampling_time_range is not None:
            self._time_to_resample[env_ids] = _rand(*self._command_resampling_time_range, (n,), dev)

    def _track_heading(self, env_ids, heading):
        """
        Turns the heading error of the environments tracking a heading target
        into their yaw rate command, and zeroes the command of the ones
        standing still.

        """
        yaw_rate = torch.clip(self._heading_control_stiffness * _wrap_to_pi(self._commands[env_ids, 3] - heading),
                              *self._command_ranges["ang_vel_z"])
        self._commands[env_ids, 2] = torch.where(self._is_heading_env[env_ids], yaw_rate, self._commands[env_ids, 2])
        self._commands[env_ids[self._is_standing_env[env_ids]], :3] = 0.

    def _sample(self, env_ids):
        raise NotImplementedError

    def _bias_commands(self, env_ids):
        """
        Skews the freshly drawn commands towards standing still, turning on the
        spot, and walking slowly in an arbitrary direction.

        """
        dev = self._device
        n = len(env_ids)

        if self._rel_standing_envs > 0.:
            self._is_standing_env[env_ids] = torch.rand(n, device=dev) <= self._rel_standing_envs

        moving = env_ids[~self._is_standing_env[env_ids]]
        n_moving = len(moving)

        if self._frac_low_speed_envs > 0.:
            is_low_speed = torch.rand(n_moving, device=dev) <= self._frac_low_speed_envs
            low_speed = moving[is_low_speed]

            direction = torch.randn(len(low_speed), 3, device=dev)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            magnitude = _rand(0., self._low_speed_threshold, (len(low_speed), 1), dev)
            self._commands[low_speed, :3] = direction * magnitude

        self._bias_moving_commands(moving)

        if self._frac_rotating_envs > 0.:
            is_rotating = torch.rand(n_moving, device=dev) <= self._frac_rotating_envs
            rotating = moving[is_rotating]

            is_slow_turn = torch.rand(len(rotating), device=dev) <= 0.5
            slow_turn = rotating[is_slow_turn]
            self._commands[slow_turn, 2] = _rand(-self._low_speed_threshold, self._low_speed_threshold,
                                                 (len(slow_turn),), dev)

            self._commands[rotating, :2] = 0.

    def _bias_moving_commands(self, moving):
        """
        Further skews the commands of the moving environments. Nothing by default.

        """

    def _environments_to_resample(self, env_ids):
        if self._command_resampling_time_range is None:
            do_resample = torch.rand(len(env_ids), device=self._device) < (1. / 500.)
            do_resample &= self._steps[env_ids] > 50
            return env_ids[do_resample]

        self._time_to_resample[env_ids] -= self._dt
        return env_ids[self._time_to_resample[env_ids] <= 0.]

    def _check_command_ranges(self, ranges):
        unknown = set(ranges) - set(self._max_command_ranges)
        if unknown:
            raise ValueError(f"unknown command ranges: {sorted(unknown)}")

        for name, (low, high) in ranges.items():
            max_low, max_high = self._max_command_ranges[name]
            if low > high:
                raise ValueError(f"the {name} command range is empty: ({low}, {high})")
            if low < max_low or high > max_high:
                raise ValueError(f"the {name} command range ({low}, {high}) is not contained in the maximum "
                                 f"range ({max_low}, {max_high}) the generator was built with")


class UniformVelocityCommands(VelocityCommandGenerator):
    """
    Velocity commands whose linear velocity along x and y, yaw rate and heading
    target are each drawn uniformly from their range, independently.

    """

    def _sample(self, env_ids):
        dev = self._device
        n = len(env_ids)
        ranges = self._command_ranges

        self._commands[env_ids, 0] = _rand(*ranges["lin_vel_x"], (n,), dev)
        self._commands[env_ids, 1] = _rand(*ranges["lin_vel_y"], (n,), dev)
        self._commands[env_ids, 3] = _rand(*ranges["heading"], (n,), dev)

        if self._rel_heading_envs < 1.:
            self._commands[env_ids, 2] = _rand(*ranges["ang_vel_z"], (n,), dev)
            self._is_heading_env[env_ids] = torch.rand(n, device=dev) <= self._rel_heading_envs


class EllipticVelocityCommands(VelocityCommandGenerator):
    """
    Velocity commands whose linear velocity along x and y and yaw rate are drawn
    uniformly from the ellipsoid inscribed in the box of their ranges. A
    fraction of the moving environments can be commanded a velocity on its
    surface.

    """

    def __init__(self, frac_max_speed_envs=0., **params):
        super().__init__(**params)
        self._frac_max_speed_envs = frac_max_speed_envs

    def _sample(self, env_ids):
        dev = self._device
        n = len(env_ids)
        center, half_width = self._ellipsoid_parameters()

        direction = torch.randn(n, 3, device=dev)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        radius = torch.rand(n, 1, device=dev).pow(1. / 3.)
        self._commands[env_ids, :3] = center + radius * half_width * direction

        self._commands[env_ids, 3] = _rand(*self._command_ranges["heading"], (n,), dev)
        if self._rel_heading_envs < 1.:
            self._is_heading_env[env_ids] = torch.rand(n, device=dev) <= self._rel_heading_envs

    def _bias_moving_commands(self, moving):
        if self._frac_max_speed_envs > 0.:
            center, half_width = self._ellipsoid_parameters()
            is_max_speed = torch.rand(len(moving), device=self._device) <= self._frac_max_speed_envs
            max_speed = moving[is_max_speed]

            direction = torch.randn(len(max_speed), 3, device=self._device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self._commands[max_speed, :3] = center + half_width * direction

    def _ellipsoid_parameters(self):
        bounds = torch.tensor([self._command_ranges[name] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                              device=self._device)
        return bounds.mean(dim=1), (bounds[:, 1] - bounds[:, 0]) / 2.
