"""
Velocity tracking task for the Unitree H2.

The Go2Walk skeleton with the reward of Seo et al. 2025, "Learning
Sim-to-Real Humanoid Locomotion in 15 Minutes" (arXiv 2512.01996), as
released in holosoma: velocity tracking, an alive bonus, a foot height
profile driven by a gait clock, and six penalties (torso rates and tilt,
action rate, weighted default pose, feet too close, feet not flat). The
penalties are multiplied by ``penalty_scale``, which a curriculum raises as
the episodes get longer. The total is not clamped at zero.

Each world carries a gait clock; the two feet read it half a cycle apart and
the policy observes the sin and cos of both phases. A world with a zero
command stands: its phases are pinned to pi, where the profile asks for both
feet on the ground, so one policy walks and stands.

The observation is scaled in the environment (OBS_SCALES).

"""

import numpy as np
import torch

from mushroom_rl.core.spaces import Box

from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.utils import uniform

from .base import H2Base, with_pushes
from .params import (
    REWARD_TERMS,
    PENALTY_TERMS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_REWARD_PARAMS,
    POSE_WEIGHTS,
    WALK_TRACKING_SIGMA,
    GAIT_FREQUENCY_RANGE,
    EPISODE_AVERAGE_WINDOW,
    WALK_COMMAND_GENERATOR,
    WALK_PUSHES,
    OBS_SCALES,
    OBS_NOISE,
)


class H2Walk(H2Base):
    """
    Args:
        num_envs (int): number of worlds;
        reward_weights (dict, None): overrides of DEFAULT_REWARD_WEIGHTS, keyed
            like the info dictionary. Every weight carries its sign;
        reward_params (dict, None): overrides of DEFAULT_REWARD_PARAMS;
        tracking_sigma (float): width of the exponential tracking kernel;
        gait_frequency_range (tuple): the gait frequency of an episode is
            drawn from this range, in Hz, when the randomization is on, and
            is its midpoint otherwise;
        penalty_scale (float): factor on the weights of PENALTY_TERMS; the
            penalty curriculum sets it during training;
        command_generator (VelocityCommandGenerator, None): defaults to a
            UniformVelocityCommands with WALK_COMMAND_GENERATOR;
        randomization_params (RandomizationParams, dict, None): the ranges,
            with WALK_PUSHES applied unless they are set;
        obs_noise (bool): whether to add uniform noise to the observation;
        **kwargs: H2Base arguments.

    """

    N_FEET = 2

    def __init__(
        self,
        num_envs,
        reward_weights=None,
        reward_params=None,
        tracking_sigma=WALK_TRACKING_SIGMA,
        gait_frequency_range=GAIT_FREQUENCY_RANGE,
        penalty_scale=1.0,
        command_generator=None,
        randomization_params=None,
        obs_noise=True,
        **kwargs,
    ):
        self._reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        unknown = set(reward_weights or {}) - set(self._reward_weights)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        self._reward_weights |= reward_weights or {}

        self._reward_params = dict(DEFAULT_REWARD_PARAMS)
        unknown = set(reward_params or {}) - set(self._reward_params)
        if unknown:
            raise ValueError(f"unknown reward parameters: {sorted(unknown)}")
        self._reward_params |= reward_params or {}

        self._tracking_sigma = float(tracking_sigma)
        self._gait_frequency_range = tuple(gait_frequency_range)
        self._penalty_scale = float(penalty_scale)
        self._command_generator = (
            UniformVelocityCommands(**WALK_COMMAND_GENERATOR) if command_generator is None else command_generator
        )
        self._obs_noise = obs_noise

        super().__init__(num_envs, randomization_params=with_pushes(randomization_params, WALK_PUSHES), **kwargs)

        self._command_generator.initialize(num_envs, self.dt, self._device)
        self._reward_info = {k: torch.zeros(num_envs, device=self._device) for k in REWARD_TERMS}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        dev = self._device
        W = self._num_envs

        # Task state.
        self._last_actions = torch.zeros(W, self._n_joints, device=dev)
        self._gait_clock = torch.zeros(W, device=dev)
        self._gait_offset = torch.zeros(W, device=dev)
        self._gait_frequency = torch.full((W,), sum(self._gait_frequency_range) / 2.0, device=dev)
        self._average_episode_length = torch.zeros((), device=dev)
        self._active = torch.ones(W, dtype=torch.bool, device=dev)

        self._pose_weights = torch.tensor(
            [next(w for key, w in POSE_WEIGHTS if key in joint) for joint in self._controlled_joints],
            dtype=torch.float32,
            device=dev,
        )

        # Appended in _create_observation, in this order.
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        bounds = self._command_generator_bounds()
        self.obs_helper.add_obs("commands", 3, -bounds, bounds)
        self.obs_helper.add_obs("actions", self._n_joints, mdp_info.action_space.low, mdp_info.action_space.high)
        self.obs_helper.add_obs("gait_phase", 2 * self.N_FEET, -1.0, 1.0)
        spec = self.randomization_obs_spec()
        for name in self._observed_randomization:
            length, low, high = spec[name]
            self.obs_helper.add_obs(name, length, low, high)

        norm_np = self._get_obs_normalization_vec()
        noise_np = self._get_noise_scale_vec()
        self._normalization_vec = torch.as_tensor(norm_np, dtype=torch.float32, device=dev)
        self._noise_scale_vec = torch.as_tensor(noise_np, dtype=torch.float32, device=dev)

        # Observation space after the default pose shift, the scaling and
        # the noise, matching what the policy sees.
        obs_low, obs_high = self.obs_helper.get_obs_limits()
        obs_low = obs_low.astype(np.float64, copy=True)
        obs_high = obs_high.astype(np.float64, copy=True)
        default_joint_np = self._default_joint_pos.cpu().numpy()
        obs_low[self._joint_pos_slice] -= default_joint_np
        obs_high[self._joint_pos_slice] -= default_joint_np
        mdp_info.observation_space = Box(obs_low * norm_np - noise_np, obs_high * norm_np + noise_np)

        return mdp_info

    def _command_generator_bounds(self):
        ranges = self._command_generator._max_command_ranges
        return np.array([max(abs(b) for b in ranges[name]) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])

    def _get_obs_normalization_vec(self):
        m = self.obs_helper.obs_idx_map
        v = np.ones(self.obs_helper.obs_low.shape[0])
        v[self._lin_vel_slice] = OBS_SCALES["base_lin_vel"]
        v[self._ang_vel_slice] = OBS_SCALES["base_ang_vel"]
        v[self._joint_pos_slice] = OBS_SCALES["joint_pos"]
        v[self._joint_vel_slice] = OBS_SCALES["joint_vel"]
        v[m["commands"]] = OBS_SCALES["commands"]
        return v

    def _get_noise_scale_vec(self):
        m = self.obs_helper.obs_idx_map
        v = np.zeros(self.obs_helper.obs_low.shape[0])
        if not self._obs_noise:
            return v
        v[self._lin_vel_slice] = OBS_NOISE["base_lin_vel"] * OBS_SCALES["base_lin_vel"]
        v[self._ang_vel_slice] = OBS_NOISE["base_ang_vel"] * OBS_SCALES["base_ang_vel"]
        v[self._joint_pos_slice] = OBS_NOISE["joint_pos"] * OBS_SCALES["joint_pos"]
        v[self._joint_vel_slice] = OBS_NOISE["joint_vel"] * OBS_SCALES["joint_vel"]
        v[m["projected_gravity"]] = OBS_NOISE["projected_gravity"]
        return v

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _gait_phase_obs(self):
        phase = self._gait_phase()
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    def _create_observation(self, obs):
        parts = [obs, self._projected_gravity(), self._command_generator.commands, self._actions, self._gait_phase_obs()]
        parts += [self.randomization_obs_value(name) for name in self._observed_randomization]
        return torch.cat(parts, dim=1)

    def _modify_observation(self, obs):
        obs = obs.clone()
        # The policy reads the joint angle the way a miscalibrated encoder
        # would, offset from the truth by the same amount the controller
        # targets are shifted by (see H2Base._preprocess_action).
        offset = self._randomizer.position_offset[:, self._ctrl_idx]
        obs[:, self._joint_pos_slice] -= self._default_joint_pos + offset
        obs *= self._normalization_vec
        if self._obs_noise:
            obs += (2.0 * torch.rand_like(obs) - 1.0) * self._noise_scale_vec
        # A diverged world is terminated by is_absorbing, but its observation
        # still enters the dataset, so it must be finite.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(obs, min=-100.0, max=100.0)

    def observation_indices(self, *names):
        """
        Where the named parts of the observation sit, as one sorted long
        tensor, for an asymmetric actor-critic.

        Names: ``base_ang_vel``, ``base_lin_vel``, ``base_vel`` (both),
        ``joint_pos``, ``joint_vel``, ``projected_gravity``, ``commands``,
        ``actions``, ``gait_phase`` and any name in ``observed_randomization``.

        """
        m = self.obs_helper.obs_idx_map
        aliases = {
            "base_ang_vel": list(range(self._ang_vel_slice.start, self._ang_vel_slice.stop)),
            "base_lin_vel": list(range(self._lin_vel_slice.start, self._lin_vel_slice.stop)),
            "joint_pos": list(range(self._joint_pos_slice.start, self._joint_pos_slice.stop)),
            "joint_vel": list(range(self._joint_vel_slice.start, self._joint_vel_slice.stop)),
        }
        indices = []
        for name in names:
            if name in aliases:
                indices += aliases[name]
            elif name in m:
                indices += list(m[name])
            else:
                raise ValueError(f"unknown observation {name!r}")
        return torch.tensor(sorted(set(indices)), dtype=torch.long, device=self._device)

    # ------------------------------------------------------------------
    # Knobs a curriculum changes during training
    # ------------------------------------------------------------------

    @property
    def command_generator(self):
        return self._command_generator

    @property
    def tracking_sigma(self):
        return self._tracking_sigma

    @tracking_sigma.setter
    def tracking_sigma(self, value):
        self._tracking_sigma = float(value)

    @property
    def tracking_stds(self):
        """The tracking kernel as the Curriculum sets it: one key, ``sigma``."""
        return dict(sigma=self._tracking_sigma)

    @tracking_stds.setter
    def tracking_stds(self, stds):
        unknown = set(stds) - {"sigma"}
        if unknown:
            raise ValueError(f"unknown tracking tolerances: {sorted(unknown)}")
        if "sigma" in stds:
            self._tracking_sigma = float(stds["sigma"])

    @property
    def penalty_scale(self):
        """Factor on the weights of PENALTY_TERMS."""
        return self._penalty_scale

    @penalty_scale.setter
    def penalty_scale(self, value):
        self._penalty_scale = float(value)

    @property
    def average_episode_length(self):
        """Running mean of the length of the episodes that ended, in control steps."""
        return self._average_episode_length.item()

    @property
    def reward_weights(self):
        return dict(self._reward_weights)

    @reward_weights.setter
    def reward_weights(self, weights):
        unknown = set(weights) - set(self._reward_weights)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        self._reward_weights.update(weights)

    @property
    def reward_keys(self):
        """The keys of the reward info dictionary."""
        return list(REWARD_TERMS)

    # ------------------------------------------------------------------
    # Gait clock
    # ------------------------------------------------------------------

    def _standing(self):
        """Worlds whose whole command, yaw included, is zero."""
        return self._command_generator.commands.norm(dim=1) < self._reward_params["standing_threshold"]

    def _gait_phase(self):
        """
        Phase of each foot in [-pi, pi), (num_envs, 2), the right foot half a
        cycle after the left. Both are pi in a standing world.

        """
        left = self._gait_clock + self._gait_offset
        phase = torch.stack([left, left + np.pi], dim=1)
        phase = torch.remainder(phase + np.pi, 2.0 * np.pi) - np.pi
        return torch.where(self._standing().unsqueeze(1), torch.full_like(phase, np.pi), phase)

    def _expected_foot_height(self, phase):
        """
        Height the profile asks of a foot at the given phase: zero at -pi,
        up to ``swing_height`` at 0 and back down to zero at pi, along a
        cubic Bezier curve (MuJoCo Playground's ``get_rz``).

        """
        height = self._reward_params["swing_height"]
        x = (phase + np.pi) / (2.0 * np.pi)

        def bezier(t):
            return t**3 + 3.0 * t**2 * (1.0 - t)

        return torch.where(x <= 0.5, height * bezier(2.0 * x), height * (1.0 - bezier(2.0 * x - 1.0)))

    # ------------------------------------------------------------------
    # Reset and step bookkeeping
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)

        idx = (
            env_indices.to(self._device).long()
            if isinstance(env_indices, torch.Tensor)
            else torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        )
        n = len(idx)
        if n == 0:
            return

        self._last_actions[idx] = 0.0
        self._gait_clock[idx] = 0.0
        if self._domain_randomization:
            self._gait_offset[idx] = uniform((-np.pi, np.pi), (n,), self._device)
            self._gait_frequency[idx] = uniform(self._gait_frequency_range, (n,), self._device)
        else:
            self._gait_offset[idx] = 0.0
            self._gait_frequency[idx] = sum(self._gait_frequency_range) / 2.0

        self._command_generator.reset(idx, self._heading()[idx])

    def step_all(self, env_mask, action):
        # The worlds outside the mask are simulated but run no episode (an
        # evaluation uses a few of them): _track_episode_length skips them.
        self._active = env_mask
        return super().step_all(env_mask, action)

    def _step_finalize(self):
        super()._step_finalize()
        self._command_generator.step(self._all_envs, self._heading())
        # The clock runs in every world; _gait_phase hides it while standing.
        advanced = self._gait_clock + 2.0 * np.pi * self._gait_frequency * self.dt
        self._gait_clock = torch.remainder(advanced, 2.0 * np.pi)

    def _track_episode_length(self, absorbing):
        """
        Fold the episodes ending at this step into the running mean of the
        episode length, which spans EPISODE_AVERAGE_WINDOW episodes per world.

        """
        done = self._active & (absorbing | (self._episode_length >= self.info.horizon))
        n = done.sum()
        mean = (self._episode_length * done).sum() / n.clamp(min=1)
        weight = (n / (EPISODE_AVERAGE_WINDOW * self._num_envs)).clamp(max=1.0)
        self._average_episode_length += weight * (mean - self._average_episode_length)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        lin_vel = self._base_lin_vel()
        ang_vel = self._base_ang_vel()
        gravity = self._projected_gravity()
        # The raw policy action: the action this method is handed is the
        # preprocessed position target for all 31 actuators.
        actions = self._actions
        dt = self.dt
        w = self._reward_weights

        r = {
            "alive": (~absorbing).float() * w["alive"] * dt,
            "tracking_lin_vel": self._reward_tracking_lin_vel(lin_vel[:, :2]) * w["tracking_lin_vel"] * dt,
            "tracking_ang_vel": self._reward_tracking_ang_vel(ang_vel[:, 2]) * w["tracking_ang_vel"] * dt,
            "feet_phase": self._reward_feet_phase() * w["feet_phase"] * dt,
            "ang_vel_xy": torch.sum(torch.square(ang_vel[:, :2]), dim=1) * w["ang_vel_xy"] * dt,
            "orientation": torch.sum(torch.square(gravity[:, :2]), dim=1) * w["orientation"] * dt,
            "action_rate": torch.sum(torch.square(self._last_actions - actions), dim=1) * w["action_rate"] * dt,
            "pose": self._reward_pose() * w["pose"] * dt,
            "feet_distance": self._reward_feet_distance() * w["feet_distance"] * dt,
            "feet_orientation": self._reward_feet_orientation() * w["feet_orientation"] * dt,
        }
        for name in PENALTY_TERMS:
            r[name] = r[name] * self._penalty_scale

        # A diverged world produces inf/NaN here, and huge finite values the
        # step before; it is terminated, and its reward must not poison the
        # batch through GAE.
        floor = self._reward_params["reward_floor"]
        total = torch.nan_to_num(sum(r.values()), nan=0.0, posinf=0.0, neginf=floor)
        total = torch.clamp(total, min=floor)
        r["termination"] = absorbing.float() * w["termination"] * dt
        total = total + r["termination"]

        self._reward_info = r
        self._last_actions = actions.clone()
        self._track_episode_length(absorbing)
        return total

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        err = torch.sum(torch.square(self._command_generator.commands[:, :2] - lin_vel_xy), dim=1)
        return torch.exp(-err / self._tracking_sigma)

    def _reward_tracking_ang_vel(self, ang_vel_z):
        err = torch.square(self._command_generator.commands[:, 2] - ang_vel_z)
        return torch.exp(-err / self._tracking_sigma)

    def _reward_feet_phase(self):
        """Exponential kernel on the distance of both feet from their height profile."""
        err = torch.square(self._foot_height() - self._expected_foot_height(self._gait_phase()))
        return torch.exp(-err.sum(dim=1) / self._reward_params["feet_phase_sigma"])

    def _reward_pose(self):
        """Squared distance of the joints from their default angle, weighted per joint."""
        dev = self._joint_pos() - self._default_joint_pos
        return torch.sum(self._pose_weights * torch.square(dev), dim=1)

    def _reward_feet_distance(self):
        """One when the feet are closer than feet_min_distance across the heading."""
        foot_pos = self._foot_pos()
        d = foot_pos[:, 0, :2] - foot_pos[:, 1, :2]
        yaw = self._heading()
        lateral = torch.abs(torch.cos(yaw) * d[:, 1] - torch.sin(yaw) * d[:, 0])
        return (lateral < self._reward_params["feet_min_distance"]).float()

    def _reward_feet_orientation(self):
        """Tilt of each foot from flat: the sine of the angle between its up axis and the vertical."""
        return torch.sqrt((1.0 - torch.square(self._foot_up())).clamp(min=0.0)).sum(dim=1)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_absorbing(self, obs):
        if not self._terminate_when_unhealthy:
            return ~self._is_finite()
        return ~self._is_healthy()

    def _create_info_dictionary(self, obs):
        info = dict(self._reward_info)
        commands = self._command_generator.commands
        info["command_x"] = commands[:, 0]
        info["command_y"] = commands[:, 1]
        info["command_yaw"] = commands[:, 2]
        return info
