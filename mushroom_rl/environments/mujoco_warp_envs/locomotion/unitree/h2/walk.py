"""
Velocity tracking task for the Unitree H2.

The Go2Walk skeleton with the reward of the old H2Walk: the Rudin set plus a
gait clock (each world carries a phase; the policy observes its sin and cos,
and a contact-schedule term pays for feet on the ground in their stance
window and in the air in their swing window, the two feet half a period
apart; with a zero command the schedule is both feet down), upright and
height terms, swing clearance, slip, a minimum distance between the feet,
and a posture term. The total is clamped at zero.

Contact is detected from the foot sole height. The observation is scaled in
the environment (OBS_SCALES), as the H2 policies were trained.

"""

import numpy as np
import torch

from mushroom_rl.core.spaces import Box

from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands

from .base import H2Base, with_pushes
from .params import (
    GAIT_OFFSETS,
    REWARD_TERMS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_REWARD_PARAMS,
    WALK_GAIT,
    WALK_TRACKING_SIGMA,
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
        gait_period (float): duration of one gait cycle, in seconds;
        duty_factor (float): fraction of the cycle each foot spends in stance;
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
        gait_period=WALK_GAIT["gait_period"],
        duty_factor=WALK_GAIT["duty_factor"],
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
        self._gait_period = gait_period
        self._duty_factor = duty_factor
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
        n_feet = self.N_FEET

        # Task state.
        self._last_actions = torch.zeros(W, self._n_joints, device=dev)
        self._last_joint_vel = torch.zeros(W, self._n_joints, device=dev)
        self._feet_air_time = torch.zeros(W, n_feet, device=dev)
        self._last_contacts = torch.zeros(W, n_feet, dtype=torch.bool, device=dev)
        self._last_foot_pos = torch.zeros(W, n_feet, 3, device=dev)
        self._gait_phase = torch.zeros(W, device=dev)
        self._gait_offsets = torch.as_tensor(GAIT_OFFSETS, dtype=torch.float32, device=dev)

        # Appended in _create_observation, in this order.
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        bounds = self._command_generator_bounds()
        self.obs_helper.add_obs("commands", 3, -bounds, bounds)
        self.obs_helper.add_obs("actions", self._n_joints, mdp_info.action_space.low, mdp_info.action_space.high)
        self.obs_helper.add_obs("gait_phase", 2, -1.0, 1.0)
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
        two_pi = 2.0 * np.pi
        return torch.stack([torch.sin(two_pi * self._gait_phase), torch.cos(two_pi * self._gait_phase)], dim=1)

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

    def _moving(self):
        """Worlds whose planar velocity command is non-zero."""
        return self._command_generator.commands[:, :2].norm(dim=1) > self._reward_params["moving_threshold"]

    def _desired_stance(self):
        """
        Contact schedule from the gait clock, (num_envs, 2) bool, True where
        the foot should be on the ground. With a zero command both feet down.

        """
        phase = (self._gait_phase.unsqueeze(1) + self._gait_offsets.unsqueeze(0)) % 1.0
        stance = phase < self._duty_factor
        return stance | ~self._moving().unsqueeze(1)

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
        if len(idx) == 0:
            return

        self._last_actions[idx] = 0.0
        self._last_joint_vel[idx] = self._joint_vel()[idx]
        self._feet_air_time[idx] = 0.0
        self._last_contacts[idx] = False
        self._last_foot_pos[idx] = self._foot_pos()[idx]
        self._gait_phase[idx] = 0.0

        self._command_generator.reset(idx, self._heading()[idx])

    def _step_finalize(self):
        super()._step_finalize()
        self._command_generator.step(self._all_envs, self._heading())
        self._gait_phase = (self._gait_phase + self.dt / self._gait_period) % 1.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        lin_vel = self._base_lin_vel()
        ang_vel = self._base_ang_vel()
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        torque = self._joint_torque()
        gravity = self._projected_gravity()
        foot_pos = self._foot_pos()
        foot_height = foot_pos[:, :, 2] - self._foot_half_height
        contact = foot_height < self._reward_params["foot_contact_height"]
        # The raw policy action: the action this method is handed is the
        # preprocessed position target for all 31 actuators.
        actions = self._actions
        dt = self.dt
        w = self._reward_weights

        r = {
            "alive": (~absorbing).float() * w["alive"] * dt,
            "tracking_lin_vel": self._reward_tracking_lin_vel(lin_vel[:, :2]) * w["tracking_lin_vel"] * dt,
            "tracking_ang_vel": self._reward_tracking_ang_vel(ang_vel[:, 2]) * w["tracking_ang_vel"] * dt,
            "lin_vel_z": torch.square(lin_vel[:, 2]) * w["lin_vel_z"] * dt,
            "ang_vel_xy": torch.sum(torch.square(ang_vel[:, :2]), dim=1) * w["ang_vel_xy"] * dt,
            "orientation": torch.sum(torch.square(gravity[:, :2]), dim=1) * w["orientation"] * dt,
            "base_height": self._reward_base_height() * w["base_height"] * dt,
            "torques": torch.sum(torch.square(torque), dim=1) * w["torques"] * dt,
            "joint_acc": torch.sum(torch.square((self._last_joint_vel - joint_vel) / dt), dim=1) * w["joint_acc"] * dt,
            "action_rate": torch.sum(torch.square(self._last_actions - actions), dim=1) * w["action_rate"] * dt,
            "joint_pos_limits": self._reward_joint_pos_limits(joint_pos) * w["joint_pos_limits"] * dt,
            "joint_deviation": self._reward_joint_deviation(joint_pos) * w["joint_deviation"] * dt,
            "feet_air_time": self._reward_feet_air_time(contact) * w["feet_air_time"] * dt,
            "feet_slip": self._reward_feet_slip(foot_pos, contact) * w["feet_slip"] * dt,
            "feet_clearance": self._reward_feet_clearance(foot_height, contact) * w["feet_clearance"] * dt,
            "feet_distance": self._reward_feet_distance(foot_pos) * w["feet_distance"] * dt,
            "gait": self._reward_gait(contact) * w["gait"] * dt,
        }

        total = torch.clamp(sum(r.values()), min=0.0)
        # A diverged world produces inf/NaN here; it is terminated, and its
        # reward must be finite so it does not poison the batch through GAE.
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        self._reward_info = r
        self._last_actions = actions.clone()
        self._last_joint_vel = joint_vel.clone()
        self._last_foot_pos = foot_pos.clone()
        return total

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        err = torch.sum(torch.square(self._command_generator.commands[:, :2] - lin_vel_xy), dim=1)
        return torch.exp(-err / self._tracking_sigma)

    def _reward_tracking_ang_vel(self, ang_vel_z):
        err = torch.square(self._command_generator.commands[:, 2] - ang_vel_z)
        return torch.exp(-err / self._tracking_sigma)

    def _reward_base_height(self):
        return torch.square(self._base_height() - self._reward_params["base_height_target"])

    def _reward_joint_pos_limits(self, joint_pos):
        below = (self._soft_joint_lower - joint_pos).clamp(min=0.0)
        above = (joint_pos - self._soft_joint_upper).clamp(min=0.0)
        return torch.sum(below + above, dim=1)

    def _reward_joint_deviation(self, joint_pos):
        dev = joint_pos[:, self._posture_idx] - self._default_joint_pos[self._posture_idx]
        return dev.abs().sum(dim=1)

    def _reward_feet_air_time(self, contact):
        # Reward long steps, paid on the first contact after a swing, while a
        # velocity command is active. Height-based contact flickers, so a
        # short swing must not turn this into a penalty.
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += self.dt
        rew = torch.sum((self._feet_air_time - self._reward_params["feet_air_time_threshold"]) * first_contact, dim=1)
        rew = torch.clamp(rew, min=0.0) * self._moving()
        self._feet_air_time *= ~contact_filt
        return rew

    def _reward_feet_slip(self, foot_pos, contact):
        """Planar speed squared of the feet that are in contact."""
        vel = (foot_pos - self._last_foot_pos) / self.dt
        return torch.sum(torch.sum(torch.square(vel[:, :, :2]), dim=2) * contact, dim=1)

    def _reward_feet_clearance(self, foot_height, contact):
        """Squared distance of the swing feet from the clearance height."""
        err = torch.square(foot_height - self._reward_params["feet_clearance_target"])
        return torch.sum(err * ~contact, dim=1) * self._moving()

    def _reward_feet_distance(self, foot_pos):
        """Penalty for feet closer together than feet_min_distance."""
        d = (foot_pos[:, 0, :2] - foot_pos[:, 1, :2]).norm(dim=1)
        return (self._reward_params["feet_min_distance"] - d).clamp(min=0.0)

    def _reward_gait(self, contact):
        """Fraction of feet whose contact state matches the gait schedule."""
        return (contact == self._desired_stance()).float().mean(dim=1)

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
