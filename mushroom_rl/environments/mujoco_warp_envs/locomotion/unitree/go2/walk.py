"""
Velocity tracking task for the Unitree Go2, the MuJoCo Warp counterpart of
new_isaac's ``Go2Isaac`` / ``QuadrupedIsaac``: same reward terms (names, signs,
formulas, defaults), same command generator, same observation content.

Differences that come from the backend, not from choice:

- contacts are read from the solver's contact list (a geom touching the floor
  with dist <= 0) instead of PhysX net contact forces above a threshold;
- foot velocities are finite differences of the foot positions over a policy
  step, Isaac reads the body velocity;
- the observation is not normalized here: like Isaac, the training script
  adds a StandardizationPreprocessor to the agent.

"""

import numpy as np
import torch

from mushroom_rl.core.spaces import Box

from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands

from .base import Go2Base
from .params import (
    BASE_REWARD_TERMS,
    OPTIONAL_REWARD_TERMS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_REWARD_PARAMS,
    DEFAULT_TRACKING_STDS,
    JOINT_MAX_VEL,
    OBS_NOISE,
)

# Optional terms that read the per-foot state (contact time, air time,
# positions, velocities).
FOOT_REWARD_TERMS = (
    "feet_air_time_high", "feet_air_time_low", "feet_clearance", "feet_clearance_lateral",
    "feet_slide", "feet_slide_low", "feet_z_velocity", "feet_air_time_symmetry", "long_contact",
    "stand_still_short_contact",
)


class Go2Walk(Go2Base):
    """
    Velocity tracking task for Unitree Go2.

    Args:
        num_envs (int): number of worlds;
        reward_weights (dict, None): overrides of DEFAULT_REWARD_WEIGHTS, keyed
            like the info dictionary. Every weight carries its sign;
        reward_params (dict, None): overrides of DEFAULT_REWARD_PARAMS;
        tracking_stds (dict, None): overrides of DEFAULT_TRACKING_STDS;
        command_generator (VelocityCommandGenerator, None): defaults to a
            UniformVelocityCommands with its default parameters;
        obs_noise (bool): whether to add uniform noise to the observation;
        **kwargs: Go2Base arguments.

    """

    def __init__(
        self,
        num_envs,
        reward_weights=None,
        reward_params=None,
        tracking_stds=None,
        command_generator=None,
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

        self._tracking_stds = dict(DEFAULT_TRACKING_STDS)
        unknown = set(tracking_stds or {}) - set(self._tracking_stds)
        if unknown:
            raise ValueError(f"unknown tracking tolerances: {sorted(unknown)}")
        self._tracking_stds |= tracking_stds or {}

        self._command_generator = UniformVelocityCommands() if command_generator is None else command_generator
        self._obs_noise = obs_noise

        super().__init__(num_envs, **kwargs)

        self._command_generator.initialize(num_envs, self.dt, self._device)

        # Every key the info dictionary will ever hold, so it is the same at
        # every step: the base terms, termination, the active optional terms.
        zero = lambda: torch.zeros(num_envs, device=self._device)  # noqa: E731
        self._reward_info = {k: zero() for k in self.reward_keys}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        dev = self._device
        W = self._num_envs

        # Task state.
        self._last_actions = torch.zeros(W, self._n_joints, device=dev)
        self._last_joint_vel = torch.zeros(W, self._n_joints, device=dev)
        self._last_contacts = torch.zeros(W, 4, dtype=torch.bool, device=dev)
        self._feet_air_time = torch.zeros(W, 4, device=dev)
        # Per-foot state of the optional foot terms.
        self._foot_air_time = torch.zeros(W, 4, device=dev)
        self._foot_last_air_time = torch.zeros(W, 4, device=dev)
        self._foot_contact_time = torch.zeros(W, 4, device=dev)
        self._foot_first_contact = torch.zeros(W, 4, dtype=torch.bool, device=dev)
        self._foot_contact = torch.zeros(W, 4, dtype=torch.bool, device=dev)
        self._foot_positions = torch.zeros(W, 4, 3, device=dev)
        self._foot_velocities = torch.zeros(W, 4, 3, device=dev)
        self._last_foot_positions = torch.zeros(W, 4, 3, device=dev)
        self._joint_max_vel = torch.as_tensor(JOINT_MAX_VEL, dtype=torch.float32, device=dev)

        # Appended in _create_observation, in this order.
        self.obs_helper.add_obs("base_pos", 3, -np.inf, np.inf)
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        bounds = self._command_generator_bounds()
        self.obs_helper.add_obs("commands", 3, -bounds, bounds)
        self.obs_helper.add_obs("actions", self._n_joints, mdp_info.action_space.low, mdp_info.action_space.high)
        spec = self.randomization_obs_spec()
        for name in self._observed_randomization:
            length, low, high = spec[name]
            self.obs_helper.add_obs(name, length, low, high)

        noise_np = self._get_noise_scale_vec()
        self._noise_scale_vec = torch.as_tensor(noise_np, dtype=torch.float32, device=dev)

        # Observation space after the default pose shift and the encoder
        # offset, widened by the noise (QuadrupedIsaac._build_observation_space).
        obs_low, obs_high = self.obs_helper.get_obs_limits()
        obs_low = obs_low.astype(np.float64, copy=True)
        obs_high = obs_high.astype(np.float64, copy=True)
        default_joint_np = self._default_joint_pos.cpu().numpy()
        offset = float(self._randomization_params["position_offset"])
        obs_low[self._joint_pos_slice] -= default_joint_np + offset
        obs_high[self._joint_pos_slice] -= default_joint_np - offset
        mdp_info.observation_space = Box(obs_low - noise_np, obs_high + noise_np)

        return mdp_info

    def _command_generator_bounds(self):
        ranges = self._command_generator._max_command_ranges
        return np.array([max(abs(b) for b in ranges[name]) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])

    def _get_noise_scale_vec(self):
        m = self.obs_helper.obs_idx_map
        v = np.zeros(self.obs_helper.obs_low.shape[0])
        if not self._obs_noise:
            return v
        v[self._lin_vel_slice] = OBS_NOISE["base_lin_vel"]
        v[self._ang_vel_slice] = OBS_NOISE["base_ang_vel"]
        v[self._joint_pos_slice] = OBS_NOISE["joint_pos"]
        v[self._joint_vel_slice] = OBS_NOISE["joint_vel"]
        v[m["projected_gravity"]] = OBS_NOISE["projected_gravity"]
        return v

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _create_observation(self, obs):
        # Only the trunk height means anything: where the robot walks does not.
        base_pos = torch.zeros(obs.shape[0], 3, device=obs.device, dtype=obs.dtype)
        base_pos[:, 2] = self._read_data("base_pos")[:, 2]
        parts = [obs, base_pos, self._projected_gravity(), self._command_generator.commands, self._actions]
        parts += [self.randomization_obs_value(name) for name in self._observed_randomization]
        return torch.cat(parts, dim=1)

    def _modify_observation(self, obs):
        obs = obs.clone()
        # The policy reads the joint angle the way a miscalibrated encoder
        # would, offset from the truth by the same amount the controller
        # targets are shifted by (see Go2Base._preprocess_action).
        offset = self._randomizer.position_offset[:, self._ctrl_idx]
        obs[:, self._joint_pos_slice] -= self._default_joint_pos + offset
        if self._obs_noise:
            obs += (2.0 * torch.rand_like(obs) - 1.0) * self._noise_scale_vec
        # A diverged world is terminated by is_absorbing, but its observation
        # still enters the dataset, so it must be finite.
        return torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

    def observation_indices(self, *names):
        """
        Where the named parts of the observation sit, as one sorted long
        tensor, for an asymmetric actor-critic.

        Names: ``base_ang_vel``, ``base_lin_vel``, ``base_vel`` (both),
        ``joint_pos``, ``joint_vel``, ``base_pos``, ``projected_gravity``,
        ``commands``, ``actions`` and any name in ``observed_randomization``.

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
    def tracking_stds(self):
        return dict(self._tracking_stds)

    @tracking_stds.setter
    def tracking_stds(self, stds):
        unknown = set(stds) - set(self._tracking_stds)
        if unknown:
            raise ValueError(f"unknown tracking tolerances: {sorted(unknown)}")
        self._tracking_stds.update(stds)

    @property
    def reward_weights(self):
        return dict(self._reward_weights)

    @reward_weights.setter
    def reward_weights(self, weights):
        unknown = set(weights) - set(self._reward_weights)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        newly_on = [k for k in OPTIONAL_REWARD_TERMS if weights.get(k, 0.) != 0. and self._reward_weights[k] == 0.]
        if newly_on:
            raise ValueError(f"{newly_on} are off; give them a weight at construction so the info "
                             "dictionary keeps the same keys")
        self._reward_weights.update(weights)

    @property
    def reward_keys(self):
        """The keys of the reward info dictionary."""
        active = [k for k in OPTIONAL_REWARD_TERMS if self._reward_weights[k] != 0.]
        return list(BASE_REWARD_TERMS) + ["termination"] + active

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
        self._last_contacts[idx] = False
        self._feet_air_time[idx] = 0.0
        self._foot_air_time[idx] = 0.0
        self._foot_last_air_time[idx] = 0.0
        self._foot_contact_time[idx] = 0.0
        self._foot_first_contact[idx] = False
        self._foot_contact[idx] = False
        self._foot_velocities[idx] = 0.0
        self._last_foot_positions[idx] = self._foot_pos()[idx]

        self._command_generator.reset(idx, self._heading()[idx])

    def _step_finalize(self):
        super()._step_finalize()
        self._command_generator.step(self._all_envs, self._heading())

    # ------------------------------------------------------------------
    # Reward (QuadrupedIsaac.reward)
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        lin_vel = self._base_lin_vel()
        ang_vel = self._base_ang_vel()
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        torque = self._joint_torque()
        # The raw policy action, as Isaac's reward sees it (the action this
        # method is handed is the preprocessed position target).
        actions = self._actions
        touch = self._floor_contacts()
        self._contacts_now = self._foot_contacts(touch)
        dt = self.dt
        w = self._reward_weights

        r = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(lin_vel[:, :2]) * w["tracking_lin_vel"] * dt,
            "tracking_ang_vel": self._reward_tracking_ang_vel(ang_vel[:, 2]) * w["tracking_ang_vel"] * dt,
            "lin_vel_z": torch.square(lin_vel[:, 2]) * w["lin_vel_z"] * dt,
            "ang_vel_xy": torch.sum(torch.square(ang_vel[:, :2]), dim=1) * w["ang_vel_xy"] * dt,
            "torques": torch.sum(torch.square(torque), dim=1) * w["torques"] * dt,
            "joint_acc": torch.sum(torch.square((self._last_joint_vel - joint_vel) / dt), dim=1) * w["joint_acc"] * dt,
            "feet_air_time": self._reward_feet_air_time() * w["feet_air_time"] * dt,
            "collision": self._lower_body_contacts(touch).float() * w["collision"] * dt,
            "action_rate": torch.sum(torch.square(self._last_actions - actions), dim=1) * w["action_rate"] * dt,
            "joint_pos_limits": self._reward_joint_pos_limits(joint_pos) * w["joint_pos_limits"] * dt,
        }

        active = [k for k in OPTIONAL_REWARD_TERMS if w[k] != 0.]
        if active:
            if any(k in FOOT_REWARD_TERMS for k in active):
                self._update_foot_state()
            for name in active:
                r[name] = getattr(self, f"_reward_{name}")(joint_pos, joint_vel, torque) * w[name] * dt

        r["termination"] = absorbing.float() * w["termination"] * dt

        total = sum(r.values())
        # A diverged world produces inf/NaN here; it is terminated, and its
        # reward must be finite so it does not poison the batch through GAE.
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        self._reward_info = r
        self._last_actions = actions.clone()
        self._last_joint_vel = joint_vel.clone()
        return total

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        commands = self._command_generator.commands
        err = torch.sum(torch.square(commands[:, :2] - lin_vel_xy), dim=1)
        std = (self._tracking_stds["lin_vel_slope"] * torch.norm(commands[:, :2], dim=1)) \
            .clamp(min=self._tracking_stds["lin_vel"])
        return torch.exp(-err / std ** 2)

    def _reward_tracking_ang_vel(self, ang_vel_z):
        commands = self._command_generator.commands
        err = torch.square(commands[:, 2] - ang_vel_z)
        std = (self._tracking_stds["ang_vel_slope"] * torch.abs(commands[:, 2])) \
            .clamp(min=self._tracking_stds["ang_vel"])
        return torch.exp(-err / std ** 2)

    def _reward_joint_pos_limits(self, joint_pos):
        below = (self._soft_joint_lower - joint_pos).clamp(min=0.0)
        above = (joint_pos - self._soft_joint_upper).clamp(min=0.0)
        return torch.sum(below + above, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps, paid on the first contact after a swing.
        contact = self._contacts_now
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += self.dt
        rew = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1)
        rew *= torch.norm(self._command_generator.commands[:, :2], dim=1) > 0.1
        self._feet_air_time *= ~contact_filt
        return rew

    # optional terms ----------------------------------------------------

    def _reward_flat_orientation(self, joint_pos, joint_vel, torque):
        return torch.sum(torch.square(self._projected_gravity()[:, :2]), dim=1)

    def _reward_joint_vel_limits(self, joint_pos, joint_vel, torque):
        soft = self._joint_max_vel * self._reward_params["joint_vel_limits_soft_ratio"]
        return torch.sum((torch.abs(joint_vel) - soft).clip(min=0., max=1.), dim=1)

    def _reward_power_draw(self, joint_pos, joint_vel, torque):
        return torch.sum(torch.abs(torque * joint_vel), dim=1)

    def _reward_similar_to_default(self, joint_pos, joint_vel, torque):
        return torch.sum(torch.abs(joint_pos - self._default_joint_pos), dim=1)

    def _reward_stand_still_deviation(self, joint_pos, joint_vel, torque):
        return self._reward_similar_to_default(joint_pos, joint_vel, torque) * self._is_standing_command()

    def _reward_base_height(self, joint_pos, joint_vel, torque):
        z = self._read_data("base_pos")[:, 2]
        return torch.square(z - self._reward_params["base_height_target"])

    def _reward_feet_air_time_high(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        rew = torch.sum((self._foot_last_air_time - p["air_time_threshold_high"]) * self._foot_first_contact, dim=1)
        return rew * (self._command_norm() > max(p["command_threshold"], p["low_speed_threshold"]))

    def _reward_feet_air_time_low(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        command_norm = self._command_norm()
        alpha = (command_norm / p["low_speed_threshold"]).clamp(0., 1.)
        threshold = p["air_time_threshold_low"] + alpha * (p["air_time_threshold_high"] - p["air_time_threshold_low"])
        rew = torch.sum((self._foot_last_air_time - threshold.unsqueeze(1)) * self._foot_first_contact, dim=1)
        rew = rew * (command_norm > p["command_threshold"])
        return rew * (command_norm < p["low_speed_threshold"])

    def _reward_feet_air_time_symmetry(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        mean_air_time = self._foot_last_air_time.mean(dim=1, keepdim=True)
        deviation = torch.square(self._foot_last_air_time - mean_air_time)
        per_foot = 1. - torch.exp(-deviation / p["air_time_symmetry_std"] ** 2)
        rew = torch.sum(per_foot * self._foot_first_contact, dim=1)
        return rew * (self._command_norm() > p["command_threshold"])

    def _reward_feet_clearance(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        command_norm = self._command_norm()
        rew = self._foot_clearance_value(p["clearance_target"], p["clearance_std"])
        rew = rew * (command_norm > p["command_threshold"])
        return rew * (command_norm < p["low_speed_threshold"])

    def _reward_feet_clearance_lateral(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        rew = self._foot_clearance_value(p["clearance_lateral_target"], p["clearance_lateral_std"])
        lateral = torch.abs(self._command_generator.commands[:, 1])
        return rew * (lateral > p["clearance_lateral_command_threshold"])

    def _reward_feet_slide(self, joint_pos, joint_vel, torque):
        return torch.sum(self._foot_velocities[:, :, :2].norm(dim=-1) * self._foot_contact, dim=1)

    def _reward_feet_slide_low(self, joint_pos, joint_vel, torque):
        slide = self._reward_feet_slide(joint_pos, joint_vel, torque)
        return slide * (self._command_norm() < self._reward_params["low_speed_threshold"])

    def _reward_feet_z_velocity(self, joint_pos, joint_vel, torque):
        return torch.sum(torch.square(self._foot_velocities[:, :, 2]) * self._foot_contact, dim=1)

    def _reward_long_contact(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        excess = (self._foot_contact_time - p["long_contact_threshold"]).clamp(min=0., max=p["long_contact_ramp_cap"])
        penalty = excess.pow(p["long_contact_ramp_power"]).sum(dim=1)
        return penalty * (self._command_norm() > p["command_threshold"])

    def _reward_stand_still_short_contact(self, joint_pos, joint_vel, torque):
        p = self._reward_params
        command_norm = self._command_norm()
        shortfall = (p["stand_still_contact_target"] - self._foot_contact_time) \
            .clamp(min=0., max=p["stand_still_ramp_cap"])
        penalty = shortfall.pow(p["stand_still_ramp_power"]).sum(dim=1)
        penalty = penalty * (1. - command_norm / p["command_threshold"]).clamp(min=0., max=1.)
        return penalty * self._is_standing_command()

    # helpers -----------------------------------------------------------

    def _update_foot_state(self):
        """Per-foot contact, air time, contact time, position and velocity."""
        contact = self._contacts_now
        self._foot_first_contact = contact & (self._foot_air_time > 0.)
        self._foot_last_air_time = torch.where(self._foot_first_contact, self._foot_air_time, self._foot_last_air_time)
        self._foot_air_time = torch.where(contact, torch.zeros_like(self._foot_air_time), self._foot_air_time + self.dt)
        self._foot_contact_time = torch.where(contact, self._foot_contact_time + self.dt,
                                              torch.zeros_like(self._foot_contact_time))
        self._foot_contact = contact

        pos = self._foot_pos()
        self._foot_velocities = (pos - self._last_foot_positions) / self.dt
        self._last_foot_positions = pos.clone()
        self._foot_positions = pos

    def _foot_clearance_value(self, target, std):
        """How close every airborne foot is to the target height, summed over the feet."""
        height = self._foot_positions[:, :, 2]
        airborne = ~self._foot_contact
        return torch.sum(torch.exp(-torch.square(height - target) / std ** 2) * airborne, dim=1)

    def _command_norm(self):
        return torch.norm(self._command_generator.commands, dim=1)

    def _is_standing_command(self):
        return self._command_norm() < self._reward_params["command_threshold"]

    # ------------------------------------------------------------------
    # Termination (Go2Isaac.is_absorbing: the trunk touched the ground)
    # ------------------------------------------------------------------

    def is_absorbing(self, obs):
        return self._trunk_contact() | ~self._is_finite()

    def _create_info_dictionary(self, obs):
        info = dict(self._reward_info)
        commands = self._command_generator.commands
        info["command_x"] = commands[:, 0]
        info["command_y"] = commands[:, 1]
        info["command_yaw"] = commands[:, 2]
        return info
