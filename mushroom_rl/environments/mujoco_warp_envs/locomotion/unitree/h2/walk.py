"""
Velocity tracking task for the Unitree H2: Unitree's own velocity task
(unitree_rl_mjlab, src/tasks/velocity) on MuJoCo Warp.

The reward has Unitree's terms, weights and parameters under their names:
velocity tracking, a pose term whose tolerance depends on the commanded
speed (tight when standing, loose on the joints a stride needs when walking
or running), a contact schedule from a gait clock, foot clearance, slip and
landing force, torso orientation and rates, angular momentum, joint limits
and accelerations, action rate, self collisions, standing still, and a
penalty for falling. One policy walks, runs and stands.

The observation is what Unitree's actor reads (pelvis angular velocity,
projected gravity, command, gait phase, joint positions and velocities, last
action) followed by what only their critic reads (pelvis linear velocity,
foot heights, air times, contacts and contact forces); the training script
hides the second part from the policy. Nothing is scaled here: the agent
standardizes its input.

"""

import numpy as np
import torch
import warp as wp

from mushroom_rl.core import register_env
from mushroom_rl.core.spaces import Box

from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands

from .base import H2Base, with_pushes
from .params import (
    REWARD_TERMS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_REWARD_PARAMS,
    POSE_STD_STANDING,
    POSE_STD_WALKING,
    POSE_STD_RUNNING,
    WALK_COMMAND_GENERATOR,
    WALK_PUSHES,
    OBS_NOISE,
)


@register_env
class H2Walk(H2Base):
    """
    Args:
        num_envs (int): number of worlds;
        reward_weights (dict, None): overrides of DEFAULT_REWARD_WEIGHTS, keyed
            like the info dictionary. Every weight carries its sign;
        reward_params (dict, None): overrides of DEFAULT_REWARD_PARAMS;
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
        self._foot_contact = torch.zeros(W, n_feet, dtype=torch.bool, device=dev)
        self._last_foot_contact = torch.zeros(W, n_feet, dtype=torch.bool, device=dev)
        self._foot_force = torch.zeros(W, n_feet, device=dev)
        self._foot_air_time = torch.zeros(W, n_feet, device=dev)
        self._self_collisions = torch.zeros(W, device=dev)
        self._gait_offsets = torch.as_tensor(self._reward_params["gait_offsets"], dtype=torch.float32, device=dev)

        self._pose_std_standing = self._per_joint(POSE_STD_STANDING)
        self._pose_std_walking = self._per_joint(POSE_STD_WALKING)
        self._pose_std_running = self._per_joint(POSE_STD_RUNNING)

        # Appended in _create_observation, in this order.
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        bounds = self._command_generator_bounds()
        self.obs_helper.add_obs("commands", 3, -bounds, bounds)
        self.obs_helper.add_obs("gait_phase", 2, -1.0, 1.0)
        self.obs_helper.add_obs("actions", self._n_joints, mdp_info.action_space.low, mdp_info.action_space.high)
        self.obs_helper.add_obs("foot_height", n_feet, -0.1, 1.5)
        self.obs_helper.add_obs("foot_air_time", n_feet, 0.0, mdp_info.horizon * mdp_info.dt)
        self.obs_helper.add_obs("foot_contact", n_feet, 0.0, 1.0)
        self.obs_helper.add_obs("foot_force", n_feet, 0.0, 10.0)
        spec = self.randomization_obs_spec()
        for name in self._observed_randomization:
            length, low, high = spec[name]
            self.obs_helper.add_obs(name, length, low, high)

        noise_np = self._get_noise_scale_vec()
        self._noise_scale_vec = torch.as_tensor(noise_np, dtype=torch.float32, device=dev)

        # Observation space after the default pose shift and the noise,
        # matching what the agent sees.
        obs_low, obs_high = self.obs_helper.get_obs_limits()
        obs_low = obs_low.astype(np.float64, copy=True)
        obs_high = obs_high.astype(np.float64, copy=True)
        default_joint_np = self._default_joint_pos.cpu().numpy()
        obs_low[self._joint_pos_slice] -= default_joint_np
        obs_high[self._joint_pos_slice] -= default_joint_np
        mdp_info.observation_space = Box(obs_low - noise_np, obs_high + noise_np)

        return mdp_info

    def _per_joint(self, table):
        """The value of every controlled joint in a (substring, value) table; first match wins."""
        values = [next(v for key, v in table if key in joint) for joint in self._controlled_joints]
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    def _command_generator_bounds(self):
        ranges = self._command_generator._max_command_ranges
        return np.array([max(abs(b) for b in ranges[name]) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])

    def _get_noise_scale_vec(self):
        m = self.obs_helper.obs_idx_map
        v = np.zeros(self.obs_helper.obs_low.shape[0])
        if not self._obs_noise:
            return v
        v[self._ang_vel_slice] = OBS_NOISE["base_ang_vel"]
        v[self._joint_pos_slice] = OBS_NOISE["joint_pos"]
        v[self._joint_vel_slice] = OBS_NOISE["joint_vel"]
        v[m["projected_gravity"]] = OBS_NOISE["projected_gravity"]
        return v

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _gait_phase_obs(self):
        """Sin and cos of the gait clock, (num_envs, 2); zero in a world commanded to stand."""
        period = self._reward_params["gait_period"]
        angle = 2.0 * np.pi * ((self._episode_length * self.dt) % period) / period
        phase = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
        standing = self._command_generator.commands.norm(dim=1) < self._reward_params["command_threshold"]
        return torch.where(standing.unsqueeze(1), torch.zeros_like(phase), phase)

    def _create_observation(self, obs):
        parts = [
            obs,
            self._projected_gravity(),
            self._command_generator.commands,
            self._gait_phase_obs(),
            self._actions,
            self._foot_pos()[:, :, 2],
            self._foot_air_time,
            self._foot_contact.float(),
            torch.log1p(self._foot_force),
        ]
        parts += [self.randomization_obs_value(name) for name in self._observed_randomization]
        return torch.cat(parts, dim=1)

    def _modify_observation(self, obs):
        obs = obs.clone()
        # The policy reads the joint angle the way a miscalibrated encoder
        # would, offset from the truth by the same amount the controller
        # targets are shifted by (see H2Base._preprocess_action).
        offset = self._randomizer.position_offset[:, self._ctrl_idx]
        obs[:, self._joint_pos_slice] -= self._default_joint_pos + offset
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
        ``gait_phase``, ``actions``, ``foot_height``, ``foot_air_time``,
        ``foot_contact``, ``foot_force`` and any name in
        ``observed_randomization``.

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

        contact, force, _ = self._contacts()
        self._last_actions[idx] = 0.0
        self._foot_contact[idx] = contact[idx]
        self._last_foot_contact[idx] = contact[idx]
        self._foot_force[idx] = force[idx]
        self._foot_air_time[idx] = 0.0
        self._self_collisions[idx] = 0.0

        self._command_generator.reset(idx, self._heading()[idx])

    def _step_init(self, obs, action):
        self._self_collisions.zero_()

    def _simulation_post_step(self):
        # The contact state after the physics, before the observation is
        # built: the critic reads it, and so does the reward.
        self._foot_contact, self._foot_force, self_force = self._contacts()
        self._foot_air_time = torch.where(
            self._foot_contact, torch.zeros_like(self._foot_air_time), self._foot_air_time + self._step_dt
        )
        # Unitree counts the physics steps of the policy step in which the
        # robot hits itself. The contacts are read once per intermediate
        # step here and taken to hold for its physics steps.
        hit = self_force > self._reward_params["self_collision_force"]
        self._self_collisions += hit.float() * self._n_substeps

    def _step_finalize(self):
        super()._step_finalize()
        self._command_generator.step(self._all_envs, self._heading())

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        p = self._reward_params
        w = self._reward_weights
        dt = self.dt
        commands = self._command_generator.commands
        # Unitree's measure of the commanded speed: planar norm plus yaw rate.
        speed = commands[:, :2].norm(dim=1) + commands[:, 2].abs()
        moving = (speed > p["command_threshold"]).float()

        lin_vel = self._base_lin_vel()
        ang_vel = self._base_ang_vel()
        joint_dev = self._joint_pos() - self._default_joint_pos
        foot_height = self._foot_pos()[:, :, 2]
        foot_speed = self._foot_vel()[:, :, :2].norm(dim=2)
        contact = self._foot_contact
        landed = contact & ~self._last_foot_contact
        # The raw policy action: the action this method is handed is the
        # preprocessed position target of every actuator.
        actions = self._actions

        r = {
            "track_linear_velocity": self._reward_track_linear_velocity(commands, lin_vel),
            "track_angular_velocity": self._reward_track_angular_velocity(commands, ang_vel),
            "body_orientation_l2": torch.sum(torch.square(self._torso_projected_gravity()[:, :2]), dim=1),
            "pose": self._reward_pose(joint_dev, speed),
            "body_ang_vel": torch.sum(torch.square(self._torso_ang_vel()[:, :2]), dim=1),
            "angular_momentum": torch.sum(torch.square(self._angular_momentum()), dim=1),
            "joint_acc_l2": torch.sum(torch.square(self._joint_acc()), dim=1),
            "joint_pos_limits": self._reward_joint_pos_limits(),
            "action_rate_l2": torch.sum(torch.square(actions - self._last_actions), dim=1),
            "foot_gait": self._reward_foot_gait(contact) * moving,
            "foot_clearance": torch.sum((foot_height - p["clearance_height"]).abs() * foot_speed, dim=1) * moving,
            "foot_slip": torch.sum(torch.square(foot_speed) * contact, dim=1) * moving,
            "soft_landing": torch.sum(self._foot_force * landed, dim=1) * moving,
            "stand_still": torch.sum(torch.square(joint_dev), dim=1) * (1.0 - moving),
            "stand_still_joint_vel": torch.sum(torch.square(self._joint_vel()), dim=1) * (1.0 - moving),
            "stand_still_base_vel": (torch.sum(torch.square(lin_vel[:, :2]), dim=1)
                                     + torch.sum(torch.square(ang_vel), dim=1)) * (1.0 - moving),
            "stand_still_com": self._com_offset() * (1.0 - moving),
            "stand_still_feet": torch.sum((~contact).float(), dim=1) * (1.0 - moving),
            "self_collisions": self._self_collisions,
        }
        r = {name: value * w[name] * dt for name, value in r.items()}

        # A diverged world produces inf/NaN here, and huge finite values the
        # step before; it is terminated, and its reward must not poison the
        # batch through GAE. The fall penalty is added after the floor.
        floor = p["reward_floor"]
        total = torch.nan_to_num(sum(r.values()), nan=0.0, posinf=0.0, neginf=floor)
        total = torch.clamp(total, min=floor)
        r["is_terminated"] = absorbing.float() * w["is_terminated"] * dt
        total = total + r["is_terminated"]

        self._reward_info = r
        self._last_actions = actions.clone()
        self._last_foot_contact = contact.clone()
        return total

    def _com_offset(self):
        """Squared planar distance between the centre of mass and the middle of the foot capsules. (num_envs,)"""
        com = wp.to_torch(self._data_wp.subtree_com)[:, self._root_body, :2]
        feet = wp.to_torch(self._data_wp.geom_xpos)[:, self._feet_geom_ids.flatten(), :2].mean(dim=1)
        return torch.sum(torch.square(com - feet), dim=1)

    def _reward_track_linear_velocity(self, commands, lin_vel):
        err = torch.sum(torch.square(commands[:, :2] - lin_vel[:, :2]), dim=1) + 2.0 * torch.square(lin_vel[:, 2])
        return torch.exp(-err / self._reward_params["lin_vel_std"] ** 2)

    def _reward_track_angular_velocity(self, commands, ang_vel):
        err = torch.square(commands[:, 2] - ang_vel[:, 2]) + 0.05 * torch.sum(torch.square(ang_vel[:, :2]), dim=1)
        return torch.exp(-err / self._reward_params["ang_vel_std"] ** 2)

    def _reward_pose(self, joint_dev, speed):
        """Kernel on the distance from the default pose, with the tolerance of the commanded speed."""
        p = self._reward_params
        standing = (speed < p["command_threshold"]).unsqueeze(1)
        running = (speed >= p["running_threshold"]).unsqueeze(1)
        std = torch.where(
            standing, self._pose_std_standing, torch.where(running, self._pose_std_running, self._pose_std_walking)
        )
        return torch.exp(-torch.mean(torch.square(joint_dev / std), dim=1))

    def _reward_joint_pos_limits(self):
        joint_pos = self._joint_pos()
        below = (self._soft_joint_lower - joint_pos).clamp(min=0.0)
        above = (joint_pos - self._soft_joint_upper).clamp(min=0.0)
        return torch.sum(below + above, dim=1)

    def _reward_foot_gait(self, contact):
        """Fraction of feet whose contact state matches the schedule of the gait clock."""
        p = self._reward_params
        cycles = (self._episode_length * self.dt / p["gait_period"]).unsqueeze(1)
        stance = (cycles + self._gait_offsets) % 1.0 < p["gait_stance"]
        return (stance == contact).float().mean(dim=1)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_absorbing(self, obs):
        if not self._terminate_when_fallen:
            return ~self._is_finite()
        return self._is_fallen() | ~self._is_finite()

    def _create_info_dictionary(self, obs):
        info = dict(self._reward_info)
        commands = self._command_generator.commands
        info["command_x"] = commands[:, 0]
        info["command_y"] = commands[:, 1]
        info["command_yaw"] = commands[:, 2]
        return info
