import numpy as np
import torch

from mushroom_rl.core.spaces import Box

from .base import H2Base


class H2Walk(H2Base):
    """
    Velocity tracking task for the Unitree H2.

    Same skeleton as Go2Walk (Rudin-style reward, every term per second and
    multiplied by dt, total clamped at zero), with the terms a biped needs
    that a quadruped does not:

    - a gait clock. Each environment carries a phase that advances by
      dt / gait_period per step; the policy observes sin and cos of it, and
      a contact-schedule reward pays for feet that are on the ground during
      their stance window and in the air during their swing window. The
      two feet are half a period apart. The stance fraction of the period
      is the duty factor: above 0.5 there is always a foot on the ground
      (walking), below 0.5 both feet are in the air part of the time
      (running). This is what separates H2Walk from H2Run, not the speed.
      With a zero command the schedule is "both feet down", so standing
      needs no separate gait logic;
    - upright and height terms (projected gravity, pelvis height);
    - swing foot clearance, foot slip, and a minimum lateral distance
      between the feet;
    - a posture term keeping hip roll/yaw, waist and arms near default.

    Contact is detected from the foot sole height, as the warp backend does
    not expose per-body contact forces.

    """

    _REWARD_KEYS = (
        "alive",
        "tracking_lin_vel",
        "tracking_ang_vel",
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "base_height",
        "torques",
        "joint_acc",
        "action_rate",
        "joint_pos_limits",
        "joint_deviation",
        "feet_air_time",
        "feet_slip",
        "feet_clearance",
        "feet_distance",
        "gait",
    )

    N_FEET = 2
    GAIT_OFFSETS = (0.0, 0.5)  # left, right; half a period apart

    def __init__(
        self,
        num_envs,
        lin_vel_x_range=(-0.6, 1.0),
        lin_vel_y_range=(-0.4, 0.4),
        heading_range=(-np.pi, np.pi),
        command_deadband=0.2,
        command_resample_interval=500,
        tracking_sigma=0.25,
        gait_period=0.8,
        duty_factor=0.6,
        foot_contact_height=0.015,
        feet_clearance_target=0.08,
        feet_air_time_threshold=0.4,
        feet_min_distance=0.18,
        base_height_target=0.96,
        alive_weight=2.0,
        tracking_lin_vel_weight=1.0,
        tracking_ang_vel_weight=0.5,
        lin_vel_z_weight=2.0,
        ang_vel_xy_weight=0.05,
        orientation_weight=1.0,
        base_height_weight=10.0,
        torque_weight=1e-6,
        joint_acc_weight=2.5e-7,
        action_rate_weight=0.01,
        joint_limit_weight=5.0,
        joint_deviation_weight=0.1,
        feet_air_time_weight=0.5,
        feet_slip_weight=0.2,
        feet_clearance_weight=2.0,
        feet_distance_weight=2.0,
        gait_weight=0.5,
        obs_noise=True,
        **kwargs,
    ):
        """
        Constructor.

        Args:
            lin_vel_x_range (tuple): sampling range of the forward velocity
                command, in m/s;
            lin_vel_y_range (tuple): sampling range of the lateral velocity
                command, in m/s;
            heading_range (tuple): sampling range of the target heading, in
                radians. The yaw rate command is derived from the heading
                error each step;
            command_deadband (float): commands with planar norm below this
                value are set to zero, so the robot also learns to stand;
            command_resample_interval (int): mean number of steps between
                command resamples within an episode;
            tracking_sigma (float): width of the exponential tracking kernel;
            gait_period (float): duration of one gait cycle, in seconds;
            duty_factor (float): fraction of the cycle each foot spends in
                stance. > 0.5 walks, < 0.5 runs;
            foot_contact_height (float): sole height below which a foot
                counts as in contact, in metres;
            feet_clearance_target (float): sole height a swing foot is pulled
                towards, in metres;
            feet_air_time_threshold (float): swing duration, in seconds,
                above which the air-time term pays out;
            feet_min_distance (float): planar distance between the feet below
                which a penalty applies, in metres;
            base_height_target (float): pelvis height the height term pulls
                towards, in metres. The keyframe stands at 1.01 and settles
                around 0.96 under load;
            alive_weight (float): reward per second for every step that does
                not terminate. Makes surviving worth more than falling; without
                it the penalties outweigh the positive terms early in training,
                the total is clamped to zero, and there is no learning signal;
            *_weight (float): weights of the reward terms, per second. Every
                term is multiplied by dt. The penalty weights start small on
                purpose: raise them once the robot walks, not before;
            obs_noise (bool): whether to add uniform noise to the observation.

        """
        self._lin_vel_x_range = lin_vel_x_range
        self._lin_vel_y_range = lin_vel_y_range
        self._heading_range = heading_range
        self._command_deadband = command_deadband
        self._command_resample_interval = command_resample_interval
        self._tracking_sigma = tracking_sigma
        self._gait_period = gait_period
        self._duty_factor = duty_factor
        self._foot_contact_height = foot_contact_height
        self._feet_clearance_target = feet_clearance_target
        self._feet_air_time_threshold = feet_air_time_threshold
        self._feet_min_distance = feet_min_distance
        self._base_height_target = base_height_target
        self._obs_noise = obs_noise

        self._w_alive = alive_weight
        self._w_tracking_lin = tracking_lin_vel_weight
        self._w_tracking_ang = tracking_ang_vel_weight
        self._w_lin_vel_z = lin_vel_z_weight
        self._w_ang_vel_xy = ang_vel_xy_weight
        self._w_orientation = orientation_weight
        self._w_base_height = base_height_weight
        self._w_torque = torque_weight
        self._w_joint_acc = joint_acc_weight
        self._w_action_rate = action_rate_weight
        self._w_joint_limit = joint_limit_weight
        self._w_joint_deviation = joint_deviation_weight
        self._w_feet_air_time = feet_air_time_weight
        self._w_feet_slip = feet_slip_weight
        self._w_feet_clearance = feet_clearance_weight
        self._w_feet_distance = feet_distance_weight
        self._w_gait = gait_weight

        super().__init__(num_envs, **kwargs)

        zero = torch.zeros(num_envs, device=self._device)
        self._reward_info = {k: zero for k in self._REWARD_KEYS}

    # ------------------------------------------------------------------
    # Observation layout
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        # Sets device, default pose, indices, slices and the action space.
        mdp_info = super()._modify_mdp_info(mdp_info)
        dev = self._device
        n_feet = self.N_FEET

        # Task state. Allocated here rather than after super().__init__()
        # because _create_observation may run before the constructor returns.
        # Commands are [vx, vy, yaw_rate, heading]; the yaw rate is recomputed
        # from the heading error every step rather than sampled directly.
        self._commands = torch.zeros(self._num_envs, 4, device=dev)
        self._last_actions = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._last_joint_vel = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._feet_air_time = torch.zeros(self._num_envs, n_feet, device=dev)
        self._last_contacts = torch.zeros(
            self._num_envs, n_feet, dtype=torch.bool, device=dev
        )
        self._last_foot_pos = torch.zeros(self._num_envs, n_feet, 3, device=dev)
        self._gait_phase = torch.zeros(self._num_envs, device=dev)
        self._gait_offsets = torch.as_tensor(
            self.GAIT_OFFSETS, dtype=torch.float32, device=dev
        )

        # Appended in _create_observation, in this order. Bounds are numpy,
        # as the observation helper requires.
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        self.obs_helper.add_obs("commands", 3, -1.0, 1.0)
        self.obs_helper.add_obs(
            "actions",
            self._n_joints,
            mdp_info.action_space.low,
            mdp_info.action_space.high,
        )
        self.obs_helper.add_obs("gait_phase", 2, -1.0, 1.0)

        norm_np = self._get_obs_normalization_vec()
        noise_np = self._get_noise_scale_vec()
        self._normalization_vec = torch.as_tensor(
            norm_np, dtype=torch.float32, device=dev
        )
        self._noise_scale_vec = torch.as_tensor(
            noise_np, dtype=torch.float32, device=dev
        )

        # Observation space after the default pose shift, the fixed scaling
        # and the observation noise, matching what the policy actually sees.
        obs_low, obs_high = self.obs_helper.get_obs_limits()
        obs_low = obs_low.astype(np.float64, copy=True)
        obs_high = obs_high.astype(np.float64, copy=True)
        default_joint_np = self._default_joint_pos.cpu().numpy()
        obs_low[self._joint_pos_slice] -= default_joint_np
        obs_high[self._joint_pos_slice] -= default_joint_np
        obs_low = obs_low * norm_np - noise_np
        obs_high = obs_high * norm_np + noise_np
        mdp_info.observation_space = Box(obs_low, obs_high)

        return mdp_info

    def _get_obs_normalization_vec(self):
        v = np.ones(self.obs_helper.obs_low.shape[0])
        v[self._lin_vel_slice] = 2.0
        v[self._ang_vel_slice] = 0.25
        v[self._joint_pos_slice] = 1.0
        v[self._joint_vel_slice] = 0.05
        v[self.obs_helper.obs_idx_map["projected_gravity"]] = 1.0
        cmd = self.obs_helper.obs_idx_map["commands"]
        v[cmd[0:2]] = 2.0
        v[cmd[2]] = 0.25
        v[self.obs_helper.obs_idx_map["actions"]] = 1.0
        v[self.obs_helper.obs_idx_map["gait_phase"]] = 1.0
        return v

    def _get_noise_scale_vec(self):
        v = np.zeros(self.obs_helper.obs_low.shape[0])
        if not self._obs_noise:
            return v
        v[self._lin_vel_slice] = 0.1 * 2.0
        v[self._ang_vel_slice] = 0.2 * 0.25
        v[self._joint_pos_slice] = 0.01 * 1.0
        v[self._joint_vel_slice] = 1.5 * 0.05
        v[self.obs_helper.obs_idx_map["projected_gravity"]] = 0.05
        return v

    def _gait_phase_obs(self):
        two_pi = 2.0 * np.pi
        return torch.stack(
            [
                torch.sin(two_pi * self._gait_phase),
                torch.cos(two_pi * self._gait_phase),
            ],
            dim=1,
        )

    def _create_observation(self, obs):
        # Joint positions are kept absolute here because reward() reads them
        # for the limit penalty; the default offset is applied in
        # _modify_observation, after the reward has been computed.
        return torch.cat(
            [
                obs,
                self._projected_gravity(),
                self._commands[:, :3],
                self._actions,
                self._gait_phase_obs(),
            ],
            dim=1,
        )

    def _modify_observation(self, obs):
        obs = obs.clone()
        obs[:, self._joint_pos_slice] -= self._default_joint_pos
        obs *= self._normalization_vec
        if self._obs_noise:
            obs += (2.0 * torch.rand_like(obs) - 1.0) * self._noise_scale_vec
        # A diverged simulation is terminated by _is_finite, but its
        # observation still enters the dataset, so it must be finite.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(obs, min=-100.0, max=100.0)

    # ------------------------------------------------------------------
    # Commands and gait clock
    # ------------------------------------------------------------------

    def _resample_commands(self, env_indices):
        n = len(env_indices)
        if n == 0:
            return

        def uniform(lo, hi):
            return torch.rand(n, device=self._device) * (hi - lo) + lo

        self._commands[env_indices, 0] = uniform(*self._lin_vel_x_range)
        self._commands[env_indices, 1] = uniform(*self._lin_vel_y_range)
        self._commands[env_indices, 3] = uniform(*self._heading_range)

        # Small commands are zeroed so the policy also learns to stand still.
        small = self._commands[env_indices, :2].norm(dim=1) < self._command_deadband
        self._commands[env_indices[small], :2] = 0.0

    def _update_yaw_command(self):
        # Teleoperation sets this flag to command a yaw rate directly instead
        # of deriving it from the heading error (same convention as Go2Walk).
        if getattr(self, "_direct_yaw", False):
            return
        heading_error = self._wrap_to_pi(self._commands[:, 3] - self._heading())
        self._commands[:, 2] = torch.clamp(0.5 * heading_error, -1.0, 1.0)

    def _moving(self):
        """Environments whose planar velocity command is non-zero."""
        return self._commands[:, :2].norm(dim=1) > 0.1

    def _desired_stance(self):
        """
        Contact schedule from the gait clock. (num_envs, 2) bool, True where
        the foot should be on the ground. With a zero command both feet
        should be down.

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

        self._resample_commands(idx)
        self._update_yaw_command()

    def _step_finalize(self):
        super()._step_finalize()

        do_resample = (
            torch.rand(self._num_envs, device=self._device)
            < 1.0 / self._command_resample_interval
        )
        do_resample &= self._episode_length > 50
        self._resample_commands(torch.nonzero(do_resample, as_tuple=True)[0])

        self._update_yaw_command()
        self._gait_phase = (self._gait_phase + self.dt / self._gait_period) % 1.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        # `action` here is what _preprocess_action returned: position targets
        # for all 31 actuators. The action-rate term is on the policy output
        # itself, which H2Base keeps in self._actions (n_joints wide).
        policy_action = self._actions
        lin_vel = next_obs[:, self._lin_vel_slice]
        ang_vel = next_obs[:, self._ang_vel_slice]
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        torque = self._joint_torque()
        gravity = self._projected_gravity()
        foot_pos = self._foot_pos()
        foot_height = foot_pos[:, :, 2] - self._foot_half_height
        contact = foot_height < self._foot_contact_height
        dt = self.dt

        r = {
            "alive": (~absorbing).float() * self._w_alive * dt,
            "tracking_lin_vel": self._reward_tracking_lin_vel(lin_vel[:, :2])
            * self._w_tracking_lin
            * dt,
            "tracking_ang_vel": self._reward_tracking_ang_vel(ang_vel[:, 2])
            * self._w_tracking_ang
            * dt,
            "lin_vel_z": lin_vel[:, 2] ** 2 * -self._w_lin_vel_z * dt,
            "ang_vel_xy": (ang_vel[:, :2] ** 2).sum(dim=1) * -self._w_ang_vel_xy * dt,
            "orientation": (gravity[:, :2] ** 2).sum(dim=1) * -self._w_orientation * dt,
            "base_height": self._reward_base_height() * -self._w_base_height * dt,
            "torques": (torque**2).sum(dim=1) * -self._w_torque * dt,
            "joint_acc": self._reward_joint_acc(joint_vel) * -self._w_joint_acc * dt,
            "action_rate": ((self._last_actions - policy_action) ** 2).sum(dim=1)
            * -self._w_action_rate
            * dt,
            "joint_pos_limits": self._reward_joint_pos_limits(joint_pos)
            * -self._w_joint_limit
            * dt,
            "joint_deviation": self._reward_joint_deviation(joint_pos)
            * -self._w_joint_deviation
            * dt,
            "feet_air_time": self._reward_feet_air_time(contact)
            * self._w_feet_air_time
            * dt,
            "feet_slip": self._reward_feet_slip(foot_pos, contact)
            * -self._w_feet_slip
            * dt,
            "feet_clearance": self._reward_feet_clearance(foot_height, contact)
            * -self._w_feet_clearance
            * dt,
            "feet_distance": self._reward_feet_distance(foot_pos)
            * -self._w_feet_distance
            * dt,
            "gait": self._reward_gait(contact) * self._w_gait * dt,
        }
        self._reward_info = r

        total = torch.clamp(sum(r.values()), min=0.0)
        # A diverged simulation produces inf/NaN here through the velocity
        # terms; that environment is terminated by _is_finite, and its reward
        # must be finite so it does not poison the whole batch through GAE.
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        self._last_actions = policy_action.clone()
        self._last_joint_vel = joint_vel.clone()
        self._last_foot_pos = foot_pos.clone()

        return total

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        err = ((self._commands[:, :2] - lin_vel_xy) ** 2).sum(dim=1)
        return torch.exp(-err / self._tracking_sigma)

    def _reward_tracking_ang_vel(self, ang_vel_z):
        err = (self._commands[:, 2] - ang_vel_z) ** 2
        return torch.exp(-err / self._tracking_sigma)

    def _reward_base_height(self):
        return (self._base_height() - self._base_height_target) ** 2

    def _reward_joint_acc(self, joint_vel):
        return (((self._last_joint_vel - joint_vel) / self.dt) ** 2).sum(dim=1)

    def _reward_joint_pos_limits(self, joint_pos):
        below = torch.clamp(self._soft_joint_lower - joint_pos, min=0.0)
        above = torch.clamp(joint_pos - self._soft_joint_upper, min=0.0)
        return (below + above).sum(dim=1)

    def _reward_joint_deviation(self, joint_pos):
        dev = (
            joint_pos[:, self._posture_idx] - self._default_joint_pos[self._posture_idx]
        )
        return dev.abs().sum(dim=1)

    def _reward_feet_air_time(self, contact):
        """
        Reward long steps: on the first contact after a swing, pay out the
        swing duration minus the threshold. Only while a velocity command
        is active, so standing is not paid for lifting feet.

        """
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += self.dt
        rew = (
            (self._feet_air_time - self._feet_air_time_threshold) * first_contact
        ).sum(dim=1)
        # Height-based contact flickers; a short swing must not turn this
        # into a penalty.
        rew = torch.clamp(rew, min=0.0)
        rew *= self._moving()
        self._feet_air_time *= ~contact_filt
        return rew

    def _reward_feet_slip(self, foot_pos, contact):
        """Planar speed squared of the feet that are in contact."""
        vel = (foot_pos - self._last_foot_pos) / self.dt
        speed2 = (vel[:, :, :2] ** 2).sum(dim=2)
        return (speed2 * contact).sum(dim=1)

    def _reward_feet_clearance(self, foot_height, contact):
        """Squared distance of the swing feet from the clearance height."""
        err = (foot_height - self._feet_clearance_target) ** 2
        return (err * ~contact).sum(dim=1) * self._moving()

    def _reward_feet_distance(self, foot_pos):
        """Penalty for feet closer together than feet_min_distance."""
        d = (foot_pos[:, 0, :2] - foot_pos[:, 1, :2]).norm(dim=1)
        return torch.clamp(self._feet_min_distance - d, min=0.0)

    def _reward_gait(self, contact):
        """Fraction of feet whose contact state matches the gait schedule."""
        return (contact == self._desired_stance()).float().mean(dim=1)

    def is_absorbing(self, obs):
        return self._terminate_when_unhealthy & ~self._is_healthy(obs)

    def _create_info_dictionary(self, obs):
        info = dict(self._reward_info)
        info["command_x"] = self._commands[:, 0]
        info["command_y"] = self._commands[:, 1]
        info["command_yaw"] = self._commands[:, 2]
        return info
