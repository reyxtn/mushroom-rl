import mujoco
import numpy as np
import torch
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.utils.mujoco import ObservationType
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import (
    Randomizer,
    RandomizationParams,
)
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.utils import uniform

from .fetch_go2_model import ensure_go2_model

from .params import (
    LEG_JOINTS,
    ACTUATORS,
    FEET_GEOMS,
    FOOT_RADIUS,
    ROOT_BODY,
    FLOOR_GEOM,
    LOWER_BODY_BODIES,
    BATCHED_MODEL_FIELDS,
    RANDOMIZATION_OBS,
    DEFAULT_JOINT_POS,
    NOMINAL_P_GAIN,
    NOMINAL_D_GAIN,
)


class Go2Base(MuJoCoWarp):
    """
    Base constructor for the Go2 quadruped in MuJoCo Warp.

    The physics split defaults to new_isaac's QuadrupedIsaac: 5 ms physics
    steps, 4 of them per 20 ms policy step, the actuation delay counted in
    physics steps, a PD law with kp=20 / kd=0.5 around the Isaac default pose.

    """

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        action_scale=0.25,
        soft_joint_limit=0.9,
        domain_randomization=True,
        randomization_params=None,
        observed_randomization=(),
        max_delay_limit=0.02,
        default_joint_pos=DEFAULT_JOINT_POS,
        nominal_p_gain=NOMINAL_P_GAIN,
        nominal_d_gain=NOMINAL_D_GAIN,
        timestep=0.005,
        n_substeps=1,
        n_intermediate_steps=4,
        use_graph_capture=False,
        nconmax=None,
        njmax=256,
        scene="scene_mjx.xml",
        **viewer_params,
    ):
        """
        Args (the ones not obvious):
            max_delay_limit (float): the largest actuation delay, in seconds,
                the action history is sized for; ``max_delay`` can never be
                raised above it;
            default_joint_pos (tuple, None): the nominal pose the actions are
                relative to, in LEG_JOINTS order. None keeps the menagerie
                "home" keyframe;
            nominal_p_gain, nominal_d_gain (float, None): overwrite the
                actuator gains of the MJCF. None keeps the MJCF's;
            timestep, n_substeps, n_intermediate_steps: the physics split. The
                delay resolution is ``timestep * n_substeps``.

        """
        xml_path = str(ensure_go2_model(scene))

        unknown = set(observed_randomization) - set(RANDOMIZATION_OBS)
        if unknown:
            raise ValueError(f"unknown randomized observations: {sorted(unknown)}")

        observation_spec = [("base_vel", ROOT_BODY, ObservationType.BODY_VEL)]
        observation_spec += [(f"{j}_pos", j, ObservationType.JOINT_POS) for j in LEG_JOINTS]
        observation_spec += [(f"{j}_vel", j, ObservationType.JOINT_VEL) for j in LEG_JOINTS]
        additional_data_spec = [
            ("base_pos", ROOT_BODY, ObservationType.BODY_POS),
            ("base_rot", ROOT_BODY, ObservationType.BODY_ROT),
        ]

        if randomization_params is None:
            randomization_params = RandomizationParams()
        elif isinstance(randomization_params, dict):
            randomization_params = RandomizationParams(**randomization_params)

        # Everything _modify_mdp_info reads: it runs inside super().__init__().
        self._observed_randomization = tuple(observed_randomization)
        self._max_delay_limit = max_delay_limit
        self._n_substeps = n_substeps
        self._randomization_params = randomization_params
        self._action_scale = action_scale
        self._soft_joint_limit = soft_joint_limit
        self._domain_randomization = domain_randomization
        self._num_envs = num_envs
        self._n_joints = len(LEG_JOINTS)
        self._requested_default_joint_pos = default_joint_pos

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=list(ACTUATORS),
            additional_data_spec=additional_data_spec,
            timestep=timestep,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

        # Nominal gains, written into the CPU model before the re-put so the
        # batched model (and so the randomizer's nominal values) carry them.
        # Position servo: force = kp * ctrl - kp * q - kd * qd.
        if nominal_p_gain is not None:
            self._model.actuator_gainprm[:, 0] = nominal_p_gain
            self._model.actuator_biasprm[:, 1] = -nominal_p_gain
        if nominal_d_gain is not None:
            self._model.actuator_biasprm[:, 2] = -nominal_d_gain

        # Re-put the model with per-world copies of the randomized fields.
        # Dev's MuJoCoWarp shares model fields across worlds; nothing has
        # captured the first _model_wp yet (graphs are built lazily).
        self._model_wp = self._mj_warp.put_model(
            self._model, batch_sizes={f: num_envs for f in BATCHED_MODEL_FIELDS}
        )
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.ITERATIONS

        self._build_randomizer(self._model, self._act_jids)

        if self._randomizer.max_delay_steps > self._max_delay_steps_limit:
            raise ValueError(
                f"max_delay {self._randomizer.params['max_delay']} s exceeds the "
                f"buffer ceiling max_delay_limit {self._max_delay_limit} s"
            )

    def _modify_mdp_info(self, mdp_info):
        m = self._model

        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise ValueError(
                "Go2Base needs position-type actuators; the loaded scene uses "
                "torque motors. Use scene='scene_mjx.xml'."
            )
        if m.nu != self._n_joints:
            raise ValueError(f"model has {m.nu} actuators, expected {self._n_joints}")
        if not np.array_equal(np.asarray(self._action_indices), np.arange(m.nu)):
            raise ValueError("actuation_spec must list every actuator in model order")

        dev = wp.to_torch(self._data_wp.qpos).device
        self._device = dev

        # The actuators drive the leg joints in LEG_JOINTS order, which is
        # the action order; checked rather than assumed.
        act_jids = m.actuator_trnid[:, 0]
        self._act_jids = act_jids
        leg_jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in LEG_JOINTS]
        if min(leg_jids) < 0 or not np.array_equal(act_jids, np.asarray(leg_jids)):
            raise ValueError("the actuators do not drive LEG_JOINTS in order")
        qpos_adr = m.jnt_qposadr[act_jids]
        self._qpos_idx = torch.as_tensor(qpos_adr, device=dev, dtype=torch.long)
        self._dof_idx = torch.as_tensor(m.jnt_dofadr[act_jids], device=dev, dtype=torch.long)
        self._ctrl_idx = torch.arange(m.nu, device=dev)

        self._feet_geom_ids = torch.as_tensor(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FEET_GEOMS], device=dev
        )
        if (self._feet_geom_ids < 0).any():
            raise ValueError("foot geoms not found in the model")

        # Nominal standing pose: the requested joint angles, with the trunk
        # lifted so the lowest foot just touches the floor (spawning with the
        # feet in the floor makes the first steps explode).
        default_qpos = m.key_qpos[0].copy()
        if self._requested_default_joint_pos is not None:
            default_qpos[qpos_adr] = np.asarray(self._requested_default_joint_pos)
            d = mujoco.MjData(m)
            d.qpos[:] = default_qpos
            mujoco.mj_kinematics(m, d)
            lowest_foot = d.geom_xpos[self._feet_geom_ids.cpu().numpy(), 2].min()
            default_qpos[2] += FOOT_RADIUS + 0.002 - lowest_foot
        self._default_qpos = torch.as_tensor(default_qpos, dtype=torch.float32, device=dev)
        self._default_joint_pos = self._default_qpos[self._qpos_idx].clone()
        self._default_ctrl = self._default_joint_pos  # every actuator is controlled

        # Joint limits, and soft limits shrunk towards the midpoint.
        rng = torch.as_tensor(m.jnt_range[act_jids], dtype=torch.float32, device=dev)
        self._joint_lower = rng[:, 0]
        self._joint_upper = rng[:, 1]
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * self._soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        # Position targets are clamped to the actuator ctrlrange.
        ctrlrange = m.actuator_ctrlrange
        self._ctrl_lower = torch.as_tensor(ctrlrange[:, 0], dtype=torch.float32, device=dev)
        self._ctrl_upper = torch.as_tensor(ctrlrange[:, 1], dtype=torch.float32, device=dev)

        self._build_contact_groups(m)

        self._actions = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._episode_length = torch.zeros(self._num_envs, dtype=torch.long, device=dev)
        self._all_envs = torch.arange(self._num_envs, device=dev)

        # Contiguous slices of the observation, for use by subclasses.
        first_pos = self.obs_helper.obs_idx_map[f"{LEG_JOINTS[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{LEG_JOINTS[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{LEG_JOINTS[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{LEG_JOINTS[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        # BODY_VEL is ordered [angular(3), linear(3)], in the body frame.
        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        # The action is a joint position offset scaled by action_scale, so
        # the action space is the joint range expressed in those units.
        default_joint_np = default_qpos[qpos_adr]
        jnt_range_np = m.jnt_range[act_jids]
        action_low = (jnt_range_np[:, 0] - default_joint_np) / self._action_scale
        action_high = (jnt_range_np[:, 1] - default_joint_np) / self._action_scale
        mdp_info.action_space = Box(action_low, action_high)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _build_contact_groups(self, m):
        """
        Geom membership of the contact groups read from the solver's contact
        list: the feet, the trunk (Go2Isaac's "body") and the thighs and calves
        without their foot sphere (Go2Isaac's "lower_body").

        """
        dev = self._device
        self._floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        if self._floor_geom < 0:
            raise ValueError(f"floor geom {FLOOR_GEOM!r} not found in the scene")

        colliding = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        self._trunk_geom_ids = torch.as_tensor(
            np.nonzero(colliding & (m.geom_bodyid == trunk))[0], device=dev, dtype=torch.long
        )

        feet = set(self._feet_geom_ids.tolist())
        membership = np.zeros((len(LOWER_BODY_BODIES), m.ngeom), dtype=np.float32)
        for i, name in enumerate(LOWER_BODY_BODIES):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"body {name!r} not found in the model")
            for g in np.nonzero(colliding & (m.geom_bodyid == body))[0]:
                if g not in feet:
                    membership[i, g] = 1.0
        self._lower_body_membership = torch.as_tensor(membership, device=dev)

    def _build_randomizer(self, m, act_jids):
        self._step_dt = m.opt.timestep * self._n_substeps
        self._max_delay_steps_limit = int(self._max_delay_limit / self._step_dt + 0.5)
        dev = self._device
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        ancestors = []
        b = trunk
        while b >= 0:
            ancestors.append(int(b))
            b = int(m.body_parentid[b]) if b != 0 else -1

        self._model_views = {
            name: wp.to_torch(getattr(self._model_wp, name)) for name in BATCHED_MODEL_FIELDS
        }
        for name, view in self._model_views.items():
            if view.shape[0] != self._num_envs:
                raise RuntimeError(
                    f"model field {name} is not batched per world (shape {tuple(view.shape)}); "
                    "check the put_model re-put with batch_sizes in Go2Base.__init__"
                )
        layout = dict(
            trunk_body=trunk,
            trunk_ancestors=ancestors,
            foot_geoms=self._feet_geom_ids,
            actuators=torch.arange(m.nu, device=dev),
            dofs=torch.as_tensor(m.jnt_dofadr[act_jids], device=dev, dtype=torch.long),
            joints=torch.as_tensor(act_jids, device=dev, dtype=torch.long),
            controlled=self._ctrl_idx,
        )
        self._randomizer = Randomizer(
            self._num_envs,
            self._model_views,
            layout,
            self._default_ctrl,
            self._action_scale,
            step_dt=self._step_dt,
            params=self._randomization_params,
        )

        # Action history for the latency, one slot per intermediate step.
        L = self._max_delay_steps_limit + 1
        self._action_history = self._default_ctrl.expand(L, self._num_envs, m.nu).clone()
        self._history_head = 0

        if self._domain_randomization:
            self._randomizer.resample_startup(self._all_envs)

    @property
    def randomizer(self):
        return self._randomizer

    @property
    def domain_randomization(self):
        return self._domain_randomization

    def set_domain_randomization(self, enabled):
        """
        Enable or disable the domain randomisation of every world.
        """
        self._domain_randomization = bool(enabled)
        if enabled:
            self._randomizer.resample_startup(self._all_envs)
        else:
            self._randomizer.reset_to_nominal()

    @property
    def max_delay_limit(self):
        """The largest delay the action history was sized for, in seconds."""
        return self._max_delay_limit

    @property
    def max_delay_steps_limit(self):
        """The number of intermediate steps the action history was sized for."""
        return self._max_delay_steps_limit

    @property
    def max_delay_steps(self):
        """Largest delay an action can currently get, in intermediate steps."""
        return self._randomizer.max_delay_steps

    @property
    def max_delay(self):
        """Largest delay an action can currently get, in seconds."""
        return self._randomizer.params["max_delay"]

    @max_delay.setter
    def max_delay(self, value):
        steps = int(value / self._step_dt + 0.5)
        if steps > self._max_delay_steps_limit:
            raise ValueError(f"max_delay {value} s exceeds max_delay_limit {self._max_delay_limit} s")
        self._randomizer.params["max_delay"] = value

    def randomization_obs_spec(self):
        """
        Length and (low, high) bounds, after scaling, of every observation in
        RANDOMIZATION_OBS. The task class uses this to register the ones in
        ``observed_randomization``.

        """
        n = self._n_joints
        p = self._randomization_params
        kp_lo, kp_hi = p["p_gain_scale"]
        kd_lo, kd_hi = p["d_gain_scale"]
        tq = p["torque_limit_factor"]
        m_lo, m_hi = p["add_trunk_mass"]
        total = float(self._model.body_subtreemass[0])
        return {
            "actual_delay": (1, 0.0, 1.0),
            "joint_calib_offset": (n, -1.0, 1.0),
            "p_gain": (n, kp_lo, kp_hi),
            "d_gain": (n, kd_lo, kd_hi),
            "mass": (1, (total + m_lo) / total, (total + m_hi) / total),
            "torque_limit": (n, 1.0 - tq, 1.0 + tq),
            "action_scaling_factor": (n, -1.0, 1.0),
            "joint_nominal_offset": (n, -1.0, 1.0),
        }

    def randomization_obs_value(self, name):
        """
        Current value of a randomized quantity for every world, (W, length),
        scaled to O(1).

        """
        r = self._randomizer
        d = r.default_parameters
        p = self._randomizer.params
        c = self._ctrl_idx
        if name == "actual_delay":
            return r.delay_steps.unsqueeze(1).float() / max(self._max_delay_steps_limit, 1)
        if name == "joint_calib_offset":
            return r.position_offset[:, c] / max(float(p["position_offset"]), 1e-3)
        if name == "p_gain":
            return r.seen_parameters["p_gain"][:, c] / d["kp"][c]
        if name == "d_gain":
            return r.seen_parameters["d_gain"][:, c] / d["kd"][c]
        if name == "mass":
            return (r.seen_parameters["mass"] / float(self._model.body_subtreemass[0])).unsqueeze(1)
        if name == "torque_limit":
            return r.seen_parameters["torque_limit"][:, c] / d["forcerange"][c, 1]
        if name == "action_scaling_factor":
            lo, hi = p["add_scaling_factor"]
            return (r.scaling_factor - d["action_scale"]) / max(abs(lo), abs(hi), 1e-3)
        if name == "joint_nominal_offset":
            lo, hi = p["add_joint_nominal_position"]
            return (r.joint_nominal_pos[:, c] - d["joint_nominal_pos"][c]) / max(abs(lo), abs(hi), 1e-3)
        raise ValueError(f"unknown randomized observation {name!r}")

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _preprocess_action(self, action):
        """
        Map the policy action (W, 12) to position targets (W, 12), once per
        policy step: the (randomized) nominal angle plus the action times the
        (randomized) scale, shifted by the encoder miscalibration. The raw
        action is kept in ``self._actions`` (observation, action rate).

        """
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action, min=-100.0, max=100.0)
        self._actions[:] = action
        r = self._randomizer
        target = r.joint_nominal_pos + r.position_offset + r.scaling_factor * action
        return torch.clamp(target, self._ctrl_lower, self._ctrl_upper)

    def _compute_action(self, obs, action):
        """
        Called at every intermediate step: returns the target the actuators
        see, the one computed ``delay`` intermediate steps ago.

        """
        L = self._action_history.shape[0]
        self._history_head = (self._history_head + 1) % L
        self._action_history[self._history_head] = action
        if not self._domain_randomization:
            return action
        delay = self._randomizer.sample_latency()
        slot = (self._history_head - delay) % L
        return self._action_history[slot, self._all_envs]

    def _set_ctrl(self, ctrl_action, env_mask):
        """
        Write the control of the active worlds without a host sync.
        """
        ctrl = wp.to_torch(self._data_wp.ctrl)
        ctrl.copy_(torch.where(env_mask.unsqueeze(1), ctrl_action.to(ctrl.dtype), ctrl))

    # ------------------------------------------------------------------
    # Quaternion helpers (w, x, y, z), batched over environments
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_rotate(q, v):
        w, xyz = q[:, :1], q[:, 1:]
        c = torch.cross(xyz, v, dim=-1)
        return v + 2.0 * w * c + 2.0 * torch.cross(xyz, c, dim=-1)

    @staticmethod
    def _quat_rotate_inverse(q, v):
        w, xyz = q[:, :1], q[:, 1:]
        c = torch.cross(xyz, v, dim=-1)
        return v - 2.0 * w * c + 2.0 * torch.cross(xyz, c, dim=-1)

    @staticmethod
    def _wrap_to_pi(angles):
        return torch.atan2(torch.sin(angles), torch.cos(angles))

    # ------------------------------------------------------------------
    # Robot state helpers
    # ------------------------------------------------------------------

    def _projected_gravity(self):
        """Gravity direction expressed in the base frame. (num_envs, 3)"""
        quat = self._read_data("base_rot")
        g = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(quat.shape[0], 3)
        return self._quat_rotate_inverse(quat, g)

    def _heading(self):
        """Yaw angle of the base forward axis, in radians. (num_envs,)"""
        quat = self._read_data("base_rot")
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(quat.shape[0], 3)
        fwd = self._quat_rotate(quat, fwd)
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _base_lin_vel(self):
        """Linear velocity of the trunk in the base frame, noise-free. (num_envs, 3)"""
        qvel = wp.to_torch(self._data_wp.qvel)
        return self._quat_rotate_inverse(self._read_data("base_rot"), qvel[:, 0:3])

    def _base_ang_vel(self):
        """Angular velocity of the trunk in the base frame (free joint qvel[3:6] is local). (num_envs, 3)"""
        return wp.to_torch(self._data_wp.qvel)[:, 3:6]

    def _joint_pos(self):
        return wp.to_torch(self._data_wp.qpos)[:, self._qpos_idx]

    def _joint_vel(self):
        return wp.to_torch(self._data_wp.qvel)[:, self._dof_idx]

    def _joint_torque(self):
        return wp.to_torch(self._data_wp.qfrc_actuator)[:, self._dof_idx]

    def _foot_pos(self):
        """World position of each foot sphere centre. (num_envs, 4, 3)"""
        return wp.to_torch(self._data_wp.geom_xpos)[:, self._feet_geom_ids]

    # ------------------------------------------------------------------
    # Contacts, from the solver's contact list (no host sync)
    # ------------------------------------------------------------------

    def _floor_contacts(self):
        """
        Which geoms of every world are touching the floor, (num_envs, ngeom)
        bool. A contact counts once the geoms interpenetrate (dist <= 0), the
        closest a contact list gets to Isaac's "net contact force > threshold".

        """
        W, ngeom = self._num_envs, self._model.ngeom
        contact = self._data_wp.contact
        geom = wp.to_torch(contact.geom).long()
        dist = wp.to_torch(contact.dist)
        world = wp.to_torch(contact.worldid).long()
        # nacon: contacts the narrowphase produced this step (ncollision is
        # the broadphase pair count, an upper bound).
        counter = self._data_wp.nacon if hasattr(self._data_wp, "nacon") else self._data_wp.ncollision
        n = wp.to_torch(counter)[0]

        valid = torch.arange(geom.shape[0], device=geom.device) < n
        valid &= dist <= 0.0
        g0, g1 = geom[:, 0], geom[:, 1]
        other = torch.where(g0 == self._floor_geom, g1, g0)
        valid &= ((g0 == self._floor_geom) | (g1 == self._floor_geom)) & (other >= 0)

        dump = W * ngeom
        flat = torch.where(valid, world.clamp(0, W - 1) * ngeom + other.clamp(0, ngeom - 1),
                           torch.full_like(other, dump))
        touch = torch.zeros(dump + 1, dtype=torch.bool, device=geom.device)
        touch[flat] = True
        return touch[:dump].view(W, ngeom)

    def _foot_contacts(self, touch=None):
        touch = self._floor_contacts() if touch is None else touch
        return touch[:, self._feet_geom_ids]

    def _trunk_contact(self, touch=None):
        touch = self._floor_contacts() if touch is None else touch
        return touch[:, self._trunk_geom_ids].any(dim=1)

    def _lower_body_contacts(self, touch=None):
        """Number of thighs and calves touching the floor. (num_envs,)"""
        touch = self._floor_contacts() if touch is None else touch
        per_body = touch.float() @ self._lower_body_membership.T
        return (per_body > 0).sum(dim=1)

    def _is_finite(self, obs=None):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.isfinite(torch.cat([qpos, qvel], dim=1)).all(dim=1)

    # ------------------------------------------------------------------
    # Reset and domain randomisation
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given worlds to the nominal pose. With the randomisation on,
        the episode's parameters are redrawn, the yaw is random, and the
        joints and base velocity get the reset noise of the params (zero in
        Isaac's setting).

        """
        super().setup(env_indices, obs)

        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        idx = (
            env_indices.to(qpos.device).long()
            if isinstance(env_indices, torch.Tensor)
            else torch.as_tensor(env_indices, device=qpos.device, dtype=torch.long)
        )
        n = len(idx)
        if n == 0:
            self._mj_warp.forward(self._model_wp, self._data_wp)
            return
        dev = qpos.device

        qpos[idx] = self._default_qpos
        qvel[idx] = 0.0

        if self._domain_randomization:
            p = self._randomizer.params
            self._randomizer.resample_reset(idx)

            yaw = uniform((-p["reset_yaw_range"], p["reset_yaw_range"]), (n,), dev)
            quat = torch.zeros(n, 4, device=dev)
            quat[:, 0] = torch.cos(0.5 * yaw)
            quat[:, 3] = torch.sin(0.5 * yaw)
            qpos[idx, 3:7] = quat

            jn = p["reset_joint_noise"]
            qpos[idx.unsqueeze(1), self._qpos_idx.unsqueeze(0)] = (
                self._default_joint_pos + uniform((-jn, jn), (n, self._n_joints), dev)
            )
            bv = p["reset_base_velocity"]
            qvel[idx, 0:2] = uniform((-bv, bv), (n, 2), dev)

        # Flush the latency buffer of the reset worlds with their nominal
        # target, so a delayed read cannot return the previous episode.
        r = self._randomizer
        self._action_history[:, idx] = (r.joint_nominal_pos + r.position_offset)[idx]

        self._actions[idx] = 0.0
        self._episode_length[idx] = 0

        self._mj_warp.forward(self._model_wp, self._data_wp)

    def _step_finalize(self):
        self._episode_length += 1
        if self._domain_randomization:
            push_idx, vel = self._randomizer.sample_disturbance(self._episode_length, self.dt)
            self._push_robots(push_idx, vel)

    def _push_robots(self, env_indices, velocities):
        """
        Add the push to the base planar velocity of the given worlds (Isaac
        adds it; it does not overwrite).
        """
        if len(env_indices) == 0:
            return
        qvel = wp.to_torch(self._data_wp.qvel)
        qvel[env_indices, 0:2] += velocities

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
