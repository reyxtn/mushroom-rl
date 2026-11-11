import math

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

from .fetch_h2_model import ensure_h2_model

from .params import (
    ALL_JOINTS,
    DEFAULT_CONTROLLED_JOINTS,
    ACTION_SCALE,
    FEET_SITES,
    FEET_BODIES,
    ROOT_BODY,
    TORSO_BODY,
    FLOOR_GEOM,
    ANGMOM_SENSOR,
    FALL_ANGLE,
    PUSH_EXTRA_VELOCITY,
    BATCHED_MODEL_FIELDS,
    RANDOMIZATION_OBS,
)


def with_pushes(randomization_params, pushes):
    """
    The randomization params with the push entries of a task applied on top,
    unless the caller set them: the pushes belong to the task while the rest
    of the randomization is shared.

    """
    given = {}
    if isinstance(randomization_params, dict):
        given = dict(randomization_params)
    elif isinstance(randomization_params, RandomizationParams):
        return randomization_params
    params = RandomizationParams(**{**pushes, **given})
    return params


class H2Base(MuJoCoWarp):
    """
    Base constructor for the H2 humanoid in MuJoCo Warp, on Unitree's own
    model: collision capsules on every link (the robot collides with itself)
    and Unitree's gains and effort limits.

    The policy controls all 29 actuated joints by default. The physics split
    is Unitree's: 5 ms physics steps, 4 per 20 ms policy step.

    """

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        controlled_joints=None,
        fall_angle=FALL_ANGLE,
        terminate_when_fallen=True,
        soft_joint_limit=0.9,
        domain_randomization=True,
        randomization_params=None,
        observed_randomization=(),
        max_delay_limit=0.02,
        timestep=None,
        n_substeps=4,
        n_intermediate_steps=1,
        use_graph_capture=False,
        nconmax=None,
        njmax=300,
        scene="scene.xml",
        **viewer_params,
    ):
        """
        Args (the ones not obvious):
            controlled_joints (tuple, None): names of the joints the policy
                controls, in action order. None is DEFAULT_CONTROLLED_JOINTS;
                the others are held at their default angle;
            fall_angle (float): tilt of the pelvis from upright, in radians,
                beyond which the robot has fallen;
            soft_joint_limit (float): fraction of every joint range, around
                its midpoint, inside which the joint is not penalized;
            max_delay_limit (float): the largest actuation delay, in seconds,
                the action history is sized for; ``max_delay`` can never be
                raised above it;
            timestep, n_substeps, n_intermediate_steps: the physics split. The
                delay resolution is ``timestep * n_substeps``. None keeps the
                model's timestep;
            scene (str): built from Unitree's MJCF on first use, see
                fetch_h2_model.py.

        """
        xml_path = str(ensure_h2_model(scene))

        controlled = tuple(controlled_joints) if controlled_joints is not None else DEFAULT_CONTROLLED_JOINTS
        unknown = set(controlled) - set(ALL_JOINTS)
        if unknown:
            raise ValueError(f"unknown joints in controlled_joints: {sorted(unknown)}")
        unknown = set(observed_randomization) - set(RANDOMIZATION_OBS)
        if unknown:
            raise ValueError(f"unknown randomized observations: {sorted(unknown)}")

        observation_spec = [("base_vel", ROOT_BODY, ObservationType.BODY_VEL)]
        observation_spec += [(f"{j}_pos", j, ObservationType.JOINT_POS) for j in controlled]
        observation_spec += [(f"{j}_vel", j, ObservationType.JOINT_VEL) for j in controlled]
        additional_data_spec = [
            ("base_pos", ROOT_BODY, ObservationType.BODY_POS),
            ("base_rot", ROOT_BODY, ObservationType.BODY_ROT),
        ]

        if randomization_params is None:
            randomization_params = RandomizationParams()
        elif isinstance(randomization_params, dict):
            randomization_params = RandomizationParams(**randomization_params)

        # Everything _modify_mdp_info reads: it runs inside super().__init__().
        self._controlled_joints = controlled
        self._n_joints = len(controlled)
        self._observed_randomization = tuple(observed_randomization)
        self._max_delay_limit = max_delay_limit
        self._n_substeps = n_substeps
        self._randomization_params = randomization_params
        self._fall_angle = fall_angle
        self._terminate_when_fallen = terminate_when_fallen
        self._soft_joint_limit = soft_joint_limit
        self._domain_randomization = domain_randomization
        self._num_envs = num_envs

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=list(ALL_JOINTS),
            additional_data_spec=additional_data_spec,
            timestep=timestep,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

        # Re-put the model with per-world copies of the randomized fields.
        # Dev's MuJoCoWarp shares model fields across worlds; nothing has
        # captured the first _model_wp yet (graphs are built lazily).
        self._model_wp = self._mj_warp.put_model(
            self._model, batch_sizes={f: num_envs for f in BATCHED_MODEL_FIELDS}
        )
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.ITERATIONS
        # mjlab's parallel line search, on the mujoco_warp versions that have the option.
        if hasattr(self._model_wp.opt, "ls_parallel"):
            self._model_wp.opt.ls_parallel = True

        self._build_randomizer(self._model, self._act_jids)

        if self._randomizer.max_delay_steps > self._max_delay_steps_limit:
            raise ValueError(
                f"max_delay {self._randomizer.params['max_delay']} s exceeds the "
                f"buffer ceiling max_delay_limit {self._max_delay_limit} s"
            )

    def _modify_mdp_info(self, mdp_info):
        m = self._model

        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise ValueError("H2Base needs position-type actuators; rebuild the model with fetch_h2_model.py")
        if m.nu != len(ALL_JOINTS):
            raise ValueError(f"model has {m.nu} actuators, expected {len(ALL_JOINTS)}")
        if not np.array_equal(np.asarray(self._action_indices), np.arange(m.nu)):
            raise ValueError("actuation_spec must list every actuator in model order")

        dev = wp.to_torch(self._data_wp.qpos).device
        self._device = dev

        def obj_id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError(f"{name!r} not in model; rebuild it with fetch_h2_model.py")
            return i

        for j in ALL_JOINTS:
            obj_id(mujoco.mjtObj.mjOBJ_JOINT, j)
        ctrl_jids = [obj_id(mujoco.mjtObj.mjOBJ_JOINT, j) for j in self._controlled_joints]
        act_jids = m.actuator_trnid[:, 0]
        self._act_jids = act_jids
        self._all_qpos_idx = torch.as_tensor(m.jnt_qposadr[act_jids], device=dev, dtype=torch.long)
        self._qpos_idx = torch.as_tensor(m.jnt_qposadr[ctrl_jids], device=dev, dtype=torch.long)
        self._dof_idx = torch.as_tensor(m.jnt_dofadr[ctrl_jids], device=dev, dtype=torch.long)

        # Every controlled joint has an actuator of the same name.
        act_ids = [obj_id(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in self._controlled_joints]
        self._ctrl_idx = torch.as_tensor(act_ids, device=dev, dtype=torch.long)

        # Nominal standing pose: the "home" keyframe the build solved so the
        # feet just touch the floor.
        key_qpos = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=dev)
        self._default_qpos = key_qpos
        self._default_ctrl = key_qpos[self._all_qpos_idx].clone()  # all actuators
        self._default_joint_pos = key_qpos[self._qpos_idx].clone()  # controlled joints

        self._action_scale = ACTION_SCALE

        # Soft joint limits, shrunk towards the midpoint of the range.
        rng = torch.as_tensor(m.jnt_range[ctrl_jids], dtype=torch.float32, device=dev)
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * self._soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        # Position targets are clamped to the actuator ctrlrange, which is
        # wider than the joint range (see fetch_h2_model.py).
        ctrlrange = m.actuator_ctrlrange
        self._ctrl_lower = torch.as_tensor(ctrlrange[:, 0], dtype=torch.float32, device=dev)
        self._ctrl_upper = torch.as_tensor(ctrlrange[:, 1], dtype=torch.float32, device=dev)

        # Feet: one site per foot for its position, and the collision
        # capsules of each foot for its contact and its friction.
        self._feet_site_ids = torch.as_tensor(
            [obj_id(mujoco.mjtObj.mjOBJ_SITE, s) for s in FEET_SITES], device=dev
        )
        feet_geoms = []
        for body in FEET_BODIES:
            bid = obj_id(mujoco.mjtObj.mjOBJ_BODY, body)
            feet_geoms.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g]])
        if not feet_geoms[0] or len(feet_geoms[0]) != len(feet_geoms[1]):
            raise ValueError("foot collision geoms not found; rebuild the model with fetch_h2_model.py")
        self._feet_geom_ids = torch.as_tensor(feet_geoms, device=dev)  # (2, geoms per foot)
        # Which foot a geom belongs to; len(FEET_SITES) for every other geom.
        self._geom_foot = torch.full((m.ngeom,), len(FEET_SITES), dtype=torch.long, device=dev)
        for foot, geoms in enumerate(feet_geoms):
            self._geom_foot[geoms] = foot
        self._feet_body_ids = torch.as_tensor(m.site_bodyid[self._feet_site_ids.cpu().numpy()], device=dev).long()
        self._root_body = int(m.body_rootid[self._feet_body_ids[0].item()])

        self._floor_geom = obj_id(mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        self._torso_body = obj_id(mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        adr = int(m.sensor_adr[obj_id(mujoco.mjtObj.mjOBJ_SENSOR, ANGMOM_SENSOR)])
        self._angmom_slice = slice(adr, adr + 3)
        self._push_extra = torch.tensor(PUSH_EXTRA_VELOCITY, dtype=torch.float32, device=dev)

        self._actions = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._episode_length = torch.zeros(self._num_envs, dtype=torch.long, device=dev)
        self._all_envs = torch.arange(self._num_envs, device=dev)

        # Contiguous slices of the observation, for use by subclasses.
        first_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        # BODY_VEL is ordered [angular(3), linear(3)], in the body frame.
        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        # The action is a joint position offset in units of the action scale,
        # so the action space is the target range in those units.
        default_np = m.key_qpos[0][m.jnt_qposadr[ctrl_jids]]
        action_low = (ctrlrange[act_ids, 0] - default_np) / self._action_scale
        action_high = (ctrlrange[act_ids, 1] - default_np) / self._action_scale
        mdp_info.action_space = Box(action_low, action_high)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _build_randomizer(self, m, act_jids):
        self._step_dt = m.opt.timestep * self._n_substeps
        self._max_delay_steps_limit = int(self._max_delay_limit / self._step_dt + 0.5)
        dev = self._device
        # Mass and centre of mass are randomized on the torso, as Unitree does.
        trunk = self._torso_body
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
                    "check the put_model re-put with batch_sizes in H2Base.__init__"
                )
        layout = dict(
            trunk_body=trunk,
            trunk_ancestors=ancestors,
            foot_geoms=self._feet_geom_ids.flatten(),
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
        Map the policy action (W, n_joints) to position targets for all
        actuators (W, nu), once per policy step: a controlled joint gets its
        (randomized) nominal angle plus the scaled action, an uncontrolled
        one its nominal angle; every target is shifted by the encoder
        miscalibration. The raw action is kept in ``self._actions``
        (observation, action rate).

        """
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action, min=-100.0, max=100.0)
        self._actions[:] = action
        # A policy step starts here. Counting it now, the observation and the
        # reward that end the step read the same episode time.
        self._episode_length += 1
        r = self._randomizer
        target = (r.joint_nominal_pos + r.position_offset).clone()
        target[:, self._ctrl_idx] += r.scaling_factor * action
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

    def _gravity_in(self, quat):
        """Gravity direction expressed in the frame of the given orientation. (num_envs, 3)"""
        g = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(quat.shape[0], 3)
        return self._quat_rotate_inverse(quat, g)

    def _projected_gravity(self):
        """Gravity direction expressed in the pelvis frame. (num_envs, 3)"""
        return self._gravity_in(self._read_data("base_rot"))

    def _torso_projected_gravity(self):
        """Gravity direction expressed in the torso frame. (num_envs, 3)"""
        return self._gravity_in(wp.to_torch(self._data_wp.xquat)[:, self._torso_body])

    def _torso_ang_vel(self):
        """Angular velocity of the torso in the world frame. (num_envs, 3)"""
        return wp.to_torch(self._data_wp.cvel)[:, self._torso_body, :3]

    def _angular_momentum(self):
        """Angular momentum of the torso subtree about its centre of mass. (num_envs, 3)"""
        return wp.to_torch(self._data_wp.sensordata)[:, self._angmom_slice]

    def _heading(self):
        """Yaw angle of the pelvis forward axis, in radians. (num_envs,)"""
        quat = self._read_data("base_rot")
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(quat.shape[0], 3)
        fwd = self._quat_rotate(quat, fwd)
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _base_lin_vel(self):
        """Linear velocity of the pelvis in its own frame, noise-free. (num_envs, 3)"""
        qvel = wp.to_torch(self._data_wp.qvel)
        return self._quat_rotate_inverse(self._read_data("base_rot"), qvel[:, 0:3])

    def _base_ang_vel(self):
        """Angular velocity of the pelvis in its own frame (free joint qvel[3:6] is local). (num_envs, 3)"""
        return wp.to_torch(self._data_wp.qvel)[:, 3:6]

    def _joint_pos(self):
        return wp.to_torch(self._data_wp.qpos)[:, self._qpos_idx]

    def _joint_vel(self):
        return wp.to_torch(self._data_wp.qvel)[:, self._dof_idx]

    def _joint_acc(self):
        return wp.to_torch(self._data_wp.qacc)[:, self._dof_idx]

    def _foot_pos(self):
        """World position of each foot site, at the sole. (num_envs, 2, 3)"""
        return wp.to_torch(self._data_wp.site_xpos)[:, self._feet_site_ids]

    def _foot_vel(self):
        """World velocity of each foot site. (num_envs, 2, 3)"""
        # cvel is the velocity [angular, linear] of the body at the centre of
        # mass of its kinematic tree.
        cvel = wp.to_torch(self._data_wp.cvel)[:, self._feet_body_ids]
        com = wp.to_torch(self._data_wp.subtree_com)[:, self._root_body]
        return cvel[:, :, 3:] + torch.cross(cvel[:, :, :3], self._foot_pos() - com.unsqueeze(1), dim=2)

    def _contacts(self):
        """
        One pass over the contact list: which feet touch the floor,
        (num_envs, 2) bool, the normal force the floor puts on each foot,
        (num_envs, 2), and the largest normal force between two parts of the
        robot, (num_envs,).

        """
        W, n_feet = self._num_envs, len(FEET_SITES)
        contact = self._data_wp.contact
        geom = wp.to_torch(contact.geom).long().clamp(0, self._model.ngeom - 1)
        world = wp.to_torch(contact.worldid).long().clamp(0, W - 1)
        # nacon: the contacts of all worlds found in this step.
        counter = self._data_wp.nacon if hasattr(self._data_wp, "nacon") else self._data_wp.ncon
        valid = torch.arange(geom.shape[0], device=geom.device) < wp.to_torch(counter)[0]

        # Normal force of a contact: the sum of its constraint rows (one row
        # without friction, the edges of the friction pyramid with it).
        efc_force = wp.to_torch(self._data_wp.efc.force)
        adr = wp.to_torch(contact.efc_address).long()
        rows = efc_force[world.unsqueeze(1), adr.clamp(0, efc_force.shape[1] - 1)]
        force = torch.where(adr >= 0, rows, torch.zeros_like(rows)).sum(dim=1) * valid

        g0, g1 = geom[:, 0], geom[:, 1]
        with_floor = (g0 == self._floor_geom) | (g1 == self._floor_geom)
        foot = self._geom_foot[torch.where(g0 == self._floor_geom, g1, g0)]
        foot = torch.where(valid & with_floor, foot, torch.full_like(foot, n_feet))
        slot = world * (n_feet + 1) + foot  # the last slot of a world collects everything else
        zeros = torch.zeros(W * (n_feet + 1), device=geom.device)
        foot_count = zeros.index_add(0, slot, valid.float()).view(W, n_feet + 1)[:, :n_feet]
        foot_force = zeros.index_add(0, slot, force).view(W, n_feet + 1)[:, :n_feet]

        own = torch.where(valid & ~with_floor, force, torch.zeros_like(force))
        self_force = torch.zeros(W, device=geom.device).scatter_reduce(0, world, own, reduce="amax")
        return foot_count > 0, foot_force, self_force

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_finite(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.isfinite(torch.cat([qpos, qvel], dim=1)).all(dim=1)

    def _is_fallen(self):
        """Pelvis tilted further than fall_angle from upright."""
        return -self._projected_gravity()[:, 2] < math.cos(self._fall_angle)

    # ------------------------------------------------------------------
    # Reset and domain randomisation
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given worlds to the standing keyframe. With the
        randomisation on, the episode's parameters are redrawn, the yaw is
        random, and the controlled joints and the pelvis velocity get the
        reset noise of the params.

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
        if self._domain_randomization:
            push_idx, vel = self._randomizer.sample_disturbance(self._episode_length, self.dt)
            self._push_robots(push_idx, vel)

    def _push_robots(self, env_indices, velocities):
        """
        Add the push to the pelvis velocity of the given worlds: the planar
        part the randomizer drew, and a vertical and an angular part up to
        PUSH_EXTRA_VELOCITY.

        """
        if len(env_indices) == 0:
            return
        qvel = wp.to_torch(self._data_wp.qvel)
        qvel[env_indices, 0:2] += velocities
        qvel[env_indices, 2:6] += self._push_extra * uniform((-1.0, 1.0), (len(env_indices), 4), self._device)

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
