import mujoco
import numpy as np
import torch
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.utils.mujoco import ObservationType
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco_warp_envs.legged_randomizer import (
    LeggedRandomizationParams,
    LeggedRandomizer,
)

from .vendor_h2 import ensure_h2_model


class H2Base(MuJoCoWarp):
    """
    Base class for the Unitree H2 humanoid in MuJoCo Warp.

    Holds everything that is a property of the robot rather than of a task:
    the model, the joint and actuator specification, the mapping from policy
    actions to joint position targets, the health check used for termination,
    and the domain randomisation. Task classes (H2Stand, H2Walk, H2Run)
    derive from this and supply the reward, the termination and the task
    specific observations.

    Domain randomisation follows the IsaacSim quadruped environments of the
    new_isaac branch (``QuadrupedRandomizer``): the trunk mass and centre of
    mass, the PD gains, the torque limits, the joint dynamics, the ground
    friction, the encoder calibration, the actuation latency, random pushes
    and the reset state. See :class:`LeggedRandomizer` for the mapping onto
    the warp model. The physical quantities live in per-world copies of the
    model arrays (``model_batch_fields``), so every world simulates its own
    robot; what the Isaac version keeps in its torch PD law (encoder offset,
    action scaling, latency) is applied here between the policy and the
    position actuators.

    Control structure: the model timestep is 0.002 s and a policy step is
    ``n_intermediate_steps * n_substeps`` physics steps. The defaults, 5
    intermediate steps of 2 physics steps, give a 50 Hz policy. The position
    target is rewritten at every intermediate step, so the actuation latency
    is drawn in units of 4 ms, up to ``max_delay_steps`` of them. The joint
    PD runs inside MuJoCo at every physics step regardless.

    The policy controls a subset of the 31 actuated joints (by default the
    legs, the waist, the shoulders and the elbows); the others are held at
    their default angle. Contact is modelled at the feet only (one box each,
    see vendor_h2.py).

    Note on construction order: the parent constructor calls _modify_mdp_info
    before it calls Environment.__init__, so everything that depends on the
    loaded model is set up inside _modify_mdp_info.

    """

    LEG_JOINTS = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_roll_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_roll_joint",
        "right_ankle_pitch_joint",
    ]
    WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    HEAD_JOINTS = ["head_pitch_joint", "head_yaw_joint"]
    ARM_JOINTS = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    UPPER_ARM_JOINTS = [j for j in ARM_JOINTS if "wrist" not in j]

    ALL_JOINTS = LEG_JOINTS + WAIST_JOINTS + HEAD_JOINTS + ARM_JOINTS
    DEFAULT_CONTROLLED_JOINTS = LEG_JOINTS + WAIST_JOINTS + UPPER_ARM_JOINTS
    POSTURE_JOINTS = (
        [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ]
        + WAIST_JOINTS
        + UPPER_ARM_JOINTS
    )

    FEET_GEOMS = ["left_foot", "right_foot"]
    ROOT_BODY = "pelvis"

    # Model fields allocated per world, so the randomizer can write them.
    BATCHED_MODEL_FIELDS = [
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_subtreemass",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "jnt_stiffness",
    ]

    # Randomized quantities the environment can expose as observations, with
    # their length per world. Values are scaled to O(1), see
    # randomization_obs_value.
    RANDOMIZATION_OBS = (
        "actual_delay",
        "joint_calib_offset",
        "p_gain",
        "d_gain",
        "mass",
        "torque_limit",
        "action_scaling_factor",
        "joint_nominal_offset",
    )

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        controlled_joints=None,
        healthy_z_range=(0.6, 1.3),
        healthy_gravity_z=-0.7,
        terminate_when_unhealthy=True,
        action_scale=0.25,
        soft_joint_limit=0.9,
        domain_randomization=True,
        randomization_params=None,
        observed_randomization=(),
        init_joint_noise=None,
        init_vel_noise=None,
        push_interval=None,
        push_max_vel=None,
        n_substeps=2,
        n_intermediate_steps=5,
        use_graph_capture=False,
        nconmax=None,
        njmax=256,
        scene="scene.xml",
        **viewer_params,
    ):
        """
        Constructor.

        Args:
            num_envs (int): number of parallel environments;
            controlled_joints (list, None): names of the joints the policy
                controls, in action order. Defaults to
                DEFAULT_CONTROLLED_JOINTS; every other actuated joint is held
                at its default angle;
            healthy_z_range (tuple): pelvis height range, in metres, in which
                the robot is considered healthy;
            healthy_gravity_z (float): upper bound on the z component of the
                gravity vector in the pelvis frame. -1 when upright, 0 when
                horizontal;
            action_scale (float): nominal scaling from policy action to joint
                position offset, in radians;
            soft_joint_limit (float): fraction of the joint range, centred on
                its midpoint, outside of which a limit penalty may apply;
            domain_randomization (bool): whether the domain randomisation is
                enabled. Off, every world runs the nominal model with no
                pushes, no latency and a deterministic reset pose;
            randomization_params (LeggedRandomizationParams, dict, None): the
                randomisation ranges; a dict is taken as overrides of the
                defaults;
            observed_randomization (tuple): names from RANDOMIZATION_OBS the
                environment appends to the observation, in this order, for
                an asymmetric critic. Appended by the task class;
            init_joint_noise, init_vel_noise, push_interval, push_max_vel:
                shorthands for the ``reset_joint_noise``,
                ``reset_base_velocity``, ``push_probability`` (as 1/interval)
                and ``push_max_velocity`` randomisation parameters, kept so
                existing configurations keep working;
            n_substeps (int): physics steps per intermediate step;
            n_intermediate_steps (int): intermediate steps per policy step.
                Sets the resolution of the actuation latency, see the class
                docstring;
            scene (str): scene file to load. Built from the Unitree URDF on
                first use, see vendor_h2.py.

        """
        xml_path = str(ensure_h2_model(scene))

        self._controlled_joints = (
            list(controlled_joints)
            if controlled_joints is not None
            else list(self.DEFAULT_CONTROLLED_JOINTS)
        )
        unknown = set(self._controlled_joints) - set(self.ALL_JOINTS)
        if unknown:
            raise ValueError(f"unknown joints in controlled_joints: {sorted(unknown)}")
        self._n_joints = len(self._controlled_joints)
        self._n_actuators = len(self.ALL_JOINTS)

        unknown = set(observed_randomization) - set(self.RANDOMIZATION_OBS)
        if unknown:
            raise ValueError(f"unknown randomized observations: {sorted(unknown)}")
        self._observed_randomization = tuple(observed_randomization)

        if isinstance(randomization_params, dict):
            randomization_params = LeggedRandomizationParams(**randomization_params)
        params = LeggedRandomizationParams() if randomization_params is None else randomization_params
        if init_joint_noise is not None:
            params["reset_joint_noise"] = init_joint_noise
        if init_vel_noise is not None:
            params["reset_base_velocity"] = init_vel_noise
        if push_interval is not None:
            params["push_probability"] = 1.0 / push_interval
        if push_max_vel is not None:
            params["push_max_velocity"] = push_max_vel
        self._randomization_params = params
        self._max_delay_steps_limit = int(params["max_delay_steps"])

        observation_spec = [("base_vel", self.ROOT_BODY, ObservationType.BODY_VEL)]
        observation_spec += [
            (f"{j}_pos", j, ObservationType.JOINT_POS) for j in self._controlled_joints
        ]
        observation_spec += [
            (f"{j}_vel", j, ObservationType.JOINT_VEL) for j in self._controlled_joints
        ]
        additional_data_spec = [
            ("base_pos", self.ROOT_BODY, ObservationType.BODY_POS),
            ("base_rot", self.ROOT_BODY, ObservationType.BODY_ROT),
        ]

        self._healthy_z_range = healthy_z_range
        self._healthy_gravity_z = healthy_gravity_z
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._action_scale = action_scale
        self._soft_joint_limit = soft_joint_limit
        self._domain_randomization = domain_randomization
        self._num_envs = num_envs

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=self.ALL_JOINTS,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            model_batch_fields=self.BATCHED_MODEL_FIELDS,
            **viewer_params,
        )

    # ------------------------------------------------------------------
    # MDP info / model dependent setup
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        m = self._model

        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise ValueError(
                "H2Base needs position-type actuators; rebuild the model with vendor_h2.py"
            )
        if m.nu != self._n_actuators:
            raise ValueError(f"model has {m.nu} actuators, expected {self._n_actuators}")
        # _set_ctrl below writes the whole ctrl row of a world at once, which
        # needs the actuation spec to cover every actuator in model order.
        if not np.array_equal(np.asarray(self._action_indices), np.arange(m.nu)):
            raise ValueError("actuation_spec must list every actuator in model order")

        dev = wp.to_torch(self._data_wp.qpos).device
        self._device = dev

        def jid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise ValueError(f"joint {name!r} not in model")
            return i

        for j in self.ALL_JOINTS:
            jid(j)
        ctrl_jids = [jid(j) for j in self._controlled_joints]
        act_jids = m.actuator_trnid[:, 0]
        self._all_qpos_idx = torch.as_tensor(m.jnt_qposadr[act_jids], device=dev, dtype=torch.long)
        self._qpos_idx = torch.as_tensor(m.jnt_qposadr[ctrl_jids], device=dev, dtype=torch.long)
        self._dof_idx = torch.as_tensor(m.jnt_dofadr[ctrl_jids], device=dev, dtype=torch.long)

        act_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in self._controlled_joints
        ]
        if min(act_ids) < 0:
            raise ValueError("every controlled joint needs an actuator of the same name")
        self._ctrl_idx = torch.as_tensor(act_ids, device=dev, dtype=torch.long)

        posture_jids = [
            self._controlled_joints.index(j)
            for j in self.POSTURE_JOINTS
            if j in self._controlled_joints
        ]
        self._posture_idx = torch.as_tensor(posture_jids, device=dev, dtype=torch.long)

        key_qpos = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=dev)
        self._default_qpos = key_qpos
        self._default_ctrl = key_qpos[self._all_qpos_idx].clone()  # all actuators
        self._default_joint_pos = key_qpos[self._qpos_idx].clone()  # controlled joints
        self._nominal_height = float(m.key_qpos[0][2])

        rng = torch.as_tensor(m.jnt_range[ctrl_jids], dtype=torch.float32, device=dev)
        self._joint_lower = rng[:, 0]
        self._joint_upper = rng[:, 1]
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * self._soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        ctrlrange = m.actuator_ctrlrange
        self._ctrl_lower = torch.as_tensor(ctrlrange[:, 0], dtype=torch.float32, device=dev)
        self._ctrl_upper = torch.as_tensor(ctrlrange[:, 1], dtype=torch.float32, device=dev)

        self._feet_geom_ids = torch.as_tensor(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.FEET_GEOMS],
            device=dev,
        )
        if (self._feet_geom_ids < 0).any():
            raise ValueError("foot geoms not found; rebuild the model with vendor_h2.py")
        self._foot_half_height = torch.as_tensor(
            m.geom_size[self._feet_geom_ids.cpu().numpy(), 2], dtype=torch.float32, device=dev
        )

        self._actions = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._episode_length = torch.zeros(self._num_envs, dtype=torch.long, device=dev)
        self._all_envs = torch.arange(self._num_envs, device=dev)

        first_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        default_np = m.key_qpos[0][m.jnt_qposadr[ctrl_jids]]
        rng_np = m.jnt_range[ctrl_jids]
        action_low = (rng_np[:, 0] - default_np) / self._action_scale
        action_high = (rng_np[:, 1] - default_np) / self._action_scale
        mdp_info.action_space = Box(action_low, action_high)

        # Domain randomisation: per-world views of the model, the randomizer
        # and the action history the latency reads from. Built before any
        # step or reset is graph-captured, and never reallocated afterwards.
        self._build_randomizer(m, act_jids)

        mdp_info = super()._modify_mdp_info(mdp_info)
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.ITERATIONS
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _build_randomizer(self, m, act_jids):
        dev = self._device
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.ROOT_BODY)
        ancestors = []
        b = trunk
        while b >= 0:
            ancestors.append(int(b))
            b = int(m.body_parentid[b]) if b != 0 else -1

        self._model_views = {
            name: wp.to_torch(getattr(self._model_wp, name)) for name in self.BATCHED_MODEL_FIELDS
        }
        for name, view in self._model_views.items():
            if view.shape[0] != self._num_envs:
                raise RuntimeError(
                    f"model field {name} is not batched per world (shape {tuple(view.shape)}); "
                    "the MuJoCoWarp base class must pass model_batch_fields to put_model"
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
        self._randomizer = LeggedRandomizer(
            self._num_envs,
            self._model_views,
            layout,
            self._default_ctrl,
            self._action_scale,
            self._randomization_params,
        )

        # Action history for the latency, one slot per intermediate step.
        L = self._max_delay_steps_limit + 1
        self._action_history = self._default_ctrl.expand(L, self._num_envs, m.nu).clone()
        self._history_head = 0

        if self._domain_randomization:
            self._randomizer.resample_startup(self._all_envs)

    # ------------------------------------------------------------------
    # Domain randomisation control
    # ------------------------------------------------------------------

    @property
    def randomizer(self):
        return self._randomizer

    @property
    def randomization_params(self):
        return self._randomization_params

    @property
    def domain_randomization(self):
        return self._domain_randomization

    def set_domain_randomization(self, enabled):
        """
        Enable or disable the domain randomisation of every world. Disabling
        it puts every world back on the nominal model; enabling it redraws
        the robot properties of every world.

        """
        self._domain_randomization = bool(enabled)
        if enabled:
            self._randomizer.resample_startup(self._all_envs)
        else:
            self._randomizer.reset_to_nominal()

    @property
    def max_delay_steps(self):
        """
        Largest number of intermediate steps an action can currently be
        delayed by. Can be lowered and raised during training (a curriculum)
        up to the value the environment was built with.

        """
        return int(self._randomization_params["max_delay_steps"])

    @max_delay_steps.setter
    def max_delay_steps(self, value):
        value = int(value)
        if value > self._max_delay_steps_limit:
            raise ValueError(
                f"the delay cannot exceed the {self._max_delay_steps_limit} intermediate steps "
                f"the environment was built with, got {value}"
            )
        self._randomization_params["max_delay_steps"] = value

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
        mass = float(self._randomizer.default_parameters["trunk_mass"])
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
        scaled to O(1): ratios to the nominal value for gains, mass and
        torque limits; offsets divided by their half-width for the encoder
        offset, the action scale and the nominal position; the delay as a
        fraction of the largest delay the environment was built with.

        """
        r = self._randomizer
        d = r.default_parameters
        p = self._randomization_params
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
        actuators (W, nu), once per policy step.

        The target of a controlled joint is its (randomized) nominal angle
        plus the action times the (randomized) scale; an uncontrolled joint
        gets its nominal angle. Every target is shifted by the encoder
        miscalibration, so that the controller, like the policy (see the
        task's _modify_observation), works in coordinates offset from the
        truth by that much.

        """
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action, min=-100.0, max=100.0)
        self._actions[:] = action
        r = self._randomizer
        target = r.joint_nominal_pos + r.position_offset
        target = target.clone()
        target[:, self._ctrl_idx] += r.scaling_factor * action
        return torch.clamp(target, self._ctrl_lower, self._ctrl_upper)

    def _compute_action(self, obs, action):
        """
        Called at every intermediate step: returns the target the actuators
        see, which is the one computed ``delay`` intermediate steps ago.

        The history is a ring buffer with one slot per intermediate step.
        With the randomisation off every delay is zero and the current
        target is returned, so the structure costs nothing but one gather.

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
        Write the control of the active worlds without a host sync. The base
        class gathers the active indices with torch.nonzero, which waits for
        the GPU; with several intermediate steps per policy step that wait
        would be paid several times. The actuation spec covers every
        actuator in order (checked in _modify_mdp_info), so a masked copy of
        the whole row is equivalent.

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
        """Gravity direction in the pelvis frame. (W, 3)"""
        quat = self._read_data("base_rot")
        g = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(quat.shape[0], 3)
        return self._quat_rotate_inverse(quat, g)

    def _heading(self):
        """Yaw angle of the pelvis forward axis, in radians. (W,)"""
        quat = self._read_data("base_rot")
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(quat.shape[0], 3)
        fwd = self._quat_rotate(quat, fwd)
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _base_height(self):
        return self._read_data("base_pos")[:, 2]

    def _joint_pos(self):
        return wp.to_torch(self._data_wp.qpos)[:, self._qpos_idx]

    def _joint_vel(self):
        return wp.to_torch(self._data_wp.qvel)[:, self._dof_idx]

    def _joint_torque(self):
        return wp.to_torch(self._data_wp.qfrc_actuator)[:, self._dof_idx]

    def _foot_pos(self):
        """World position of each foot box centre. (W, 2, 3)"""
        return wp.to_torch(self._data_wp.geom_xpos)[:, self._feet_geom_ids, :]

    def _foot_height(self):
        """Height of each foot sole above the ground. (W, 2)"""
        return self._foot_pos()[:, :, 2] - self._foot_half_height

    def _foot_up(self):
        """z component of each foot's up axis, 1 when flat on the ground. (W, 2)"""
        xmat = wp.to_torch(self._data_wp.geom_xmat)[:, self._feet_geom_ids, :, :]
        return xmat[:, :, 2, 2]

    # ------------------------------------------------------------------
    # Health / termination
    # ------------------------------------------------------------------

    def _is_finite(self, obs):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.isfinite(torch.cat([qpos, qvel], dim=1)).all(dim=1)

    def _is_within_z_range(self, obs):
        min_z, max_z = self._healthy_z_range
        z = self._base_height()
        return (z >= min_z) & (z <= max_z)

    def _is_upright(self, obs):
        return self._projected_gravity()[:, 2] <= self._healthy_gravity_z

    def _is_healthy(self, obs):
        return self._is_finite(obs) & self._is_within_z_range(obs) & self._is_upright(obs)

    # ------------------------------------------------------------------
    # Reset and domain randomisation
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given worlds to the standing keyframe. With the
        randomisation on, the episode's parameters are redrawn, the yaw and
        the controlled joints get uniform noise and the pelvis a random
        planar velocity.

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

        qpos[idx] = self._default_qpos
        qvel[idx] = 0.0

        if self._domain_randomization:
            p = self._randomization_params
            self._randomizer.resample_reset(idx)

            yaw = (torch.rand(n, device=qpos.device) * 2 - 1) * p["reset_yaw_range"]
            quat = torch.zeros(n, 4, device=qpos.device)
            quat[:, 0] = torch.cos(0.5 * yaw)
            quat[:, 3] = torch.sin(0.5 * yaw)
            qpos[idx, 3:7] = quat

            noise = (torch.rand(n, self._n_joints, device=qpos.device) * 2 - 1) * p["reset_joint_noise"]
            qpos[idx.unsqueeze(1), self._qpos_idx.unsqueeze(0)] = self._default_joint_pos + noise
            qvel[idx, 0:2] = (torch.rand(n, 2, device=qpos.device) * 2 - 1) * p["reset_base_velocity"]

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
        """Overwrite the pelvis planar velocity of the given worlds."""
        if len(env_indices) == 0:
            return
        qvel = wp.to_torch(self._data_wp.qvel)
        qvel[env_indices, 0:2] = velocities

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
