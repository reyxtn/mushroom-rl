import mujoco
import numpy as np
import torch
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp

from .vendor_h2 import ensure_h2_model


class H2Base(MuJoCoWarp):
    """
    Base class for the Unitree H2 humanoid in MuJoCo Warp.

    Holds everything that is a property of the robot rather than of a task:
    the model, the joint and actuator specification, the mapping from policy
    actions to joint position targets, the health check used for termination,
    and the domain randomisation applied on reset and during episodes. Task
    classes (H2Stand, H2Walk, H2Run) derive from this and supply the reward,
    the termination and the task specific observations.

    Same structure as Go2Base, with two differences that matter for a
    humanoid:

    - The policy controls a subset of the 31 actuated joints (by default the
      legs, the waist, the shoulders and the elbows). Every other joint is
      held at its default angle by its position actuator. Joint indices are
      resolved by name through the model's qpos and dof addresses rather than
      assumed to be contiguous after the free joint.
    - Contact is only modelled at the feet (one box each, see vendor_h2.py),
      so the foot "sole height" used by the tasks is the box centre minus
      its half height.

    The model is built from the Unitree URDF on first use; see vendor_h2.py.
    Its position actuators run the PD controller inside MuJoCo at every
    physics step, with per-joint gains set in vendor_h2.py.

    Note on construction order: the parent constructor calls _modify_mdp_info
    before it calls Environment.__init__, so everything that depends on the
    loaded model is set up inside _modify_mdp_info.

    """

    # Joint groups, in the order they appear in the URDF (which is also the
    # actuator order in the generated MJCF).
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

    # Every actuated joint, in actuator order. The actuator of a joint has
    # the joint's name.
    ALL_JOINTS = LEG_JOINTS + WAIST_JOINTS + HEAD_JOINTS + ARM_JOINTS

    # Joints the policy controls by default. The head and the wrists are held
    # at their default angle; they add action dimensions without helping
    # locomotion.
    DEFAULT_CONTROLLED_JOINTS = LEG_JOINTS + WAIST_JOINTS + UPPER_ARM_JOINTS

    # Joints that should stay near their default angle during locomotion.
    # Used by the tasks for a deviation penalty; kept here because it is a
    # property of the robot, not of one task.
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
        init_joint_noise=0.05,
        init_vel_noise=0.2,
        push_interval=750,
        push_max_vel=0.5,
        n_substeps=10,
        n_intermediate_steps=1,
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
                DEFAULT_CONTROLLED_JOINTS. Every other actuated joint is held
                at its default angle;
            healthy_z_range (tuple): pelvis height range, in metres, in which
                the robot is considered healthy. The pelvis is at 1.01 in the
                keyframe and settles around 0.96 under load;
            healthy_gravity_z (float): upper bound on the z component of the
                gravity vector in the pelvis frame. -1 when upright, 0 when
                horizontal; -0.7 is roughly 45 degrees of lean;
            action_scale (float): scaling from policy action to joint
                position offset relative to the default pose, in radians;
            soft_joint_limit (float): fraction of the joint range, centred on
                its midpoint, outside of which a limit penalty may apply;
            domain_randomization (bool): whether to randomise the initial
                pose and apply random pushes during episodes;
            init_joint_noise (float): half-width of the uniform noise added
                to the controlled joints at reset, in radians;
            init_vel_noise (float): half-width of the uniform planar velocity
                given to the pelvis at reset, in m/s;
            push_interval (int): mean number of steps between random pushes;
            push_max_vel (float): magnitude bound of the planar velocity set
                by a push, in m/s;
            n_substeps (int): physics steps per intermediate step. With the
                0.002 s model timestep and n_substeps=10 the defaults give a
                50 Hz policy rate; the joint PD runs inside MuJoCo at every
                physics step regardless;
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

        # Pelvis velocity in the pelvis frame, then the controlled joints'
        # positions and velocities. World position and orientation are not
        # observed: the policy sees orientation only through the projected
        # gravity vector, which is what the IMU provides on the real robot.
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
        self._init_joint_noise = init_joint_noise
        self._init_vel_noise = init_vel_noise
        self._push_interval = push_interval
        self._push_max_vel = push_max_vel
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
            **viewer_params,
        )

    # ------------------------------------------------------------------
    # MDP info / model dependent setup
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        m = self._model

        # The action is mapped to a joint position target, which only works
        # with position-type actuators (MuJoCo runs the PD at every physics
        # step). vendor_h2.py always builds those; this guards against a
        # hand-edited model.
        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise ValueError(
                "H2Base needs position-type actuators; rebuild the model with vendor_h2.py"
            )
        if m.nu != self._n_actuators:
            raise ValueError(
                f"model has {m.nu} actuators, expected {self._n_actuators}"
            )

        dev = wp.to_torch(self._data_wp.qpos).device
        self._device = dev

        def jid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise ValueError(f"joint {name!r} not in model")
            return i

        # Addresses of the controlled joints in qpos / qvel, resolved by
        # name. Every hinge has one qpos and one dof entry.
        for j in self.ALL_JOINTS:
            jid(j)  # every listed joint must exist
        ctrl_jids = [jid(j) for j in self._controlled_joints]
        # qpos address of the joint driven by each actuator, in actuator
        # order, so the default control vector never depends on ALL_JOINTS
        # matching the actuator order in the file.
        act_jids = m.actuator_trnid[:, 0]
        self._all_qpos_idx = torch.as_tensor(
            m.jnt_qposadr[act_jids], device=dev, dtype=torch.long
        )
        self._qpos_idx = torch.as_tensor(
            m.jnt_qposadr[ctrl_jids], device=dev, dtype=torch.long
        )
        self._dof_idx = torch.as_tensor(
            m.jnt_dofadr[ctrl_jids], device=dev, dtype=torch.long
        )

        # Position of each controlled joint within the actuator vector. The
        # actuators are in ALL_JOINTS order and named after their joint.
        act_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            for j in self._controlled_joints
        ]
        if min(act_ids) < 0:
            raise ValueError(
                "every controlled joint needs an actuator of the same name"
            )
        self._ctrl_idx = torch.as_tensor(act_ids, device=dev, dtype=torch.long)

        posture_jids = [
            self._controlled_joints.index(j)
            for j in self.POSTURE_JOINTS
            if j in self._controlled_joints
        ]
        self._posture_idx = torch.as_tensor(posture_jids, device=dev, dtype=torch.long)

        # Nominal standing pose from the "home" keyframe.
        key_qpos = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=dev)
        self._default_qpos = key_qpos
        self._default_ctrl = key_qpos[self._all_qpos_idx].clone()  # all actuators
        self._default_joint_pos = key_qpos[self._qpos_idx].clone()  # controlled joints
        self._nominal_height = float(m.key_qpos[0][2])

        # Joint limits of the controlled joints, and soft limits shrunk
        # towards the midpoint.
        rng = torch.as_tensor(m.jnt_range[ctrl_jids], dtype=torch.float32, device=dev)
        self._joint_lower = rng[:, 0]
        self._joint_upper = rng[:, 1]
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * self._soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        # Position targets are clamped to the actuator ctrlrange, which the
        # vendor script sets to the joint range.
        ctrlrange = m.actuator_ctrlrange
        self._ctrl_lower = torch.as_tensor(
            ctrlrange[:, 0], dtype=torch.float32, device=dev
        )
        self._ctrl_upper = torch.as_tensor(
            ctrlrange[:, 1], dtype=torch.float32, device=dev
        )

        self._feet_geom_ids = torch.as_tensor(
            [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in self.FEET_GEOMS
            ],
            device=dev,
        )
        if (self._feet_geom_ids < 0).any():
            raise ValueError(
                "foot geoms not found; rebuild the model with vendor_h2.py"
            )
        # The foot geoms are boxes; the sole is half the box height below
        # the geom centre.
        self._foot_half_height = torch.as_tensor(
            m.geom_size[self._feet_geom_ids.cpu().numpy(), 2],
            dtype=torch.float32,
            device=dev,
        )

        self._actions = torch.zeros(self._num_envs, self._n_joints, device=dev)
        self._episode_length = torch.zeros(self._num_envs, dtype=torch.long, device=dev)

        # Contiguous slices of the observation, for use by subclasses.
        first_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{self._controlled_joints[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        # BODY_VEL is ordered [angular(3), linear(3)].
        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        # Action space: joint position offsets in units of action_scale. The
        # parent's action space spans the raw actuator ctrlrange of all 31
        # actuators, which is not what the policy outputs.
        default_np = m.key_qpos[0][m.jnt_qposadr[ctrl_jids]]
        rng_np = m.jnt_range[ctrl_jids]
        action_low = (rng_np[:, 0] - default_np) / self._action_scale
        action_high = (rng_np[:, 1] - default_np) / self._action_scale
        mdp_info.action_space = Box(action_low, action_high)

        mdp_info = super()._modify_mdp_info(mdp_info)
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.ITERATIONS
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _preprocess_action(self, action):
        """
        Map the policy action (num_envs, n_joints) to position targets for
        all actuators (num_envs, n_actuators). Uncontrolled joints get their
        default angle.

        """
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action, min=-100.0, max=100.0)
        self._actions[:] = action
        target = self._default_ctrl.expand(action.shape[0], -1).clone()
        target[:, self._ctrl_idx] = (
            self._default_joint_pos + self._action_scale * action
        )
        return torch.clamp(target, self._ctrl_lower, self._ctrl_upper)

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
        """Gravity direction in the pelvis frame. (num_envs, 3)"""
        quat = self._read_data("base_rot")
        g = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(quat.shape[0], 3)
        return self._quat_rotate_inverse(quat, g)

    def _heading(self):
        """Yaw angle of the pelvis forward axis, in radians. (num_envs,)"""
        quat = self._read_data("base_rot")
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(
            quat.shape[0], 3
        )
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
        """World position of each foot box centre. (num_envs, 2, 3)"""
        return wp.to_torch(self._data_wp.geom_xpos)[:, self._feet_geom_ids, :]

    def _foot_height(self):
        """Height of each foot sole above the ground. (num_envs, 2)"""
        return self._foot_pos()[:, :, 2] - self._foot_half_height

    def _foot_up(self):
        """
        z component of each foot's up axis, 1 when the sole is flat on the
        ground. (num_envs, 2)

        """
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
        return (
            self._is_finite(obs) & self._is_within_z_range(obs) & self._is_upright(obs)
        )

    # ------------------------------------------------------------------
    # Reset and domain randomisation
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given environments to the standing keyframe. With domain
        randomisation the controlled joints get additive uniform noise and
        the pelvis a small random planar velocity.

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
            noise = (
                torch.rand(n, self._n_joints, device=qpos.device) * 2 - 1
            ) * self._init_joint_noise
            qpos[idx.unsqueeze(1), self._qpos_idx.unsqueeze(0)] = (
                self._default_joint_pos + noise
            )
            qvel[idx, 0:2] = (
                torch.rand(n, 2, device=qpos.device) * 2 - 1
            ) * self._init_vel_noise

        self._actions[idx] = 0.0
        self._episode_length[idx] = 0

        self._mj_warp.forward(self._model_wp, self._data_wp)

    def _step_finalize(self):
        self._episode_length += 1

        if self._domain_randomization:
            do_push = (
                torch.rand(self._num_envs, device=self._device)
                < 1.0 / self._push_interval
            )
            do_push &= self._episode_length > 50
            self._push_robots(torch.nonzero(do_push, as_tuple=True)[0])

    def _push_robots(self, env_indices):
        """Overwrite the pelvis planar velocity of the given environments."""
        if len(env_indices) == 0:
            return
        qvel = wp.to_torch(self._data_wp.qvel)
        vel = (
            torch.rand(len(env_indices), 2, device=self._device) * 2 - 1
        ) * self._push_max_vel
        qvel[env_indices, 0:2] = vel

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
