import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mj_warp

from mushroom_rl.core.spaces import Box
from mushroom_rl.utils.mujoco import ObservationType
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import (
    Randomizer,
)

from .fetch_go2_model import ensure_go2_model

from .params import (
    LEG_JOINTS,
    ACTUATORS,
    FEET_GEOMS,
    ROOT_BODY,
    BATCHED_MODEL_FIELDS,
    RANDOMIZATION_OBS,
)


class Go2Base(MuJoCoWarp):
    """
    Base constructor for the Go2 quadruped in MuJoCo Warp.
    """

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        healthy_z_range=(0.15, 0.45),
        healthy_gravity_z=-0.6,
        terminate_when_unhealthy=True,
        action_scale=0.25,
        soft_joint_limit=0.9,
        domain_randomization=True,
        randomization_params=None,
        observed_randomization=(),
        max_delay_limit=0.04,
        n_substeps=10,
        n_intermediate_steps=1,
        use_graph_capture=False,
        nconmax=None,
        njmax=256,
        scene="scene_mjx.xml",
        **viewer_params,
    ):

        xml_path = str(ensure_go2_model(scene))

        unknown = set(observed_randomization) - set(RANDOMIZATION_OBS)

        if unknown:
            raise ValueError(f"unknown randomized observations: {sorted(unknown)}")
        self._observed_randomization = tuple(observed_randomization)
        self._max_delay_limit = max_delay_limit

        observation_spec = [("base_vel", ROOT_BODY, ObservationType.BODY_VEL)]
        observation_spec += [
            (f"{j}_pos", j, ObservationType.JOINT_POS) for j in LEG_JOINTS
        ]
        observation_spec += [
            (f"{j}_vel", j, ObservationType.JOINT_VEL) for j in LEG_JOINTS
        ]
        additional_data_spec = [
            ("base_pos", ROOT_BODY, ObservationType.BODY_POS),
            ("base_rot", ROOT_BODY, ObservationType.BODY_ROT),
        ]
        self._n_substeps = n_substeps
        self._randomization_params = randomization_params

        self._healthy_z_range = healthy_z_range
        self._healthy_gravity_z = healthy_gravity_z
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._action_scale = action_scale
        self._soft_joint_limit = soft_joint_limit
        self._domain_randomization = domain_randomization
        self._num_envs = num_envs

        # Must precede super().__init__(): the parent calls _modify_mdp_info
        # partway through its own constructor, and it uses this.
        self._n_joints = len(LEG_JOINTS)

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=ACTUATORS,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

        self._model_wp = self._mj_warp.put_model(
            self._model, batch_sizes={f: num_envs for f in BATCHED_MODEL_FIELDS}
        )

        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~self._mj_warp.OverflowType.ITERATIONS

        self._model_wp = mj_warp.put_model(
            self._model, batch_sizes={f: num_envs for f in BATCHED_MODEL_FIELDS}
        )
        self._model_wp.opt.warn_overflow &= ~mj_warp.OverflowType.LS_ITERATIONS
        self._model_wp.opt.warn_overflow &= ~mj_warp.OverflowType.ITERATIONS

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
        leg_jids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in LEG_JOINTS
        ]
        if min(leg_jids) < 0 or not np.array_equal(act_jids, np.asarray(leg_jids)):
            raise ValueError("the actuators do not drive LEG_JOINTS in order")
        self._qpos_idx = torch.as_tensor(
            m.jnt_qposadr[act_jids], device=dev, dtype=torch.long
        )
        self._dof_idx = torch.as_tensor(
            m.jnt_dofadr[act_jids], device=dev, dtype=torch.long
        )
        self._ctrl_idx = torch.arange(m.nu, device=dev)

        # Nominal standing pose, taken from the "home" keyframe of the MJCF.
        # qpos0 has the legs fully extended and the base at 0.445, which is
        # not a pose the robot can hold; the keyframe is the one to use.
        key_qpos = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=dev)
        self._default_qpos = key_qpos
        self._default_joint_pos = key_qpos[self._qpos_idx].clone()
        self._default_ctrl = self._default_joint_pos  # every actuator is controlled

        # Joint limits from the model, and soft limits shrunk towards the
        # midpoint.
        rng = torch.as_tensor(m.jnt_range[act_jids], dtype=torch.float32, device=dev)
        self._joint_lower = rng[:, 0]
        self._joint_upper = rng[:, 1]
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * self._soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        # Position targets are clamped to the actuator ctrlrange.
        ctrlrange = m.actuator_ctrlrange
        self._ctrl_lower = torch.as_tensor(
            ctrlrange[:, 0], dtype=torch.float32, device=dev
        )
        self._ctrl_upper = torch.as_tensor(
            ctrlrange[:, 1], dtype=torch.float32, device=dev
        )

        self._feet_geom_ids = torch.as_tensor(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FEET_GEOMS],
            device=dev,
        )
        if (self._feet_geom_ids < 0).any():
            raise ValueError("foot geoms not found in the model")

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

        # BODY_VEL is ordered [angular(3), linear(3)].
        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        # The action is a joint position offset scaled by action_scale, so
        # the action space is the joint range expressed in those units. The
        # space computed by the parent is the raw ctrl range, which is not
        # what the policy outputs.
        default_joint_np = m.key_qpos[0][m.jnt_qposadr[act_jids]]
        jnt_range_np = m.jnt_range[act_jids]
        action_low = (jnt_range_np[:, 0] - default_joint_np) / self._action_scale
        action_high = (jnt_range_np[:, 1] - default_joint_np) / self._action_scale
        mdp_info.action_space = Box(action_low, action_high)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _build_randomizer(self, m, act_jids):
        self._step_dt = m.opt.timestep * self._n_substeps
        self._max_delay_steps_limit = int(round(self._max_delay_limit / self._step_dt))
        dev = self._device
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        ancestors = []
        b = trunk
        while b >= 0:
            ancestors.append(int(b))
            b = int(m.body_parentid[b]) if b != 0 else -1

        self._model_views = {
            name: wp.to_torch(getattr(self._model_wp, name))
            for name in BATCHED_MODEL_FIELDS
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
        self._action_history = self._default_ctrl.expand(
            L, self._num_envs, m.nu
        ).clone()
        self._history_head = 0

        if self._domain_randomization:
            self._randomizer.resample_startup(self._all_envs)
