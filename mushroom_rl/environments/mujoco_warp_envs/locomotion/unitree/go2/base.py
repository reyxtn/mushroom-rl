import mujoco
import numpy as np
import torch
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import Randomizer
from mushroom_rl.utils.mujoco import ObservationType
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco_warp_envs.randomizer import {
    RandomizationParams,
    Randomizer,
}

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
        init_joint_noise=0.1,
        init_vel_noise=0.2,
        push_interval=750,
        push_max_vel=1.0,
        n_substeps=10,
        n_intermediate_steps=1,
        use_graph_capture=False,
        nconmax=None,
        njmax=256,
        scene="scene_mjx.xml",
        **viewer_params,
    ):
        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=self.ACTUATORS,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            model_batch_fields=self.BATCHED_MODEL_FIELDS,
            **viewer_params,
        )
