from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands

from .base import with_pushes
from .walk import H2Walk
from .params import (
    RUN_REWARD_WEIGHTS,
    RUN_REWARD_PARAMS,
    RUN_GAIT,
    RUN_TRACKING_SIGMA,
    RUN_COMMAND_GENERATOR,
    RUN_PUSHES,
)


class H2Run(H2Walk):
    """
    Running task for the Unitree H2.

    H2Walk with the gait changed, not just the speed: the duty factor drops
    below 0.5 so the contact schedule asks for a flight phase in every cycle,
    the cycle is shorter, the swing feet are pulled higher, and the vertical
    velocity and height terms are relaxed because a running gait bounces. The
    command range starts where walking stops. Observation and action layout
    are identical to H2Walk, so a walking policy can start the training.

    """

    def __init__(
        self,
        num_envs,
        reward_weights=None,
        reward_params=None,
        tracking_sigma=RUN_TRACKING_SIGMA,
        gait_period=RUN_GAIT["gait_period"],
        duty_factor=RUN_GAIT["duty_factor"],
        randomization_params=None,
        command_generator=None,
        **kwargs,
    ):
        super().__init__(
            num_envs,
            reward_weights={**RUN_REWARD_WEIGHTS, **(reward_weights or {})},
            reward_params={**RUN_REWARD_PARAMS, **(reward_params or {})},
            tracking_sigma=tracking_sigma,
            gait_period=gait_period,
            duty_factor=duty_factor,
            randomization_params=with_pushes(randomization_params, RUN_PUSHES),
            command_generator=(
                UniformVelocityCommands(**RUN_COMMAND_GENERATOR) if command_generator is None else command_generator
            ),
            **kwargs,
        )
