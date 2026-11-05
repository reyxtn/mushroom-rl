from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands

from .base import with_pushes
from .walk import H2Walk
from .params import STAND_REWARD_WEIGHTS, STAND_COMMAND_GENERATOR, STAND_PUSHES


class H2Stand(H2Walk):
    """
    Balance-in-place task for the Unitree H2.

    H2Walk with the velocity command pinned to zero, so the observation and
    action layout are identical and a standing policy can start the walk
    training. With a zero command the gait schedule already asks for both
    feet down, so the differences are the weights (STAND_REWARD_WEIGHTS) and
    stronger, more frequent pushes (STAND_PUSHES, unless the randomization
    params set their own).

    """

    def __init__(self, num_envs, reward_weights=None, randomization_params=None, command_generator=None, **kwargs):
        super().__init__(
            num_envs,
            reward_weights={**STAND_REWARD_WEIGHTS, **(reward_weights or {})},
            randomization_params=with_pushes(randomization_params, STAND_PUSHES),
            command_generator=(
                UniformVelocityCommands(**STAND_COMMAND_GENERATOR) if command_generator is None else command_generator
            ),
            **kwargs,
        )
