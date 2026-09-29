import torch

from .walk import H2Walk


class H2Stand(H2Walk):
    """
    Balance-in-place task for the Unitree H2.

    A subclass of H2Walk with the velocity command pinned to zero, so the
    observation and action layout are identical to the walking task and a
    standing policy can be loaded as the starting point for walk training.
    With a zero command the gait schedule already asks for both feet down,
    so the differences are the weights: no step reward, more weight on
    height and orientation, and stronger, more frequent pushes.

    """

    def __init__(
        self,
        num_envs,
        push_interval=250,
        push_max_vel=0.8,
        base_height_weight=20.0,
        orientation_weight=2.0,
        feet_air_time_weight=0.0,
        feet_clearance_weight=0.0,
        gait_weight=1.0,
        **kwargs,
    ):
        """
        Constructor. Arguments not listed are forwarded to H2Walk; the
        command ranges are fixed at zero and cannot be overridden.

        """
        for k in ("lin_vel_x_range", "lin_vel_y_range", "heading_range"):
            kwargs.pop(k, None)
        super().__init__(
            num_envs,
            lin_vel_x_range=(0.0, 0.0),
            lin_vel_y_range=(0.0, 0.0),
            heading_range=(0.0, 0.0),
            push_interval=push_interval,
            push_max_vel=push_max_vel,
            base_height_weight=base_height_weight,
            orientation_weight=orientation_weight,
            feet_air_time_weight=feet_air_time_weight,
            feet_clearance_weight=feet_clearance_weight,
            gait_weight=gait_weight,
            **kwargs,
        )

    def _resample_commands(self, env_indices):
        """Zero velocity; hold the heading the robot has at reset."""
        if len(env_indices) == 0:
            return
        self._commands[env_indices, :3] = 0.0
        self._commands[env_indices, 3] = self._heading()[env_indices]
