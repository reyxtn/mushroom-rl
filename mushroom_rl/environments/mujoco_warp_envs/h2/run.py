from .walk import H2Walk


class H2Run(H2Walk):
    """
    Running task for the Unitree H2.

    A subclass of H2Walk that changes the gait, not just the speed: the
    duty factor drops below 0.5, so the contact schedule asks for a flight
    phase in every cycle, the cycle is shorter, the swing feet are pulled
    higher, and the vertical velocity and height terms are relaxed because
    a running gait bounces. The command range starts where walking stops.

    Observation and action layout are identical to H2Walk and H2Stand, so a
    walking policy can be loaded as the starting point.

    """

    def __init__(
        self,
        num_envs,
        lin_vel_x_range=(1.0, 3.0),
        lin_vel_y_range=(-0.3, 0.3),
        command_deadband=0.5,
        tracking_sigma=0.5,
        gait_period=0.5,
        duty_factor=0.35,
        feet_clearance_target=0.15,
        feet_air_time_threshold=0.2,
        lin_vel_z_weight=0.5,
        base_height_weight=2.0,
        feet_air_time_weight=1.0,
        gait_weight=1.0,
        push_max_vel=0.3,
        **kwargs,
    ):
        """
        Constructor. Arguments not listed are forwarded to H2Walk.

        """
        super().__init__(
            num_envs,
            lin_vel_x_range=lin_vel_x_range,
            lin_vel_y_range=lin_vel_y_range,
            command_deadband=command_deadband,
            tracking_sigma=tracking_sigma,
            gait_period=gait_period,
            duty_factor=duty_factor,
            feet_clearance_target=feet_clearance_target,
            feet_air_time_threshold=feet_air_time_threshold,
            lin_vel_z_weight=lin_vel_z_weight,
            base_height_weight=base_height_weight,
            feet_air_time_weight=feet_air_time_weight,
            gait_weight=gait_weight,
            push_max_vel=push_max_vel,
            **kwargs,
        )
