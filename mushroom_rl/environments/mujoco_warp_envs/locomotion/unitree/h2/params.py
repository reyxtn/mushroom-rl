"""
Constants of the Unitree H2 and of its locomotion tasks, kept free of
warp/mujoco imports so the real-robot side can read them too.

Joint order is the actuator order of the model built by fetch_h2_model.py
(the URDF joint order). Map it explicitly onto the SDK when deploying.

"""

import math

LEG_JOINTS = (
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
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
HEAD_JOINTS = ("head_pitch_joint", "head_yaw_joint")
ARM_JOINTS = (
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
)
UPPER_ARM_JOINTS = tuple(j for j in ARM_JOINTS if "wrist" not in j)

# Every actuated joint, in actuator order. The policy controls a subset; the
# others are held at their default angle.
ALL_JOINTS = LEG_JOINTS + WAIST_JOINTS + HEAD_JOINTS + ARM_JOINTS
DEFAULT_CONTROLLED_JOINTS = LEG_JOINTS + WAIST_JOINTS + UPPER_ARM_JOINTS

# Joints the posture term keeps near their default angle.
POSTURE_JOINTS = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
) + WAIST_JOINTS + UPPER_ARM_JOINTS

# One box per foot (see fetch_h2_model.py); left, right.
FEET_GEOMS = ("left_foot", "right_foot")
GAIT_OFFSETS = (0.0, 0.5)  # half a period apart
ROOT_BODY = "pelvis"
FLOOR_GEOM = "floor"

# Model fields allocated per world, so the randomizer can write them.
BATCHED_MODEL_FIELDS = (
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
)

# Randomized quantities the environment can expose as observations.
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

# ---------------------------------------------------------------------------
# Reward. The Rudin set plus the terms a biped needs: a gait clock, upright
# and height, swing clearance, slip, feet distance, posture. Every weight
# carries its sign, every term is multiplied by dt, the total is clamped at
# zero (the alive term is what keeps the early signal positive).
# ---------------------------------------------------------------------------

REWARD_TERMS = (
    "alive",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "base_height",
    "torques",
    "joint_acc",
    "action_rate",
    "joint_pos_limits",
    "joint_deviation",
    "feet_air_time",
    "feet_slip",
    "feet_clearance",
    "feet_distance",
    "gait",
    "no_fly",
    "stand_still",
)

DEFAULT_REWARD_WEIGHTS = dict(
    alive=2.0,
    tracking_lin_vel=1.0,
    tracking_ang_vel=0.5,
    lin_vel_z=-2.0,
    ang_vel_xy=-0.05,
    orientation=-1.0,
    base_height=-10.0,
    torques=-1e-6,
    joint_acc=-2.5e-7,
    action_rate=-0.01,
    joint_pos_limits=-5.0,
    joint_deviation=-0.1,
    feet_air_time=0.5,
    feet_slip=-0.2,
    feet_clearance=-2.0,
    feet_distance=-2.0,
    gait=0.5,
    no_fly=-1.0,
    stand_still=-0.25,
)

DEFAULT_REWARD_PARAMS = dict(
    foot_contact_height=0.015,  # sole height below which a foot is in contact
    feet_clearance_target=0.08,
    feet_air_time_threshold=0.4,
    feet_min_distance=0.18,
    base_height_target=0.96,  # the keyframe stands at 1.01 and settles here under load
    moving_threshold=0.1,  # planar command norm above which the gait terms apply
    stand_still_base_scale=1.0,  # base m/s and rad/s next to the joint radians in stand_still
)

# Balance in place: no step reward, more weight on height and orientation.
STAND_REWARD_WEIGHTS = dict(
    base_height=-20.0,
    orientation=-2.0,
    feet_air_time=0.0,
    feet_clearance=0.0,
    gait=1.0,
    stand_still=-0.5,
)

# Gait clock: period in seconds and duty factor. Duty > 0.5 always keeps a
# foot on the ground.
WALK_GAIT = dict(gait_period=0.8, duty_factor=0.6)
WALK_TRACKING_SIGMA = 0.25

# Command generators, the sampler of the old H2 tasks: uniform ranges, a
# dead zone, per-step resampling (1/500 after 50 steps), every world tracking
# a heading target with the yaw rate clipped to +-1 rad/s.
WALK_COMMAND_GENERATOR = dict(
    max_command_ranges=dict(lin_vel_x=(-3.0, 3.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)),
    command_ranges=dict(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)),
    command_dead_zone=0.2,
    command_resampling_time_range=None,
    heading_control_stiffness=0.5,
    rel_heading_envs=1.0,
    rel_standing_envs=0.15,
)
STAND_COMMAND_GENERATOR = dict(
    WALK_COMMAND_GENERATOR,
    command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
    rel_standing_envs=1.0,
)

# Pushes of the old tasks (randomizer push_probability / push_max_velocity).
STAND_PUSHES = dict(push_probability=1.0 / 250.0, push_max_velocity=0.8)
WALK_PUSHES = dict(push_probability=1.0 / 750.0, push_max_velocity=0.5)

# Observation scaling, and noise half-widths in unscaled units.
OBS_SCALES = dict(base_lin_vel=2.0, base_ang_vel=0.25, joint_pos=1.0, joint_vel=0.05, commands=(2.0, 2.0, 0.25))
OBS_NOISE = dict(base_lin_vel=0.1, base_ang_vel=0.2, joint_pos=0.01, joint_vel=1.5, projected_gravity=0.05)

YAW_RANGE = math.pi
