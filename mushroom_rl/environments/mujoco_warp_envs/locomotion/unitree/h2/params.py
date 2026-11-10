"""
Constants of the Unitree H2 and of its locomotion task, kept free of
warp/mujoco imports so the real-robot side can read them too.

Joint order is the actuator order of the model built by fetch_h2_model.py
(the URDF joint order). Map it explicitly onto the SDK when deploying.

"""

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

# One box per foot (see fetch_h2_model.py); left, right.
FEET_GEOMS = ("left_foot", "right_foot")
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
# Reward: the locomotion recipe of Seo et al. 2025 (arXiv 2512.01996), with
# the terms, weights and parameters of its released PPO preset (holosoma,
# g1_29dof_loco). Every weight carries its sign and every term is multiplied
# by dt. The total is not clamped at zero; the penalty curriculum keeps the
# early signal positive instead.
# ---------------------------------------------------------------------------

REWARD_TERMS = (
    "alive",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_phase",
    "ang_vel_xy",
    "orientation",
    "action_rate",
    "pose",
    "feet_distance",
    "feet_orientation",
    "termination",
)

# The terms the penalty curriculum scales (H2Walk.penalty_scale).
PENALTY_TERMS = (
    "ang_vel_xy",
    "orientation",
    "action_rate",
    "pose",
    "feet_distance",
    "feet_orientation",
)

DEFAULT_REWARD_WEIGHTS = dict(
    alive=1.0,
    tracking_lin_vel=2.0,
    tracking_ang_vel=1.5,
    feet_phase=5.0,
    ang_vel_xy=-1.0,
    orientation=-10.0,
    action_rate=-2.0,
    pose=-0.5,
    feet_distance=-10.0,
    feet_orientation=-5.0,
    termination=-1000.0,
)

DEFAULT_REWARD_PARAMS = dict(
    swing_height=0.09,  # peak of the foot height profile
    feet_phase_sigma=0.008,  # width of the foot height kernel, in m^2
    feet_min_distance=0.125,  # lateral; 0.63 of the 0.198 the feet stand apart, as 0.15 is on the G1
    standing_threshold=0.1,  # command norm below which the world stands
    reward_floor=-1.0,  # per step: a diverging world must not put a huge value in the batch
)

# Weight of every joint in the pose term, (substring of joint name, weight).
# First match wins. The joints that swing the leg are free, the ones that
# splay it are not, and the upper body is held.
POSE_WEIGHTS = (
    ("hip_pitch", 0.01),
    ("hip_roll", 1.0),
    ("hip_yaw", 5.0),
    ("knee", 0.01),
    ("ankle", 5.0),
    ("waist", 50.0),
    ("head", 50.0),
    ("shoulder", 50.0),
    ("elbow", 50.0),
    ("wrist", 50.0),
)

WALK_TRACKING_SIGMA = 0.25

# The mean episode length (the signal the penalty curriculum reads) runs over
# this many finished episodes per world: 1000 episodes at 4096 worlds, as in
# the paper.
EPISODE_AVERAGE_WINDOW = 0.25

# Gait clock: the two feet follow the same height profile half a cycle
# apart. Under domain randomization the cadence of an episode is drawn from
# this range, in Hz; without it the midpoint is used.
GAIT_FREQUENCY_RANGE = (0.8, 1.2)

# Commands: redrawn every 10 s, a fifth of the worlds standing still, as in
# the paper; the ranges, the dead zone and the heading target are the H2's.
WALK_COMMAND_GENERATOR = dict(
    max_command_ranges=dict(
        lin_vel_x=(-3.0, 3.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
    ),
    command_ranges=dict(
        lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    ),
    command_dead_zone=0.2,
    command_resampling_time_range=(10.0, 10.0),
    heading_control_stiffness=0.5,
    rel_heading_envs=1.0,
    rel_standing_envs=0.2,
)

# Pushes (randomizer push_probability / push_max_velocity).
WALK_PUSHES = dict(push_probability=1.0 / 750.0, push_max_velocity=0.5)

# Observation scaling, and noise half-widths in unscaled units.
OBS_SCALES = dict(
    base_lin_vel=2.0,
    base_ang_vel=0.25,
    joint_pos=1.0,
    joint_vel=0.05,
    commands=(2.0, 2.0, 0.25),
)
OBS_NOISE = dict(
    base_lin_vel=0.1,
    base_ang_vel=0.2,
    joint_pos=0.01,
    joint_vel=1.5,
    projected_gravity=0.05,
)
