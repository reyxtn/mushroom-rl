"""
Constants of the Unitree H2 and of its velocity task, kept free of
warp/mujoco imports so the real-robot side can read them too. The values are
Unitree's own (unitree_rl_mjlab: h2_constants.py, velocity_env_cfg.py and
config/h2/env_cfgs.py).

Joint order is the actuator order of the model built by fetch_h2_model.py.
Map it explicitly onto the SDK when deploying.

"""

import math

from .fetch_h2_model import FEET_BODIES, FEET_SITES, ROOT_BODY  # noqa: F401

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

# Every actuated joint, in actuator order. The policy controls all of them.
ALL_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS
DEFAULT_CONTROLLED_JOINTS = ALL_JOINTS

# Radians of joint target per unit of action, for every joint. Unitree's
# h2_constants.py also computes a scale per joint (0.25 * effort limit / kp),
# but their H2 task never applies it and runs on this default.
ACTION_SCALE = 0.25

TORSO_BODY = "torso_link"
FLOOR_GEOM = "floor"
ANGMOM_SENSOR = "root_angmom"  # angular momentum of the subtree of the torso

# The robot is terminated when the pelvis tilts further than this.
FALL_ANGLE = math.radians(70.0)

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
# Reward: the terms, weights and parameters of Unitree's H2 velocity task,
# under the names they have there. Every weight carries its sign and every
# term is multiplied by dt.
# ---------------------------------------------------------------------------

REWARD_TERMS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "body_orientation_l2",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "is_terminated",
    "joint_acc_l2",
    "joint_pos_limits",
    "action_rate_l2",
    "foot_gait",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "stand_still",
    "self_collisions",
)

DEFAULT_REWARD_WEIGHTS = dict(
    track_linear_velocity=1.0,
    track_angular_velocity=1.0,
    body_orientation_l2=-1.0,
    pose=1.0,
    body_ang_vel=-0.05,
    angular_momentum=-0.025,
    is_terminated=-200.0,
    joint_acc_l2=-2.5e-7,
    joint_pos_limits=-10.0,
    action_rate_l2=-0.05,
    foot_gait=0.5,
    foot_clearance=-1.0,
    foot_slip=-0.25,
    soft_landing=-1e-3,
    stand_still=-1.0,
    self_collisions=-1.0,
)

DEFAULT_REWARD_PARAMS = dict(
    lin_vel_std=math.sqrt(0.25),  # tracking kernels: exp(-err / std^2)
    ang_vel_std=math.sqrt(0.5),
    command_threshold=0.1,  # |v_xy| + |w_z| below which the world stands
    running_threshold=1.5,  # and above which it runs
    gait_period=0.6,  # seconds
    gait_offsets=(0.0, 0.5),  # left, right, in cycles
    gait_stance=0.56,  # fraction of the cycle a foot is on the ground
    clearance_height=0.10,  # of the foot site while it moves
    self_collision_force=10.0,  # N, above which a contact of the robot with itself counts
    reward_floor=-10.0,  # per step, before the fall penalty: a diverging world must not flood the batch
)

# Tolerance of the pose term, per joint, (substring of joint name, std). First
# match wins. Standing holds the default pose tightly; walking and running
# free the joints a stride needs.
POSE_STD_STANDING = (("joint", 0.05),)
POSE_STD_WALKING = (
    ("hip_pitch", 0.5),
    ("hip_roll", 0.15),
    ("hip_yaw", 0.15),
    ("knee", 0.5),
    ("ankle_roll", 0.1),
    ("ankle_pitch", 0.15),
    ("waist_yaw", 0.15),
    ("waist_roll", 0.1),
    ("waist_pitch", 0.1),
    ("shoulder_pitch", 0.15),
    ("shoulder_roll", 0.1),
    ("shoulder_yaw", 0.1),
    ("elbow", 0.1),
    ("wrist", 0.1),
)
POSE_STD_RUNNING = (
    ("hip_pitch", 0.5),
    ("hip_roll", 0.25),
    ("hip_yaw", 0.25),
    ("knee", 0.5),
    ("ankle_roll", 0.1),
    ("ankle_pitch", 0.25),
    ("waist_yaw", 0.25),
    ("waist_roll", 0.1),
    ("waist_pitch", 0.1),
    ("shoulder_pitch", 0.25),
    ("shoulder_roll", 0.1),
    ("shoulder_yaw", 0.1),
    ("elbow", 0.1),
    ("wrist", 0.1),
)

# Commands: redrawn every 3 to 8 s, every world tracking a heading target,
# one in twenty standing still, a linear command below 0.1 m/s set to zero.
# The training script starts from narrower ranges and widens them to these.
WALK_COMMAND_GENERATOR = dict(
    max_command_ranges=dict(lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)),
    command_ranges=dict(lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)),
    command_dead_zone=0.1,
    command_resampling_time_range=(3.0, 8.0),
    heading_control_stiffness=0.5,
    rel_heading_envs=1.0,
    rel_standing_envs=0.05,
)

# Pushes: every 5 to 6 s, never in the first 5 s (250 steps) of an episode,
# the randomizer adds up to push_max_velocity to the planar velocity of the
# pelvis; the environment adds the rest of Unitree's push on top, up to these
# values: linear z in m/s, then roll, pitch and yaw rate in rad/s.
WALK_PUSHES = dict(push_interval_range=(5.0, 6.0), push_min_episode_length=250, push_max_velocity=0.5)
PUSH_EXTRA_VELOCITY = (0.4, 0.52, 0.52, 0.78)

# Uniform observation noise half-widths, on what the actor reads.
OBS_NOISE = dict(base_ang_vel=0.2, projected_gravity=0.05, joint_pos=0.01, joint_vel=1.5)
