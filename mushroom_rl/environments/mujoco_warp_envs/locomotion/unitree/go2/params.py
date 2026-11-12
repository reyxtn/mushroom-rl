"""
Constants of the Unitree Go2 and of its walking task, kept free of warp/mujoco
imports so the real-robot side can read them too.

Joint order is the SIM order (FL, FR, RL, RR), which is also the action order
of the new_isaac Go2Isaac. It is NOT the Unitree SDK motor order (FR, FL, RR,
RL): map it explicitly when deploying.

"""

import math

LEG_JOINTS = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

ACTUATORS = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)

# Nominal standing pose, the one new_isaac's Go2Isaac (and legged_gym's Go2)
# expresses the actions relative to. Replaces the menagerie "home" keyframe,
# whose legs are straighter (0, 0.9, -1.8).
DEFAULT_JOINT_POS = (
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
)

# Nominal PD gains of the control law, as in new_isaac (QuadrupedIsaac's
# nominal_p_gain / nominal_d_gain) and legged_gym. The menagerie actuators
# ship with kp=50; Go2Base overwrites them with these before batching.
NOMINAL_P_GAIN = 20.0
NOMINAL_D_GAIN = 0.5

# Maximum joint speed, rad/s, per joint (hip, thigh, calf) x 4. The menagerie
# MJCF has no velocity limits; these are the Go2 datasheet values. Only the
# joint_vel_limits reward term reads them. Check against the Isaac USD.
JOINT_MAX_VEL = (30.1, 30.1, 15.7) * 4

# Foot geoms (spheres of radius 0.0175, priority 1).
FEET_GEOMS = ("FL", "FR", "RL", "RR")
FOOT_RADIUS = 0.0175
ROOT_BODY = "base"
FLOOR_GEOM = "floor"

# Bodies whose (non-foot) geoms touching the ground count as a collision,
# the "lower_body" group of Go2Isaac.
LOWER_BODY_BODIES = (
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)

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
# Reward. Names, signs and defaults are those of new_isaac's QuadrupedIsaac:
# every weight carries its sign, and every term is multiplied by dt.
# ---------------------------------------------------------------------------

BASE_REWARD_TERMS = (
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy", "torques",
    "joint_acc", "feet_air_time", "collision", "action_rate", "joint_pos_limits",
)

OPTIONAL_REWARD_TERMS = (
    "flat_orientation", "joint_vel_limits", "power_draw", "similar_to_default",
    "stand_still_deviation", "base_height", "feet_air_time_high", "feet_air_time_low",
    "feet_clearance", "feet_clearance_lateral", "feet_slide", "feet_slide_low", "feet_z_velocity",
    "feet_air_time_symmetry", "long_contact", "stand_still_short_contact",
)

# QuadrupedIsaac's defaults: the legged_gym set, every optional term off.
DEFAULT_REWARD_WEIGHTS = dict(
    tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-2.0, ang_vel_xy=-0.05, torques=-0.0002,
    joint_acc=-2.5e-7, feet_air_time=1.0, collision=-1.0, action_rate=-0.01, joint_pos_limits=-10.0,
    termination=-500.,
    **{name: 0. for name in OPTIONAL_REWARD_TERMS},
)

# The weights new_isaac's go2_curriculum.py trains with. The terms that keep
# the trunk level while turning are flat_orientation (tilt), ang_vel_xy (roll
# and pitch rate) and base_height; the lateral clearance and the slide terms
# make side-stepping the cheaper way to change direction.
GO2_REWARD_WEIGHTS = dict(
    tracking_lin_vel=1.2, tracking_ang_vel=0.6, lin_vel_z=-1.6, ang_vel_xy=-0.07,
    torques=-1e-5, joint_acc=-2.5e-7, action_rate=-0.005, joint_pos_limits=-0.1,
    collision=-1.0, feet_air_time=0., flat_orientation=-0.2, joint_vel_limits=-0.1,
    power_draw=-4e-6, similar_to_default=-0.085, stand_still_deviation=-1.4,
    base_height=-50., feet_air_time_high=0.125, feet_air_time_low=0.75,
    feet_air_time_symmetry=-0.25, feet_clearance=0.085, feet_clearance_lateral=0.1,
    feet_slide=-0.04, feet_slide_low=-0.2, feet_z_velocity=-0.03, long_contact=-0.7,
    stand_still_short_contact=-0.5,
)

DEFAULT_REWARD_PARAMS = dict(
    command_threshold=0.05, low_speed_threshold=0.5, air_time_threshold_high=0.5,
    air_time_threshold_low=0.25, air_time_symmetry_std=0.05, clearance_target=0.03, clearance_std=0.02,
    clearance_lateral_target=0.05, clearance_lateral_std=0.02,
    clearance_lateral_command_threshold=0.3, long_contact_threshold=0.4, long_contact_ramp_power=2.,
    long_contact_ramp_cap=1., base_height_target=0.3, joint_vel_limits_soft_ratio=0.9,
    stand_still_contact_target=0.5, stand_still_ramp_power=2., stand_still_ramp_cap=0.5,
)

# Tolerance of the two tracking terms: exp(-err / std^2), with
# std = max(base, slope * |command|).
DEFAULT_TRACKING_STDS = dict(lin_vel=0.5, lin_vel_slope=0., ang_vel=0.5, ang_vel_slope=0.)

# The command generator new_isaac's go2_curriculum.py trains with.
GO2_COMMAND_GENERATOR = dict(
    max_command_ranges=dict(lin_vel_x=(-2.5, 2.5), lin_vel_y=(-2.5, 2.5), ang_vel_z=(-1.5, 1.5)),
    command_ranges=dict(ang_vel_z=(-1.5, 1.5)),
    command_resampling_time_range=(5., 10.),
    heading_control_stiffness=1., rel_heading_envs=0.5, rel_standing_envs=0.1, command_dead_zone=0.,
    frac_rotating_envs=0.15, frac_low_speed_envs=0.35, low_speed_threshold=0.5,
)

# Observation noise half-widths, unscaled (new_isaac normalizes with a
# StandardizationPreprocessor in the agent, not in the environment).
OBS_NOISE = dict(base_lin_vel=0.1, base_ang_vel=0.2, joint_pos=0.01, joint_vel=1.5, projected_gravity=0.05)

YAW_RANGE = math.pi
