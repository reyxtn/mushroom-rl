LEG_JOINTS = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

ACTUATORS = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]

# Foot geoms, used for the contact proxy. Spheres of radius 0.0175 with
# priority 1, so their friction wins every foot-floor contact.
FEET_GEOMS = ["FL", "FR", "RL", "RR"]
ROOT_BODY = "base"

# Model fields allocated per world, so the randomizer can write them.
BATCHED_MODEL_FIELDS = [
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
]
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


REWARD_KEYS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "torques",
    "joint_acc",
    "feet_air_time",
    "action_rate",
    "joint_pos_limits",
)
