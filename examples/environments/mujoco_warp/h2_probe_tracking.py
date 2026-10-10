"""
Measure how well a trained H2 policy follows fixed velocity commands.

For each command in COMMANDS, N_ENVS worlds start from the keyframe, the
command is held fixed (no resampling, no pushes, no observation noise), and
the greedy policy runs for --steps steps. Reported per command:

    survive   fraction of worlds still upright at the end
    vx, vy    mean achieved base velocity (base frame), over upright worlds,
              after the first second (to skip the acceleration phase)
    yaw       mean achieved yaw rate
    tilt      mean pelvis tilt from vertical, degrees
    sway      std of the pelvis roll/pitch rate, rad/s (wobble)
    slip      mean planar speed of feet that are on the ground, m/s

    python h2_probe_tracking.py --agent /path/to/walk.msh
    python h2_probe_tracking.py --agent /path/to/run.msh --task run

"""

import argparse
import math

import torch
import warp as wp

from mushroom_rl.core import Agent
from mushroom_rl.environments.mujoco_warp_envs import H2Run, H2Stand, H2Walk
from mushroom_rl.utils.torch_utils import TorchUtils

TASKS = {"stand": H2Stand, "walk": H2Walk, "run": H2Run}

# (vx, vy, yaw rate). Covers every direction the walk task was trained on.
COMMANDS = {
    "walk": [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-0.3, 0.0, 0.0),
        (-0.6, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, -0.3, 0.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.0, -0.5),
        (0.5, 0.0, 0.5),
    ],
    "run": [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 0.0, 0.5)],
    "stand": [(0.0, 0.0, 0.0)],
}


def load_agent(path, device):
    TorchUtils.set_default_device(device)
    agent = Agent.load(path)
    policy = agent.policy
    if hasattr(policy, "_mu") and hasattr(policy._mu, "network"):
        policy._mu.network.to(device)
    if hasattr(policy, "_log_sigma"):
        policy._log_sigma.data = policy._log_sigma.data.to(device)
    return agent


def set_command(env, vx, vy, yaw):
    env._direct_yaw = True
    env._commands[:, 0] = vx
    env._commands[:, 1] = vy
    env._commands[:, 2] = yaw
    env._commands[:, 3] = 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--task", choices=sorted(TASKS), default="walk")
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=500, help="policy steps per command (50 Hz)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--privileged", nargs="*", default=["actual_delay", "joint_calib_offset"],
                   help="privileged observation entries the policy was trained with (h2_curriculum.py "
                        "--privileged); pass none for an older symmetric checkpoint")
    p.add_argument("--dr", action="store_true",
                   help="probe with the domain randomization on (default: nominal model, no pushes)")
    args = p.parse_args()

    agent = load_agent(args.agent, args.device)
    env = TASKS[args.task](
        num_envs=args.n_envs,
        use_graph_capture=True,
        domain_randomization=args.dr,
        obs_noise=False,
        command_resample_interval=10**9,
        observed_randomization=tuple(n for n in args.privileged if n.lower() != "none"),
    )
    trained = tuple(agent.mdp_info.observation_space.shape)
    if trained != tuple(env.info.observation_space.shape):
        raise SystemExit(f"agent trained on observations {trained}, environment builds "
                         f"{tuple(env.info.observation_space.shape)}: fix --privileged")
    dev = env._device
    mask = torch.ones(args.n_envs, dtype=torch.bool, device=dev)
    warmup = int(1.0 / env.dt)

    print(f"{'command (vx vy yaw)':>22} | {'survive':>7} {'vx':>6} {'vy':>6} {'yaw':>6} "
          f"{'tilt':>6} {'sway':>6} {'slip':>6}")
    print("-" * 80)
    for vx, vy, yaw in COMMANDS[args.task]:
        torch.manual_seed(0)
        obs, _ = env.reset_all(mask)
        set_command(env, vx, vy, yaw)
        obs, _ = env.reset_all(mask)  # rebuild obs with the command in it
        set_command(env, vx, vy, yaw)

        alive = mask.clone()
        acc = {k: torch.zeros((), device=dev) for k in ("vx", "vy", "yaw", "tilt", "slip")}
        rate_sq = torch.zeros((), device=dev)
        n = torch.zeros((), device=dev)
        last_feet = env._foot_pos().clone()

        for t in range(args.steps):
            set_command(env, vx, vy, yaw)
            obs, _, absorbing, _ = env.step_all(mask, agent.policy.draw_action_greedy(obs))
            qvel = wp.to_torch(env._data_wp.qvel)
            alive &= torch.isfinite(qvel).all(dim=1) & ~absorbing

            feet = env._foot_pos()
            if t >= warmup:
                quat = env._read_data("base_rot")
                lin = env._quat_rotate_inverse(quat, qvel[:, 0:3])
                ang = qvel[:, 3:6]
                g = env._projected_gravity()
                tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
                foot_h = feet[:, :, 2] - env._foot_half_height
                contact = foot_h < env._foot_contact_height
                speed = ((feet[:, :, :2] - last_feet[:, :, :2]) / env.dt).norm(dim=2)
                slip = (speed * contact).sum(1) / contact.sum(1).clamp(min=1)

                w = alive.float()
                acc["vx"] += (lin[:, 0] * w).sum()
                acc["vy"] += (lin[:, 1] * w).sum()
                acc["yaw"] += (ang[:, 2] * w).sum()
                acc["tilt"] += (tilt * w).sum()
                acc["slip"] += (slip * w).sum()
                rate_sq += ((ang[:, :2] ** 2).sum(1) * w).sum()
                n += w.sum()
            last_feet = feet.clone()

        n = n.clamp(min=1)
        m = {k: (v / n).item() for k, v in acc.items()}
        sway = math.sqrt((rate_sq / n).item())
        print(f"{vx:+6.2f} {vy:+6.2f} {yaw:+6.2f} | {alive.float().mean().item():7.2f} "
              f"{m['vx']:+6.2f} {m['vy']:+6.2f} {m['yaw']:+6.2f} "
              f"{m['tilt']:6.1f} {sway:6.2f} {m['slip']:6.2f}")


if __name__ == "__main__":
    main()
