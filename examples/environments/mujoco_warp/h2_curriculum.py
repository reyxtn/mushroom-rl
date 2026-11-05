"""
Trains the Unitree H2 locomotion curriculum, stand -> walk -> run, for one
seed, with domain randomization and an asymmetric actor-critic, in MuJoCo
Warp. The go2_curriculum.py bookkeeping (seeds, eval, best checkpoint after
the schedule completes) plus the stage chain: every stage warm-starts from
the best checkpoint of the previous one, and a stage below its gate stops
the chain.

    python h2_curriculum.py --seed 1
    for s in 1 2 3 4 5; do python h2_curriculum.py --seed $s --run-dir runs/h2_dr; done
    python h2_curriculum.py --seed 1 --stages walk run --init-agent runs/h2_dr/seed_1/stand.msh
    python h2_curriculum.py --seed 1 --no-dr                      # nominal model, nothing random
    python h2_curriculum.py --seed 1 --old-dr --no-curriculum --privileged none   # the old h2_sweep.sh setup
    python h2_curriculum.py --seed 1 --dr add_trunk_mass="(-5,10)" --dr max_delay=0.01
    python h2_curriculum.py --seed 1 --walk-arg "reward_weights={'feet_slip': -2.0}"

Every seed writes to ``<run-dir>/seed_<seed>/``: ``<stage>.msh`` (best
checkpoint of the stage), ``<stage>.best`` (its epoch and metrics),
``<stage>.log`` (one line per evaluation), ``config.txt`` and ``status``;
the layout ``h2_pick_best.sh`` reads.

Within a stage, after every policy update a schedule widens the velocity
commands and the actuation delay and anneals the tracking tolerance; the
environment only accepts new values, when to change them is decided by
:class:`Curriculum`. The best checkpoint is only looked for once the
schedule has reached its final values. ``--no-curriculum`` applies the
final values from the first update.

The asymmetry: the policy network reads the observation minus the entries
named by ``--privileged`` (by default the actuation delay and the encoder
offset the randomization drew, which the real robot cannot measure), while
the critic, which only runs in simulation, reads all of it.

"""

import argparse
import ast
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.core import Agent, Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO, RudinPPO
from mushroom_rl.environments.mujoco_warp_envs import H2Run, H2Stand, H2Walk
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import RandomizationParams
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.networks import MaskedActorNetwork
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.curriculum import Curriculum
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.h2.params import RANDOMIZATION_OBS
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

TASKS = {"stand": H2Stand, "walk": H2Walk, "run": H2Run}
STAGES = ("stand", "walk", "run")

# ----------------------------------------------------------------------
# Defaults per stage. Everything here can be overridden from the command
# line: epochs with --<stage>-epochs, environment arguments with
# --<stage>-arg KEY=VALUE, the gate with --<stage>-min-ep-len.
# ----------------------------------------------------------------------

STAGE_EPOCHS = dict(stand=80, walk=120, run=80)
STAGE_MIN_EP_LEN = dict(stand=200.0, walk=700.0, run=0.0)

# The dr_sweep3 recipe: survival worth less than tracking, posture and
# stillness paid for, gentler pushes than the tasks' own. Stand keeps the
# task defaults (the large alive bonus is what makes it learn to balance).
STAGE_REWARD_WEIGHTS = dict(
    stand=dict(),
    walk=dict(
        alive=0.5,
        tracking_lin_vel=2.0,
        tracking_ang_vel=1.0,
        joint_deviation=-0.5,
        feet_slip=-1.0,
        ang_vel_xy=-0.3,
        action_rate=-0.03,
    ),
    run=dict(
        alive=0.5,
        tracking_lin_vel=2.0,
        tracking_ang_vel=1.0,
        feet_slip=-1.0,
        action_rate=-0.03,
    ),
)
STAGE_REWARD_PARAMS = dict(
    stand=dict(),
    walk=dict(feet_air_time_threshold=0.25),  # below the 0.32 s scheduled swing, so following the gait clock pays
    run=dict(),
)
STAGE_PUSHES = dict(
    stand=dict(push_max_velocity=0.5),  # H2Stand's 1/250 interval, gentler than its 0.8 m/s
    walk=dict(push_max_velocity=0.3),
    run=dict(),
)

# Within-stage schedules, as fractions of the stage's steps per environment
# (turned into the step counts Curriculum takes by stage_schedule). The
# command ranges and the delay ceiling switch in stages; the tracking kernel
# is interpolated between the two ends of its window. Delays in seconds.
STAGE_CURRICULUM = dict(
    stand=dict(
        command_steps=(),
        command_ranges=[None],
        delays=[0.0, 0.01, 0.02],
        delay_steps=(0.3, 0.6),
        tracking_steps=None,
        tracking_stds=None,
    ),
    walk=dict(
        command_steps=(0.3, 0.6),
        command_ranges=[
            dict(lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2)),
            dict(lin_vel_x=(-0.5, 0.8), lin_vel_y=(-0.3, 0.3)),
            dict(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.4, 0.4)),
        ],
        delays=[0.0, 0.01, 0.02],
        delay_steps=(0.3, 0.6),
        tracking_steps=(0.1, 0.5),
        tracking_stds=(dict(sigma=0.5), dict(sigma=0.25)),
    ),
    run=dict(
        command_steps=(0.3, 0.6),
        command_ranges=[
            dict(lin_vel_x=(1.0, 1.5), lin_vel_y=(-0.2, 0.2)),
            dict(lin_vel_x=(1.0, 2.2), lin_vel_y=(-0.3, 0.3)),
            dict(lin_vel_x=(1.0, 3.0), lin_vel_y=(-0.3, 0.3)),
        ],
        delays=[0.0, 0.01, 0.02],
        delay_steps=(0.3, 0.6),
        tracking_steps=None,  # H2Run's own tracking_sigma (0.5) throughout
        tracking_stds=None,
    ),
)

# Domain randomization ranges for the H2: the Isaac Go2 values scaled for a
# 75 kg biped. ``max_delay`` is in seconds; every entry is a
# RandomizationParams field. Pushes are per stage (STAGE_PUSHES on top of the
# tasks' own). Any entry can be overridden with --dr KEY=VALUE.
DR_DEFAULTS = dict(
    add_trunk_mass=(-2.0, 6.0),
    add_com_displacement=(-0.03, 0.03),
    p_gain_scale=(0.85, 1.15),
    d_gain_scale=(0.85, 1.15),
    ground_friction_factor=5.0 / 9.0,
    max_delay=0.02,
    mixed_chance=0.0,
    position_offset=0.02,
    reset_yaw_range=math.pi,
    reset_joint_noise=0.1,
    reset_base_velocity=0.2,
    # The H2 model's spring reference (qpos_spring) is the all-zero pose, not
    # the standing keyframe, so a joint stiffness would pull the robot away
    # from its nominal pose. Kept inert.
    joint_stiffness=(0.0, 0.0),
)

# What the old H2Base did under the name "domain randomization": random
# pushes, joint noise and a planar velocity at reset, on the nominal model.
# --old-dr applies this on top of DR_DEFAULTS and --dr overrides.
OLD_DR = dict(
    add_trunk_mass=(0.0, 0.0),
    add_com_displacement=(0.0, 0.0),
    p_gain_scale=(1.0, 1.0),
    d_gain_scale=(1.0, 1.0),
    ground_friction_factor=0.0,
    max_delay=0.0,
    position_offset=0.0,
    reset_yaw_range=0.0,
)

DEFAULT_PRIVILEGED = ("actual_delay", "joint_calib_offset")


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

def build_agent(mdp, args, batch_size, privileged):
    obs_shape = mdp.info.observation_space.shape
    network_input_shape = (args.history_length,) + obs_shape if args.history_length > 1 else obs_shape

    policy_kwargs = dict(std_0=args.std_0, n_features=args.n_features, activation=args.activation,
                         gain_scale=args.gain_scale)
    if privileged:
        hidden = set(mdp.observation_indices(*privileged).tolist())
        observed = torch.tensor([i for i in range(obs_shape[0]) if i not in hidden], dtype=torch.long)
        policy = GaussianTorchPolicy(MaskedActorNetwork, network_input_shape, mdp.info.action_space.shape,
                                     observed_indices=observed, **policy_kwargs)
    else:
        policy = GaussianTorchPolicy(ActorNetwork, network_input_shape, mdp.info.action_space.shape,
                                     **policy_kwargs)

    adam = {"fused": True} if args.fused_adam else {}
    critic_params = dict(
        network=ActorNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": args.critic_lr, **adam}},
        loss=F.mse_loss,
        n_features=args.critic_features,
        activation=args.activation,
        gain_scale=args.gain_scale,
        batch_size=batch_size,
        input_shape=network_input_shape,
        output_shape=(1,),
    )
    alg_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": args.actor_lr, **adam}},
        n_epochs_policy=args.n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=args.eps_ppo,
        lam=args.lam,
        ent_coeff=args.ent_coeff,
        critic_fit_params=dict(n_epochs=args.n_epochs_critic),
        history_length=args.history_length,
    )
    if args.alg == "rudin":
        alg = RudinPPO
        alg_params.update(clip_grad_norm=args.clip_grad_norm, schedule="adaptive", desired_kl=args.desired_kl)
    else:
        alg = PPO
    # The H2 observation is scaled in the environment, so no preprocessor.
    return alg(mdp.info, policy, critic_params=critic_params, **alg_params)


def warm_start(agent, path, reset_std=None):
    """
    Copy the policy and critic weights of a saved agent with the same
    observation and action layout; the optimizers start fresh. The std can
    be reset, as a converged policy has little exploration left.

    """
    loaded = Agent.load(path)
    for name, mine, theirs in (("policy", agent.policy, loaded.policy), ("critic", agent._V, loaded._V)):
        w = theirs.get_weights()
        if w.shape != mine.get_weights().shape:
            raise SystemExit(
                f"cannot warm-start the {name} from {path}: {w.shape[0]} weights saved, "
                f"{mine.get_weights().shape[0]} expected. The observation layout (privileged entries, "
                "history length) or the network size differs from the saved agent's."
            )
        mine.set_weights(w)
    if reset_std is not None:
        agent.policy._log_sigma.data.fill_(math.log(reset_std))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def dataset_tracking(mdp, dataset):
    """
    Mean |v_xy - cmd| and |w_z - cmd| over the evaluation dataset, in m/s and
    rad/s (the observation is scaled, so the scaling is undone), and the fall
    rate.

    """
    scale = mdp._normalization_vec
    cmd_idx = mdp.observation_indices("commands")
    lin_idx = mdp.observation_indices("base_lin_vel")
    ang_idx = mdp.observation_indices("base_ang_vel")
    commands = dataset.state[:, cmd_idx] / scale[cmd_idx]
    lin_vel = dataset.state[:, lin_idx] / scale[lin_idx]
    ang_vel = dataset.state[:, ang_idx] / scale[ang_idx]
    lin_err = torch.linalg.norm(commands[:, :2] - lin_vel[:, :2], dim=1).mean().item()
    ang_err = (commands[:, 2] - ang_vel[:, 2]).abs().mean().item()
    n_episodes = max(int(dataset.n_episodes), 1)
    falls = dataset.absorbing.sum().item() / n_episodes
    return lin_err, ang_err, falls


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def parse_kv(items):
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def stage_schedule(stage, total_steps):
    """STAGE_CURRICULUM[stage] with its fractions turned into step counts."""
    c = STAGE_CURRICULUM[stage]
    steps = lambda fractions: [int(f * total_steps) for f in fractions]  # noqa: E731
    # Without command stages the delay switches on its own fractions, with a
    # single (unchanged) command range repeated for every delay stage.
    command_steps = steps(c["command_steps"] or c["delay_steps"])
    ranges = list(c["command_ranges"])
    if len(ranges) == 1:
        ranges = ranges * (len(command_steps) + 1)
    return dict(
        command_steps=command_steps,
        command_ranges=ranges,
        delays=c["delays"],
        tracking_steps=steps(c["tracking_steps"]) if c["tracking_steps"] else (0, 1),
        tracking_stds=c["tracking_stds"],
    )


def run_stage(stage, args, seed_dir, init_path, dr_params, privileged):
    # Every stage starts from the same RNG state, as it did when each stage
    # was its own process.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_envs = args.n_envs
    n_steps_per_fit = n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch
    n_epochs = getattr(args, f"{stage}_epochs")
    batch_size = n_steps_per_fit // args.n_minibatches
    n_fits = n_epochs * args.fits_per_epoch
    total_steps = n_fits * args.fragment_length

    env_kwargs = parse_kv(args.env_arg)
    env_kwargs.update(parse_kv(getattr(args, f"{stage}_arg")))
    dr_on = env_kwargs.pop("domain_randomization", not args.no_dr)
    reward_weights = {**STAGE_REWARD_WEIGHTS[stage], **env_kwargs.pop("reward_weights", {})}
    reward_params = {**STAGE_REWARD_PARAMS[stage], **env_kwargs.pop("reward_params", {})}

    # The latency is in seconds. max_delay is also the ceiling the action
    # history is sized for, so the curriculum's delays are clamped to it
    # (--dr max_delay=0.01 caps the schedule at 10 ms, --old-dr at none).
    params = dict(dr_params, **STAGE_PUSHES[stage])
    if not dr_on:
        params["max_delay"] = 0.0

    env_cls = TASKS[stage]
    mdp = env_cls(
        num_envs=n_envs,
        use_graph_capture=args.graph_capture,
        domain_randomization=dr_on,
        randomization_params=params,
        max_delay_limit=params["max_delay"],
        observed_randomization=tuple(n for n in privileged if n in RANDOMIZATION_OBS),
        reward_weights=reward_weights,
        reward_params=reward_params,
        **env_kwargs,
    )
    curriculum = Curriculum(mdp, enabled=not args.no_curriculum, **stage_schedule(stage, total_steps))
    print(f"[curriculum] {stage}: intermediate step {mdp._step_dt * 1e3:.0f} ms, latency up to "
          f"{mdp.max_delay_steps_limit} step(s) = {params['max_delay'] * 1e3:.0f} ms"
          + ("" if dr_on else " (randomization off)")
          + ("" if args.no_curriculum else f"; best checkpoint looked for from step {curriculum.complete_at} on"))

    wandb_kwargs = None
    if args.wandb:
        run_name = Path(args.run_dir).name
        wandb_kwargs = Logger.default_wandb_kwargs(
            args.wandb_project,
            group=f"{run_name}/seed_{args.seed}",
            name=f"{run_name}-s{args.seed}-{stage}",
            config=dict(stage=stage, seed=args.seed, init_agent=init_path, n_envs=n_envs, n_epochs=n_epochs,
                        n_steps_per_fit=n_steps_per_fit, batch_size=batch_size, alg=args.alg,
                        history_length=args.history_length, privileged=list(privileged), tf32=args.tf32,
                        domain_randomization=dr_on, dr_params=params, env_kwargs=env_kwargs,
                        reward_weights=reward_weights, reward_params=reward_params,
                        curriculum=STAGE_CURRICULUM[stage], curriculum_enabled=not args.no_curriculum,
                        actor_lr=args.actor_lr, critic_lr=args.critic_lr, std_0=args.std_0,
                        ent_coeff=args.ent_coeff, n_epochs_critic=args.n_epochs_critic),
        )
    alg_name = "RudinPPO" if args.alg == "rudin" else "PPO"
    logger = Logger(f"{alg_name}_{mdp.name()}", results_dir=str(seed_dir / "logs"), seed=args.seed,
                    wandb_kwargs=wandb_kwargs, console_log_level=logging.WARNING)

    agent = build_agent(mdp, args, batch_size, privileged)
    if init_path is not None:
        warm_start(agent, init_path, reset_std=args.reset_std)
    core = Core(agent, mdp, callbacks_fit=[curriculum], logger=logger)

    log = open(seed_dir / f"{stage}.log", "w")
    best = dict(J=-float("inf"))

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(agent.history_manager.parse_initial_state(dataset)).mean().item()
        lin_err, ang_err, falls = dataset_tracking(mdp, dataset)
        alive = 1.0 - falls
        cur = curriculum.state()
        logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L, V=V, lin_vel_err=lin_err,
                              ang_vel_err=ang_err, alive_frac=alive, **cur)
        line = (f"{time.strftime('%d/%m/%Y %H:%M:%S')} [INFO] Epoch {epoch} | J: {J} R: {R} entropy: {E} "
                f"mean_ep_len: {L} V: {V} lin_vel_err: {lin_err} ang_vel_err: {ang_err} alive_frac: {alive} "
                + " ".join(f"{k}: {v}" for k, v in cur.items()))
        log.write(line + "\n")
        log.flush()
        # Returns from before the schedule is complete are not comparable
        # with the ones after it, so the best checkpoint is only tracked
        # once it is.
        if curriculum.complete:
            logger.log_best_agent(agent, J)
            if J >= best["J"]:  # the logger saves on >= too
                best.update(J=J, epoch=epoch, ep_len=L, lin_err=lin_err, alive=alive)
        del dataset

    evaluate(0)
    for it in trange(n_epochs, leave=False, desc=f"seed {args.seed} {stage}", ncols=80):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        if (it + 1) % args.eval_every == 0 or it + 1 == n_epochs:
            evaluate(it + 1)
    log.close()

    if "epoch" not in best:
        raise SystemExit(f"{stage}: no evaluation ran after the schedule completed; "
                         f"lower --eval-every or raise --{stage}-epochs")
    ckpts = sorted((seed_dir / "logs").rglob("agent*-best.msh"), key=os.path.getmtime)
    if not ckpts:
        raise SystemExit(f"no checkpoint saved for {stage}")
    out = seed_dir / f"{stage}.msh"
    shutil.copy(ckpts[-1], out)
    (seed_dir / f"{stage}.best").write_text(
        f"{best['epoch']} {best['J']} {best['ep_len']} {best['lin_err']} {best['alive']}\n"
    )
    print(f"[curriculum] {stage} done: best epoch {best['epoch']} J={best['J']:.3f} ep_len={best['ep_len']:.1f} "
          f"lin_vel_err={best['lin_err']:.3f} alive={best['alive']:.3f} -> {out}")

    logger.finish()  # one wandb run per stage
    mdp.stop()
    del core, agent, mdp
    torch.cuda.empty_cache()
    return out, best


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", default="runs/h2_curriculum", help="sweep folder; this seed goes to seed_<seed>/")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--init-agent", default=None, help="checkpoint the first stage warm-starts from")
    p.add_argument("--reset-std", type=float, default=None,
                   help="policy std to reset to after a warm start (default: keep the checkpoint's)")
    p.add_argument("--force", action="store_true", help="ignore the gates between stages")
    for s in STAGES:
        p.add_argument(f"--{s}-epochs", type=int, default=STAGE_EPOCHS[s])
        p.add_argument(f"--{s}-min-ep-len", type=float, default=STAGE_MIN_EP_LEN[s],
                       help=f"best mean episode length {s} needs for the next stage to start")
        p.add_argument(f"--{s}-arg", action="append", default=[], metavar="KEY=VALUE",
                       help=f"{TASKS[s].__name__} constructor argument (Python literal), applied after --env-arg; "
                            "reward_weights={...} and reward_params={...} merge into the stage's")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="constructor argument for every stage, e.g. --env-arg obs_noise=False")

    g = p.add_argument_group("domain randomization")
    g.add_argument("--no-dr", action="store_true",
                   help="nothing random at all: nominal model, no pushes, no reset noise, no latency")
    g.add_argument("--old-dr", action="store_true",
                   help="only what the old H2Base randomized: pushes and reset noise on the nominal model")
    g.add_argument("--dr", action="append", default=[], metavar="KEY=VALUE",
                   help="override a DR_DEFAULTS entry, e.g. --dr position_offset=0.01 or --dr max_delay=0.01 "
                        "(seconds); applied after --old-dr")
    g.add_argument("--no-curriculum", action="store_true",
                   help="final command ranges, delay and tracking sigma from the first update")
    g.add_argument("--privileged", nargs="*", default=list(DEFAULT_PRIVILEGED),
                   help="observation entries only the critic sees (names from H2Walk.observation_indices, "
                        "e.g. actual_delay joint_calib_offset base_lin_vel); pass none for a symmetric agent")

    g = p.add_argument_group("algorithm")
    g.add_argument("--alg", choices=("rudin", "ppo"), default="ppo",
                   help="plain PPO, or RudinPPO (adaptive learning rate from the policy KL, gradient clipping)")
    g.add_argument("--history-length", type=int, default=1, help="stacked observations the networks read")
    g.add_argument("--n-envs", type=int, default=4096)
    g.add_argument("--fragment-length", type=int, default=24, help="on-policy steps per environment per fit")
    g.add_argument("--fits-per-epoch", type=int, default=10)
    g.add_argument("--n-minibatches", type=int, default=16)
    g.add_argument("--n-epochs-policy", type=int, default=5)
    g.add_argument("--n-epochs-critic", type=int, default=1,
                   help="critic passes per fit; 1 was clearly better than 10 on the H2 under domain randomization")
    g.add_argument("--n-episodes-test", type=int, default=256)
    g.add_argument("--eval-every", type=int, default=1,
                   help="evaluate (and consider for the best checkpoint) every N epochs, and at the last")
    g.add_argument("--actor-lr", type=float, default=3e-4)
    g.add_argument("--critic-lr", type=float, default=3e-4)
    g.add_argument("--desired-kl", type=float, default=0.01)
    g.add_argument("--clip-grad-norm", type=float, default=1.0)
    g.add_argument("--eps-ppo", type=float, default=0.2)
    g.add_argument("--lam", type=float, default=0.95)
    g.add_argument("--std-0", type=float, default=0.3)
    g.add_argument("--ent-coeff", type=float, default=0.005)
    g.add_argument("--n-features", type=int, nargs="+", default=[512, 256, 128])
    g.add_argument("--critic-features", type=int, nargs="+", default=[512, 256, 128])
    g.add_argument("--activation", default="relu")
    g.add_argument("--gain-scale", type=float, default=1.0)
    g.add_argument("--no-fused-adam", action="store_false", dest="fused_adam")

    g = p.add_argument_group("compute")
    g.add_argument("--tf32", action="store_true",
                   help="run the matrix multiplications of the networks in TF32 on the tensor cores")
    g.add_argument("--no-graph-capture", action="store_false", dest="graph_capture")
    g.add_argument("--no-wandb", action="store_false", dest="wandb")
    g.add_argument("--wandb-project", default="mushroom_rl_h2")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "MuJoCo Warp requires a CUDA device."
    TorchUtils.set_default_device("cuda:0")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dr_params = dict(DR_DEFAULTS)
    if args.old_dr:
        dr_params.update(OLD_DR)
    dr_params.update(parse_kv(args.dr))
    unknown = set(dr_params) - set(DR_DEFAULTS) - set(RandomizationParams().as_dict())
    if unknown:
        raise SystemExit(f"unknown --dr entries: {sorted(unknown)}")
    if "max_delay_steps" in dr_params:
        raise SystemExit("the latency is given in seconds: --dr max_delay=0.02, not max_delay_steps")
    privileged = tuple(n for n in args.privileged if n.lower() != "none")

    run_dir = Path(args.run_dir)
    seed_dir = run_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    config_text = "\n".join([
        f"stages={' '.join(args.stages)} alg={args.alg} history_length={args.history_length} tf32={args.tf32}",
        f"epochs stand={args.stand_epochs} walk={args.walk_epochs} run={args.run_epochs} "
        f"fits_per_epoch={args.fits_per_epoch} n_envs={args.n_envs} n_epochs_critic={args.n_epochs_critic}",
        f"domain_randomization={not args.no_dr} old_dr={args.old_dr} curriculum_enabled={not args.no_curriculum} "
        f"dr_params={dr_params}",
        f"privileged={list(privileged)}",
        f"gates stand_min_ep_len={args.stand_min_ep_len} walk_min_ep_len={args.walk_min_ep_len}",
        f"env_arg={args.env_arg} stand_arg={args.stand_arg} walk_arg={args.walk_arg} run_arg={args.run_arg}",
        f"stage_reward_weights={STAGE_REWARD_WEIGHTS} stage_reward_params={STAGE_REWARD_PARAMS} "
        f"stage_pushes={STAGE_PUSHES}",
        f"curriculum={STAGE_CURRICULUM}",
        f"std_0={args.std_0} ent_coeff={args.ent_coeff} actor_lr={args.actor_lr} critic_lr={args.critic_lr}",
        f"git={git}", "",
    ])
    # One per seed, always (a reused --run-dir would otherwise keep the first
    # run's settings), and one for the run folder if it has none yet.
    (seed_dir / "config.txt").write_text(config_text)
    if not (run_dir / "config.txt").exists():
        (run_dir / "config.txt").write_text(config_text)

    init_path = args.init_agent
    status = "ok"
    try:
        for i, stage in enumerate(args.stages):
            out, best = run_stage(stage, args, seed_dir, init_path, dr_params, privileged)
            init_path = str(out)
            gate = getattr(args, f"{stage}_min_ep_len")
            if i + 1 < len(args.stages) and not args.force and best["ep_len"] < gate:
                status = f"gated: {stage} best ep_len {best['ep_len']:.1f} < {gate}"
                print(f"[curriculum] {status}; not starting {args.stages[i + 1]} (use --force to override)")
                break
    except Exception as e:  # noqa: BLE001 - recorded in status for the sweep
        (seed_dir / "status").write_text(f"failed: {type(e).__name__}: {e}\n")
        raise
    (seed_dir / "status").write_text(status + "\n")


if __name__ == "__main__":
    main()
