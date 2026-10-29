"""
Trains the Unitree Go2 velocity tracking task for one seed, with domain
randomization, an asymmetric actor-critic and a within-run curriculum, in
MuJoCo Warp. The counterpart of the new_isaac ``go2_curriculum.py`` for the
warp backend: same reward weights, command generator, curriculum, network
sizes, RudinPPO settings, 8-step history, privileged critic entries and
StandardizationPreprocessor. What this script adds on top is the run
bookkeeping (seeds, eval, best checkpoint after the schedule completes).

    python go2_curriculum.py --seed 1
    for s in 1 2 3 4 5; do python go2_curriculum.py --seed $s --run-dir runs/go2_dr; done
    ./h2_pick_best.sh runs/go2_dr                                 # picks walk.msh across seeds
    python go2_curriculum.py --seed 1 --no-dr                     # nominal model, nothing random
    python go2_curriculum.py --seed 1 --old-dr --no-curriculum --privileged none   # the go2_ppo.py setup
    python go2_curriculum.py --seed 1 --dr add_trunk_mass="(-3,6)" --dr max_delay=0.01
    python go2_curriculum.py --seed 1 --privileged actual_delay joint_calib_offset base_lin_vel \\
        --history-length 8                                        # the Isaac script's asymmetry

Every seed writes to ``<run-dir>/seed_<seed>/``: ``walk.msh`` (best
checkpoint), ``walk.best`` (its epoch and metrics), ``walk.log`` (one line
per evaluation), ``config.txt`` and ``status``; the layout
``h2_pick_best.sh`` reads.

After every policy update a schedule widens the velocity commands and the
actuation delay and anneals the tracking tolerance, as in the Isaac script;
the environment only accepts new values, when to change them is decided by
:class:`Curriculum`. The best checkpoint is only looked for once the
schedule has reached its final values: earlier evaluations run on easier
commands and a wider tracking kernel, so their returns are higher for
reasons that have nothing to do with the policy. ``--no-curriculum``
applies the final values from the first update.

The asymmetry: the policy network reads the observation minus the entries
named by ``--privileged`` (by default the actuation delay and the encoder
offset the randomization drew, which the real robot cannot measure), while
the critic, which only runs in simulation, reads all of it. Any part of the
observation can be hidden this way, e.g. ``base_lin_vel`` as in the Isaac
script; only the names in ``go2.params.RANDOMIZATION_OBS`` are appended to
the observation.

"""

import argparse
import logging
import ast
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
from mushroom_rl.environments.mujoco_warp_envs import Go2Walk
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import RandomizationParams
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.networks import MaskedActorNetwork
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.curriculum import Curriculum
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.commands import UniformVelocityCommands
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.go2.params import (
    RANDOMIZATION_OBS,
    GO2_REWARD_WEIGHTS,
    GO2_COMMAND_GENERATOR,
)
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

STAGE = "walk"  # the file names h2_pick_best.sh expects

# The new_isaac schedule, verbatim. Thresholds count control steps per
# environment (dataset rows): 60 epochs x 50 fits x 24 steps = 72000, so the
# commands widen at 1/3 and 2/3 of the run and the tracking tolerance tightens
# between 11 % and 28 %. Fewer epochs than 60 means the schedule never
# completes. Delays: Isaac's 2, 3, 4 physics steps of 5 ms, in seconds.
CURRICULUM = dict(
    command_steps=[24000, 48000],
    command_ranges=[(-1., 1.), (-2., 2.), (-2.5, 2.5)],
    delays=[0.010, 0.015, 0.020],
    tracking_steps=(8000, 20000),
    tracking_stds=(dict(lin_vel=0.5, lin_vel_slope=0., ang_vel=0.5, ang_vel_slope=0.),
                   dict(lin_vel=0.05, lin_vel_slope=0.2, ang_vel=0.1, ang_vel_slope=0.5)),
)

# Domain randomization ranges: the Isaac Go2 values (the randomizer's own
# defaults), listed so config.txt records them. ``max_delay`` is in seconds;
# every entry is a RandomizationParams field, pushes included (1 m/s every
# 750 steps on average by default: push_max_velocity / push_probability).
# Any entry can be overridden with --dr KEY=VALUE.
DR_DEFAULTS = dict(
    add_trunk_mass=(-2.0, 4.0),
    add_com_displacement=(-0.05, 0.05),
    p_gain_scale=(0.85, 1.15),
    d_gain_scale=(0.85, 1.15),
    ground_friction_factor=5.0 / 9.0,
    max_delay=0.02,
    mixed_chance=0.0,
    position_offset=0.04,
    reset_yaw_range=math.pi,
    # Isaac resets to the exact default pose, standing still, when the
    # randomization is on.
    reset_joint_noise=0.0,
    reset_base_velocity=0.0,
    # The menagerie model's spring reference (qpos_spring) is the all-zero
    # pose, not the standing keyframe, so a joint stiffness would pull the
    # robot away from its nominal pose. Kept inert.
    joint_stiffness=(0.0, 0.0),
)

# What the merged Go2Base did under the name "domain randomization": random
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

DEFAULT_PRIVILEGED = ("base_lin_vel", "base_pos", "actual_delay", "joint_calib_offset")


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
    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)
    # Isaac normalizes the observation in the agent, not in the environment.
    agent.add_agent_preprocessor(StandardizationPreprocessor(mdp.info, backend="torch"))
    return agent


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
    """Isaac's metrics on the evaluation dataset: mean |v_xy - cmd| and |w_z - cmd|, and the fall rate."""
    commands = dataset.state[:, mdp.observation_indices("commands")]
    lin_vel = dataset.state[:, mdp.observation_indices("base_lin_vel")]
    ang_vel = dataset.state[:, mdp.observation_indices("base_ang_vel")]
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


def train(args, seed_dir, dr_params, privileged):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_envs = args.n_envs
    n_steps_per_fit = n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch
    batch_size = n_steps_per_fit // args.n_minibatches
    n_fits = args.n_epochs * args.fits_per_epoch

    env_kwargs = parse_kv(args.env_arg)
    dr_on = env_kwargs.pop("domain_randomization", not args.no_dr)

    # The latency is in seconds. max_delay is also the ceiling the action
    # history is sized for, so the curriculum's delays are clamped to it
    # (--dr max_delay=0.01 caps the schedule at 10 ms, --old-dr at none).
    params = dict(dr_params)
    if not dr_on:
        params["max_delay"] = 0.0

    mdp = Go2Walk(
        num_envs=n_envs,
        use_graph_capture=args.graph_capture,
        domain_randomization=dr_on,
        randomization_params=RandomizationParams(**params),
        max_delay_limit=params["max_delay"],
        observed_randomization=tuple(n for n in privileged if n in RANDOMIZATION_OBS),
        reward_weights=GO2_REWARD_WEIGHTS,
        command_generator=UniformVelocityCommands(**GO2_COMMAND_GENERATOR),
        **env_kwargs,
    )
    curriculum = Curriculum(mdp, enabled=not args.no_curriculum, **CURRICULUM)
    total_steps = n_fits * args.fragment_length
    if not args.no_curriculum and total_steps < curriculum.complete_at:
        print(f"[curriculum] WARNING: {total_steps} steps per env in this run, the schedule completes at "
              f"{curriculum.complete_at}: no best checkpoint will be saved")
    print(f"[curriculum] intermediate step {mdp._step_dt * 1e3:.0f} ms, latency up to "
          f"{mdp.max_delay_steps_limit} step(s) = {params['max_delay'] * 1e3:.0f} ms"
          + ("" if dr_on else " (randomization off)")
          + ("" if args.no_curriculum else f"; best checkpoint looked for from step {curriculum.complete_at} on"))

    wandb_kwargs = None
    if args.wandb:
        run_name = Path(args.run_dir).name
        wandb_kwargs = Logger.default_wandb_kwargs(
            args.wandb_project,
            group=run_name,
            name=f"{run_name}-s{args.seed}",
            config=dict(seed=args.seed, init_agent=args.init_agent, n_envs=n_envs, n_epochs=args.n_epochs,
                        n_steps_per_fit=n_steps_per_fit, batch_size=batch_size, alg=args.alg,
                        history_length=args.history_length, privileged=list(privileged), tf32=args.tf32,
                        domain_randomization=dr_on, dr_params=dr_params, env_kwargs=env_kwargs,
                        curriculum=CURRICULUM, curriculum_enabled=not args.no_curriculum,
                        actor_lr=args.actor_lr, critic_lr=args.critic_lr, std_0=args.std_0,
                        ent_coeff=args.ent_coeff, n_epochs_critic=args.n_epochs_critic),
        )
    alg_name = "RudinPPO" if args.alg == "rudin" else "PPO"
    logger = Logger(f"{alg_name}_{mdp.name()}", results_dir=str(seed_dir / "logs"), seed=args.seed,
                       wandb_kwargs=wandb_kwargs, console_log_level=logging.WARNING)

    agent = build_agent(mdp, args, batch_size, privileged)
    if args.init_agent is not None:
        warm_start(agent, args.init_agent, reset_std=args.reset_std)
    core = Core(agent, mdp, callbacks_fit=[curriculum], logger=logger)

    log = open(seed_dir / f"{STAGE}.log", "w")
    best = dict(J=-float("inf"))
    last = {}

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(agent.history_manager.parse_initial_state(dataset)).mean().item()
        lin_err, ang_err, falls = dataset_tracking(mdp, dataset)
        diverged, alive, terms = 0.0, 1.0 - falls, {}
        cur = curriculum.state()
        logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L, V=V, lin_vel_err=lin_err,
                              ang_vel_err=ang_err, diverged_frac=diverged, alive_frac=alive, **cur, **terms)
        last.update(J=J, lin=lin_err, alive=alive)
        line = (f"{time.strftime('%d/%m/%Y %H:%M:%S')} [INFO] Epoch {epoch} | J: {J} R: {R} entropy: {E} "
                f"mean_ep_len: {L} V: {V} lin_vel_err: {lin_err} ang_vel_err: {ang_err} "
                f"diverged_frac: {diverged} alive_frac: {alive} "
                + " ".join(f"{k}: {v}" for k, v in {**cur, **terms}.items()))
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

    # Evaluation steps all n_envs worlds until n_episodes_test of them end,
    # which with a policy that survives costs as much as several epochs of
    # training; so only every eval_every epochs, and always at the last one.
    evaluate(0)
    for it in trange(args.n_epochs, leave=False, desc=f"seed {args.seed}", ncols=80):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        if (it + 1) % args.eval_every == 0 or it + 1 == args.n_epochs:
            evaluate(it + 1)
    log.close()

    if "epoch" not in best:
        raise SystemExit("no evaluation ran after the schedule completed; lower --eval-every or raise --n-epochs")
    ckpts = sorted((seed_dir / "logs").rglob("agent*-best.msh"), key=os.path.getmtime)
    if not ckpts:
        raise SystemExit("no checkpoint saved")
    out = seed_dir / f"{STAGE}.msh"
    shutil.copy(ckpts[-1], out)
    (seed_dir / f"{STAGE}.best").write_text(
        f"{best['epoch']} {best['J']} {best['ep_len']} {best['lin_err']} {best['alive']}\n"
    )
    print(f"[curriculum] done: best epoch {best['epoch']} J={best['J']:.3f} ep_len={best['ep_len']:.1f} "
          f"lin_vel_err={best['lin_err']:.3f} alive={best['alive']:.3f} -> {out}")

    logger.finish()
    mdp.stop()
    return out, best


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", default="runs/go2_curriculum", help="sweep folder; this seed goes to seed_<seed>/")
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--init-agent", default=None, help="checkpoint to warm-start from (same observation layout)")
    p.add_argument("--reset-std", type=float, default=None,
                   help="policy std to reset to after the warm start (default: keep the checkpoint's)")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="Go2Walk constructor argument (Python literal), e.g. --env-arg obs_noise=False")

    g = p.add_argument_group("domain randomization")
    g.add_argument("--no-dr", action="store_true",
                   help="nothing random at all: nominal model, no pushes, no reset noise, no latency")
    g.add_argument("--old-dr", action="store_true",
                   help="only what the merged Go2Base randomized: pushes and reset noise on the nominal model")
    g.add_argument("--dr", action="append", default=[], metavar="KEY=VALUE",
                   help="override a DR_DEFAULTS entry, e.g. --dr position_offset=0.02 or --dr max_delay=0.01 "
                        "(seconds); applied after --old-dr")
    g.add_argument("--no-curriculum", action="store_true",
                   help="final command ranges, delay and tracking sigma from the first update")
    g.add_argument("--privileged", nargs="*", default=list(DEFAULT_PRIVILEGED),
                   help="observation entries only the critic sees (names from Go2Walk.observation_indices, "
                        "e.g. actual_delay joint_calib_offset base_lin_vel); pass none for a symmetric agent")

    g = p.add_argument_group("algorithm")
    g.add_argument("--alg", choices=("rudin", "ppo"), default="rudin",
                   help="plain PPO, or RudinPPO (adaptive learning rate from the policy KL, gradient clipping)")
    g.add_argument("--history-length", type=int, default=8, help="stacked observations the networks read")
    g.add_argument("--n-envs", type=int, default=4096)
    g.add_argument("--fragment-length", type=int, default=24, help="on-policy steps per environment per fit")
    g.add_argument("--fits-per-epoch", type=int, default=50)
    g.add_argument("--n-minibatches", type=int, default=4)
    g.add_argument("--n-epochs-policy", type=int, default=5)
    g.add_argument("--n-epochs-critic", type=int, default=5,
                   help="critic passes per fit (the Isaac Go2 script's 5)")
    g.add_argument("--n-episodes-test", type=int, default=256)
    g.add_argument("--eval-every", type=int, default=1,
                   help="evaluate (and consider for the best checkpoint) every N epochs, and at the last")
    g.add_argument("--actor-lr", type=float, default=1e-3)
    g.add_argument("--critic-lr", type=float, default=1e-3)
    g.add_argument("--desired-kl", type=float, default=0.01)
    g.add_argument("--clip-grad-norm", type=float, default=1.0)
    g.add_argument("--eps-ppo", type=float, default=0.2)
    g.add_argument("--lam", type=float, default=0.95)
    g.add_argument("--std-0", type=float, default=1.0)
    g.add_argument("--ent-coeff", type=float, default=0.01)
    g.add_argument("--n-features", type=int, nargs="+", default=[512, 512, 256])
    g.add_argument("--critic-features", type=int, nargs="+", default=[256, 256, 128])
    g.add_argument("--activation", default="elu")
    g.add_argument("--gain-scale", type=float, default=0.5)
    g.add_argument("--no-fused-adam", action="store_false", dest="fused_adam")

    g = p.add_argument_group("compute")
    g.add_argument("--tf32", action="store_true",
                   help="run the matrix multiplications of the networks in TF32 on the tensor cores")
    g.add_argument("--no-graph-capture", action="store_false", dest="graph_capture")
    g.add_argument("--no-wandb", action="store_false", dest="wandb")
    g.add_argument("--wandb-project", default="mushroom_rl_go2")
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
        f"alg={args.alg} history_length={args.history_length} tf32={args.tf32}",
        f"n_epochs={args.n_epochs} fits_per_epoch={args.fits_per_epoch} n_envs={args.n_envs} "
        f"n_epochs_critic={args.n_epochs_critic}",
        f"domain_randomization={not args.no_dr} old_dr={args.old_dr} curriculum_enabled={not args.no_curriculum} "
        f"dr_params={dr_params}",
        f"privileged={list(privileged)}",
        f"env_arg={args.env_arg}",
        f"curriculum={CURRICULUM}",
        f"std_0={args.std_0} ent_coeff={args.ent_coeff} actor_lr={args.actor_lr} critic_lr={args.critic_lr}",
        f"git={git}", "",
    ])
    # One per seed, always (a reused --run-dir would otherwise keep the first
    # run's settings), and one for the run folder if it has none yet.
    (seed_dir / "config.txt").write_text(config_text)
    if not (run_dir / "config.txt").exists():
        (run_dir / "config.txt").write_text(config_text)

    try:
        train(args, seed_dir, dr_params, privileged)
    except Exception as e:  # noqa: BLE001 - recorded in status for the sweep
        (seed_dir / "status").write_text(f"failed: {type(e).__name__}: {e}\n")
        raise
    (seed_dir / "status").write_text("ok\n")


if __name__ == "__main__":
    main()
