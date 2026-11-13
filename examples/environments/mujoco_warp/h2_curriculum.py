"""
Trains the Unitree H2 velocity policy for one seed in MuJoCo Warp, the way
Unitree trains it (unitree_rl_mjlab, Unitree-H2-Flat): their model, reward,
commands, randomization and PPO settings, on this repository's environment
and RudinPPO. One policy walks, runs and stands. The bookkeeping (seeds,
eval, best checkpoint after the schedule completes) is go2_curriculum.py's.

    python h2_curriculum.py --seed 1
    for s in 1 2 3; do python h2_curriculum.py --seed $s --run-dir runs/h2; done
    python h2_curriculum.py --seed 1 --epochs 2 --n-envs 256 --no-wandb          # smoke test
    python h2_curriculum.py --seed 1 --no-dr                                     # nominal model, nothing random
    python h2_curriculum.py --seed 1 --dr position_offset=0.03
    python h2_curriculum.py --seed 1 --env-arg "reward_weights={'foot_gait': 1.0}"
    python h2_curriculum.py --seed 1 --init-agent runs/h2/seed_1/walk.msh --no-curriculum   # continue a run

Every seed writes to ``<run-dir>/seed_<seed>/``: ``walk.msh`` (best
checkpoint), ``walk.best`` (its epoch and metrics), ``walk.log`` (one line
per evaluation), ``config.txt`` and ``status``; the layout
``h2_pick_best.sh`` reads. ``walk_last.msh`` is the agent at the last
evaluation, for a look at a run that is still training.

One schedule runs after every policy update: the velocity commands start
narrow and widen to their full ranges halfway through the run, as in
Unitree's task. The environment only accepts new values, when to change them
is decided by :class:`Curriculum`. The best checkpoint is only looked for
once the commands are at their final ranges. ``--no-curriculum`` applies
them from the first update.

The asymmetry: the policy network reads the observation minus the entries
named by ``--privileged`` (by default what only Unitree's critic reads: the
pelvis linear velocity and the foot heights, air times, contacts and contact
forces), while the critic, which only runs in simulation, reads all of it.

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
from mushroom_rl.environments.mujoco_warp_envs import H2Walk
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.randomizer import RandomizationParams
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.networks import MaskedActorNetwork
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.curriculum import Curriculum
from mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.h2.params import RANDOMIZATION_OBS
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

# ----------------------------------------------------------------------
# Defaults. The reward weights and parameters are the task's own
# (h2/params.py); override them with --env-arg reward_weights={...}.
# ----------------------------------------------------------------------

# With 10 fits per epoch this is Unitree's 10000 iterations.
EPOCHS = 1000

# The command ranges switch at this fraction of the run's steps per
# environment (turned into the step count Curriculum takes by schedule):
# Unitree's two stages, the second from iteration 5000 of 10000. The yaw
# rate range is the full one from the start. No actuation delay.
CURRICULUM = dict(
    command_steps=(0.5,),
    command_ranges=[
        dict(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5)),
        dict(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5)),
    ],
    delays=[0.0, 0.0],
)

# Domain randomization, Unitree's: foot friction, encoder bias and the centre
# of mass of the torso; the pushes are the task's own (h2/params.py). Every
# entry is a RandomizationParams field and can be overridden with
# --dr KEY=VALUE. The friction is 0.6 times a factor in [0.5, 1.5], where
# Unitree draws it in [0.3, 1.6].
DR_DEFAULTS = dict(
    ground_friction_factor=0.5,
    position_offset=0.015,
    add_com_displacement=(-0.05, 0.05),
    add_trunk_mass=(0.0, 0.0),
    p_gain_scale=(1.0, 1.0),
    d_gain_scale=(1.0, 1.0),
    max_delay=0.0,
    mixed_chance=0.0,
    reset_yaw_range=math.pi,
    reset_joint_noise=0.0,
    reset_base_velocity=0.0,
    joint_stiffness=(0.0, 0.0),
)

# What only Unitree's critic reads.
DEFAULT_PRIVILEGED = ("base_lin_vel", "foot_height", "foot_air_time", "foot_contact", "foot_force")


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
        output_shape=(),
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
    # Unitree normalizes the observation in the networks; here the agent does.
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
    # The weights only make sense on inputs standardized with the statistics
    # they were trained with.
    agent._history_manager._preprocessors = loaded._history_manager._preprocessors
    if reset_std is not None:
        agent.policy._log_sigma.data.fill_(math.log(reset_std))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def dataset_tracking(mdp, dataset):
    """Mean |v_xy - cmd| and |w_z - cmd| over the evaluation dataset, in m/s and rad/s, and the fall rate."""
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


def schedule(total_steps):
    """CURRICULUM with its fractions turned into step counts; the tracking kernel has no schedule."""
    return dict(
        CURRICULUM,
        command_steps=[int(f * total_steps) for f in CURRICULUM["command_steps"]],
        tracking_steps=(0, 1),
        tracking_stds=None,
    )


def train(args, seed_dir, dr_params, privileged):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_envs = args.n_envs
    n_steps_per_fit = n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch
    n_epochs = args.epochs
    batch_size = n_steps_per_fit // args.n_minibatches
    n_fits = n_epochs * args.fits_per_epoch
    total_steps = n_fits * args.fragment_length

    env_kwargs = parse_kv(args.env_arg)
    dr_on = env_kwargs.pop("domain_randomization", not args.no_dr)

    # The latency is in seconds. max_delay is also the ceiling the action
    # history is sized for.
    params = dict(dr_params)
    if not dr_on:
        params["max_delay"] = 0.0

    mdp = H2Walk(
        num_envs=n_envs,
        use_graph_capture=args.graph_capture,
        domain_randomization=dr_on,
        randomization_params=params,
        max_delay_limit=params["max_delay"],
        observed_randomization=tuple(n for n in privileged if n in RANDOMIZATION_OBS),
        **env_kwargs,
    )
    curriculum = Curriculum(mdp, enabled=not args.no_curriculum, **schedule(total_steps))
    print(f"[curriculum] policy step {mdp.dt * 1e3:.0f} ms, observation {mdp.info.observation_space.shape[0]}, "
          f"action {mdp.info.action_space.shape[0]}"
          + ("" if dr_on else ", randomization off")
          + ("" if args.no_curriculum else f"; best checkpoint looked for from step {curriculum.complete_at} on"))

    wandb_kwargs = None
    if args.wandb:
        run_name = Path(args.run_dir).name
        wandb_kwargs = Logger.default_wandb_kwargs(
            args.wandb_project,
            group=run_name,
            name=f"{run_name}-s{args.seed}",
            config=dict(seed=args.seed, init_agent=args.init_agent, n_envs=n_envs, n_epochs=n_epochs,
                        n_steps_per_fit=n_steps_per_fit, batch_size=batch_size, alg=args.alg,
                        history_length=args.history_length, privileged=list(privileged), tf32=args.tf32,
                        domain_randomization=dr_on, dr_params=params, env_kwargs=env_kwargs,
                        reward_weights=mdp.reward_weights, curriculum=CURRICULUM,
                        curriculum_enabled=not args.no_curriculum,
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

    log = open(seed_dir / "walk.log", "w")
    best = dict(J=-float("inf"))
    last = dict()

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
        last.update(J=J, epoch=epoch, ep_len=L, lin_err=lin_err, alive=alive)
        agent.save(str(seed_dir / "walk_last.msh"))
        # Returns from before the schedule is complete are not comparable
        # with the ones after it (the commands get harder), so the best
        # checkpoint is only tracked once it is.
        if curriculum.complete:
            logger.log_best_agent(agent, J)
            if J >= best["J"]:  # the logger saves on >= too
                best.update(last)
        del dataset

    evaluate(0)
    for it in trange(n_epochs, leave=False, desc=f"seed {args.seed}", ncols=80):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        if (it + 1) % args.eval_every == 0 or it + 1 == n_epochs:
            evaluate(it + 1)
    log.close()

    out = seed_dir / "walk.msh"
    status = "ok"
    if "epoch" in best:
        ckpts = sorted((seed_dir / "logs").rglob("agent*-best.msh"), key=os.path.getmtime)
        if not ckpts:
            raise SystemExit("no checkpoint saved")
        shutil.copy(ckpts[-1], out)
    else:
        # No evaluation ran after the schedule completed, so no best
        # checkpoint was tracked.
        best = last
        agent.save(str(out))
        status = "schedule incomplete: last agent kept"
    (seed_dir / "walk.best").write_text(
        f"{best['epoch']} {best['J']} {best['ep_len']} {best['lin_err']} {best['alive']}\n"
    )
    print(f"[curriculum] done ({status}): epoch {best['epoch']} J={best['J']:.3f} ep_len={best['ep_len']:.1f} "
          f"lin_vel_err={best['lin_err']:.3f} alive={best['alive']:.3f} -> {out}")

    logger.finish()
    mdp.stop()
    return status


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", default="runs/h2_curriculum", help="sweep folder; this seed goes to seed_<seed>/")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--init-agent", default=None, help="checkpoint to warm-start from")
    p.add_argument("--reset-std", type=float, default=None,
                   help="policy std to reset to after a warm start (default: keep the checkpoint's)")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="H2Walk constructor argument (Python literal), e.g. --env-arg obs_noise=False or "
                        "--env-arg \"reward_weights={'foot_gait': 1.0}\"")

    g = p.add_argument_group("domain randomization")
    g.add_argument("--no-dr", action="store_true",
                   help="nothing random at all: nominal model, no pushes, no reset yaw")
    g.add_argument("--dr", action="append", default=[], metavar="KEY=VALUE",
                   help="override a DR_DEFAULTS entry or any other RandomizationParams field, "
                        "e.g. --dr position_offset=0.03 or --dr push_max_velocity=1.0")
    g.add_argument("--no-curriculum", action="store_true", help="full command ranges from the first update")
    g.add_argument("--privileged", nargs="*", default=list(DEFAULT_PRIVILEGED),
                   help="observation entries only the critic sees (names from H2Walk.observation_indices); "
                        "pass none for a symmetric agent")

    g = p.add_argument_group("algorithm")
    g.add_argument("--alg", choices=("rudin", "ppo"), default="rudin",
                   help="RudinPPO (adaptive learning rate from the policy KL, gradient clipping), or plain PPO")
    g.add_argument("--history-length", type=int, default=1, help="stacked observations the networks read")
    g.add_argument("--n-envs", type=int, default=4096)
    g.add_argument("--fragment-length", type=int, default=24, help="on-policy steps per environment per fit")
    g.add_argument("--fits-per-epoch", type=int, default=10)
    g.add_argument("--n-minibatches", type=int, default=4)
    g.add_argument("--n-epochs-policy", type=int, default=5)
    g.add_argument("--n-epochs-critic", type=int, default=5)
    g.add_argument("--n-episodes-test", type=int, default=256)
    g.add_argument("--eval-every", type=int, default=20,
                   help="evaluate (and consider for the best checkpoint) every N epochs, and at the last")
    g.add_argument("--actor-lr", type=float, default=1e-3)
    g.add_argument("--critic-lr", type=float, default=1e-3)
    g.add_argument("--desired-kl", type=float, default=0.01)
    g.add_argument("--clip-grad-norm", type=float, default=1.0)
    g.add_argument("--eps-ppo", type=float, default=0.2)
    g.add_argument("--lam", type=float, default=0.95)
    g.add_argument("--std-0", type=float, default=1.0)
    g.add_argument("--ent-coeff", type=float, default=0.01)
    g.add_argument("--n-features", type=int, nargs="+", default=[512, 256, 128])
    g.add_argument("--critic-features", type=int, nargs="+", default=[512, 256, 128])
    g.add_argument("--activation", default="elu")
    g.add_argument("--gain-scale", type=float, default=0.5)
    g.add_argument("--no-fused-adam", action="store_false", dest="fused_adam")

    g = p.add_argument_group("compute")
    g.add_argument("--device", default="cuda:0", help="cuda:0, or cpu for a slow smoke test without a GPU")
    g.add_argument("--tf32", action="store_true",
                   help="run the matrix multiplications of the networks in TF32 on the tensor cores")
    g.add_argument("--no-graph-capture", action="store_false", dest="graph_capture")
    g.add_argument("--no-wandb", action="store_false", dest="wandb")
    g.add_argument("--wandb-project", default="mushroom_rl_h2")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        args.graph_capture = False
        args.fused_adam = False
    else:
        assert torch.cuda.is_available(), "no CUDA device; pass --device cpu for a smoke test"
    TorchUtils.set_default_device(args.device)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dr_params = dict(DR_DEFAULTS)
    dr_params.update(parse_kv(args.dr))
    unknown = set(dr_params) - set(RandomizationParams().as_dict())
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
        f"alg={args.alg} history_length={args.history_length} tf32={args.tf32} init_agent={args.init_agent}",
        f"epochs={args.epochs} fits_per_epoch={args.fits_per_epoch} n_envs={args.n_envs} "
        f"n_minibatches={args.n_minibatches} n_epochs_policy={args.n_epochs_policy} "
        f"n_epochs_critic={args.n_epochs_critic}",
        f"domain_randomization={not args.no_dr} curriculum_enabled={not args.no_curriculum} dr_params={dr_params}",
        f"privileged={list(privileged)}",
        f"env_arg={args.env_arg}",
        f"curriculum={CURRICULUM}",
        f"std_0={args.std_0} ent_coeff={args.ent_coeff} actor_lr={args.actor_lr} critic_lr={args.critic_lr} "
        f"activation={args.activation} n_features={args.n_features} critic_features={args.critic_features}",
        f"git={git}", "",
    ])
    # One per seed, always (a reused --run-dir would otherwise keep the first
    # run's settings), and one for the run folder if it has none yet.
    (seed_dir / "config.txt").write_text(config_text)
    if not (run_dir / "config.txt").exists():
        (run_dir / "config.txt").write_text(config_text)

    try:
        status = train(args, seed_dir, dr_params, privileged)
    except Exception as e:  # noqa: BLE001 - recorded in status for the sweep
        (seed_dir / "status").write_text(f"failed: {type(e).__name__}: {e}\n")
        raise
    (seed_dir / "status").write_text(status + "\n")


if __name__ == "__main__":
    main()
