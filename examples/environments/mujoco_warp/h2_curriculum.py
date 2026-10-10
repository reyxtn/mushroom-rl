"""
Trains the Unitree H2 locomotion curriculum, stand -> walk -> run, for one
seed, with domain randomization and an asymmetric actor-critic, in MuJoCo
Warp. The counterpart of the new_isaac ``go2_curriculum.py`` for the warp
backend, plus the stage chain that used to live in ``h2_sweep.sh``.

    python h2_curriculum.py --seed 1
    for s in 1 2 3 4 5; do python h2_curriculum.py --seed $s; done
    python h2_curriculum.py --seed 1 --tf32 --alg rudin --history-length 1
    python h2_curriculum.py --seed 1 --stages walk run --init-agent runs/x/seed_1/stand.msh
    python h2_curriculum.py --seed 1 --no-dr                     # nominal model, no pushes
    python h2_curriculum.py --seed 1 --dr add_trunk_mass="(-5,10)" --dr max_delay_steps=2

Every seed writes to ``<run-dir>/seed_<seed>/``: ``<stage>.msh`` (best
checkpoint of the stage), ``<stage>.best`` (its epoch and metrics),
``<stage>.log`` (one line per evaluation) and ``status``; the layout
``h2_pick_best.sh`` reads. ``<run-dir>/config.txt`` records the settings.

Two curricula run at once. Across stages, every stage warm-starts from the
best checkpoint of the previous one, and a stage below its gate stops the
chain. Within a stage, after every policy update a schedule widens the
velocity commands and the actuation delay in steps and anneals the tracking
tolerance, as in the Go2 script; the environment only accepts new values,
when to change them is decided here.

The asymmetry: the policy network reads the observation minus the entries
named by ``--privileged`` (by default the actuation delay and the encoder
offset the randomization drew, which the real robot cannot measure), while
the critic, which only runs in simulation, reads all of it.

"""

import argparse
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
import warp as wp
from tqdm import trange

from mushroom_rl.core import Agent, Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO, RudinPPO
from mushroom_rl.environments.mujoco_warp_envs import H2Run, H2Stand, H2Walk
from mushroom_rl.environments.mujoco_warp_envs.h2 import MaskedActorNetwork
from mushroom_rl.environments.mujoco_warp_envs.legged_randomizer import LeggedRandomizationParams
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

STAGE_EPOCHS = dict(stand=80, walk=120, run=200)
STAGE_MIN_EP_LEN = dict(stand=200.0, walk=700.0, run=0.0)

# Reward and push settings of the previous sweeps: survival worth less than
# tracking, posture and stillness paid for. Stand keeps the env defaults
# (it needs the large alive bonus to learn to balance at all).
STAGE_ENV_ARGS = dict(
    stand={},
    walk=dict(
        alive_weight=0.5,
        tracking_lin_vel_weight=2.0,
        tracking_ang_vel_weight=1.0,
        joint_deviation_weight=0.5,
        feet_slip_weight=1.0,
        ang_vel_xy_weight=0.3,
        action_rate_weight=0.03,
        push_max_vel=0.3,
    ),
    run=dict(
        alive_weight=0.5,
        tracking_lin_vel_weight=2.0,
        tracking_ang_vel_weight=1.0,
        feet_slip_weight=1.0,
        action_rate_weight=0.03,
    ),
)

# Within-stage schedules, as fractions of the stage's policy updates. The
# command ranges and the delay ceiling switch in stages; the tracking
# tolerance is interpolated between the two ends of its window.
STAGE_CURRICULUM = dict(
    stand=dict(
        command_stages=(),
        command_ranges=[None],
        delay_stages=(0.3, 0.6),
        delay_steps=[0, 2, 4],
        sigma_window=None,
        sigmas=None,
    ),
    walk=dict(
        command_stages=(0.3, 0.6),
        command_ranges=[
            dict(lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2)),
            dict(lin_vel_x=(-0.5, 0.8), lin_vel_y=(-0.3, 0.3)),
            dict(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.4, 0.4)),
        ],
        delay_stages=(0.3, 0.6),
        delay_steps=[0, 2, 4],
        sigma_window=(0.1, 0.5),
        sigmas=(0.5, 0.15),
    ),
    run=dict(
        command_stages=(0.3, 0.6),
        command_ranges=[
            dict(lin_vel_x=(1.0, 1.5), lin_vel_y=(-0.2, 0.2)),
            dict(lin_vel_x=(1.0, 2.2), lin_vel_y=(-0.3, 0.3)),
            dict(lin_vel_x=(1.0, 3.0), lin_vel_y=(-0.3, 0.3)),
        ],
        delay_stages=(0.3, 0.6),
        delay_steps=[0, 2, 4],
        sigma_window=(0.1, 0.5),
        sigmas=(0.5, 0.25),
    ),
)

# Domain randomization ranges for the H2. The Isaac Go2 defaults are for a
# 15 kg robot; these are scaled for 75 kg and a biped. Pushes are set per
# stage through push_max_vel / push_interval in STAGE_ENV_ARGS and the
# task defaults. Any entry can be overridden with --dr KEY=VALUE.
DR_DEFAULTS = dict(
    add_trunk_mass=(-2.0, 6.0),
    add_com_displacement=(-0.03, 0.03),
    p_gain_scale=(0.85, 1.15),
    d_gain_scale=(0.85, 1.15),
    ground_friction_factor=5.0 / 9.0,
    max_delay_steps=4,
    mixed_chance=0.0,
    position_offset=0.02,
    reset_yaw_range=math.pi,
)

DEFAULT_PRIVILEGED = ("actual_delay", "joint_calib_offset")


class Curriculum:
    """
    The schedule a stage is tightened along, applied after every policy
    update. Ported from the new_isaac Go2 script; thresholds are fractions
    of the stage's updates rather than step counts, so they do not depend on
    the epoch count.

    """

    def __init__(self, mdp, n_fits, command_stages, command_ranges, delay_stages, delay_steps,
                 sigma_window, sigmas):
        self._mdp = mdp
        self._n_fits = max(int(n_fits), 1)
        self._command_stages = tuple(command_stages)
        self._command_ranges = list(command_ranges)
        self._delay_stages = tuple(delay_stages)
        self._delay_steps = list(delay_steps)
        self._sigma_window = sigma_window
        self._sigmas = sigmas
        self._fit = 0
        self.apply()

    def __call__(self, dataset):
        self._fit += 1
        self.apply()

    @property
    def fraction(self):
        return min(self._fit / self._n_fits, 1.0)

    @property
    def command_stage(self):
        return sum(self.fraction >= t for t in self._command_stages)

    @property
    def delay_stage(self):
        return sum(self.fraction >= t for t in self._delay_stages)

    @property
    def progress(self):
        if self._sigma_window is None:
            return 1.0
        start, end = self._sigma_window
        return min(max((self.fraction - start) / (end - start), 0.0), 1.0)

    def apply(self):
        ranges = self._command_ranges[min(self.command_stage, len(self._command_ranges) - 1)]
        if ranges is not None:
            self._mdp.command_ranges = ranges
        if self._mdp.domain_randomization:
            self._mdp.max_delay_steps = self._delay_steps[min(self.delay_stage, len(self._delay_steps) - 1)]
        if self._sigmas is not None:
            start, end = self._sigmas
            self._mdp.tracking_sigma = start + self.progress * (end - start)

    def state(self):
        return dict(
            cmd_stage=self.command_stage,
            delay_steps=self._mdp.max_delay_steps,
            sigma=self._mdp.tracking_sigma,
            progress=self.progress,
        )


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
    return alg(mdp.info, policy, critic_params=critic_params, **alg_params)


def warm_start(agent, path, reset_std=None):
    """
    Copy the policy and critic weights of a saved agent. Every H2 task
    shares the observation and action layout, so the weights transfer as
    they are; the optimizers start fresh. The std can be reset, as a
    policy that has converged on one task has too little exploration left
    for the next.

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

def measure_tracking(mdp, agent, n_envs, n_steps=200):
    """
    Greedy roll-out on the training environment (randomization included):
    mean squared velocity tracking errors over the worlds still alive, the
    fraction that diverged, the fraction standing at the end, and every
    reward term per second.

    """
    dev = mdp._device
    mask = torch.ones(n_envs, dtype=torch.bool, device=dev)
    obs, _ = mdp.reset_all(mask)
    alive = mask.clone()
    diverged = torch.zeros_like(mask)
    lin_sum = torch.zeros((), device=dev)
    ang_sum = torch.zeros((), device=dev)
    n = torch.zeros((), device=dev)
    term_sums = {k: torch.zeros((), device=dev) for k in mdp._REWARD_KEYS}
    for _ in range(n_steps):
        obs, _, absorbing, _ = mdp.step_all(mask, agent.policy.draw_action_greedy(obs))
        qvel = wp.to_torch(mdp._data_wp.qvel)
        finite = torch.isfinite(qvel).all(dim=1)
        diverged |= ~finite
        alive &= finite & ~absorbing
        quat = mdp._read_data("base_rot")
        lin = mdp._quat_rotate_inverse(quat, qvel[:, 0:3])
        ang = qvel[:, 3:6]
        lin_err = ((mdp._commands[:, :2] - lin[:, :2]) ** 2).sum(dim=1)
        ang_err = (mdp._commands[:, 2] - ang[:, 2]) ** 2
        lin_sum += torch.where(alive, lin_err, 0.0).sum()
        ang_sum += torch.where(alive, ang_err, 0.0).sum()
        n += alive.sum()
        for k in term_sums:
            term_sums[k] += torch.where(alive, mdp._reward_info[k], 0.0).sum()
    n = n.clamp(min=1)
    terms = {f"r_{k}": (v / n / mdp.dt).item() for k, v in term_sums.items()}
    return (lin_sum / n).item(), (ang_sum / n).item(), diverged.float().mean().item(), alive.float().mean().item(), terms


# ----------------------------------------------------------------------
# One stage
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


def run_stage(stage, args, seed_dir, init_path, dr_params, privileged):
    n_envs = args.n_envs
    n_steps_per_fit = n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch
    n_epochs = getattr(args, f"{stage}_epochs")
    batch_size = n_steps_per_fit // args.n_minibatches
    n_fits = n_epochs * args.fits_per_epoch

    env_kwargs = dict(STAGE_ENV_ARGS[stage])
    env_kwargs.update(parse_kv(args.env_arg))
    env_kwargs.update(parse_kv(getattr(args, f"{stage}_arg")))
    dr_on = env_kwargs.pop("domain_randomization", not args.no_dr)

    mdp = TASKS[stage](
        num_envs=n_envs,
        use_graph_capture=args.graph_capture,
        domain_randomization=dr_on,
        randomization_params=LeggedRandomizationParams(**dr_params),
        observed_randomization=privileged,
        **env_kwargs,
    )
    curriculum = Curriculum(mdp, n_fits, **STAGE_CURRICULUM[stage])

    wandb_kwargs = None
    if args.wandb:
        wandb_kwargs = Logger.default_wandb_kwargs(
            args.wandb_project,
            config=dict(stage=stage, seed=args.seed, init_agent=init_path, n_envs=n_envs, n_epochs=n_epochs,
                        n_steps_per_fit=n_steps_per_fit, batch_size=batch_size, alg=args.alg,
                        history_length=args.history_length, privileged=list(privileged), tf32=args.tf32,
                        domain_randomization=dr_on, dr_params=dr_params, env_kwargs=env_kwargs,
                        curriculum=STAGE_CURRICULUM[stage], actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                        std_0=args.std_0, ent_coeff=args.ent_coeff),
        )
    alg_name = "RudinPPO" if args.alg == "rudin" else "PPO"
    logger = Logger(f"{alg_name}_{mdp.name()}", results_dir=str(seed_dir / "logs"), seed=args.seed,
                    wandb_kwargs=wandb_kwargs)

    agent = build_agent(mdp, args, batch_size, privileged)
    if init_path is not None:
        warm_start(agent, init_path, reset_std=args.reset_std)
    core = Core(agent, mdp, callbacks_fit=[curriculum], logger=logger)

    stage_log = open(seed_dir / f"{stage}.log", "w")
    best = dict(J=-float("inf"))

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(dataset.get_init_states()).mean().item()
        lin_err, ang_err, diverged, alive, terms = measure_tracking(mdp, agent, n_envs)
        cur = curriculum.state()
        logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L, V=V, lin_vel_err=lin_err,
                              ang_vel_err=ang_err, diverged_frac=diverged, alive_frac=alive, **cur, **terms)
        logger.log_best_agent(agent, J)
        line = (f"{time.strftime('%d/%m/%Y %H:%M:%S')} [INFO] Epoch {epoch} | J: {J} R: {R} entropy: {E} "
                f"mean_ep_len: {L} V: {V} lin_vel_err: {lin_err} ang_vel_err: {ang_err} "
                f"diverged_frac: {diverged} alive_frac: {alive} "
                + " ".join(f"{k}: {v}" for k, v in {**cur, **terms}.items()))
        stage_log.write(line + "\n")
        stage_log.flush()
        if J > best["J"]:
            best.update(J=J, epoch=epoch, ep_len=L, lin_err=lin_err, alive=alive)
        del dataset

    evaluate(0)
    for it in trange(n_epochs, leave=False, desc=f"seed {args.seed} {stage}"):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        evaluate(it + 1)
    stage_log.close()

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
                   help="policy std to reset to after every warm start (default: keep the checkpoint's)")
    p.add_argument("--force", action="store_true", help="ignore the gates between stages")
    for s in STAGES:
        p.add_argument(f"--{s}-epochs", type=int, default=STAGE_EPOCHS[s])
        p.add_argument(f"--{s}-min-ep-len", type=float, default=STAGE_MIN_EP_LEN[s],
                       help=f"gate: the next stage runs only if {s} reached this episode length")
        p.add_argument(f"--{s}-arg", action="append", default=[], metavar="KEY=VALUE",
                       help=f"environment argument for the {s} stage only")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="environment argument for every stage (Python literal)")

    g = p.add_argument_group("domain randomization")
    g.add_argument("--no-dr", action="store_true", help="nominal model, no pushes, deterministic resets")
    g.add_argument("--dr", action="append", default=[], metavar="KEY=VALUE",
                   help="override a LeggedRandomizationParams entry, e.g. --dr position_offset=0.04")
    g.add_argument("--privileged", nargs="*", default=list(DEFAULT_PRIVILEGED),
                   help="observation entries only the critic sees (names from H2Walk.observation_indices, "
                        "e.g. actual_delay joint_calib_offset base_lin_vel); pass none for a symmetric agent")

    g = p.add_argument_group("algorithm")
    g.add_argument("--alg", choices=("rudin", "ppo"), default="rudin",
                   help="RudinPPO (adaptive learning rate from the policy KL, gradient clipping) or plain PPO")
    g.add_argument("--history-length", type=int, default=1, help="stacked observations the networks read")
    g.add_argument("--n-envs", type=int, default=4096)
    g.add_argument("--fragment-length", type=int, default=24, help="on-policy steps per environment per fit")
    g.add_argument("--fits-per-epoch", type=int, default=10)
    g.add_argument("--n-minibatches", type=int, default=16)
    g.add_argument("--n-epochs-policy", type=int, default=5)
    g.add_argument("--n-epochs-critic", type=int, default=5)
    g.add_argument("--n-episodes-test", type=int, default=256)
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.tf32:
        # 10-bit mantissa matmuls on the tensor cores; the physics in warp is
        # untouched. See the notes in the pull request before relying on it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dr_params = dict(DR_DEFAULTS)
    dr_params.update(parse_kv(args.dr))
    privileged = tuple(n for n in args.privileged if n.lower() != "none")

    run_dir = Path(args.run_dir)
    seed_dir = run_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    config = run_dir / "config.txt"
    if not config.exists():
        config.write_text(
            "\n".join(
                [f"stages={' '.join(args.stages)} alg={args.alg} history_length={args.history_length} tf32={args.tf32}",
                 f"epochs stand={args.stand_epochs} walk={args.walk_epochs} run={args.run_epochs} "
                 f"fits_per_epoch={args.fits_per_epoch} n_envs={args.n_envs}",
                 f"domain_randomization={not args.no_dr} dr_params={dr_params}",
                 f"privileged={list(privileged)}",
                 f"gates stand_min_ep_len={args.stand_min_ep_len} walk_min_ep_len={args.walk_min_ep_len}",
                 f"env_arg={args.env_arg} stand_arg={args.stand_arg} walk_arg={args.walk_arg} run_arg={args.run_arg}",
                 f"stage_env_args={STAGE_ENV_ARGS}",
                 f"curriculum={STAGE_CURRICULUM}",
                 f"std_0={args.std_0} ent_coeff={args.ent_coeff} actor_lr={args.actor_lr} critic_lr={args.critic_lr}",
                 f"git={git}", ""]
            )
        )

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
