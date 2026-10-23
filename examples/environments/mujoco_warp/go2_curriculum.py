"""
Trains the Unitree Go2 velocity tracking task for one seed, with domain
randomization, an asymmetric actor-critic and a within-run curriculum, in
MuJoCo Warp. The counterpart of the new_isaac ``go2_curriculum.py`` for the
warp backend, and of ``h2_curriculum.py`` for the Go2 (one task, so no
stage chain).

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
script; only the names in ``Go2Base.RANDOMIZATION_OBS`` are appended to
the observation.

"""

import argparse
import ast
import inspect
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
from mushroom_rl.environments.mujoco_warp_envs import Go2Walk
from mushroom_rl.environments.mujoco_warp_envs.legged_randomizer import LeggedRandomizationParams
from mushroom_rl.environments.mujoco_warp_envs.networks import MaskedActorNetwork
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

STAGE = "walk"  # the file names h2_pick_best.sh expects

# Within-run schedule, as fractions of the policy updates. The command
# ranges and the delay ceiling switch in stages; the tracking tolerance is
# interpolated between the two ends of its window. Delays are in seconds
# and converted to the environment's intermediate steps (20 ms at the
# default n_substeps=10 / n_intermediate_steps=1 split). The last command
# range is the environment's default, +-1 m/s in x and y.
CURRICULUM = dict(
    command_stages=(0.3, 0.6),
    command_ranges=[
        dict(lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5)),
        dict(lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.8, 0.8)),
        dict(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0)),
    ],
    delay_stages=(0.3, 0.6),
    delays=[0.0, 0.01, 0.02],
    sigma_window=(0.1, 0.5),
    sigmas=(0.5, 0.25),
)

# Domain randomization ranges: the Isaac Go2 values (the randomizer's own
# defaults), listed so config.txt records them. ``max_delay`` is in seconds
# (converted to intermediate steps before the environment is built); every
# other entry is a LeggedRandomizationParams field. Pushes are the
# environment's: 1 m/s every 750 steps on average (push_max_vel /
# push_interval kwargs). Any entry can be overridden with --dr KEY=VALUE.
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

DEFAULT_PRIVILEGED = ("actual_delay", "joint_calib_offset")


# ----------------------------------------------------------------------
# Curriculum. Lives in the training script on purpose, as in the Isaac
# go2_curriculum.py: the environment only accepts new values, when to
# change them is decided here.
# ----------------------------------------------------------------------

MODEL_TIMESTEP = 0.002  # option/@timestep of the H2 and Go2 models


def intermediate_dt(env_cls, env_kwargs, timestep=MODEL_TIMESTEP):
    """
    Duration of one intermediate step, the unit of the actuation latency:
    the model timestep times ``n_substeps`` (from ``env_kwargs`` or the
    environment's default).

    """
    if "n_substeps" in env_kwargs:
        return timestep * int(env_kwargs["n_substeps"])
    for cls in env_cls.__mro__:  # the task classes forward **kwargs to the base
        param = inspect.signature(cls.__init__).parameters.get("n_substeps")
        if param is not None and param.default is not inspect.Parameter.empty:
            return timestep * int(param.default)
    raise ValueError(f"{env_cls.__name__} has no n_substeps default to read the latency unit from")


def delay_to_steps(delay_s, idt):
    """Seconds to intermediate steps, half up: 10 ms at a 20 ms step is one step, not none."""
    return int(float(delay_s) / idt + 0.5)


class Curriculum:
    """
    The schedule a stage is tightened along, applied after every policy
    update (a ``Core`` fit callback). Thresholds are fractions of the stage's
    updates rather than step counts, so they do not depend on the epoch
    count.

    ``complete`` is True once every schedule has reached its final value.
    The stage's best checkpoint should only be looked for from then on:
    before, the commands are easier and the tracking kernel wider, so the
    returns are higher for reasons that have nothing to do with the policy.

    With ``enabled=False`` every schedule sits at its final value from the
    start and ``complete`` is True throughout.

    Args:
        mdp: the environment; needs ``command_ranges``, ``tracking_sigma``,
            ``max_delay_steps`` setters, ``max_delay_steps_limit`` and
            ``domain_randomization``;
        n_fits (int): policy updates in the stage;
        command_stages (tuple): fractions at which the command ranges switch
            to the next entry of ``command_ranges``;
        command_ranges (list): dicts for the ``command_ranges`` setter, or
            None to leave the ranges alone;
        delay_stages (tuple): fractions at which the latency ceiling switches
            to the next entry of ``delays``;
        delays (list): latency ceilings in seconds, converted with ``idt``
            and clamped to the environment's limit;
        sigma_window (tuple, None): fractions between which the tracking
            sigma is interpolated from ``sigmas[0]`` to ``sigmas[1]``;
        sigmas (tuple, None): the two ends of the sigma schedule;
        idt (float): duration of one intermediate step, in seconds;
        enabled (bool, True): see above.

    """

    def __init__(self, mdp, n_fits, command_stages, command_ranges, delay_stages, delays,
                 sigma_window, sigmas, idt, enabled=True):
        self._mdp = mdp
        self._n_fits = max(int(n_fits), 1)
        self._command_stages = tuple(command_stages)
        self._command_ranges = list(command_ranges)
        self._delay_stages = tuple(delay_stages)
        self._delay_steps = [min(delay_to_steps(d, idt), mdp.max_delay_steps_limit) for d in delays]
        self._sigma_window = sigma_window
        self._sigmas = sigmas
        self._enabled = enabled
        self._fit = 0
        thresholds = list(self._command_stages) + list(self._delay_stages)
        if sigma_window is not None:
            thresholds.append(sigma_window[1])
        self._complete_at = max(thresholds) if thresholds else 0.0
        self.apply()

    def __call__(self, dataset):
        self._fit += 1
        self.apply()

    @property
    def fraction(self):
        if not self._enabled:
            return 1.0
        return min(self._fit / self._n_fits, 1.0)

    @property
    def complete_at(self):
        return self._complete_at

    @property
    def complete(self):
        return self.fraction >= self._complete_at

    @property
    def command_stage(self):
        return sum(self.fraction >= t for t in self._command_stages)

    @property
    def delay_stage(self):
        return sum(self.fraction >= t for t in self._delay_stages)

    @property
    def delay_steps(self):
        """The latency ceilings of the schedule, in intermediate steps."""
        return list(self._delay_steps)

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
            complete=float(self.complete),
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

    # The latency is specified in seconds and the environment counts it in
    # intermediate steps, whose length depends on how the 20 ms policy step
    # is split; max_delay sizes the action history, the schedule is clamped
    # to it.
    idt = intermediate_dt(Go2Walk, env_kwargs)
    params = dict(dr_params)
    max_delay = params.pop("max_delay")
    params["max_delay_steps"] = delay_to_steps(max_delay if dr_on else 0.0, idt)

    mdp = Go2Walk(
        num_envs=n_envs,
        use_graph_capture=args.graph_capture,
        domain_randomization=dr_on,
        randomization_params=LeggedRandomizationParams(**params),
        observed_randomization=tuple(n for n in privileged if n in Go2Walk.RANDOMIZATION_OBS),
        **env_kwargs,
    )
    if abs(mdp.dt / mdp._n_intermediate_steps - idt) > 1e-9:
        raise SystemExit(f"intermediate step is {mdp.dt / mdp._n_intermediate_steps * 1e3:.1f} ms, "
                         f"expected {idt * 1e3:.1f} ms: is the model timestep still 0.002?")
    curriculum = Curriculum(mdp, n_fits, idt=idt, enabled=not args.no_curriculum, **CURRICULUM)
    print(f"[curriculum] intermediate step {idt * 1e3:.0f} ms, latency up to "
          f"{params['max_delay_steps']} step(s) = {params['max_delay_steps'] * idt * 1e3:.0f} ms"
          + ("" if dr_on else " (randomization off)")
          + ("" if args.no_curriculum else f"; best checkpoint looked for from {curriculum.complete_at:.0%} on"))

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
                    wandb_kwargs=wandb_kwargs)

    agent = build_agent(mdp, args, batch_size, privileged)
    if args.init_agent is not None:
        warm_start(agent, args.init_agent, reset_std=args.reset_std)
    core = Core(agent, mdp, callbacks_fit=[curriculum], logger=logger)

    log = open(seed_dir / f"{STAGE}.log", "w")
    best = dict(J=-float("inf"))

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(dataset.get_init_states()).mean().item()
        lin_err, ang_err, diverged, alive, terms = measure_tracking(mdp, agent, n_envs, args.tracking_steps)
        cur = curriculum.state()
        logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L, V=V, lin_vel_err=lin_err,
                              ang_vel_err=ang_err, diverged_frac=diverged, alive_frac=alive, **cur, **terms)
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

    # Evaluation steps all n_envs worlds until n_episodes_test of them end
    # plus tracking_steps more, which with a policy that survives costs as
    # much as several epochs of training; so only every eval_every epochs,
    # and always at the last one.
    evaluate(0)
    for it in trange(args.n_epochs, leave=False, desc=f"seed {args.seed}"):
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
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--init-agent", default=None, help="checkpoint to warm-start from (same observation layout)")
    p.add_argument("--reset-std", type=float, default=None,
                   help="policy std to reset to after the warm start (default: keep the checkpoint's)")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="Go2Walk constructor argument (Python literal), e.g. --env-arg push_max_vel=0.5")

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
    g.add_argument("--alg", choices=("rudin", "ppo"), default="ppo",
                   help="plain PPO, or RudinPPO (adaptive learning rate from the policy KL, gradient clipping)")
    g.add_argument("--history-length", type=int, default=1, help="stacked observations the networks read")
    g.add_argument("--n-envs", type=int, default=4096)
    g.add_argument("--fragment-length", type=int, default=24, help="on-policy steps per environment per fit")
    g.add_argument("--fits-per-epoch", type=int, default=50)
    g.add_argument("--n-minibatches", type=int, default=16)
    g.add_argument("--n-epochs-policy", type=int, default=5)
    g.add_argument("--n-epochs-critic", type=int, default=5,
                   help="critic passes per fit. 5 is the Isaac Go2 script's value; go2_ppo.py used mushroom's "
                        "default 10, and on the H2 under domain randomization 1 worked best. Untested on the Go2")
    g.add_argument("--n-episodes-test", type=int, default=256)
    g.add_argument("--eval-every", type=int, default=1,
                   help="evaluate (and consider for the best checkpoint) every N epochs, and at the last")
    g.add_argument("--tracking-steps", type=int, default=200,
                   help="policy steps of the greedy roll-out that measures lin/ang_vel_err at every evaluation")
    g.add_argument("--actor-lr", type=float, default=3e-4)
    g.add_argument("--critic-lr", type=float, default=3e-4)
    g.add_argument("--desired-kl", type=float, default=0.01)
    g.add_argument("--clip-grad-norm", type=float, default=1.0)
    g.add_argument("--eps-ppo", type=float, default=0.2)
    g.add_argument("--lam", type=float, default=0.95)
    g.add_argument("--std-0", type=float, default=1.0)
    g.add_argument("--ent-coeff", type=float, default=0.01)
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
    unknown = set(dr_params) - set(DR_DEFAULTS) - set(LeggedRandomizationParams().as_dict())
    if unknown:
        raise SystemExit(f"unknown --dr entries: {sorted(unknown)}")
    if "max_delay_steps" in dr_params:
        raise SystemExit("the latency is given in seconds: --dr max_delay=0.02, not max_delay_steps")
    pushes = {"push_max_velocity", "push_probability", "push_interval_range"} & set(dr_params)
    if pushes:
        raise SystemExit(f"{sorted(pushes)}: pushes are set by the environment kwargs push_max_vel / "
                         "push_interval, which override these; use --env-arg push_max_vel=0.5")
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
