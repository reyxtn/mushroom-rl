"""
This script trains the Unitree H2 locomotion tasks with PPO in MuJoCo Warp.

Three tasks share one observation and action layout, so a policy trained on
one can start the next:

    python h2_ppo.py --task stand
    python h2_ppo.py --task walk --init-agent logs/PPO_H2Stand/.../agent-best.msh
    python h2_ppo.py --task run  --init-agent logs/PPO_H2Walk/.../agent-best.msh

Hyperparameters are the Go2 ones. They have not been tuned for the H2.

"""

import argparse
import ast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.environments.mujoco_warp_envs import H2Stand, H2Walk, H2Run
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

TASKS = {"stand": H2Stand, "walk": H2Walk, "run": H2Run}


def experiment(
    task,
    n_epochs,
    n_steps,
    n_steps_per_fit,
    n_episodes_test,
    n_envs,
    use_graph_capture=True,
    use_wandb=True,
    seed=None,
    init_agent=None,
    std_0=0.3,
    ent_coeff=0.0,
    env_kwargs=None,
):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    assert torch.cuda.is_available(), "MuJoCo Warp requires a CUDA device."
    assert n_envs >= 2, "n_envs must be at least 2."

    TorchUtils.set_default_device("cuda:0")

    # MDP
    mdp = TASKS[task](
        num_envs=n_envs, use_graph_capture=use_graph_capture, **(env_kwargs or {})
    )

    # Settings, copied from go2_ppo.py. Not tuned for the H2.
    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = [512, 256, 128]
    n_minibatches = 16
    batch_size = n_steps_per_fit // n_minibatches
    n_epochs_policy = 5
    eps = 0.2
    lam = 0.95
    # std_0 and ent_coeff come from the arguments. The Go2 values (1.0, 0.01)
    # knock a biped over at the first step: 1.0 * action_scale is 0.25 rad
    # of noise on every joint, and with a weak early reward the entropy
    # bonus alone keeps widening it.

    # Logging
    wandb_kwargs = None
    if use_wandb:
        wandb_kwargs = Logger.default_wandb_kwargs(
            "mushroom_rl_h2",
            config=dict(
                task=task,
                init_agent=init_agent,
                n_envs=n_envs,
                n_epochs=n_epochs,
                n_steps=n_steps,
                n_steps_per_fit=n_steps_per_fit,
                n_episodes_test=n_episodes_test,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                n_features=n_features,
                batch_size=batch_size,
                n_epochs_policy=n_epochs_policy,
                eps_ppo=eps,
                lam=lam,
                std_0=std_0,
                ent_coeff=ent_coeff,
                env_kwargs=env_kwargs,
                graph_capture=use_graph_capture,
            ),
        )

    logger = Logger(
        f"{PPO.name()}_{mdp.name()}",
        results_dir="./logs",
        seed=seed,
        wandb_kwargs=wandb_kwargs,
    )
    logger.log_experiment_info(
        PPO,
        mdp,
        n_epochs=n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=n_episodes_test,
        n_envs=n_envs,
    )

    # Policy
    policy = GaussianTorchPolicy(
        ActorNetwork,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=std_0,
        n_features=n_features,
    )

    # Agent
    critic_params = dict(
        network=ActorNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    agent = PPO(
        mdp.info,
        policy,
        critic_params=critic_params,
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps,
        lam=lam,
        ent_coeff=ent_coeff,
    )

    if init_agent is not None:
        # Warm start from a policy trained on another H2 task. The tasks
        # share the observation and action layout, so the network weights
        # transfer as they are; only the policy and the critic are copied,
        # the optimisers start fresh.
        loaded = PPO.load(init_agent)
        agent.policy.set_weights(loaded.policy.get_weights())
        agent._V.set_weights(loaded._V.get_weights())

    # Algorithm. No StandardizationPreprocessor: the environment applies the
    # fixed observation scaling of the reference, which is also what is used
    # at deployment time.
    core = Core(agent, mdp, logger=logger)

    def measure_tracking(n_steps=200):
        """
        Mean squared velocity tracking error over the worlds still alive,
        plus the fraction of worlds that diverged and the fraction still
        standing at the end. The error alone flatters a policy that falls,
        because fallen worlds stop counting; read it with alive_frac.

        A world stops counting once it terminates (fall, or non-finite state):
        it is never reset here, so its state afterwards is meaningless, and a
        diverged world would turn the whole mean into NaN. The fraction of
        worlds that went non-finite is returned separately, so divergence is
        visible instead of hidden.

        """
        mask = torch.ones(n_envs, dtype=torch.bool, device=mdp._device)
        obs, _ = mdp.reset_all(mask)
        alive = mask.clone()
        diverged = torch.zeros_like(mask)
        lin_sum = torch.zeros((), device=mdp._device)
        ang_sum = torch.zeros((), device=mdp._device)
        n = torch.zeros((), device=mdp._device)
        term_sums = {k: torch.zeros((), device=mdp._device) for k in mdp._REWARD_KEYS}
        for _ in range(n_steps):
            obs, _, absorbing, _ = mdp.step_all(
                mask, agent.policy.draw_action_greedy(obs)
            )
            qvel = wp.to_torch(mdp._data_wp.qvel)
            finite = torch.isfinite(qvel).all(dim=1)
            diverged |= ~finite
            alive &= finite & ~absorbing
            quat = mdp._read_data("base_rot")
            lin = mdp._quat_rotate_inverse(quat, qvel[:, 0:3])
            ang = qvel[:, 3:6]
            lin_err = ((mdp._commands[:, :2] - lin[:, :2]) ** 2).sum(dim=1)
            ang_err = (mdp._commands[:, 2] - ang[:, 2]) ** 2
            # where() rather than multiplying by the mask: NaN * 0 is NaN.
            lin_sum += torch.where(alive, lin_err, 0.0).sum()
            ang_sum += torch.where(alive, ang_err, 0.0).sum()
            n += alive.sum()
            for k in term_sums:
                term_sums[k] += torch.where(alive, mdp._reward_info[k], 0.0).sum()
        n = n.clamp(min=1)
        # Per-second values (the env multiplies every term by dt), so they
        # compare directly with the weights. Penalties are negative.
        terms = {f"r_{k}": (v / n / mdp.dt).item() for k, v in term_sums.items()}
        return (
            (lin_sum / n).item(),
            (ang_sum / n).item(),
            diverged.float().mean().item(),
            alive.float().mean().item(),
            terms,
        )

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(dataset.get_init_states()).mean().item()

        lin_err, ang_err, diverged_frac, alive_frac, terms = measure_tracking()

        logger.log_evaluation(
            epoch,
            J=J,
            R=R,
            entropy=E,
            mean_ep_len=L,
            V=V,
            lin_vel_err=lin_err,
            ang_vel_err=ang_err,
            diverged_frac=diverged_frac,
            alive_frac=alive_frac,
            **terms,
        )
        logger.log_best_agent(agent, J)

    # RUN
    evaluate(0)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        evaluate(it + 1)


def parse_env_args(items):
    kwargs = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env-arg expects KEY=VALUE, got {item!r}")
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value  # plain string
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="walk")
    parser.add_argument(
        "--std-0",
        type=float,
        default=0.3,
        help="initial policy std, in action units (x action_scale rad)",
    )
    parser.add_argument("--ent-coeff", type=float, default=0.0)
    parser.add_argument(
        "--env-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="constructor argument for the environment, repeatable, e.g. "
        "--env-arg alive_weight=3.0 --env-arg torque_weight=1e-7. "
        "VALUE is parsed as a Python literal",
    )
    parser.add_argument(
        "--init-agent",
        default=None,
        help="path of a saved agent from another H2 task to start from",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4096, help="number of parallel environments"
    )
    parser.add_argument(
        "--fragment-length",
        type=int,
        default=24,
        help="on-policy steps collected per environment per fit. "
        "n_steps_per_fit is derived from this so that "
        "changing --n-envs does not silently change the "
        "GAE horizon",
    )
    parser.add_argument(
        "--fits-per-epoch", type=int, default=50, help="policy updates per epoch"
    )
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--n-episodes-test", type=int, default=256)
    parser.add_argument(
        "--no-graph-capture",
        action="store_false",
        dest="graph_capture",
        help="disable CUDA graph capture",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="wandb",
        help="disable Weights & Biases logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the experiment, random when not given",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_steps_per_fit = args.n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch

    experiment(
        task=args.task,
        n_epochs=args.n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        n_envs=args.n_envs,
        use_graph_capture=args.graph_capture,
        use_wandb=args.wandb,
        seed=args.seed,
        init_agent=args.init_agent,
        std_0=args.std_0,
        ent_coeff=args.ent_coeff,
        env_kwargs=parse_env_args(args.env_arg),
    )
