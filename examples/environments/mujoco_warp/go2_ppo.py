"""
This script shows how to train the Unitree Go2 velocity tracking task with PPO in MuJoCo Warp.


"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.environments.mujoco_warp_envs import Go2Walk
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(
    n_epochs,
    n_steps,
    n_steps_per_fit,
    n_episodes_test,
    n_envs,
    use_graph_capture=True,
    use_wandb=True,
    seed=None,
):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    assert torch.cuda.is_available(), "MuJoCo Warp requires a CUDA device."
    assert n_envs >= 2, "n_envs must be at least 2."

    TorchUtils.set_default_device("cuda:0")

    # MDP
    mdp = Go2Walk(num_envs=n_envs, use_graph_capture=use_graph_capture)

    # Settings. The A1 reference uses lr 1e-3, but relies on RudinPPO to adapt
    # it from the policy KL divergence; with vanilla PPO the learning rate is
    # fixed, so a smaller and more conservative value is used here.
    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = [512, 256, 128]
    n_minibatches = 16
    batch_size = n_steps_per_fit // n_minibatches
    n_epochs_policy = 5
    eps = 0.2
    lam = 0.95
    std_0 = 1.0
    ent_coeff = 0.01

    # Logging
    wandb_kwargs = None
    if use_wandb:
        wandb_kwargs = Logger.default_wandb_kwargs(
            "mushroom_rl_go2",
            config=dict(
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

    # Algorithm. No StandardizationPreprocessor: the environment applies the
    # fixed observation scaling of the reference, which is also what is used
    # at deployment time.
    core = Core(agent, mdp, logger=logger)

    def measure_tracking(n_steps=200):
        mask = torch.ones(n_envs, dtype=torch.bool, device=mdp._device)
        obs, _ = mdp.reset_all(mask)
        lin_acc = ang_acc = 0.0
        for _ in range(n_steps):
            obs, *_ = mdp.step_all(mask, agent.policy.draw_action_greedy(obs))
            qvel = wp.to_torch(mdp._data_wp.qvel)
            quat = mdp._read_data("base_rot")
            lin = mdp._quat_rotate_inverse(quat, qvel[:, 0:3])
            ang = qvel[:, 3:6]
            lin_acc += ((mdp._commands[:, :2] - lin[:, :2]) ** 2).sum(
                dim=1
            ).mean().item() / n_steps
            ang_acc += ((mdp._commands[:, 2] - ang[:, 2]) ** 2).mean().item() / n_steps
        return lin_acc, ang_acc

    def evaluate(epoch):
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        L = dataset.episodes_length.float().mean().item()
        V = agent._V(dataset.get_init_states()).mean().item()

        lin_err, ang_err = measure_tracking()

        logger.log_evaluation(
            epoch,
            J=J,
            R=R,
            entropy=E,
            mean_ep_len=L,
            V=V,
            lin_vel_err=lin_err,
            ang_vel_err=ang_err,
        )
        logger.log_best_agent(agent, J)

    # RUN
    evaluate(0)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        evaluate(it + 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        n_epochs=args.n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        n_envs=args.n_envs,
        use_graph_capture=args.graph_capture,
        use_wandb=args.wandb,
        seed=args.seed,
    )
