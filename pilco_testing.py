# MC-PILCO training loop with Pendulum

import gpytorch
import torch
import numpy as np
import gymnasium as gym
import tqdm
from collections import defaultdict

from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import SerialEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import RandomPolicy, check_env_specs
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage

from torch_pilco.model_learning.dynamical_models import (
    ExactDynamicalModel,
    ExactFit,
)
from torch_pilco.policy_learning.controllers import SumOfGaussians
from torch_pilco.rewards import pendulum_cost
from torch_pilco.policy_learning.rollout import GPyTorchEnv


def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()

# make the batch version of our gym environment
def make_env():
    return GymEnv("Pendulum-v1")

def main():
    # if torch.cuda.is_available():
    #     print("GPU is available. Using GPU backend.")
    #     device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():
    #     print("MPS is available. Using MPS backend.")
    #     device = torch.device("mps")
    # else:
    #     print("MPS not available. Falling back to CPU.")
    #     device = torch.device("cpu")

    device = torch.device("cpu")

    frames_per_batch = 150
    env = GymEnv("Pendulum-v1")

    random_policy = RandomPolicy(env.action_spec)
    action_dim = env.action_space.shape[0]
    x = env.reset()
    state_dim = x['observation'].shape[0]

    num_particles = 400
    num_basis = 100

    num_pilco_training_loops = 5

    control_policy = SumOfGaussians(
        state_dim,
        action_dim,
        num_basis,
        u_max=env.action_space.high[0],
        dtype=torch.float32,
    ) 
    batched_policy = torch.vmap(control_policy, in_dims=0)
    # Store each interaction with the environment
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

    # Generate a random trajectory from the environment
    collector = SyncDataCollector(
        env,
        policy=random_policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch,
    )

    penv = SerialEnv(1, make_env)
    print(check_env_specs(penv))

    for _ in range(num_pilco_training_loops):
        # Put the data into the replay buffer
        for data in collector:
            # convert the tensordict from collector to a version
            # suitable for dynamical model
            replay_buffer.extend(data)

        # Now grab some data and fit the GP
        # Use the whole buffer for data
        states, actions = build_pendulum_training_data(replay_buffer.sample(len(replay_buffer)))

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=states.shape[1]
        )
        model = ExactDynamicalModel(
            states,
            actions,
            likelihood,
        )
        # Find optimal model hyperparameters
        ExactFit(model.to(torch.float64), print_loss=False, n_training_iter=100)

        gp_env = GPyTorchEnv(
            model,
            env,
            pendulum_cost,
            replay_buffer,
            device=device,
            batch_size=(num_particles,)
        )
        print(check_env_specs(gp_env))
        batched_policy = torch.vmap(control_policy, in_dims=0)
        policy = TensorDictModule(
            batched_policy,
            in_keys=["observation"],
            out_keys=["action"],
        )
        optim = torch.optim.Adam(control_policy.parameters(), lr=2e-3)
        if num_pilco_training_loops == 0:
            N = 2_000
        else:
            N = 4_000
        pbar = tqdm.tqdm(range(N // num_particles))
        optim = torch.optim.Adam(control_policy.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N)
        logs = defaultdict(list)

        for _ in pbar:
            rollout = gp_env.rollout(35, policy)
            traj_return = rollout["next", "reward"].mean(dim=0).sum()
            traj_return.backward()
            gn = torch.nn.utils.clip_grad_norm_(control_policy.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            pbar.set_description(
                f"reward: {traj_return: 4.4f}, "
                f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
            )
            logs["return"].append(traj_return.item())
            logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean(dim=0).item())
            scheduler.step()
        penv.reset()
        collector = SyncDataCollector(
            penv,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=frames_per_batch,
        )

if __name__ == '__main__':
    main()