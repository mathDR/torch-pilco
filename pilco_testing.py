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
from torchrl.envs import GymEnv
from torchrl.envs.transforms import BatchSizeTransform
from torchrl.envs.utils import RandomPolicy, check_env_specs
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage

from botorch.fit import fit_gpytorch_mll

from torch_pilco.model_learning.dynamical_models import (
    ApproximateDynamicalModel,
    ApproximateFit,
    #ExactDynamicalModel,
    #ExactFit,
)
from torch_pilco.policy_learning.controllers import SumOfGaussians
from torch_pilco.rewards import pendulum_cost
from torch_pilco.policy_learning.rollout import GPyTorchEnv

from torch_pilco.model_learning.multitask_truncated_normal_likelihood import MultitaskTruncatedNormalLikelihood


def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()


def main():
    if torch.cuda.is_available():
        print("GPU is available. Using GPU backend.")
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using MPS backend.")
        device = torch.device("mps")
    else:
        print("MPS not available. Falling back to CPU.")
        device = torch.device("cpu")

    device = torch.device("cpu")

    frames_per_batch = 45
    total_frames = 7*frames_per_batch
    # make the batch version of our gym environment
    base_env = GymEnv("Pendulum-v1") 
    env = base_env.append_transform(BatchSizeTransform(reshape_fn=lambda x: x.unsqueeze(0)))
    print(check_env_specs(env))
    

    random_policy = RandomPolicy(env.action_spec)
    action_dim = env.action_space.shape[0]
    x = env.reset()
    state_dim = x['observation'].shape[1]

    num_particles = 400
    num_basis = 32

    num_pilco_training_loops = 5

    control_policy = SumOfGaussians(
        state_dim,
        action_dim,
        num_basis,
        u_max=env.action_space.high[0],
        dtype=torch.float32,
        device=device,
    ) 
    batched_policy = torch.vmap(control_policy, in_dims=0)
    policy = TensorDictModule(
        batched_policy,
        in_keys=["observation"],
        out_keys=["action"],
    )

    # Store each interaction with the environment
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

    # Generate a random trajectory from the environment
    collector = SyncDataCollector(
        env,
        policy=random_policy,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=True,
    )
    bounds = torch.tensor([
        [-1., 1.],   # Task 1: x
        [-1., 1.],   # Task 2: y
        [-8., 8.],   # Task 3: theta dot
    ])

    for _ in range(num_pilco_training_loops):
        # Put the data into the replay buffer
        for data in collector:
            # convert the tensordict from collector to a version
            # suitable for dynamical model
            replay_buffer.extend(data)

        # Now grab some data and fit the GP
        # Use the whole buffer for data
        states, actions = build_pendulum_training_data(replay_buffer.sample(len(replay_buffer)))
        states = states.reshape(-1,state_dim)
        actions = actions.reshape(-1,action_dim)

        # We should take the bounds of the environmental outputs as inputs to force the model
        # To given outputs in this range.  Otherwise we may generate nonsensical values -- this
        # means we should do a nice randomization over the environment -- to try to get close to the 
        # bounds

        #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        #    num_tasks=states.shape[1]
        #)
        likelihood = MultitaskTruncatedNormalLikelihood(num_tasks=states.shape[1], bounds=bounds)
        # model = ExactDynamicalModel(
        #     states,
        #     actions,
        #     likelihood,
        # )
        #model.float()
        model = ApproximateDynamicalModel(
            states,
            actions,
            likelihood,
            num_inducing_points=50,
        )
        # Find optimal model hyperparameters
        #ExactFit(model)
        ApproximateFit(model)

        gp_env = GPyTorchEnv(
            model,
            state_dim,
            action_dim,
            env.action_space.low,
            env.action_space.high,
            pendulum_cost,
            replay_buffer,
            device=device,
            batch_size=(num_particles,)
        )
        print(check_env_specs(gp_env))

        if num_pilco_training_loops == 0:
            N = 2_000
        else:
            N = 4_000

        pbar = tqdm.tqdm(range(N // num_particles))
        # Initalize the optimizer on the original control_policy parameters
        optim = torch.optim.Adam(control_policy.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N)
        logs = defaultdict(list)

        for _ in pbar:
            rollout = gp_env.rollout(35, policy)
            breakpoint()
            v = rollout["next", "reward"]
            w = v.mean(dim=0)
            #assert w.max() < 16.3, f"Whoops! {v.max()}, {states.min(dim=0)}, {states.max(dim=0)}, {actions.min()}, {actions.max()}" 
            traj_return = w.sum()
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
        env.reset()
        # Now sample from true environment with optimized policy
        collector = SyncDataCollector(
            env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
        )

if __name__ == '__main__':
    main()