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
    ExactDynamicalModel,
    ExactFit,
)
from torch_pilco.policy_learning.controllers import SumOfGaussians
from torch_pilco.rewards import pendulum_cost
from torch_pilco.policy_learning.rollout import GPyTorchEnv

from torch_pilco.model_learning.multitask_truncated_gaussian_likelihood import MultitaskTruncatedGaussianLikelihood


def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()


def main():

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
    num_basis = 64

    num_pilco_training_loops = 5

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
    breakpoint()


    for _ in range(num_pilco_training_loops):
        # Put the data into the replay buffer
        for data in collector:
            # convert the tensordict from collector to a version
            # suitable for dynamical model
            replay_buffer.extend(data)

        # We should take the bounds of the environmental outputs as inputs to force the model
        # To given outputs in this range.  Otherwise we may generate nonsensical values -- this
        # means we should do a nice randomization over the environment -- to try to get close to the 
        # bounds


        if num_pilco_training_loops == 0:
            # Now grab some data and fit the GP
            # Use the whole buffer for data
            states, actions = build_pendulum_training_data(replay_buffer.sample(len(replay_buffer)))
            states = states.reshape(-1,state_dim).to(device)
            actions = actions.reshape(-1,action_dim).to(device)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
               num_tasks=states.shape[1]
            ).double().to(device)
            model = ExactDynamicalModel(
                states,
                actions,
                likelihood,
            )
        else:
            breakpoint()
            # Predict on the new values and only keep the ones we cannot predict...
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
               num_tasks=states.shape[1]
            ).double().to(device)
            model = ExactDynamicalModel(
                states,
                actions,
                likelihood,
            )
        model.double().to(device)
        ExactFit(model)

        env.reset()
        # Now sample from true environment with optimized policy
        collector = SyncDataCollector(
            env,
            policy=random_policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
        )

if __name__ == '__main__':
    main()