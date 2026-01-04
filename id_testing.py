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

from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch


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

    frames_per_batch = 50
    total_frames = 16*frames_per_batch
    # make the batch version of our gym environment
    base_env = GymEnv("Pendulum-v1") 
    env = base_env.append_transform(BatchSizeTransform(reshape_fn=lambda x: x.unsqueeze(0)))
    print(check_env_specs(env))
    
    random_policy = RandomPolicy(env.action_spec)
    action_dim = env.action_space.shape[0]
    x = env.reset()
    state_dim = x['observation'].shape[1]


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

    for _ in range(1):
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

        # Compute intrinsic dimension estimate:
        est_dimensions = []
        for k in range(2,10):
            est_dimensions.append(mle_id(states, k=2, averaging_of_inverses = True))
            if k==2:
                d2nn = twonn_pytorch(states, return_xy=False)
        print(est_dimensions[0],d2nn)
        breakpoint()




if __name__ == '__main__':
    main()