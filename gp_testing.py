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
from torch_pilco.model_learning.likelihoods import (
    TruncatedGaussianLikelihood,

)
from torch_pilco.policy_learning.controllers import SumOfGaussians
from torch_pilco.rewards import pendulum_cost
from torch_pilco.policy_learning.rollout import GPyTorchEnv


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

    frames_per_batch = 35
    total_frames = 6*frames_per_batch
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

        bounds = torch.vstack((states.min(dim=0)[0], states.max(dim=0)[0]))

        # We should take the bounds of the environmental outputs as inputs to force the model
        # To given outputs in this range.  Otherwise we may generate nonsensical values -- this
        # means we should do a nice randomization over the environment

        # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        #     num_tasks=states.shape[1]
        # )
        likelihood = TruncatedGaussianLikelihood(bounds=bounds)
        # model = ExactDynamicalModel(
        #     states,
        #     actions,
        #     likelihood,
        # )
        amodel = ApproximateDynamicalModel(
            states,
            actions,
            likelihood,
            num_inducing_points=50,
        )
        mll = gpytorch.mlls.VariationalELBO(amodel.likelihood, amodel, amodel.num_inducing_points)
        ApproximateFit(amodel)
        print(-mll(amodel(amodel.training_data), amodel.training_outputs))
        breakpoint()
        # Find optimal model hyperparameters
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # print(-mll(model(model.training_data), model.training_outputs))
        # ExactFit(model)
        # breakpoint()
        # print(-mll(model(model.training_data), model.training_outputs))
        #  now check how well we predict the training data
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # # The model should be called with current state + action to predict next state
        #     model_input = torch.vmap(
        #         model.data_to_gp_input,
        #         in_dims=(0,0)
        #     )(states.unsqueeze(1), actions.unsqueeze(1))
        #     with gpytorch.settings.cholesky_jitter(1e-4):
        #         new_states = states + torch.cat([model.sample(mi) for mi in model_input]).float()



if __name__ == '__main__':
    main()