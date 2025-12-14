# End to end MC Pilco for pendulum

import gpytorch
import torch
import numpy as np
import gymnasium as gym

from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import RandomPolicy
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from torch.func import vmap, functional_call

from model_learning.dynamical_models import DynamicalModel, fit


def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()


def update_actions(action: torch.Tensor, actions: torch.Tensor,) -> torch.Tensor:
    """Append action to the front of actions and pop off last actions
    value.
    """
    return torch.cat(
        [action[:, None, :], actions[:, :-1, :]], dim=1
    )


def update_states(state: torch.Tensor, states: torch.Tensor,) -> torch.Tensor:
    """Append state to the front of states and pop off last
        states value.
    """
    return torch.cat(
        [
            state[:, None, :],
            states[:, :-1, :]
        ],
        dim=1
    )


device = "cuda:0" if torch.cuda.is_available() else "cpu"
frames_per_batch = 100

env = GymEnv("Pendulum-v1")
random_policy = RandomPolicy(env.action_spec)

# Generate a random trajectory from the environment
collector = SyncDataCollector(
    env,
    policy=random_policy,
    frames_per_batch=frames_per_batch,
    total_frames=frames_per_batch,
)
# Now determine how many frames are stacked for the dynamical model input:
position_memory: int = 2
control_memory: int = 1

num_particles = 400

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

for data in collector:
    # convert the tensordict from collector to a version
    # suitable for dynamical model
    replay_buffer.extend(data)
    states, actions = build_pendulum_training_data(data)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=states.shape[1]
    )
    model = DynamicalModel(
        states,
        actions,
        likelihood,
        position_memory=position_memory,
        control_memory=control_memory,
    )
    # Find optimal model hyperparameters
    fit(model, likelihood)
# Define a functional format for rollout
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())


def functional_forward(x):
    return functional_call(model, (params, buffers), x)


# Generate an initial condition for rollout
# (pick a random value from replay buffer?)
sample = replay_buffer.sample(2+max(control_memory, position_memory))

states, actions = build_pendulum_training_data(sample)
test_x = model.data_to_gp_input(states, actions)
breakpoint()
# Now need to be able to sample from this model:
# Create the posterior:
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    posterior = model(test_x)
# Now sample from it
#     # To draw a single sample from the posterior
# single_sample = posterior.sample()

# # To draw multiple samples (e.g., 10 samples)
# num_samples = 10
multiple_samples = posterior.sample(sample_shape=torch.Size([num_particles]))
breakpoint()
# Generate some random actions
multiple_actions = 4.0*torch.rand(num_particles)[..., None, None] - 2.0
xx_test = torch.cat((multiple_samples, multiple_actions), dim=2)
# Maybe use torch.vmap? https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
breakpoint()

batched_output = vmap(functional_forward, in_dims=0)(xx_test)
# Need to be able to generate an initial condition
# Should sample from the posterior num_particles times.
# Then generate some actions
# Then update the initial condition to append these actions
# Then should batch call model on those outputs with a single sample