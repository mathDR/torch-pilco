import gpytorch
import torch
import numpy as np
import gymnasium as gym

from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
import torchopt
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import RandomPolicy
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage

from torch_pilco.model_learning.dynamical_models import (
    DynamicalModel,
    fit,
)
from torch_pilco.policy_learning.controllers import SumOfGaussians
from torch_pilco.policy_learning.rollout import policy_rollout
from torch_pilco.policy_learning.fit_controller import fit_controller
from torch_pilco.rewards import pendulum_cost

def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
frames_per_batch = 100

env = GymEnv("Pendulum-v1")
random_policy = RandomPolicy(env.action_spec)
action_dim = env.action_space.shape[0]
x = env.reset()
state_dim = x['observation'].shape[0]

position_memory: int = 2
control_memory: int = 1
num_particles = 400
num_basis = 100

control_policy = SumOfGaussians(
    state_dim * (1 + position_memory),
    action_dim,
    num_basis,
    u_max=env.action_space.high[0],
)

# Generate a random trajectory from the environment
collector = SyncDataCollector(
    env,
    policy=random_policy,
    frames_per_batch=frames_per_batch,
    total_frames=frames_per_batch,
)
# Now determine how many frames are stacked for the dynamical model input:

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

# %load src/torch_pilco/pendulum.py
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
    fit(model, likelihood, print_loss = False)

# Rollout
#replay_buffer_sample = replay_buffer.sample(queue_size)
#initial_states, initial_actions = build_pendulum_training_data(replay_buffer_sample)
#model_input = data_to_gp_input(initial_states, initial_actions, control_memory, position_memory)
#with torch.no_grad(), gpytorch.settings.fast_pred_var():
#    posterior = model(model_input)
#next_states = posterior.sample(sample_shape=torch.Size([num_particles]))
# # Now inflate running_states (and running_actions) for each particle
#running_states = torch.tile(initial_states,(num_particles,1,1))
#running_actions = torch.tile(initial_actions,(num_particles,1,1))

# params = dict(control_policy.named_parameters())
# buffers = dict(control_policy.named_buffers())
breakpoint()
#print(policy_rollout(
#    policy_params=params,
#    policy_buffers=buffers,
#    policy=control_policy,
#    init_states=running_states,
#    init_actions=running_actions,
#    model=model,
#    timesteps=torch.arange(10),
#    obj_func=pendulum_cost
#))
controller = fit_controller(
    policy=control_policy,
    replay_buffer=replay_buffer,
    num_particles=num_particles,
    gp_model=model,
    timesteps=torch.arange(10),
    obj_func=pendulum_cost,
    max_steps=10000,
)
