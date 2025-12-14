""" Optimize a controller on a given cost function."""
import functools
import gpytorch
import torch
from typing import Callable, Tuple

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from tensordict import TensorDict
import torchopt

from torch_pilco.policy_learning.controllers import Policy
from torch_pilco.model_learning.dynamical_models import (
    DynamicalModel,
    data_to_gp_input,
    data_to_policy_input,
)
from torch_pilco.policy_learning.rollout import policy_rollout 


def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()


def fit_controller(  # noqa: PLR0913
    *,
    policy: Policy,
    replay_buffer: ReplayBuffer,
    queue_size: int,
    control_memory: int,
    position_memory: int,
    num_particles: int,
    gp_model: DynamicalModel,
    timesteps: torch.Tensor,
    obj_func: Callable[[torch.Tensor, torch.Tensor], float],
    optimizer: torchopt.optim.func.base.FuncOptimizer,
    max_steps: int = 100,
    patience: int = 7,
) -> Policy:
    """The optimization loop for fitting the policy parameters.

    It returns the optimized policy and early stops based on an
    estimation of validation loss.  Unlike for neural networks, we
    do not split the data, but instead restart the rollout from
    different particles.

    Each step of the optimizer reinializes states and actions from the
    replay buffer.
    """
    replay_buffer_sample = replay_buffer.sample(queue_size)
    states, actions = build_pendulum_training_data(replay_buffer_sample)
    model_input = data_to_gp_input(states, actions, control_memory, position_memory)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp_model(model_input)
    next_states = posterior.sample(sample_shape=torch.Size([num_particles]))
    # Now inflate running_states (and running_actions) for each particle
    init_states = torch.tile(states,(num_particles,1,1))
    init_actions = torch.tile(actions,(num_particles,1,1))

    params = dict(policy.named_parameters())
    buffers = dict(policy.named_buffers())

    # Optimization step.

    for optimization_iteration in range(max_steps):
        loss = policy_rollout(params,buffers,policy,init_states,init_actions,gp_model,timesteps,obj_func)
        params = optimizer.step(loss, params)


    return params