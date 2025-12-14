""" Rollout a Policy given a model and cost function."""

import functools
import gpytorch
import torch
from torch.func import functional_call
from typing import Callable, Dict, Tuple

from torch_pilco.policy_learning.controllers import Policy
from torch_pilco.model_learning.dynamical_models import (
    DynamicalModel,
    data_to_gp_input,
    data_to_policy_input,
) 

def policy_rollout_with_std(
    policy_params: Dict,
    policy_buffers: Dict,
    policy: Policy,
    init_states: torch.Tensor,
    init_actions: torch.Tensor,
    model: DynamicalModel,
    timesteps: torch.Tensor,
    obj_func: Callable[[torch.Tensor, torch.Tensor], float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The function that produces the rollout.
    It starts from the init_samples and utilizes the policy to
    generate actions that allow for sampling from the model to
    get the next samples.
    """
    def call_single_policy(params: Dict, buffers: Dict, states: torch.Tensor, timestep: float) -> torch.Tensor:
        # functional_call replaces the module's internal state with the provided params/buffers
        return functional_call(policy, (params, buffers), (states, timestep))
    batched_policy = torch.vmap(call_single_policy, in_dims=(None, None, 0, None))

    data_to_policy_input_closure = functools.partial(
        data_to_policy_input,
        position_memory=model.position_memory
        
    )
    data_to_gp_input_closure = functools.partial(
        data_to_gp_input,
        control_memory=model.control_memory,
        position_memory=model.position_memory
    )

    
    def update_actions(new_action: torch.Tensor, old_actions: torch.Tensor,) -> torch.Tensor:
        """Append action to the rear of actions and pop off first actions value.
        """
        return torch.cat(
            (old_actions[:, 1:, :], new_action), dim=1
        )


    def update_states(new_state: torch.Tensor, old_states: torch.Tensor,) -> torch.Tensor:
        """Append state to the rear of states and pop off first states value.
        """
        return torch.cat((old_states[:, 1:, :], new_state), dim=1)

    
    def one_rollout_step(
        carry: Tuple[Dict, Dict, float, float, torch.Tensor, torch.Tensor],
        timestep: float,
    ) -> Tuple[Dict, Dict, float, float, torch.Tensor, torch.Tensor]:
        num_particles = 400

        policy_params, policy_buffers, total_cost, total_var, running_states, running_actions = carry

        # Compute the action from the most recent state
        action_inputs = torch.vmap(data_to_policy_input_closure, in_dims=0)(running_states)
        next_actions = batched_policy(policy_params, policy_buffers, action_inputs, timestep,)
        running_actions = update_actions(next_actions, running_actions)
        model_input = torch.vmap(data_to_gp_input_closure, in_dims=(0,0))(running_states,running_actions)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posteriors = [model(model_input[i,:,:]) for i in range(num_particles)]
        next_states = torch.cat([posteriors[i].sample() for i in range(num_particles)]).unsqueeze(1)
        running_states = update_states(next_states, running_states)

        full_cost = torch.vmap(obj_func, in_dims=(0,0))(next_states, next_actions)
        cost = torch.mean(full_cost)
        var = torch.var(full_cost)

        return (policy_params, policy_buffers, total_cost+cost, total_var+var, running_states, running_actions)

    total_cost = 0
    total_var = 0

    s = init_states
    a = init_actions
    for t in timesteps:
        policy_params, policy_buffers, total_cost, total_var, s, a = one_rollout_step(
            (policy_params, policy_buffers, total_cost, total_var, s, a),
            t
        )
    return total_cost, total_var


def policy_rollout(
    policy_params: Dict,
    policy_buffers: Dict,
    policy: Policy,
    init_states: torch.Tensor,
    init_actions: torch.Tensor,
    model: DynamicalModel,
    timesteps: torch.Tensor,
    obj_func: Callable[[torch.Tensor, torch.Tensor], float],
) -> float:
    """Just return the mean of the policy rollout."""
    mu, _ = policy_rollout_with_std(
        policy_params,
        policy_buffers,
        policy,
        init_states,
        init_actions,
        model,
        timesteps,
        obj_func,
    )
    return mu

