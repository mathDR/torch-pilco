# MC-PILCO training loop with Pendulum
import matplotlib.pyplot as plt
import gpytorch
import torch
from torch import nn
import numpy as np
import gymnasium as gym
import tqdm
from collections import defaultdict

from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    GymEnv,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.transforms import BatchSizeTransform
from torchrl.envs.utils import RandomPolicy, check_env_specs, ExplorationType, set_exploration_type
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from botorch.fit import fit_gpytorch_mll

from torch_pilco.model_learning.dynamical_models import (
    ApproximateDynamicalModel,
    ApproximateFit,
    ExactDynamicalModel,
    ExactFit,
)
from torch_pilco.rewards import pendulum_cost
from torch_pilco.policy_learning.rollout import GPyTorchEnv

def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return data_tensordict['observation'].float(), data_tensordict['action'].float()


def main():
    device = torch.device("cpu")

    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0

    # Environmental Data Collection
    frames_per_batch = 45
    total_frames = 7*frames_per_batch

    # Surrogate Data Collection
    surrogate_frames_per_batch = 1000
    # For a complete training, bring the number of frames up to 1M
    surrogate_total_frames = 10_000

    ##PPO Parameters
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    base_env = GymEnv("InvertedDoublePendulum-v4", device=device)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    print("normalization constant shape:", env.transform[0].loc.shape)
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)
    print(check_env_specs(env))

    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)
    
    random_policy = RandomPolicy(env.action_spec)
    action_dim = env.action_space.shape[0]
    x = env.reset()
    state_dim = x['observation'].shape[0]

    num_pilco_training_loops = 5

    # Store each interaction with the environment
    environment_replay_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

    # Generate a random trajectory from the environment
    true_env_collector = SyncDataCollector(
        env,
        policy=random_policy,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=True,
    )

    # Policy
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    policy_tensordict_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    for _ in range(num_pilco_training_loops):
        # Put the data into the replay buffer
        for data in true_env_collector:
            # convert the tensordict from true_env_collector to a version
            # suitable for dynamical model
            environment_replay_buffer.extend(data)

        # Now grab some data and fit the GP
        # Use the whole buffer for data
        states, actions = build_pendulum_training_data(environment_replay_buffer.sample(len(environment_replay_buffer)))
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
        model.double().to(device)
        print("Fitting GP Surrogate model.")
        ExactFit(model)
        # Should save this model and load it for testing...
        print("GP Surrogate model fit.")

        gp_env = GPyTorchEnv(
            model,
            state_dim,
            action_dim,
            env.action_space.low,
            env.action_space.high,
            pendulum_cost,
            environment_replay_buffer,
            device=device,
            batch_size=(surrogate_frames_per_batch,)
        )
        print(check_env_specs(gp_env))

        policy_module = ProbabilisticActor(
            module=policy_tensordict_module,
            spec=gp_env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": gp_env.action_spec_unbatched.space.low,
                "high": gp_env.action_spec_unbatched.space.high,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )

        surrogate_collector = SyncDataCollector(
            gp_env,
            policy_module,
            frames_per_batch=surrogate_frames_per_batch,
            total_frames=surrogate_total_frames,
            split_trajs=False,
            device=device,
        )

        surrogate_replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=surrogate_frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )

        loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coeff=entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coeff=1.0,
            loss_critic_type="smooth_l1",
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )

        if num_pilco_training_loops == 0:
            N = 2_000
        else:
            N = 4_000

        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(surrogate_collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                surrogate_replay_buffer.extend(data_view.cpu())
                for _ in range(surrogate_frames_per_batch // sub_batch_size):
                    subdata = surrogate_replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = gp_env.rollout(1000, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()
        
        # Now sample from true environment with optimized policy
        true_env_collector = SyncDataCollector(
            env,
            policy=policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
        )
        breakpoint()

if __name__ == '__main__':
    main()