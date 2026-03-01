# PPO-MC-PILCO training loop with Pendulum
import matplotlib.pyplot as plt
import gpytorch
import torch
from torch import nn

import tqdm
from collections import defaultdict

from torchrl.collectors import Collector
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    GymEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.envs.utils import RandomPolicy, check_env_specs, ExplorationType, set_exploration_type
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives.reinforce import ReinforceLoss

from tqdm import tqdm

from torch_pilco.model_learning.dynamical_models import (
    ExactDynamicalModel,
    ExactFit,
    RewardModel,
)
from torch_pilco.policy_learning.gp_env import GPyTorchEnv

def build_pendulum_training_data(
    data_tensordict: TensorDict,
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        data_tensordict["observation"].float(),
        data_tensordict["action"].float(),
        data_tensordict["next"]["reward"].float(),
    )


def main():
    # if torch.cuda.is_available():
    #     print("GPU is available. Using GPU backend.")
    #     device = torch.device("cuda:0")
    # else:
    #     print("GPU not available. Falling back to CPU.")
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    num_cells = 64  # number of cells in each layer i.e. output dim.
    lr = 2e-3
    max_grad_norm = 1.0

    num_epochs = 10

    # Environmental Data Collection
    horizon_steps = 45
    total_frames = 7*horizon_steps

    # Surrogate Data Collection
    num_particles = 7

    #base_env = GymEnv("InvertedDoublePendulum-v5", healthy_reward = 0, device=device)
    #base_env  GymEnv("InvertedPendulum-v5", device=device)
    base_env = GymEnv("Pendulum-v1", device=device)
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    print(check_env_specs(env))

    random_policy = RandomPolicy(env.action_spec)
    action_dim = env.action_space.shape[0]
    x = env.reset()
    state_dim = x['observation'].shape[0]

    num_pilco_training_loops = 1

    # Store each interaction with the environment
    environment_replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(10000),
        sampler=SamplerWithoutReplacement(),
    )

    # Generate a random trajectory from the environment
    true_env_collector = Collector(
        env,
        policy=random_policy,
        total_frames=total_frames,
        frames_per_batch=horizon_steps,
        reset_at_each_iter=True,
    )

    for pilco_iteration in range(num_pilco_training_loops):
        # Put the data into the replay buffer
        for data in true_env_collector:
            # convert the tensordict from true_env_collector to a version
            # suitable for dynamical model
            environment_replay_buffer.extend(data)

        # Now grab some data and fit the GP
        # Use the whole buffer for data
        all_data = environment_replay_buffer.sample(len(environment_replay_buffer))

        states, actions, rewards = build_pendulum_training_data(all_data)
        states = states.reshape(-1, state_dim).to(device)
        actions = actions.reshape(-1, action_dim).to(device)

        surrogate_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
           num_tasks=states.shape[1]
        ).double().to(device)
        reward_likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)

        surrogate_model = ExactDynamicalModel(
            states,
            actions,
            surrogate_likelihood,
        )
        surrogate_model.load_state_dict(torch.load(f'gp_model_weights_{pilco_iteration}.pth', weights_only=True))
        surrogate_model.double().to(device)
        surrogate_model.eval()
        # surrogate_model.double().to(device)
        # print("Fitting GP Surrogate model.")
        # ExactFit(surrogate_model)
        # print("GP Surrogate model fit.")
        # torch.save(surrogate_model.state_dict(), f'gp_model_weights_{pilco_iteration}.pth')

        reward_model = RewardModel(states, actions, rewards.to(device), reward_likelihood)
        reward_model.double().to(device)
        ExactFit(reward_model)

        base_gp_env = GPyTorchEnv(
            surrogate_model,
            state_dim,
            action_dim,
            env.action_space.low,
            env.action_space.high,
            reward_model,
            auto_reset=True,
        )
        gp_env = TransformedEnv(
            base_gp_env,
            Compose(
                StepCounter(max_steps=horizon_steps),
            )
        ).to(device)
        gp_env.compile()
        print(check_env_specs(gp_env))
        breakpoint()

        # Policy
        actor_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(gp_env.action_spec.shape[-1], device=device),
        )
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["action"]
        )
        policy_module.compile(mode="default")

        act = policy_module(gp_env.reset())
        breakpoint()

        optim = torch.optim.Adam(policy_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // horizon_steps, 0.0
        )

        # pbar = tqdm(total=total_frames)
        # eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for _ in range(num_epochs):
            init_td = gp_env.reset(gp_env.gen_params(batch_size=[num_particles]))
            rollout = gp_env.rollout(horizon_steps, policy_module, tensordict=init_td, auto_reset=False)
            loss = torch.neg(rollout["next", "reward"].mean(dim=0).sum())
            print(loss.item())

            # Optimization: backward, grad clipping and optimization step
            loss.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            #torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()

        # Now sample from true environment with optimized policy
        true_env_collector = Collector(
            env,
            policy=policy_module,
            frames_per_batch=horizon_steps,
            total_frames=total_frames,
            reset_at_each_iter=True,
        )
    print("Finished Training")
    breakpoint()

if __name__ == '__main__':
    main()