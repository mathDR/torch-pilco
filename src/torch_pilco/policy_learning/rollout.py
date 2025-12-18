""" Convert the Fitted GPyTorch model to a TorchRL enviornment."""
import gpytorch
import numpy as np
from tensordict import TensorDict
import torch
import torchrl
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    UnboundedContinuous,
    BoundedContinuous,
    ReplayBuffer,
)
from typing import Callable
from torch_pilco.model_learning.dynamical_models import DynamicalModel



class GPyTorchEnv(EnvBase):
    # Wraps an existing GPyTorch model as a TorchRL environment
    def __init__( 
        self,
        trained_model: DynamicalModel,
        env: torchrl.envs.libs.gym.GymEnv,
        reward_func: Callable[[torch.Tensor, torch.Tensor], float],
        replay_buffer: ReplayBuffer,
        device="cpu",
        batch_size: tuple | torch.Size | None = None,
    ) -> None:
        super(GPyTorchEnv, self).__init__(batch_size=batch_size)
        
        # custom property intialization - unique to this environment
        self.dtype = np.float32
        self.gp_model = trained_model.to(device)
        self.gp_model.eval() # Set model to evaluation mode
        self.reward_func = reward_func
        self.replay_buffer = replay_buffer

        self.device = device

        self.state_size = env.observation_space.shape[0]
        assert self.state_size == self.gp_model.num_outputs, "Number of GP outputs needs to match true environment state."
        self.action_size = env.action_space.shape[0]
        
        # specs
        self.action_spec = BoundedContinuous(
            low=torch.tile(torch.from_numpy(env.action_space.low),(self.batch_size[0],self.action_size)),
            high=torch.tile(torch.from_numpy(env.action_space.high),(self.batch_size[0],self.action_size)),
            device=self.device,
            dtype=torch.float32,
        )

        observation_spec = UnboundedContinuous(
            shape=torch.Size([self.batch_size[0], self.state_size])
        ) # unlimited observation space
        # Observation spec should be same and batch_size per https://github.com/pytorch/rl/issues/1766
        self.observation_spec = Composite(
            observation=observation_spec,
            shape=self.batch_size,
        ) # has to be CompositeSpec per the docs

        self.state_spec = self.observation_spec.clone()

        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([self.batch_size[0], 1])
        ) # unlimited reward space(even though we could limit it to (-17, 0] for the pendulum)

    def gen_states(self, batch_size: int) -> None:
        # init new state from the replay buffer
        replay_buffer_sample = self.replay_buffer.sample(batch_size)
        self.state = replay_buffer_sample["observation"].float().reshape(self.batch_size[0],self.state_size)
    
    def _reset(self, tensordict: TensorDict | None = None):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single state based on the replay_buffer
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            self.gen_states(batch_size=self.batch_size[0])
        else:
            self.state = tensordict["observation"].float().reshape(self.batch_size[0],self.state_size)
        
        out_tensordict = TensorDict({}, batch_size=self.batch_size)
        out_tensordict.set("observation", self.state)

        return out_tensordict

    def _step(
        self,
        tensordict: TensorDict
    ) -> TensorDict:
        action = tensordict["action"]
        action = action.reshape((self.batch_size[0], self.action_size))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The model should be called with current state + action to predict next state
            model_input = torch.vmap(
                self.gp_model.data_to_gp_input,
                in_dims=(0,0)
            )(self.state.unsqueeze(1), action.unsqueeze(1))
            self.state = torch.cat([self.gp_model(mi).sample() for mi in model_input])

        reward = torch.vmap(self.reward_func, in_dims=(0,0))(self.state, action).float()

        out_tensordict = TensorDict(
            {
                "observation": self.state,
                "reward": reward,
                #"done": False
            },
            batch_size=self.batch_size
        )
        return out_tensordict

    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng