""" Convert the Fitted GPyTorch model to a TorchRL enviornment."""
import gpytorch
from tensordict import TensorDict
from typing import Union
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
from torch_pilco.model_learning.dynamical_models import (
    ExactDynamicalModel,
    ApproximateDynamicalModel,
)


class GPyTorchEnv(EnvBase):
    # Wraps an existing GPyTorch model as a TorchRL environment
    def __init__( 
        self,
        trained_model: ExactDynamicalModel | ApproximateDynamicalModel,
        state_size: int,
        action_size: int,
        action_space_low: float,
        action_space_high: float,
        reward_func: Callable[[torch.Tensor, torch.Tensor], Union[torch.float32, torch.float64]],
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float,
        batch_size: tuple | torch.Size | None = None,
        **kwargs,
    ) -> None:
        super(GPyTorchEnv, self).__init__(batch_size=batch_size, **kwargs)
        
        self.device = device
        self.dtype = dtype
        
        # custom property intialization - unique to this environment
        self.gp_model = trained_model.to(device)
        self.gp_model.eval() # Set model to evaluation mode
        self.reward_func = reward_func # can we populate this with the env.reward function?

        self.state_size = state_size
        assert self.state_size == self.gp_model.num_outputs, "Number of GP outputs needs to match true environment state."
        self.action_size = action_size
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self._set_specs(action_space_low, action_space_high)

    def _set_specs(self, action_space_low: float, action_space_high: float) -> None:
        # specs
        self.action_spec = BoundedContinuous(
            low=torch.tile(torch.from_numpy(action_space_low), (self.batch_size[0], self.action_size)),
            high=torch.tile(torch.from_numpy(action_space_high), (self.batch_size[0], self.action_size)),
            device=self.device,
            dtype=self.dtype,
        )

        observation_spec = UnboundedContinuous(
            shape=torch.Size([self.batch_size[0], self.state_size]),
            device=self.device,
            dtype=self.dtype,
        ) # unlimited observation space

        # Observation spec should be same and batch_size per https://github.com/pytorch/rl/issues/1766
        self.observation_spec = Composite(
            observation=observation_spec,
            shape=self.batch_size,
        ) # has to be CompositeSpec per the docs

        self.state_spec = self.observation_spec.clone()

        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([self.batch_size[0], 1]),
            device=self.device,
            dtype=self.dtype,
        ) # unlimited reward space

    def gen_states(self, sample_size: int) -> None:
        # init new state from the replay buffer
        self.state = self.gp_model.training_outputs[
            torch.randperm(sample_size)
        ].reshape(self.batch_size[0], self.state_size).float()
    
    def _reset(self, tensordict: TensorDict | None = None):
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single state based on the gaussian process training data
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            self.gen_states(sample_size=self.batch_size[0])
        else:
            self.state = tensordict["observation"].reshape(self.batch_size[0], self.state_size).float()
        
        out_tensordict = TensorDict(
            {
                "observation": self.state,
                "step_count": self.step_count,
                "truncated": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
                "done": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
                "terminated": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
            },
            batch_size=self.batch_size
        )

        return out_tensordict

    def _step(
        self,
        tensordict: TensorDict
    ) -> TensorDict:
        self.step_count += 1
        action = tensordict["action"]
        action = action.reshape((self.batch_size[0], self.action_size))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The model should be called with current state + action to predict next state
            model_input = torch.vmap(
                self.gp_model.data_to_gp_input,
                in_dims=0,
            )(self.state.unsqueeze(1), action.unsqueeze(1)).double()
            with gpytorch.settings.cholesky_jitter(1e-4):
                self.state = torch.cat(
                    [self.gp_model.likelihood(self.gp_model(mi)).rsample() for mi in model_input]
                ).float()
            reward = torch.cat([self.reward_func(mi).sample() for mi in model_input]).float()

        out_tensordict = TensorDict(
            {
                "observation": self.state,
                "reward": reward,
                "step_count": self.step_count,
                "truncated": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
                "done": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
                "terminated": torch.zeros((*self.batch_size, 1), dtype=torch.bool).to(self.device),
            },
            batch_size=self.batch_size
        )
        return out_tensordict

    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng