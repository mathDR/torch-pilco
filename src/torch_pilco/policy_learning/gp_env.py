""" Convert the Fitted GPyTorch model to a TorchRL enviornment."""
import gpytorch
from tensordict import TensorDict, TensorDictBase
from typing import Union
import torch
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    UnboundedContinuous,
    BoundedContinuous,
)
from typing import Callable, List
from torch_pilco.model_learning.dynamical_models import (
    ExactDynamicalModel,
    ApproximateDynamicalModel,
)


class GPyTorchEnv(EnvBase):
    # Wraps an existing GPyTorch model as a TorchRL environment
    batch_locked = False

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
        batch_size=None,
        **kwargs,
    ) -> None:
        super(GPyTorchEnv, self).__init__(batch_size=batch_size, device=device, **kwargs)
        
        self.dtype = dtype
        
        # custom property intialization - unique to this environment
        self.gp_model = trained_model.to(device)
        self.gp_model.eval() # Set model to evaluation mode

        self.reward_func = reward_func.to(device)
        self.reward_func.eval()

        self.state_size = state_size
        assert self.state_size == self.gp_model.num_outputs, "Number of GP outputs needs to match true environment state."
        self.action_size = action_size
        self._set_specs(action_space_low, action_space_high)

    def _set_specs(
            self,
            action_space_low: float,
            action_space_high: float
    ) -> None:
        # specs
        self.action_spec = BoundedContinuous(
            low=torch.from_numpy(action_space_low),
            high=torch.from_numpy(action_space_high),
            device=self.device,
            dtype=self.dtype,
        )

        observation_spec = UnboundedContinuous(
            shape=torch.Size([self.state_size]),
            device=self.device,
            dtype=self.dtype,
        ) # unlimited observation space

        self.observation_spec = Composite(
            observation=observation_spec,
            shape=[],
        ) # has to be CompositeSpec per the docs

        self.state_spec = self.observation_spec.clone()
        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([1]),
            device=self.device,
            dtype=self.dtype,
        ) # unlimited reward space

    def gen_params(self, batch_size: List | None = None) -> TensorDictBase:
        # init new state from the training data
        if batch_size is None:
            batch_size = []
        num_outputs = len(self.gp_model.training_outputs)
        td = TensorDict(
            {
                "observation": self.gp_model.training_outputs[
                    torch.randperm(num_outputs)[0]
                ]
            }
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td
    
    def _reset(
            self,
            tensordict: TensorDict | None = None
    ):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single state based on the gaussian process training data
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant parameters to get started.
            tensordict = self.gen_params()
        
        self.state = tensordict["observation"].float()
        
        
        out_tensordict = TensorDict(
            {
                "observation": self.state,
            },
            batch_size=tensordict.shape
        )

        return out_tensordict

    def _step(
        self,
        tensordict: TensorDict
    ) -> TensorDict:
        action = tensordict["action"]

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The model should be called with current state + action to predict next state
            model_input = torch.vmap(
                self.gp_model.data_to_gp_input,
                in_dims=(0, 0),
            )(torch.atleast_2d(self.state).unsqueeze(1), torch.atleast_2d(action).unsqueeze(1)).double()
            with gpytorch.settings.cholesky_jitter(1e-4):
                state = torch.cat(
                    [self.gp_model.likelihood(self.gp_model(mi)).rsample() for mi in model_input]
                ).squeeze(1).float()
            self.state = state.reshape(self.state.shape)
        reward = torch.cat([self.reward_func(mi).rsample() for mi in model_input]).float()
            
        out_tensordict = TensorDict(
            {
                "observation": self.state,
                "reward": reward,
            },
            batch_size=tensordict.shape
        )
        return out_tensordict

    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng