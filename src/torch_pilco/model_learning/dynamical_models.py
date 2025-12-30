""" The main model class. """

__all__ = ["DynamicalModel"]
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_mll
import gpytorch
import numpy as np


class ExactDynamicalModel(gpytorch.models.ExactGP, GPyTorchModel):
    """The base class for forward model of the system dynamics (uses Cholesky Deompositions).

    Heavily borrows from the gpytorch Multitask GP Regression example:
    https://github.com/cornellius-gp/gpytorch/blob/main/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.ipynb

    Args:
        states (ArrayLike): The input states $x_t$.
        actions (ArrayLike): The input controls $u_t.$
        kernel_funcs (AbstractKernel): The kernel function(s) for each GP. If
          there are multiple outputs but a single kernel is passed, each GP
          will get this kernel.
        mean_funcs (AbstractMeanFunction): The mean function(s) for each GP.
          If there are multiple outputs, but a single mean is passed, each
          GP will get this mean. If left blank, will default to zero mean
        likelihood (AbstractLikelihood): The likelihood of the posterior. If
          there are multiple outputs and a single likelihood is passed, each
          GP will get that likelihood. If left blank, will default to a
          Gaussian Likelihood.
        models: The GP model(s) for each output dimension.
        position_memory (Int): the number of previous states that are included
          to form the GP inputs.
        control_memory (Int): the number of previous actions that are included
          to form the GP inputs.
        name (string): The name of the model.
    """

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        bounds: torch.Tensor | None = None,
    ):

        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self._num_outputs = self.training_outputs.shape[1]
        self.input_dimension = self.training_data.shape[1]

        state_dim = states.shape[1]

        super(ExactDynamicalModel, self).__init__(
            self.training_data,
            self.training_outputs,
            likelihood,
        )
        self.bounds = bounds # Should be shape (2, num_outputs)
        if bounds is not None:
            assert bounds.shape == (2, self._num_outputs)

        self.likelihood = likelihood

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=state_dim
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=state_dim, rank=1
        )

    @property
    def num_outputs(self) -> int:
        """Read-only property required by the BoTorch Model API."""
        return self._num_outputs
        
    def data_to_gp_output(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms data into PILCO data format."""
        val = torch.diff(states, n=1, dim=0)
        if val.ndim == 1:
            val = torch.atleast_2d(val).T
        return val

    def data_to_gp_input(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms data into PILCO data format."""
        val = torch.hstack((states, actions))
        if val.ndim == 1:
            val = torch.atleast_2d(val).T
        return val

    def data_to_gp_input_output(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms data into PILCO data format."""
        return (
            self.data_to_gp_input(
                states[1:],
                actions[1:],
            ),
            self.data_to_gp_output(
                states,
                actions,
            ),
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return  gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x,
            covar_x
        )

    def sample(self, x: torch.Tensor, num_samples: int=1) -> torch.Tensor:
        # Samples from the GP and rescales given the bounds
        # Apply sigmoid to samples to enforce [0, 1] then rescale to self.bounds
        samples = self(x).rsample()
        if self.bounds is not None:
            # Simple sigmoid squashing to [0, 1]
            samples = torch.sigmoid(samples) 
            # Rescale to [min, max]
            low, high = self.bounds[0], self.bounds[1]
            samples = low + (high - low) * samples
        return samples

def ExactFit(
    model: ExactDynamicalModel,
) -> None:
    # Put in training mode
    model.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Set model to evaluation mode
    model.eval()
