""" The main model class. """

__all__ = ["DynamicalModel", "IMGPR", "IMSVGPR"]
import torch
import gpytorch
import numpy as np


def data_to_gp_output(
    states: torch.Tensor,
    actions: torch.Tensor,
    control_memory: int,
    position_memory: int,
) -> torch.Tensor:
    """Transforms data into PILCO data format.
        Assumes states are passed in oldest to newest, so we
        first flip them and then take their first difference
    """
    cutoff = max(control_memory, position_memory) - 1
    return torch.flip(
        torch.diff(states, n=1, dim=0),
        dims=[0]
    ).float()[:-cutoff, :]


def data_to_gp_input(
    states: torch.Tensor,
    actions: torch.Tensor,
    control_memory: int,
    position_memory: int,
) -> torch.Tensor:
    """Transforms all training data into PILCO data format.
        Assumes states are passed in oldest to newest, so we
        first flip them and then take their first difference
    """
    reordered_states = torch.flip(states, dims=[0])
    states_diff = torch.diff(reordered_states, n=1, dim=0)
    reordered_actions = torch.flip(actions, dims=[0])

    delta_states = states_diff.unfold(
        dimension=0,
        size=position_memory,
        step=1,
    ).flatten(start_dim=1, end_dim=2)
    cat_actions = reordered_actions.unfold(
        dimension=0,
        size=control_memory + 1,
        step=1,
    ).flatten(start_dim=1, end_dim=2)
    # We can use python booleans here since position_memory and control_memory
    # are defined at initialization time
    if position_memory > control_memory:
        cutoff = position_memory-control_memory
        return torch.cat(
            (
                reordered_states[0:-position_memory, :],
                delta_states[:, :],
                cat_actions[0:-cutoff, :]
            ),
            dim=1
        ).float()
    elif control_memory < position_memory:
        cutoff = control_memory-position_memory
        return torch.cat(
            (
                reordered_states[0:-control_memory, :],
                delta_states[0:-cutoff, :],
                cat_actions[:, :]
            ),
            dim=1
        ).float()
    else:
        return torch.cat(
            (
                reordered_states[0:-control_memory, :],
                delta_states[:, :],
                cat_actions[:, :]
            ),
            dim=1
        ).float()


def data_to_policy_input(
    states: torch.Tensor,
    position_memory: int,
) -> torch.Tensor:
    """Transforms data into policy data format.
        Assumes states are passed in oldest to newest, so we
        first flip them and then take their first difference
    """
    reordered_states = torch.flip(states, dims=[0])
    states_diff = torch.diff(reordered_states, n=1, dim=0)
    return torch.hstack(
        [
            torch.ravel(reordered_states[0, :]),
            torch.ravel(states_diff[0:position_memory, :])
        ]
    ).float()


class DynamicalModel(gpytorch.models.ExactGP):
    """The base class for forward model of the system dynamics.

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
        position_memory: int = 2,
        control_memory: int = 1,
    ):
        self.position_memory = position_memory
        self.control_memory = control_memory

        io_data = self.data_to_gp_input_output(
            states, actions
        )
        self.training_data, self.training_outputs = io_data
        self.num_outputs = self.training_outputs.shape[1]
        self.input_dimension = self.training_data.shape[1]

        max_lookback = max(self.control_memory,self.position_memory)
        state_dim = states.shape[1]
        action_dim = actions.shape[1]
        num_samples = states.shape[0]

        assert self.training_data.shape == (
            num_samples-max_lookback,
            state_dim * (1+self.position_memory) +
            action_dim * (1+self.control_memory)
        )

        assert self.training_outputs.shape == (
            num_samples-max_lookback, state_dim
        )

        super(DynamicalModel, self).__init__(
            self.training_data,
            self.training_outputs,
            likelihood,
        )

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=state_dim
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=state_dim, rank=1
        )

    def data_to_gp_input_output(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms data into PILCO data format."""
        return (
            data_to_gp_input(
                states,
                actions,
                self.control_memory,
                self.position_memory,
            ),
            data_to_gp_output(
                states,
                actions,
                self.control_memory,
                self.position_memory,
            ),
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x,
            covar_x
        )


def fit(
    model,
    likelihood,
    *,
    print_loss: bool = False,
    n_training_iter: int = 100
) -> None:
    # Put in training mode
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_training_iter):
        optimizer.zero_grad()
        output = model(model.training_data)
        loss = -mll(output, model.training_outputs)
        loss.backward()
        if print_loss:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_training_iter, loss.item()))
        optimizer.step()
    # Put in evaluation mode
    model.eval()
    likelihood.eval()
