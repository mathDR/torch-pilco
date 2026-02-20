""" The main model class. """

__all__ = ["DynamicalModel", "ExactDynamicalModel"]
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_mll
import gpytorch


class DynamicalModel(torch.nn.Module):
    """ Generic methods all dynamical models need."""

    _num_outputs: int

    @property
    def num_outputs(self) -> int:
        """Read-only property required by the BoTorch Model API."""
        return self._num_outputs

    def data_to_gp_output(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms data into PILCO data format."""
        val = states
        if val.ndim == 1:
            val = torch.atleast_2d(val)
        return val[1:,:]

    def data_to_gp_input(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms data into PILCO data format."""
        val = torch.hstack((states, actions))
        if val.ndim == 1:
            val = torch.atleast_2d(val)
        return val

    def data_to_gp_input_output(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms data into PILCO data format."""
        return (
            self.data_to_gp_input(
                states,
                actions,
            )[:-1, :],
            self.data_to_gp_output(
                states,
            ),
        )


class RewardModel(DynamicalModel, gpytorch.models.ExactGP, GPyTorchModel):
    training_data: torch.Tensor
    training_outputs: torch.Tensor
    state_dim: int
    input_dimension: int
    likelihood: gpytorch.likelihoods.GaussianLikelihood

    mean_module: gpytorch.means.Mean
    covar_module: gpytorch.kernels.Kernel

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float,
    ):

        self.training_data = self.data_to_gp_input(
            states, actions
        )
        self.training_outputs = rewards.squeeze(1)

        self.input_dimension = self.training_data.shape[1]
        self.state_dim = states.shape[1]

        self._num_outputs = 1
        super().__init__(self.training_data, self.training_outputs, likelihood)

        self.likelihood = likelihood

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
        #     num_mixtures=4,
        #     ard_num_dims=self.input_dimension
        # )
        # self.covar_module.initialize_from_data(self.training_data, self.training_outputs)
        self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.input_dimension)


    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def sample(self, x: torch.Tensor, num_samples: int=1) -> torch.Tensor:
        # Samples from the GP
        samples = self(x).rsample(torch.Size([num_samples]))
        return samples


class ExactDynamicalModel(DynamicalModel, gpytorch.models.ExactGP, GPyTorchModel):
    """Forward model of the system dynamics with LCM kernel (uses Cholesky Deompositions).

    Heavily borrows from the gpytorch Multitask GP Regression example:
    https://github.com/cornellius-gp/gpytorch/blob/main/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.ipynb

    Args:
        states (torch.Tensor): The input states $x_t$.
        actions (torch.Tensor): The input controls $u_t.$
        likelihood (AbstractLikelihood): The likelihood of the posterior. If
          there are multiple outputs and a single likelihood is passed, each
          GP will get that likelihood. If left blank, will default to a
          Gaussian Likelihood.

        TODO: allow for passing in a list of kernels and/or means
    """
    training_data: torch.Tensor
    training_outputs: torch.Tensor
    state_dim: int
    input_dimension: int
    bounds: torch.Tensor
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood

    mean_module: gpytorch.means.MultitaskMean
    covar_module: gpytorch.kernels.MultitaskKernel

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float,
    ):

        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self.input_dimension = self.training_data.shape[1]
        self.state_dim = states.shape[1]

        self._num_outputs = self.training_outputs.shape[1]
        super().__init__(self.training_data, self.training_outputs, likelihood)

        self.likelihood = likelihood

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.state_dim
        )
        # Put a prior on the covariance structure.  The LKJ with eta = 1 assumes (prior)  noncorrelated outputs
        # sd_prior_instance = gpytorch.priors.HalfNormalPrior(0.5)
        # task_covar_prior = gpytorch.priors.LKJCovariancePrior(
        #     n=self.state_dim, 
        #     eta=1.0,
        #     sd_prior=sd_prior_instance,
        # )
        # base_kernels = []
        # for i in range(self._num_outputs):
        #     k = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3, ard_num_dims=self.input_dimension)
        #     k.initialize_from_data(self.training_data, self.training_outputs[i])
        #     base_kernels.append(k)
        self.covar_module = gpytorch.kernels.LCMKernel(
            base_kernels=[gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dimension) for i in range(self._num_outputs)],
            num_tasks=self.state_dim,
            rank=1,
            #task_covar_prior=task_covar_prior
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return  gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def sample(self, x: torch.Tensor, num_samples: int=1) -> torch.Tensor:
        # Samples from the GP
        samples = self(x).rsample(torch.Size([num_samples]))
        return samples


class ApproximateDynamicalModel(DynamicalModel, gpytorch.models.ApproximateGP, GPyTorchModel):
    """Forward model of the system dynamics with LCM kernel.

    Heavily borrows from the gpytorch Multitask GP Regression example:
    https://github.com/cornellius-gp/gpytorch/blob/main/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.ipynb

    Args:
        states (torch.Tensor): The input states $x_t$.
        actions (torch.Tensor): The input controls $u_t.$
        likelihood (AbstractLikelihood): The likelihood of the posterior. If
          there are multiple outputs and a single likelihood is passed, each
          GP will get that likelihood. If left blank, will default to a
          Gaussian Likelihood.

        TODO: allow for passing in a list of kernels and/or means
    """
    mean_module: gpytorch.means.MultitaskMean
    covar_module: gpytorch.kernels.MultitaskKernel
    num_inducing_points: int

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        *,
        num_inducing_points: int,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float,
    ):

        num_latents = states.shape[1]

        inducing_points = torch.rand(num_latents, num_inducing_points, states.shape[1]+actions.shape[1])

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_latents,
            num_latents=num_latents,
            latent_dim=-1,
        )

        gpytorch.models.ApproximateGP.__init__(self, variational_strategy)

        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self.input_dimension = self.training_data.shape[1]
        self.state_dim = states.shape[1]

        self._num_outputs = self.training_outputs.shape[1]

        self.num_inducing_points = num_inducing_points

        self.likelihood = likelihood

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
            task_covar_prior=task_covar_prior
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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


def ApproximateFit(
    model: ApproximateDynamicalModel,
) -> None:
    # Put in training mode
    # model.train()

    # # "Loss" for GPs - the marginal log likelihood
    # mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, model.num_inducing_points)
    # fit_gpytorch_mll(mll)

    num_epochs = 250

    model.train()
    #variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=model.training_outputs.size(0), lr=0.1)
    #hyperparameter_optimizer = torch.optim.Adam(model.hyperparameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=model.training_outputs.size(0))

    # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
    # effective for VI.
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        #variational_ngd_optimizer.zero_grad()
        #hyperparameter_optimizer.zero_grad()
        optimizer.zero_grad()
        output = model(model.training_data)
        loss = -mll(output, model.training_outputs)
        loss.backward()
        #variational_ngd_optimizer.step()
        #hyperparameter_optimizer.step()
        optimizer.step()

    # Set model to evaluation mode
    model.eval()
