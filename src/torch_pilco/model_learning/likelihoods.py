import torch
import gpytorch
from botorch.utils.probability import TruncatedMultivariateNormal
from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase
from gpytorch.module import Module
from gpytorch.constraints import GreaterThan


import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.utils.probability import TruncatedMultivariateNormal

class TruncatedMultitaskLikelihood(MultitaskGaussianLikelihood):
    def __init__(self, num_tasks, bounds_lower, bounds_upper, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)
        # Bounds should match the shape of the multitask output
        self.register_buffer("bounds_lower", bounds_lower)
        self.register_buffer("bounds_upper", bounds_upper)

    def forward(self, function_samples, *params, **kwargs):
        # function_samples is the latent f(x)
        # Add noise from the base MultitaskGaussianLikelihood
        noise = self.noise_covar.noise
        
        # Construct the truncated distribution
        # Note: TruncatedMultivariateNormal expects loc and covariance
        return TruncatedMultivariateNormal(
            loc=function_samples,
            covariance_matrix=torch.diag_embed(noise), 
            lower_bound=self.bounds_lower,
            upper_bound=self.bounds_upper
        )


class TruncatedGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
        self,
        num_tasks: int,
        noise_covar: Module,
        rank: int | None = 0,
        batch_shape: torch.Size = torch.Size(),
        bounds: torch.Tensor | None=None,
        noise_constraint: torch.Tensor=None,
        batch_shape:torch.Size=torch.Size(),
        **kwargs
    ) -> None:
        super().__init__(noise_covar=noise_covar)
        
        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda m: m._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks
        self.rank = rank
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = torch.tensor([-torch.inf, torch.inf])
        
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self) -> torch.Tensor:
        return self.raw_noise_constraint.transform(self.raw_noise)

    def forward(self, function_samples, *args, **kwargs):
        """
        Computes the conditional distribution p(y | f).
        function_samples: Latent function values (f)
        """
        # Define the mean and covariance for the truncation
        # Typically f(x) acts as the mean of the observed distribution
        mean = function_samples
        
        # Add observational noise to the diagonal
        # This creates the covariance matrix for the Truncated MVN
        num_samples = mean.size(-1)
        covar = torch.eye(num_samples).to(mean) * self.noise
        
        return TruncatedMultivariateNormal(
            loc=mean,
            covariance_matrix=covar,
            bounds=self.bounds
        )
