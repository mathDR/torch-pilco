#!/usr/bin/env python3

from typing import Any, Optional, Union

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
)
from torch import Tensor
from torch.distributions import Normal

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import base_distributions, MultitaskMultivariateNormal
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood
from gpytorch.priors import Prior
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, Noise
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import Distribution

from torch_pilco.model_learning.multitask_truncated_multivariate_normal import MultitaskTruncatedMultivariateNormal



class MultitaskTruncatedGaussianLikelihood(Likelihood):
    r"""
    A convenient extension of the :class:`~gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.

    .. note::
        At least one of :attr:`has_global_noise` or :attr:`has_task_noise` should be specified.

    :param num_tasks: Number of tasks.
    :param bounds: the bounds for each task
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    :param has_global_noise: Whether to include a :math:`\sigma^2 \mathbf I_{nt}` term in the noise model.
    :param has_task_noise: Whether to include task-specific noise terms, which add
        :math:`\mathbf I_n \otimes \mathbf D_T` into the noise model.

    :ivar torch.Tensor task_noise_covar: The inter-task noise covariance matrix
    :ivar torch.Tensor task_noises: (Optional) task specific noise variances (added onto the `task_noise_covar`)
    :ivar torch.Tensor noise: (Optional) global noise variance (added onto the `task_noise_covar`)
    """

    def __init__(
        self,
        num_tasks: int,
        bounds: Tensor,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        self.bounds = bounds
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def noise(self) -> Optional[Tensor]:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Union[float, Tensor]) -> None:
        self._set_noise(value)

    @property
    def task_noises(self) -> Optional[Tensor]:
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    @task_noises.setter
    def task_noises(self, value: Union[float, Tensor]) -> None:
        if self.rank == 0:
            self._set_task_noises(value)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    def _set_noise(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

    @property
    def task_noise_covar(self) -> Tensor:
        if self.rank > 0:
            return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2))
        else:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value: Tensor) -> None:
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if self.rank > 0:
            with torch.no_grad():
                self.task_noise_covar_factor.data = to_linear_operator(value).pivoted_cholesky(rank=self.rank)
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self) -> Tensor:
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def _shaped_noise_covar(
        self, shape: torch.Size, add_noise: Optional[bool] = True, interleaved: bool = True, *params: Any, **kwargs: Any
    ) -> LinearOperator:
        if not self.has_task_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=shape[-2] * self.num_tasks)
            return noise

        if self.rank == 0:
            task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
            task_var_lt = DiagLinearOperator(task_noises)
            dtype, device = task_noises.dtype, task_noises.device
            ckl_init = KroneckerProductDiagLinearOperator
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt = RootLinearOperator(task_noise_covar_factor)
            dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device
            ckl_init = KroneckerProductLinearOperator

        eye_lt = ConstantDiagLinearOperator(
            torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2]
        )
        task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)  # pyre-ignore[6]

        # to add the latent noise we exploit the fact that
        # I \kron D_T + \sigma^2 I_{NT} = I \kron (D_T + \sigma^2 I)
        # which allows us to move the latent noise inside the task dependent noise
        # thereby allowing exploitation of Kronecker structure in this likelihood.
        if add_noise and self.has_global_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=task_var_lt.shape[-1])
            task_var_lt = task_var_lt + noise

        if interleaved:
            covar_kron_lt = ckl_init(eye_lt, task_var_lt)
        else:
            covar_kron_lt = ckl_init(task_var_lt, eye_lt)
        
        return covar_kron_lt
        
    def expected_log_prob(
        self, 
        observations: Tensor, 
        function_dist: MultitaskMultivariateNormal, 
        *args: Any, 
        **kwargs: Any
    ) -> Tensor:
        """
        Computes expected log probability under the truncated distribution.
        This is used during training with variational inference.
        """
        # IMPORTANT: Use marginal(), NOT forward()
        # forward() calls the parent class which expects .shape attribute
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        covar_kron_lt = self._shaped_noise_covar(
            mean.shape, add_noise=self.has_global_noise, interleaved=function_dist._interleaved
        )
        covar = covar + covar_kron_lt

        truncated_dist = MultitaskTruncatedMultivariateNormal(
            mean,
            covar,
            bounds=self.bounds,
            interleaved=function_dist._interleaved
        )
        
        # Compute log probability of observations
        log_prob = truncated_dist.log_prob(observations)
        
        return log_prob

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> MultitaskTruncatedMultivariateNormal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        noise = noise.reshape(*noise.shape[:-1], *function_samples.shape[-2:])
        return MultitaskTruncatedMultivariateNormal(
            function_samples,
            noise.sqrt(),
            bounds=self.bounds,
        )

# Example usage
if __name__ == "__main__":
    num_tasks = 5
    num_data = 10
    
    # Define bounds: each task has [lower, upper] bounds
    bounds = torch.tensor([
        [0.0, 1.0],   # Task 0: [0, 1]
        [-1.0, 1.0],  # Task 1: [-1, 1]
        [0.0, 5.0],   # Task 2: [0, 5]
        [-2.0, 2.0],  # Task 3: [-2, 2]
        [0.0, 10.0],  # Task 4: [0, 10]
    ])
    
    # Create likelihood
    likelihood = MultitaskTruncatedGaussianLikelihood(num_tasks=num_tasks, bounds=bounds)
    breakpoint()
    
    # Simulate GP output - create MultitaskMultivariateNormal directly
    # This is what you'd typically get from a MultitaskGP model
    mean = torch.randn(num_data, num_tasks)  # [10, 5]
    
    # Create a simple block-diagonal covariance (each data point has independent task covariance)
    task_covar = torch.randn(num_tasks, num_tasks)
    task_covar = task_covar @ task_covar.T + 0.1 * torch.eye(num_tasks)
    
    # Repeat for each data point to create block diagonal structure
    full_covar = torch.zeros(num_data * num_tasks, num_data * num_tasks)
    for i in range(num_data):
        full_covar[i*num_tasks:(i+1)*num_tasks, i*num_tasks:(i+1)*num_tasks] = task_covar
    
    # Create the MultitaskMultivariateNormal
    function_dist = MultitaskMultivariateNormal(
        mean=mean,
        covariance_matrix=full_covar,
        interleaved=True
    )
    
    # Get truncated distribution
    truncated_dist = likelihood(function_dist)
    
    # Sample from truncated distribution
    samples = truncated_dist.rsample()
    print(f"Samples shape: {samples.shape}")
    print(f"Sample values:\n{samples[:3]}")
    print(f"\nBounds check - all samples within bounds: {((samples >= bounds[:, 0]) & (samples <= bounds[:, 1])).all()}")