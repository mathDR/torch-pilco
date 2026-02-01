#!/usr/bin/env python3

from typing import Any, Union

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

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.priors import Prior
from gpytorch.lazy import LazyEvaluatedKernelTensor

from torch_pilco.model_learning.multitask_truncated_multivariate_normal import MultitaskTruncatedMultivariateNormal


class MultitaskTruncatedGaussianLikelihood(Likelihood):
    r"""
    Multitask Truncated Gaussian Likellihood that allows for apriori Box Constraints on the latent outputs.
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    .. note::
        At least one of :attr:`has_global_noise` or :attr:`has_task_noise` should be specified.

    :param num_tasks: Number of tasks.
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
        task_prior: Prior | None = None,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        self.bounds=bounds
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
    def noise(self) -> Tensor | None:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Union[float, Tensor]) -> None:
        self._set_noise(value)

    @property
    def task_noises(self) -> Tensor | None:
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
        self, shape: torch.Size, add_noise: bool | None = True, interleaved: bool = True, *params: Any, **kwargs: Any
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
    

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> MultitaskTruncatedMultivariateNormal:
        covar_kron_lt = self._shaped_noise_covar(
            function_samples.shape, add_noise=self.has_global_noise,
        )
        return MultitaskTruncatedMultivariateNormal(loc=function_samples, covariance_matrix=covar_kron_lt, bounds=self.bounds)
