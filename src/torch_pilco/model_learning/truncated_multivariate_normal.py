#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Tuple, Union

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import DiagLinearOperator, LinearOperator, RootLinearOperator
from torch import Tensor
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal as TruncMultivariateNormal

from torch.distributions.utils import lazy_property

from gpytorch import settings
from gpytorch.utils.warnings import NumericalWarning
from gpytorch.distributions import Distribution


class TruncatedMultivariateNormal(TruncMultivariateNormal):#, Distribution):
    """
    Constructs a truncated multivariate normal random variable, based on mean, covariance and bounds.
    Can be truncated multivariate, or a batch of truncated multivariate normals

    Passing a vector mean corresponds to a truncated multivariate normal.
    Passing a matrix mean corresponds to a batch of truncated multivariate normals.

    :param mean: `... x N` mean of tmvn distribution.
    :param covariance_matrix: `... x N X N` covariance matrix of tmvn distribution.
    :param bounds: `N x 2` tensor of bounds.
    :param validate_args: If True, validate `mean` anad `covariance_matrix` arguments. (Default: False.)

    :ivar torch.Tensor covariance_matrix: The covariance matrix, represented as a dense :class:`torch.Tensor`
    :ivar ~linear_operator.LinearOperator lazy_covariance_matrix: The covariance matrix, represented
        as a :class:`~linear_operator.LinearOperator`.
    :ivar torch.Tensor mean: The mean.
    :ivar torch.Tensor stddev: The standard deviation.
    :ivar torch.Tensor variance: The variance.
    """

    def __init__(
        self,
        mean: Union[Tensor, LinearOperator],
        covariance_matrix: Union[Tensor, LinearOperator],
        bounds: Tensor,
        validate_args: bool = False,
    ):
        self._islazy = False
        if isinstance(mean, LinearOperator):
            mean = mean.to_dense()
        if isinstance(covariance_matrix, LinearOperator):
            covariance_matrix=covariance_matrix.to_dense()
        super().__init__(loc=mean, covariance_matrix=covariance_matrix, bounds=bounds, validate_args=validate_args)
        
    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        :param sample_shape: the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return sample_shape + self._batch_shape

    @staticmethod
    def _repr_sizes(loc: Tensor, covariance_matrix: Union[Tensor, LinearOperator], bounds: Tensor) -> str:
        return f"TruncatedMultivariateNormal(loc: {loc.size()}, scale: {covariance_matrix.size()}, bounds: {bounds.size()})"

    def add_jitter(self, noise: float = 1e-4) -> TruncatedMultivariateNormal:
        r"""
        Adds a small constant diagonal to the MVN covariance matrix for numerical stability.

        :param noise: The size of the constant diagonal.
        """
        return self.__class__(
            loc=self.loc,
            covariance_matrix=self.lazy_covariance_matrix.add_jitter(noise),
            bounds=self.bounds
        )


    @lazy_property
    def covariance_matrix(self) -> Tensor:
        if self.islazy:
            return self.covariance_matrix.to_dense()
        else:
            return super().covariance_matrix

    def expand(self, batch_size: torch.Size) -> TruncatedMultivariateNormal:
        r"""
        See :py:meth:`torch.distributions.Distribution.expand
        <torch.distributions.distribution.Distribution.expand>`.
        """
        # NOTE: Pyro may call this method with list[int] instead of torch.Size.
        batch_size = torch.Size(batch_size)
        new_loc = self.loc.expand(batch_size + self.loc.shape[-1:])
        new_bounds = self.bounds.expand(batch_size + self.loc.shape[-1:])
        if self.islazy:
            new_covar = self.covariance_matrix.expand(batch_size + self.covariance_matrix.shape[-2:])
            new = self.__class__(loc=new_loc, covariance_matrix=new_covar, bounds=new_bounds)
            if self.__unbroadcasted_scale_tril is not None:
                # Reuse the scale tril if available.
                new.__unbroadcasted_scale_tril = self.__unbroadcasted_scale_tril.expand(
                    batch_size + self.__unbroadcasted_scale_tril.shape[-2:]
                )
        else:
            # Non-lazy TMVN is represented using scale_tril in PyTorch.
            # Constructing it from scale_tril will avoid unnecessary computation.
            # Initialize using  __new__, so that we can skip __init__ and use scale_tril.
            new = self.__new__(type(self))
            new._islazy = False
            new_scale_tril = self.__unbroadcasted_scale_tril.expand(
                batch_size + self.__unbroadcasted_scale_tril.shape[-2:]
            )
            super(TruncatedMultivariateNormal, new).__init__(loc=new_loc, scale_tril=new_scale_tril, bounds=new_bounds)
            # Set the covar matrix, since it is always available for GPyTorch TMVN.
            new.covariance_matrix = self.covariance_matrix.expand(batch_size + self.covariance_matrix.shape[-2:])
        return new

    def unsqueeze(self, dim: int) -> TruncatedMultivariateNormal:
        r"""
        Constructs a new TruncatedMultivariateNormal with the batch shape unsqueezed
        by the given dimension.
        For example, if `self.batch_shape = torch.Size([2, 3])` and `dim = 0`, then
        the returned TruncatedMultivariateNormal will have `batch_shape = torch.Size([1, 2, 3])`.
        If `dim = -1`, then the returned TruncatedMultivariateNormal will have
        `batch_shape = torch.Size([2, 3, 1])`.
        """
        if dim > len(self.batch_shape) or dim < -len(self.batch_shape) - 1:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-len(self.batch_shape) - 1}, {len(self.batch_shape)}], but got {dim})."
            )
        if dim < 0:
            # If dim is negative, get the positive equivalent.
            dim = len(self.batch_shape) + dim + 1

        new_loc = self.loc.unsqueeze(dim)
        new_bounds = self.bounds.unsqueeze(dim)
        if self.islazy:
            new_covar = self.covariance_matrix.unsqueeze(dim)
            new = self.__class__(loc=new_loc, covariance_matrix=new_covar, bounds=new_bounds)
            if self.__unbroadcasted_scale_tril is not None:
                # Reuse the scale tril if available.
                new.__unbroadcasted_scale_tril = self.__unbroadcasted_scale_tril.unsqueeze(dim)
        else:
            # Non-lazy TMVN is represented using scale_tril in PyTorch.
            # Constructing it from scale_tril will avoid unnecessary computation.
            # Initialize using  __new__, so that we can skip __init__ and use scale_tril.
            new = self.__new__(type(self))
            new._islazy = False
            new_scale_tril = self.__unbroadcasted_scale_tril.unsqueeze(dim)
            super(TruncatedMultivariateNormal, new).__init__(loc=new_loc, scale_tril=new_scale_tril, bounds=new_bounds)
            # Set the covar matrix, since it is always available for GPyTorch MVN.
            new.covariance_matrix = self.covariance_matrix.unsqueeze(dim)
        return new

    @lazy_property
    def lazy_covariance_matrix(self) -> LinearOperator:
        if self.islazy:
            return self.covariance_matrix
        else:
            return to_linear_operator(super().covariance_matrix)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""
        See :py:meth:`torch.distributions.Distribution.log_prob
        <torch.distributions.distribution.Distribution.log_prob>`.
        """
        if settings.fast_computations.log_prob.off():
            return super().log_prob(value)
        else:
            raise RuntimeError(
                "We cannot use fast_computations for log_prob in TruncatedMultivariateNormal. Please "
                "set `settings.fast_computations.log_prob` to `on`."
            )


    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return TruncMultivariateNormal.rsample(self, sample_shape=sample_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        r"""
        Generates a `sample_shape` shaped sample or `sample_shape`
        shaped batch of samples if the distribution parameters
        are batched.

        Note that these samples are not reparameterized and therefore cannot be backpropagated through.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :return: A `*sample_shape x *batch_shape x N` tensor of i.i.d. samples.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    @property
    def stddev(self) -> Tensor:
        # self.variance is guaranteed to be positive, because we do clamping.
        return self.variance.sqrt()

    @property
    def variance(self) -> Tensor:
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            diag = self.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)
            diag = diag.view(diag.shape[:-1] + self._event_shape)
            variance = diag.expand(self._batch_shape + self._event_shape)
        else:
            variance = super().variance

        # Check to make sure that variance isn't lower than minimum allowed value (default 1e-6).
        # This ensures that all variances are positive
        min_variance = settings.min_variance.value(variance.dtype)
        if variance.lt(min_variance).any():
            warnings.warn(
                f"Negative variance values detected. "
                "This is likely due to numerical instabilities. "
                f"Rounding negative variances up to {min_variance}.",
                NumericalWarning,
            )
            variance = variance.clamp_min(min_variance)
        return variance

    def __getitem__(self, idx) -> TruncatedMultivariateNormal:
        r"""
        Constructs a new TruncatedMultivariateNormal that represents a random variable
        modified by an indexing operation.

        The mean, covariance matrix and bounds arguments are indexed accordingly.

        :param idx: Index to apply to the mean and bounds. The covariance matrix is indexed accordingly.
        """

        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) > self.loc.dim() and Ellipsis in idx:
            idx = tuple(i for i in idx if i != Ellipsis)
            if len(idx) < self.loc.dim():
                raise IndexError("Multiple ambiguous ellipsis in index!")

        rest_idx = idx[:-1]
        last_idx = idx[-1]
        new_mean = self.loc[idx]
        new_bounds= self.bounds[idx]

        if len(idx) <= self.loc.dim() - 1 and (Ellipsis not in rest_idx):
            # We are only indexing the batch dimensions in this case
            new_cov = self.lazy_covariance_matrix[idx]
        elif len(idx) > self.loc.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # In this case we know last_idx corresponds to the last dimension
            # of mean and the last two dimensions of lazy_covariance_matrix
            if isinstance(last_idx, int):
                new_cov = DiagLinearOperator(
                    self.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)[(*rest_idx, last_idx)]
                )
            elif isinstance(last_idx, slice):
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, last_idx)]
            elif last_idx is (...):
                new_cov = self.lazy_covariance_matrix[rest_idx]
            else:
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, slice(None, None, None))][..., last_idx]
        return self.__class__(loc=new_mean, covariance_matrix=new_cov, bounds=new_bounds)
