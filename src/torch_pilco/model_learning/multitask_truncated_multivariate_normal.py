#!/usr/bin/env python3
from __future__ import annotations

from typing import Union
import torch
from torch import Tensor
from linear_operator import LinearOperator
from linear_operator.operators import (
    DiagLinearOperator,
)

from torch_pilco.model_learning.truncated_multivariate_normal import TruncatedMultivariateNormal


class MultitaskTruncatedMultivariateNormal(TruncatedMultivariateNormal):
    """
    Constructs a multi-output truncated multivariate Normal random variable, based on mean, covariance and bounds
    Can be multi-output truncated multivariate, or a batch of multi-output truncated multivariate Normal

    Passing a vector mean corresponds to a multi-output truncated multivariate Normal
    Passing a matrix mean corresponds to a batch of truncated multivariate Normals

    :param torch.Tensor loc:  An `n x t` or batch `b x n x t` matrix of means for the TMVN distribution.
    :param ~linear_operator.operators.LinearOperator covariance_matrix: An `... x NT x NT` (batch) matrix.
        covariance matrix of TMVN distribution.
    :param torch.Tensor bounds: A `t x 2` tensor of bounds.
    :param bool validate_args: (default=False) If True, validate `mean`, `covariance_matrix` and `bounds` arguments.
    :param bool interleaved: (default=True) If True, covariance matrix is interpreted as block-diagonal w.r.t.
        inter-task covariances for each observation. If False, it is interpreted as block-diagonal
        w.r.t. inter-observation covariance for each task.
    """

    def __init__(self,
        loc: Union[Tensor, LinearOperator],
        covariance_matrix: Union[Tensor, LinearOperator],
        bounds: Tensor,
        validate_args: bool = False,
        interleaved: bool = True,
    ):
        if not torch.is_tensor(loc) and not isinstance(loc, LinearOperator):
            raise RuntimeError("The mean of a MultitaskTruncatedMultivariateNormal must be a Tensor or LinearOperator")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LinearOperator):
            raise RuntimeError("The covariance of a MultitaskTruncatedMultivariateNormal must be a Tensor or LinearOperator")
        
        if not torch.is_tensor(bounds):
            raise RuntimeError("The bounds of a MultitaskTruncatedMultivariateNormal must be a Tensor")

        if loc.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        # Ensure that shapes are broadcasted appropriately across the mean and covariance
        # Means can have singleton dimensions for either the `n` or `t` dimensions
        batch_shape = torch.broadcast_shapes(loc.shape[:-2], covariance_matrix.shape[:-2])
        if loc.shape[-2:].numel() != covariance_matrix.size(-1):
            if covariance_matrix.size(-1) % loc.shape[-2:].numel():
                raise RuntimeError(
                    f"mean shape {loc.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
            elif loc.size(-2) == 1:
                loc = loc.expand(*batch_shape, covariance_matrix.size(-1) // loc.size(-1), loc.size(-1))
            elif loc.size(-1) == 1:
                loc = loc.expand(*batch_shape, loc.size(-2), covariance_matrix.size(-2) // loc.size(-2))
            else:
                raise RuntimeError(
                    f"mean shape {loc.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
        else:
            loc = loc.expand(*batch_shape, *loc.shape[-2:])

        self._output_shape = loc.shape
        # TODO: Instead of transpose / view operations, use a PermutationLinearOperator (see #539)
        # to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_tmvn = loc.reshape(*loc.shape[:-2], -1)
        else:
            mean_tmvn = loc.transpose(-1, -2).reshape(*loc.shape[:-2], -1)
        if loc.dim() == 3:
            # loc is (b, n, t) bounds needs to be (b, n*t, 2)
            b, n, _ = loc.shape
            extended_bounds = bounds.repeat(b, n, 1)
        else:
            # loc is of shape (n, t) bounds needs to be (n*t,2)
            n, _ = loc.shape
            extended_bounds = bounds.repeat(n, 1)
        breakpoint()
        super().__init__(mean=mean_tmvn, covariance_matrix=covariance_matrix, bounds=extended_bounds, validate_args=validate_args)


    @property
    def event_shape(self):
        return self._output_shape[-2:]

    def expand(self, batch_size: int) -> MultitaskTruncatedMultivariateNormal:
        new_mean = self.loc.expand(torch.Size(batch_size) + self.loc.shape[-2:])
        new_covar = self.covariance_matrix.expand(torch.Size(batch_size) + self.covariance_matrix.shape[-2:])
        new_bounds = self.bounds.expand(torch.Size(batch_size))
        res = self.__class__(loc=new_mean, covariance_matrix=new_covar, bounds=new_bounds, interleaved=self._interleaved)
        return res

    def log_prob(self, value):
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = value.shape[:-2] + value.shape[:-3:-1]
            value = value.view(new_shape).transpose(-1, -2).contiguous()
        return super().log_prob(value.reshape(*value.shape[:-2], -1))

    @property
    def mean(self):
        mean = self.loc
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return mean.view(new_shape).transpose(-1, -2).contiguous()
        return mean.view(self._output_shape)

    @property
    def num_tasks(self):
        return self._output_shape[-1]

    def rsample(self, sample_shape=torch.Size()):
        samples = super().rsample(sample_shape=sample_shape)
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = sample_shape + self._output_shape[:-2] + self._output_shape[:-3:-1]
            return samples.view(new_shape).transpose(-1, -2).contiguous()
        return samples.view(sample_shape + self._output_shape)

    @property
    def variance(self):
        var = super().variance
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return var.view(new_shape).transpose(-1, -2).contiguous()
        return var.view(self._output_shape)

    # @property
    # def confidence_region(self) -> Tuple[Tensor, Tensor]:
    #     """
    #     Returns 2 standard deviations above and below the mean.

    #     :return: Pair of tensors of size `... x N`, where N is the
    #         dimensionality of the random variable. The first (second) Tensor is the
    #         lower (upper) end of the confidence region.  We clip this against the bounds
    #         since it is not analytic to compute this accurately.
    #     """
    #     std2 = self.variance.sqrt().mul_(2)
    #     mu = self.mean
    #     return (
    #         torch.max(torch.min(mu.sub(std2), self.bounds[:mu.shape[-1],1]), self.bounds[:mu.shape[-1],0]),
    #         torch.max(torch.min(mu.add(std2), self.bounds[:mu.shape[-1],1]), self.bounds[:mu.shape[-1],0]),
    #     )

    def __getitem__(self, idx) -> TruncatedMultivariateNormal:
        """
        Constructs a new TruncatedMultivariateNormal that represents a random variable
        modified by an indexing operation.

        The mean, covariance matrix, and bounds arguments are indexed accordingly.

        :param Any idx: Index to apply to the mean. The covariance matrix is indexed accordingly.
        :returns: If indices specify a slice for samples and tasks, returns a
            MultitaskTruncatedMultivariateNormal, else returns a TruncatedMultivariateNormal.
        """

        # Normalize index to a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        if ... in idx:
            # Replace ellipsis '...' with explicit indices
            ellipsis_location = idx.index(...)
            if ... in idx[ellipsis_location + 1 :]:
                raise IndexError("Only one ellipsis '...' is supported!")
            prefix = idx[:ellipsis_location]
            suffix = idx[ellipsis_location + 1 :]
            infix_length = self.loc.dim() - len(prefix) - len(suffix)
            if infix_length < 0:
                raise IndexError(f"Index {idx} has too many dimensions")
            idx = prefix + (slice(None),) * infix_length + suffix
        elif len(idx) == self.loc.dim() - 1:
            # Normalize indices ignoring the task-index to include it
            idx = idx + (slice(None),)

        new_mean = self.loc[idx]
        new_bounds = self.bounds[idx]

        # We now create a covariance matrix appropriate for new_mean
        if len(idx) <= self.loc.dim() - 2:
            # We are only indexing the batch dimensions in this case
            return MultitaskTruncatedMultivariateNormal(
                loc=new_mean,
                covariance_matrix=self.lazy_covariance_matrix[idx],
                bounds=new_bounds,
                interleaved=self._interleaved,
            )
        elif len(idx) > self.loc.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # We have an index that extends over all dimensions
            batch_idx = idx[:-2]
            if self._interleaved:
                row_idx = idx[-2]
                col_idx = idx[-1]
                num_rows = self._output_shape[-2]
                num_cols = self._output_shape[-1]
            else:
                row_idx = idx[-1]
                col_idx = idx[-2]
                num_rows = self._output_shape[-1]
                num_cols = self._output_shape[-2]

            if isinstance(row_idx, int) and isinstance(col_idx, int):
                # Single sample with single task
                row_idx = _normalize_index(row_idx, num_rows)
                col_idx = _normalize_index(col_idx, num_cols)
                new_cov = DiagLinearOperator(
                    self.lazy_covariance_matrix.diagonal()[batch_idx + (row_idx * num_cols + col_idx,)]
                )
                return TruncatedMultivariateNormal(loc=new_mean, covariance_matrix=new_cov, bounds=new_bounds)
            elif isinstance(row_idx, int) and isinstance(col_idx, slice):
                # A block of the covariance matrix
                row_idx = _normalize_index(row_idx, num_rows)
                col_idx = _normalize_slice(col_idx, num_cols)
                new_slice = slice(
                    col_idx.start + row_idx * num_cols,
                    col_idx.stop + row_idx * num_cols,
                    col_idx.step,
                )
                new_cov = self.lazy_covariance_matrix[batch_idx + (new_slice, new_slice)]
                return TruncatedMultivariateNormal(loc=new_mean, covariance_matrix=new_cov, bounds=new_bounds)
            elif isinstance(row_idx, slice) and isinstance(col_idx, int):
                # A block of the reversely interleaved covariance matrix
                row_idx = _normalize_slice(row_idx, num_rows)
                col_idx = _normalize_index(col_idx, num_cols)
                new_slice = slice(row_idx.start + col_idx, row_idx.stop * num_cols + col_idx, row_idx.step * num_cols)
                new_cov = self.lazy_covariance_matrix[batch_idx + (new_slice, new_slice)]
                return TruncatedMultivariateNormal(loc=new_mean, covariance_matrix=new_cov, bounds=new_bounds)
            elif (
                isinstance(row_idx, slice)
                and isinstance(col_idx, slice)
                and row_idx == col_idx == slice(None, None, None)
            ):
                new_cov = self.lazy_covariance_matrix[batch_idx]
                return MultitaskTruncatedMultivariateNormal(
                    loc=new_mean,
                    covariance_matrix=new_cov,
                    bounds=new_bounds,
                    interleaved=self._interleaved,
                    validate_args=False,
                )
            elif isinstance(row_idx, slice) or isinstance(col_idx, slice):
                # slice x slice or indices x slice or slice x indices
                if isinstance(row_idx, slice):
                    row_idx = torch.arange(num_rows)[row_idx]
                if isinstance(col_idx, slice):
                    col_idx = torch.arange(num_cols)[col_idx]
                row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing="ij")
                indices = (row_grid * num_cols + col_grid).reshape(-1)
                new_cov = self.lazy_covariance_matrix[batch_idx + (indices,)][..., indices]
                return MultitaskTruncatedMultivariateNormal(
                    loc=new_mean, covariance_matrix=new_cov, bounds=new_bounds, interleaved=self._interleaved, validate_args=False
                )
            else:
                # row_idx and col_idx have pairs of indices
                indices = row_idx * num_cols + col_idx
                new_cov = self.lazy_covariance_matrix[batch_idx + (indices,)][..., indices]
                return TruncatedMultivariateNormal(
                    loc=new_mean,
                    covariance_matrix=new_cov,
                    bounds=new_bounds,
                )

    def __repr__(self) -> str:
        return f"MultitaskTruncatedMultivariateNormal(mean shape: {self._output_shape}, covar shape: {self.covariance_matrix.shape}, bounds shape: {self.bounds.shape})"


def _normalize_index(i: int, dim_size: int) -> int:
    if i < 0:
        return dim_size + i
    else:
        return i


def _normalize_slice(s: slice, dim_size: int) -> slice:
    start = s.start
    if start is None:
        start = 0
    elif start < 0:
        start = dim_size + start
    stop = s.stop
    if stop is None:
        stop = dim_size
    elif stop < 0:
        stop = dim_size + stop
    step = s.step
    if step is None:
        step = 1
    return slice(start, stop, step)
