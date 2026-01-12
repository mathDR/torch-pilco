import torch
from torch import Tensor
import gpytorch
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal
from gpytorch.distributions import Distribution


class MultitaskTruncatedMultivariateNormal(TruncatedMultivariateNormal, Distribution):
    def __init__(self, mean, covariance_matrix, bounds, validate_args=False, interleaved=True):
        """
        Constructs a multi-output truncated multivariate Normal random variable, based on mean, covariance
        and bounds.
        Can be multi-output truncated multivariate, or a batch of multi-output truncated multivariate Normal

        Passing a matrix mean corresponds to a multi-output multivariate Normal
        Passing a matrix mean corresponds to a batch of multivariate Normals

        Params:
            mean (:obj:`torch.tensor`): An `n x t` or batch `b x n x t` matrix of means for the TMVN distribution.
            covar (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyTensor`): An `nt x nt` or batch `b x nt x nt`
                covariance matrix of TMVN distribution.
            bounds (:obj:`torch.tensor`): A `t x 2` or batch `b x t x 2` tensor for the bounds for each task.
            validate_args (:obj:`bool`): If True, validate `mean` and `covariance_matrix` arguments.
            interleaved (:obj:`bool`): If True, covariance matrix is interpreted as block-diagonal w.r.t.
                inter-task covariances for each observation. If False, it is interpreted as block-diagonal
                w.r.t. inter-observation covariance for each task.
        """
        if not torch.is_tensor(mean) and not isinstance(mean, gpytorch.lazy.LazyTensor):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, gpytorch.lazy.LazyTensor):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if mean.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        if bounds.shape[-1] != 2:
            raise RuntimeError("bounds should be a matrix having shape t x 2")

        self._output_shape = mean.shape
        # TODO: Instead of transpose / view operations, use a PermutationLazyTensor (see #539) to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_mvn = mean.reshape(*mean.shape[:-2], -1)
        else:
            mean_mvn = mean.transpose(-1, -2).reshape(*mean.shape[:-2], -1)
        super().__init__(loc=mean_mvn, covariance_matrix=covariance_matrix, bounds=bounds, validate_args=validate_args)
        self.bounds = bounds

    @property
    def event_shape(self):
        return self._output_shape[-2:]

    @classmethod
    def from_independent_tmvns(cls, tmvns):
        if len(tmvns) < 2:
            raise ValueError("Must provide at least 2 TMVNs to form a MultitaskTruncatedMultivariateNormal")
        if any(isinstance(tmvn, MultitaskTruncatedMultivariateNormal) for tmvn in tmvns):
            raise ValueError("Cannot accept MultitaskTruncatedMultivariateNormals")
        if not all(m.batch_shape == tmvns[0].batch_shape for m in tmvns[1:]):
            raise ValueError("All TruncatedMultivariateNormals must have the same batch shape")
        if not all(m.event_shape == tmvns[0].event_shape for m in tmvns[1:]):
            raise ValueError("All TruncatedMultivariateNormals must have the same event shape")
        mean = torch.stack([tmvn.loc for tmvn in tmvns], -1)
        # TODO: To do the following efficiently, we don't want to evaluate the
        # covariance matrices. Instead, we want to use the lazies directly in the
        # BlockDiagLazyTensor. This will require implementing a new BatchLazyTensor:

        # https://github.com/cornellius-gp/gpytorch/issues/468
        covar_blocks_lazy = gpytorch.lazy.CatLazyTensor(
            *[tmvn.lazy_covariance_matrix.unsqueeze(0) for tmvn in tmvns],
            dim=0,
            output_device=mean.device
        )
        covar_lazy = gpytorch.lazy.BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return cls(loc=mean, covariance_matrix=covar_lazy, bounds=tmvns.bounds, interleaved=False)

    def expand(self, batch_size):
        new_mean = self.loc.expand(torch.Size(batch_size) + self.loc.shape[-2:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(loc=new_mean, covariance_matrix=new_covar, bounds=self.bounds, interleaved=self._interleaved)
        return res

    def log_prob(self, value):
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = value.shape[:-2] + value.shape[:-3:-1]
            value = value.view(new_shape).transpose(-1, -2).contiguous()
        return super().log_prob(value.view(*value.shape[:-2], -1))

    @property
    def mean(self):
        mean = super().loc
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

    def confidence_region(self) -> tuple[Tensor, Tensor]:
        """
        Return approximate confidence region (mean ± 2 std).
        Note: For truncated distributions, this is an approximation.
        """
        std = torch.sqrt(self.variance)
        lower = self.mean - 2 * std
        upper = self.mean + 2 * std
        
        # Clip to bounds to ensure validity
        lower = torch.max(lower, self._bounds[..., 0])
        upper = torch.min(upper, self._bounds[..., 1])
        
        return lower, upper
