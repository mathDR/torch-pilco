import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal
from torch import Tensor
from torch.distributions import Distribution
from typing import Any


class BatchedTruncatedDistribution(Distribution):
    """
    A batched truncated multivariate normal distribution.
    Handles each batch element independently since TruncatedMultivariateNormal
    doesn't properly support batching.
    """
    
    def __init__(self, mean: Tensor, covar: Tensor, bounds: Tensor) -> None:
        self.loc = mean  # Use 'loc' to match PyTorch distribution API
        self.covariance_matrix = covar
        self._bounds = bounds
        batch_size = mean.shape[0]
        num_tasks = mean.shape[1]
        
        # Initialize the Distribution base class
        super().__init__(
            batch_shape=torch.Size([batch_size]),
            event_shape=torch.Size([num_tasks]),
            validate_args=False
        )
        
        # Create individual truncated distributions for each batch element
        self.dists = []
        for i in range(batch_size):
            dist = TruncatedMultivariateNormal(
                loc=mean[i],
                covariance_matrix=covar[i],
                bounds=bounds[i],
                validate_args=False
            )
            self.dists.append(dist)
    
    @property
    def mean(self) -> Tensor:
        """Return the mean of the distribution"""
        return self.loc
    
    @property
    def variance(self) -> Tensor:
        """Return the diagonal variance"""
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        # Sample from each batch element independently
        samples = []
        for dist in self.dists:
            sample = dist.rsample(sample_shape)
            samples.append(sample)
        
        # Stack along batch dimension
        return torch.stack(samples, dim=-2)
    
    def log_prob(self, value: Tensor) -> Tensor:
        # Compute log prob for each batch element
        log_probs = []
        for i, dist in enumerate(self.dists):
            lp = dist.log_prob(value[..., i, :])
            log_probs.append(lp)
        
        return torch.stack(log_probs, dim=-1)
    
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


class TruncatedMultitaskLikelihood(MultitaskGaussianLikelihood):
    """
    Multitask Gaussian likelihood with box constraints per task.
    
    Args:
        num_tasks: Number of tasks
        bounds: Tensor of shape [num_tasks, 2] where bounds[:, 0] are lower bounds
                and bounds[:, 1] are upper bounds for each task
        rank: Rank of task covariance matrix (default: 0 for diagonal)
        **kwargs: Additional arguments for MultitaskGaussianLikelihood
    """
    
    def __init__(
        self, 
        num_tasks: int, 
        bounds: Tensor, 
        rank: int = 0, 
        **kwargs: Any
    ) -> None:
        super().__init__(num_tasks=num_tasks, rank=rank, **kwargs)
        
        # Validate and register bounds
        if bounds.shape != (num_tasks, 2):
            raise ValueError(f"bounds must have shape [{num_tasks}, 2], got {bounds.shape}")
        if (bounds[:, 0] >= bounds[:, 1]).any():
            raise ValueError("Lower bounds must be strictly less than upper bounds")
        
        self.register_buffer('bounds', bounds)
    
    def marginal(
        self, 
        function_dist: MultitaskMultivariateNormal, 
        *args: Any, 
        **kwargs: Any
    ) -> BatchedTruncatedDistribution:
        """
        Returns a wrapper that provides rsample and log_prob for truncated distribution.
        
        Since TruncatedMultivariateNormal doesn't handle batching well, we create
        a custom distribution wrapper that handles each batch element independently.
        
        Args:
            function_dist: MultitaskMultivariateNormal from GP
            
        Returns:
            BatchedTruncatedDistribution with rsample() and log_prob() methods
        """
        # MultitaskMultivariateNormal mean has shape [batch_size, num_tasks]
        mean = function_dist.mean
        
        # Infer actual dimensions from mean shape
        batch_size = mean.shape[0]
        actual_num_tasks = mean.shape[1]
        
        # Validate that bounds match the actual number of tasks
        if self.bounds.shape[0] != actual_num_tasks:
            raise ValueError(
                f"Bounds dimension {self.bounds.shape[0]} does not match "
                f"actual number of tasks {actual_num_tasks}. "
                f"Initialize likelihood with num_tasks={actual_num_tasks}"
            )
        
        # Get the base_dist if available
        if hasattr(function_dist, 'base_dist'):
            base_covar = function_dist.base_dist.covariance_matrix
        else:
            # Fallback: extract from lazy covariance
            full_covar = function_dist.lazy_covariance_matrix.to_dense()
            # Extract block diagonal
            base_covar = torch.zeros(batch_size, actual_num_tasks, actual_num_tasks, 
                                     device=mean.device, dtype=mean.dtype)
            for i in range(batch_size):
                idx_start = i * actual_num_tasks
                idx_end = (i + 1) * actual_num_tasks
                base_covar[i] = full_covar[idx_start:idx_end, idx_start:idx_end]
        
        # Add observation noise to each task's diagonal
        noise_eye = torch.eye(actual_num_tasks, device=mean.device, dtype=mean.dtype)
        noise_matrix = self.noise.unsqueeze(-1) * noise_eye  # [num_tasks, num_tasks]
        noise_matrix = noise_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        full_covar = base_covar + noise_matrix
        
        # Expand bounds to match actual dimensions [batch_size, num_tasks, 2]
        bounds = self.bounds.unsqueeze(0).expand(batch_size, -1, -1)
        
        return BatchedTruncatedDistribution(mean, full_covar, bounds)
    
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
        truncated_dist = self.marginal(function_dist, *args, **kwargs)
        
        # Compute log probability of observations
        log_prob = truncated_dist.log_prob(observations)
        
        return log_prob


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
    likelihood = TruncatedMultitaskLikelihood(num_tasks=num_tasks, bounds=bounds)
    
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
    
    print(f"Function dist mean shape: {function_dist.mean.shape}")
    
    # Get truncated distribution
    truncated_dist = likelihood(function_dist)
    
    # Sample from truncated distribution
    samples = truncated_dist.rsample()
    print(f"Samples shape: {samples.shape}")
    print(f"Sample values:\n{samples[:3]}")
    print(f"\nBounds check - all samples within bounds: {((samples >= bounds[:, 0]) & (samples <= bounds[:, 1])).all()}")