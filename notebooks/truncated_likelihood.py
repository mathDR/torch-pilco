#!/usr/bin/env python
# coding: utf-8

# # Implement Test of Multitask Multivariate Truncated Normal Likelihood

# ## Imports
import torch
import gpytorch
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal as TruncMultivariateNormal
from torch.distributions import constraints
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal

from matplotlib import pyplot as plt
from torch_pilco.model_learning.multitask_truncated_multivariate_normal import MultitaskTruncatedMultivariateNormal
from torch_pilco.model_learning.multitask_truncated_gaussian_likelihood import MultitaskTruncatedGaussianLikelihood
from torch_pilco.model_learning.truncated_multivariate_normal import TruncatedMultivariateNormal

seed_value = 4
torch.manual_seed(seed_value)


# ## Build Data
train_x = torch.linspace(0, 1, 100)[:,torch.newaxis].double()

def compute_y(input_x: torch.Tensor) -> torch.Tensor:
    return torch.hstack([
        torch.sin(input_x * (2 * torch.pi)) + torch.randn(input_x.size()) * 0.2,
        torch.cos(input_x * (2 * torch.pi)) + torch.randn(input_x.size()) * 0.2,
        torch.sin(input_x * (2 * torch.pi)) + 2 * torch.cos(input_x * (2 * torch.pi)) + torch.randn(input_x.size()) * 0.2,
        -torch.cos(input_x * (2 * torch.pi)) + torch.randn(input_x.size()) * 0.2,
    ])
train_y = compute_y(train_x)
num_latents = 3
num_tasks = 4


# ## Construct GP Model
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 25, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=4,
            num_latents=3,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


model = MultitaskGPModel().double()


# ## Build Likelihood
bounds = torch.tensor([
    [-1.25, 1.35],   # Task 1: sin
    [-1.45, 1.25],   # Task 2: cos  
    [-2.6, 2.5],   # Task 3: sin + 2*cos
    [-1.7, 1.3],   # Task 4: -cos
])
assert (torch.all(train_y.min(dim=0)[0] >= bounds[:, 0]) and torch.all(train_y.max(dim=0)[0] <= bounds[:, 1]))

likelihood = MultitaskTruncatedGaussianLikelihood(
    num_tasks=num_tasks,
    bounds=bounds,
).double()

output = model(train_x)
pred_dist = likelihood(output)
breakpoint()
with torch.no_grad(): print(pred_dist.rsample().shape)
# ## Construct and Optimize Marginal Likelihood 
# Should profile this
# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
# num_epochs = 2

# model.train()
# likelihood.train()
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.1)

# # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# # effective for VI.
# for i in range(num_epochs):
#     # Within each iteration, we will go over each minibatch of data
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     if i%10 == 0:
#         print(f'Iteration {i}')
#     print(f'loss={loss.item()}')
#     loss.backward()
#     optimizer.step()

# # Set into eval mode
# model.eval()
# likelihood.eval()


# # ## Evaluate at Test Data
# test_x = torch.linspace(0, 1, 51)[:,torch.newaxis]
# test_y = compute_y(test_x)

# # # Make predictions
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     predictive_dist = likelihood(model(test_x))
#     mu = predictive_dist.mean
#     #lower, upper = predictive_dist.confidence_region()
#     std2 = predictive_dist.variance.sqrt().mul_(2)
#     lower, upper = torch.max(torch.min(mu.sub(std2), bounds[:,1]), bounds[:,0]), torch.max(torch.min(mu.add(std2), bounds[:,1]), bounds[:,0])

# samples = predictive_dist.rsample()
# print(samples.shape)
