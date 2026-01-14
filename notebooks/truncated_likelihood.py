#!/usr/bin/env python

import torch
import gpytorch
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal
from torch.distributions import constraints
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal

from matplotlib import pyplot as plt

from torch_pilco.model_learning.multitask_truncated_normal_likelihood import MultitaskTruncatedNormalLikelihood


seed_value = 4
torch.manual_seed(seed_value)


# ## Build Data
train_x = torch.linspace(0, 1, 100)[:,torch.newaxis]

train_y = torch.hstack([
    torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * torch.pi)) + 2 * torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
])

print(train_x.shape, train_y.shape)


num_latents = 3
num_tasks = 4

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


model = MultitaskGPModel()

bounds = torch.tensor([
    [-1.5, 1.5],   # Task 1: sin
    [-1.5, 1.5],   # Task 2: cos  
    [-2.65, 4.5],   # Task 3: sin + 2*cos
    [-1.7, 1.5],   # Task 4: -cos
])
assert (torch.all(train_y.min(dim=0)[0] >= bounds[:, 0]) and torch.all(train_y.max(dim=0)[0] <= bounds[:, 1]))

# Create the likelihood
likelihood = MultitaskTruncatedNormalLikelihood(
    num_tasks=num_tasks,
    bounds=bounds,
)


mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
num_epochs = 50

model.train()
likelihood.train()
hyperparameter_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
    hyperparameter_optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    print(f'loss={loss.item()}')
    loss.backward()
    hyperparameter_optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()


# ## Evaluate at Test Data
test_x = torch.linspace(0, 1, 51)
test_y = model(test_x)

# predictive_dist = TruncatedMultivariateNormal(
#     loc=test_y.mean.flatten(),
#     covariance_matrix=test_y.covariance_matrix,
#     bounds = torch.tile(bounds, (51,1))
# )

# mu = predictions.mean.reshape((51,4))
# std = torch.sqrt(torch.diagonal(predictions.covariance_matrix, dim1=-2, dim2=-1)).reshape((51,4))

# lower = mu - 2 * std
# upper = mu + 2 * std


breakpoint()
# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# # Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    predictive_dist = likelihood(model(test_x))
    mu = predictive_dist.mean
    lower, upper = predictive_dist.confidence_region()

for task, ax in enumerate(axs):
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.numpy(), mu[:, task].detach().numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x.numpy(), lower[:, task].detach().numpy(), upper[:, task].detach().numpy(), alpha=0.5)
    ax.axhline(bounds[task,0])
    ax.axhline(bounds[task,1])
    ax.set_ylim([-5, 5])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')

fig.tight_layout()

plt.show()

