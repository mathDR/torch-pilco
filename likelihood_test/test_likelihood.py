import torch
import gpytorch
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from botorch.utils.probability.truncated_multivariate_normal import TruncatedMultivariateNormal
from torch.distributions import constraints
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal

from matplotlib import pyplot as plt

from torch_pilco.model_learning.my_likelihood import TruncatedMultitaskLikelihood

train_x = torch.linspace(0, 1, 100)[:,torch.newaxis]

train_y = torch.hstack([
    torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * torch.pi)) + 2 * torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
])

print(train_x.shape, train_y.shape)


# In[57]:


num_latents = 3
num_tasks = 4

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 16, 1)

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
base_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)


bounds = torch.tensor([
    [-1.5, 1.5],   # Task 1: sin
    [-1.5, 1.5],   # Task 2: cos  
    [-3.5, 3.5],   # Task 3: sin + 2*cos
    [-1.5, 1.5],   # Task 4: -cos
])


# Create the likelihood
likelihood = TruncatedMultitaskLikelihood(
    num_tasks=num_tasks,
    bounds=bounds,
)
base_mll = gpytorch.mlls.VariationalELBO(base_likelihood, model, num_data=train_y.size(0))
print(-base_mll(model(train_x), train_y))


mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
print(-mll(model(train_x), train_y))
breakpoint()
num_epochs = 50

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
with gpytorch.settings.num_likelihood_samples(100):
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        print(f'loss={loss.item()}')
        loss.backward()
        optimizer.step()
breakpoint()
# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

for task, ax in enumerate(axs):
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.numpy(), mean[:, task].numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')

fig.tight_layout()
plt.show()






