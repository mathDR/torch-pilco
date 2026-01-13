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


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


seed_value = 4
torch.manual_seed(seed_value)


# ## Build Data

# In[4]:


train_x = torch.linspace(0, 1, 100)[:,torch.newaxis]

train_y = torch.hstack([
    torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * torch.pi)) + 2 * torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2,
])

print(train_x.shape, train_y.shape)


# # Build GP Model

# In[5]:


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


# ## Initialize Model and Likelihood

# In[6]:


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


# In[7]:


base_likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)


# In[8]:


base_likelihood(model(train_x)).covariance_matrix.shape


# In[9]:


likelihood(model(train_x))


# ## Try Computing marginal likelihood

# In[10]:


mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))


# In[11]:


mll(model(train_x), train_y)


# ## Optimize Hyperparameters of GP Model

# In[12]:


# variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)
# hyperparameter_optimizer = torch.optim.Adam([
#     {'params': model.hyperparameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.1)
hyperparameter_optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)
#
hyperparameter_scheduler = torch.optim.lr_scheduler.StepLR(
    hyperparameter_optimizer,
    step_size=5,
    gamma=0.1
)
# variational_ngd_scheduler = torch.optim.lr_scheduler.StepLR(
#     variational_ngd_optimizer,
#     step_size=5,
#     gamma=0.1
# )


# In[13]:


mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
num_epochs = 50

model.train()
likelihood.train()

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
with gpytorch.settings.num_likelihood_samples(25):
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        #variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        print(f'loss={loss.item()}')
        loss.backward()
        #variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()
# Set into eval mode
model.eval()
likelihood.eval()


# ## Evaluate at Test Data

# In[14]:


test_x = torch.linspace(0, 1, 51)
test_y = model(test_x)


# In[15]:


test_y.mean.shape, test_y.covariance_matrix.shape, 51*4


# In[45]:


test_y.mean.flatten().shape


# In[17]:


predictions = TruncatedMultivariateNormal(loc=test_y.mean.flatten(), covariance_matrix=test_y.covariance_matrix, bounds = torch.tile(bounds, (51,1)))


# In[18]:


mu = predictions.mean.reshape((51,4))


# In[19]:


std = torch.sqrt(torch.diagonal(predictions.covariance_matrix, dim1=-2, dim2=-1)).reshape((51,4))


# In[20]:


lower = mu - 2 * std
upper = mu + 2 * std


# In[23]:


# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# # Make predictions
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = torch.linspace(0, 1, 51)
#     predictions = likelihood(model(test_x))
#     mean = predictions.mean
#     lower, upper = predictions.confidence_region()

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


# In[ ]:

