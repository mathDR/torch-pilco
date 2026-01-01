import torch
import gpytorch
from minecraft import MinecraftKernel
from matplotlib import pyplot as plt

# 1. Generate Synthetic Data (Two frequency components)
train_x = torch.linspace(0, 10, 100)
train_y = torch.sin(train_x) + 0.5 * torch.sin(4 * train_x) + torch.randn(100) * 0.1

# 2. Define the GP Model
class MinecraftGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Using 2 blocks to capture the two frequencies
        self.covar_module = MinecraftKernel(num_blocks=2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 3. Training
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MinecraftGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# 4. Inference and Plotting
model.eval()
with torch.no_grad():
    test_x = torch.linspace(0, 12, 200)
    pred = model(test_x)
    
    plt.figure(figsize=(10, 4))
    plt.plot(train_x, train_y, 'k*', label="Data")
    plt.plot(test_x, pred.mean, 'b', label="Minecraft Kernel Mean")
    plt.fill_between(test_x, pred.mean - 2, pred.mean + 2, alpha=0.2, label="Confidence")
    plt.legend()
    plt.show()
