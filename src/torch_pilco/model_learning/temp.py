import torch
import gpytorch
from gpytorch.mlls import VariationalELBO
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.utils.probability import TruncatedMultivariateNormal

class TruncatedMultitaskLikelihood(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    def __init__(self, num_tasks, lower, upper, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)

    def expected_log_prob(self, target, variational_dist_f, **kwargs):
        # target: (N, num_tasks), variational_dist_f: distribution q(f)
        # We sample from q(f) to estimate E_q [log p(y|f)]
        samples = variational_dist_f.rsample(torch.Size([gpytorch.settings.num_likelihood_samples.value()]))
        
        # noise_covar provides the covariance matrix (typically diagonal for multitask)
        noise = self.noise_covar.noise.unsqueeze(-1)
        
        # Construct the truncated distribution for the samples
        res = TruncatedMultivariateNormal(
            loc=samples,
            covariance_matrix=torch.diag_embed(noise.expand_as(samples)),
            lower_bound=self.lower,
            upper_bound=self.upper
        )
        # Compute log_prob and average over MC samples
        return res.log_prob(target).mean(0)

# 1. Initialize Model and Likelihood
# Ensure 'lower' and 'upper' match your task output dimensions
likelihood = TruncatedMultitaskLikelihood(num_tasks=2, lower=lower_bounds, upper=upper_bounds)
model = SingleTaskVariationalGP(train_X, likelihood=likelihood)

# 2. Setup the Variational ELBO loss
mll = VariationalELBO(likelihood, model, num_data=train_X.size(0))

# 3. Standard Optimization Loop
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, train_Y)
    loss.backward()
    optimizer.step()
