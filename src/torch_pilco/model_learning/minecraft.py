import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

class MinecraftKernel(Kernel):
    is_stationary = True

    def __init__(self, num_blocks=1, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        
        # Register hyperparameters for each block
        self.register_parameter(name="raw_amplitudes", parameter=torch.nn.Parameter(torch.ones(num_blocks)))
        self.register_parameter(name="raw_frequencies", parameter=torch.nn.Parameter(torch.zeros(num_blocks)))
        self.register_parameter(name="raw_bandwidths", parameter=torch.nn.Parameter(torch.ones(num_blocks)))

        # Ensure amplitudes and bandwidths remain positive
        self.register_constraint("raw_amplitudes", Positive())
        self.register_constraint("raw_bandwidths", Positive())

    @property
    def amplitudes(self): return self.raw_amplitudes_constraint.transform(self.raw_amplitudes)
    @property
    def frequencies(self): return self.raw_frequencies
    @property
    def bandwidths(self): return self.raw_bandwidths_constraint.transform(self.raw_bandwidths)

    def forward(self, x1, x2, diag=False, **params):
        tau = self.covar_dist(x1, x2, square_dist=False, diag=diag, **params)
        res = torch.zeros_like(tau)
        eps = 1e-8 # Prevent division by zero in sinc
        
        for i in range(self.num_blocks):
            # Sinc(x) in PyTorch is sin(pi*x)/(pi*x)
            sinc_term = torch.sinc(self.bandwidths[i] * tau + eps)
            cos_term = torch.cos(2 * torch.pi * self.frequencies[i] * tau)
            res += self.amplitudes[i] * sinc_term * cos_term
        return res


class MultiOutputMinecraftKernel(Kernel):
    """
    Multi-output Minecraft kernel for 2 correlated tasks.
    Each 'block' has shared bandwidth and frequency but task-specific 
    amplitudes and phase shifts.
    """
    def __init__(self, num_tasks=2, num_blocks=1, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.num_blocks = num_blocks

        # Shared parameters per block
        self.register_parameter("raw_freq", torch.nn.Parameter(torch.zeros(num_blocks)))
        self.register_parameter("raw_bw", torch.nn.Parameter(torch.ones(num_blocks)))
        
        # Task-specific parameters: Amplitudes (rank-1 covariance) and Phases
        self.register_parameter("raw_amps", torch.nn.Parameter(torch.ones(num_blocks, num_tasks)))
        self.register_parameter("phases", torch.nn.Parameter(torch.zeros(num_blocks, num_tasks)))

        self.register_constraint("raw_bw", Positive())

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # x1, x2 expected to have task indices in the last dimension
        # For simplicity in this example, we assume standard GPyTorch Multi-task indexing
        # where the last column of X contains integer task IDs.
        
        tau = self.covar_dist(x1[:, :-1], x2[:, :-1], square_dist=False, diag=diag)
        i = x1[:, -1].long()
        j = x2[:, -1].long()
        
        res = torch.zeros_like(tau)
        bw = self.raw_bw_constraint.transform(self.raw_bw)
        
        for b in range(self.num_blocks):
            # Spectral block in time domain: sinc(B*tau)
            sinc_term = torch.sinc(bw[b] * tau + 1e-8)
            
            # Correlated component: cos(2*pi*f*tau + (phi_i - phi_j))
            # Amplitude is typically modeled as A_i * A_j
            phase_diff = self.phases[b, i] - self.phases[b, j]
            cos_term = torch.cos(2 * torch.pi * self.raw_freq[b] * tau + phase_diff)
            amp_term = self.raw_amps[b, i] * self.raw_amps[b, j]
            
            res += amp_term * sinc_term * cos_term
        return res
