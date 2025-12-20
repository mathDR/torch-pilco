" Controllers for different use cases."
import numpy as np
import torch
from typing import Callable


class Policy(torch.nn.Module):
    """
    Superclass of policy objects
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        flg_squash: bool=False,
        u_max: float = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | int | str = torch.device("cpu")
    ):
        super(Policy, self).__init__()
        # model parameters
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        # set squashing function
        if flg_squash:
            self.f_squash = lambda x: self.squashing(x, u_max)
        else:
            # assign the identity function
            self.f_squash = lambda x: x

    def forward(
        self,
        states: torch.Tensor,
        timepoint: float | None = None,
        p_dropout: float=0.0
    ):
        raise NotImplementedError()

    def forward_np(
        self,
        states: torch.Tensor,
        timepoint: float=None,
    ):
        """
        Numpy implementation of the policy
        """
        input_tc = self(
            states=torch.tensor(states, dtype=self.dtype, device=self.device),
            timepoint=timepoint
        )
        return input_tc.detach().cpu().numpy()

    def to(
        self,
        device: torch.device | int | str
    ):
        """
        Move the model parameters to 'device'
        """
        super(Policy, self).to(device)
        self.device = device

    def squashing(
        self,
        u: torch.Tensor | float,
        u_max: torch.Tensor | float,
    ):
        """
        Squash the inputs inside (-u_max, +u_max)
        """
        if np.isscalar(u_max):
            return u_max * torch.tanh(u / u_max)
        else:
            u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)
            return u_max * torch.tanh(u / u_max)

    def get_np_policy(self):
        """
        Returns a function handle to the numpy version of the policy
        """
        f = lambda state, timepoint: self.forward_np(state, timepoint)

        return f

    def reinit(self, scaling: float=1):
        raise NotImplementedError()


class RandomExploration(Policy):
    """
    Random control action, uniform dist. (-u_max, +u_max)
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        flg_squash: bool=False,
        u_max: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | int | str = torch.device("cpu")
    ):

        super(RandomExploration, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            flg_squash=flg_squash,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.u_max = u_max

    def forward(
        self,
        states: torch.Tensor,
        timepoint: float | None = None,
    ):
        # returns random control action
        rand_u = self.u_max * (
            2.0 * np.random.rand(self.input_dim) - 1
        ).reshape([-1, self.input_dim])
        return torch.tensor(rand_u, dtype=self.dtype, device=self.device)


class SumOfSinusoids(Policy):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        num_sin: int,
        omega_min: float,
        omega_max: float,
        amplitude_min: float,
        amplitude_max: float,
        flg_squash: bool=False,
        u_max: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | int | str = torch.device("cpu")
    ):
        super(SumOfSinusoids, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            flg_squash=flg_squash,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.num_sin = num_sin
        amplitude_min = np.array(amplitude_min)
        amplitude_max = np.array(amplitude_max)
        # generate random parameters
        self.amplitudes = torch.nn.Parameter(
            torch.tensor(
                amplitude_min + (amplitude_max - amplitude_min) * np.random.rand(num_sin, input_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.omega = torch.nn.Parameter(
            torch.tensor(
                np.random.choice([-1, 1], [num_sin, input_dim])
                * (omega_min + (omega_max - omega_min) * np.random.rand(num_sin, input_dim)),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.phases = torch.nn.Parameter(
            torch.tensor(
                np.random.choice([-1, 1], [num_sin, input_dim]) * (np.pi * (np.random.rand(num_sin, input_dim) - 0.5)),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        states: torch.Tensor,
        timepoint: float,
    ):
        # returns the sinusoid values at time `timepoint`
        return self.f_squash(
            torch.sum(
                self.amplitudes * (torch.sin(self.omega * timepoint + self.phases)),
                dim=0
            ).reshape([-1, self.input_dim])
        )


class SumOfGaussians(Policy):
    """
    Control policy: sum of 'num_basis' gaussians
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        num_basis: int,
        flg_train_lengthscales: bool=True,
        lengthscales_init=None,
        flg_train_centers: bool=True,
        centers_init: torch.Tensor | float| None = None,
        centers_init_min: float=-1,
        centers_init_max: float=1,
        weight_init: torch.Tensor | float| None = None,
        flg_train_weight: bool=True,
        flg_bias: bool=False,
        bias_init: torch.Tensor | float| None = None,
        flg_train_bias: bool=False,
        flg_squash: bool=False,
        u_max: float = 1.0,
        scale_factor: torch.Tensor | float| None = None,
        flg_drop: bool=True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | int | str = torch.device("cpu")
    ):
        super(SumOfGaussians, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            flg_squash=flg_squash,
            u_max=u_max,
            dtype=dtype,
            device=device
        )
        # set number of gaussian basis functions
        self.num_basis = num_basis
        # get initial log lengthscales
        if lengthscales_init is None:
            lengthscales_init = np.ones((1,state_dim))
        self.log_lengthscales = torch.nn.Parameter(
            torch.tensor(np.log(lengthscales_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_lengthscales,
        )
        # get initial centers
        if centers_init is None:
            centers_init = centers_init_min * np.ones([num_basis, state_dim]) + (
                centers_init_max - centers_init_min
            ) * np.random.rand(num_basis, state_dim)
        self.centers = torch.nn.Parameter(
            torch.tensor(centers_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_centers
        )
        # initilize the linear ouput layer
        self.f_linear = torch.nn.Linear(in_features=num_basis, out_features=input_dim, bias=flg_bias)
        # check weight initialization
        if not (weight_init is None):
            self.f_linear.weight.data = torch.tensor(weight_init, dtype=dtype, device=device)
        else:
            self.f_linear.weight.data = torch.tensor(np.ones([input_dim, num_basis]), dtype=dtype, device=device)

        self.f_linear.weight.requires_grad = flg_train_weight
        # check bias initialization
        if flg_bias:
            self.f_linear.bias.requires_grad = flg_train_bias
            if not (bias_init is None):
                self.f_linear.bias.data = torch.tensor(bias_init)
        # set type and device
        self.f_linear.type(self.dtype)
        self.f_linear.to(self.device)

        if scale_factor is None:
            scale_factor = np.ones(state_dim)
        self.scale_factor = torch.tensor(scale_factor, dtype=self.dtype, device=self.device).reshape([1, -1])

        # set dropout
        if flg_drop == True:
            self.f_drop = torch.nn.functional.dropout
        else:
            self.f_drop = lambda x, p: x

    def reinit(
        self,
        lengthscales_par: torch.Tensor,
        centers_par: torch.Tensor,
        weight_par: torch.Tensor,
    ):
        self.log_lengthscales.data = torch.tensor(
            np.log(lengthscales_par), dtype=self.dtype, device=self.device
        ).reshape([1, -1])
        self.centers.data = (
            torch.tensor(centers_par, dtype=self.dtype, device=self.device)
            * 2
            * (torch.rand(self.num_basis, self.state_dim, dtype=self.dtype, device=self.device) - 0.5)
        )
        self.f_linear.weight.data = weight_par * (
            torch.rand(self.input_dim, self.num_basis, dtype=self.dtype, device=self.device) - 0.5
        )

    def forward(
        self,
        states: torch.Tensor,
        timepoint: float | None = None,
        p_dropout: float=0.0
    ):
        """
        Returns a linear combination of gaussian functions
        with input given by the the distances between that state
        and the vector of centers of the gaussian functions
        """
        # get the lengthscales from log
        lengthscales = torch.exp(self.log_lengthscales)
        states = torch.atleast_2d(states)
        states = states / self.scale_factor

        # normalize states and centers
        norm_states = states / lengthscales
        norm_centers = self.centers / lengthscales
        
        # get the square distance
        dist = torch.square(norm_states.unsqueeze(1)-norm_centers).sum(dim=2)

        # apply exp and get output
        exp_dist_dropped = self.f_drop(torch.exp(-0.5*dist), p_dropout)
        inputs = self.f_linear(exp_dist_dropped)

        # returns the constrained control action
        return self.f_squash(inputs).squeeze(0)

class SumOfGaussiansWithAngles(SumOfGaussians):
    """
    Extends sum of gaussians policy. Angle indices are mapped in cos and sin before computing the policy
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        num_basis: int,
        angle_indices: torch.Tensor,
        non_angle_indices: torch.Tensor,
        flg_train_lengthscales: bool=True,
        lengthscales_init=None,
        flg_train_centers: bool=True,
        centers_init: torch.Tensor | float| None = None,
        centers_init_min: float=-1,
        centers_init_max: float=1,
        weight_init: torch.Tensor | float| None = None,
        flg_train_weight: bool=True,
        flg_bias: bool=False,
        bias_init: torch.Tensor | float| None = None,
        flg_train_bias: bool=False,
        flg_squash: bool=False,
        u_max: float = 1.0,
        scale_factor: torch.Tensor | float| None = None,
        flg_drop: bool=True,
        dtype: torch.dtype = torch.float64,
        device: torch.device | int | str = torch.device("cpu"),
    ):
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        self.num_angle_indices = angle_indices.size
        self.num_non_angle_indices = non_angle_indices.size
        super(SumOfGaussiansWithAngles, self).__init__(
            state_dim=state_dim + self.num_angle_indices,
            input_dim=input_dim,
            num_basis=num_basis,
            flg_train_lengthscales=flg_train_lengthscales,
            lengthscales_init=lengthscales_init,
            flg_train_centers=flg_train_centers,
            centers_init=centers_init,
            centers_init_min=centers_init_min,
            centers_init_max=centers_init_max,
            weight_init=weight_init,
            flg_train_weight=flg_train_weight,
            flg_bias=flg_bias,
            bias_init=bias_init,
            flg_train_bias=flg_train_bias,
            flg_squash=flg_squash,
            u_max=u_max,
            flg_drop=flg_drop,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        states: torch.Tensor,
        timepoint: float,
        p_dropout: float=0.0
    ):
        # build a state with non angle features and cos,sin of angle features
        states = states.reshape([-1, self.state_dim - self.num_angle_indices])
        new_state = torch.cat(
            [
                states[:, self.non_angle_indices],
                torch.cos(states[:, self.angle_indices]),
                torch.sin(states[:, self.angle_indices]),
            ],
            1,
        )
        # call the forward method of the superclass
        return super().forward(new_state, timepoint=timepoint, p_dropout=p_dropout)
    