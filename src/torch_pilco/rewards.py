import torch


def pendulum_cost(
    states: torch.Tensor,
    actions: torch.Tensor,
) -> float:
    """
    Replicated Cost function from gymnasium:
        -(theta**2 + 0.1*theta_dt**2 + 0.001*torque**2)
    but we minimize it, so we return the negation.
    """
    states = torch.atleast_2d(states)
    actions = torch.atleast_2d(actions)

    x = states[0,0]
    y = states[0,1]
    angle_velocity = states[0,2]
    torque = actions[0,0]
    theta = torch.atan2(y, x)

    return (
        torch.square(theta) +
        0.1*torch.square(angle_velocity) +
        0.001*torch.square(torque)
    )