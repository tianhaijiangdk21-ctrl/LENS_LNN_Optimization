"""
Reward function for laser cladding RL (Eq. 6 in document).
R = 10 * exp(-5|Δh|) + 20*(1 - crack_risk) - 0.1 * P * Δt
"""

import torch


def compute_reward(delta_h: torch.Tensor, crack_risk: torch.Tensor, P: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Args:
        delta_h: height deviation (mm), shape [batch] or scalar tensor
        crack_risk: crack probability (0~1), same shape
        P: laser power (kW), same shape
        dt: control interval (s)
    Returns:
        reward tensor of same shape
    """
    term1 = 10 * torch.exp(-5 * torch.abs(delta_h))
    term2 = 20 * (1 - crack_risk)
    term3 = 0.1 * P * dt
    return term1 + term2 - term3