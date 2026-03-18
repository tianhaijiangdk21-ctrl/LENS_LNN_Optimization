"""
Physics-Informed Liquid Neural Network (LNN) module.
Implements equations (1)-(3) from the document:
    τ dS/dt = -S + tanh(W(t)S + I(t))   with W(t) = W0 * exp(-α·σ_thermal)
Includes physics loss function (Fourier heat + thermo-mechanical).
"""

import torch
import torch.nn as nn


class LiquidCell(nn.Module):
    """
    Single liquid neuron layer with thermal stress modulated weights.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tau: float = 1.0, alpha: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.alpha = alpha

        # Base weight matrix W0 (learnable)
        self.W0 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        # Input projection matrix
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, S: torch.Tensor, I: torch.Tensor, sigma_thermal: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        One Euler step.
        Args:
            S: state [batch, hidden_dim]
            I: external input [batch, input_dim]
            sigma_thermal: thermal stress [batch, 1]
            dt: time step size
        Returns:
            next state [batch, hidden_dim]
        """
        batch_size = S.size(0)
        # Modulated weight: W(t) = W0 * exp(-alpha * sigma_thermal)
        sigma = sigma_thermal.view(batch_size, 1, 1)          # [batch, 1, 1]
        W_t = self.W0 * torch.exp(-self.alpha * sigma)       # [batch, hidden_dim, hidden_dim]

        I_proj = self.input_proj(I)                           # [batch, hidden_dim]
        S_unsq = S.unsqueeze(-1)                              # [batch, hidden_dim, 1]
        W_S = torch.bmm(W_t, S_unsq).squeeze(-1)              # [batch, hidden_dim]

        dS_dt = (-S + torch.tanh(W_S + I_proj)) / self.tau
        S_next = S + dt * dS_dt
        return S_next


class PhysicsInformedLNN(nn.Module):
    """
    Full LNN model that processes a sequence of inputs and outputs crack risk and residual stress.
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.liquid = LiquidCell(input_dim, hidden_dim)

        # Head to predict residual stress σ_residual
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # Sigmoid to convert σ_residual to crack risk (0~1) assuming a reference stress of 500 MPa
        self.crack_head = nn.Sigmoid()

    def forward(self, static_params: torch.Tensor, I_seq: torch.Tensor, sigma_thermal_seq: torch.Tensor) -> tuple:
        """
        Args:
            static_params: [batch, static_dim] – current process parameters (P, v, F)
            I_seq: [batch, seq_len, input_dim] – external input sequence (e.g., sensor readings)
            sigma_thermal_seq: [batch, seq_len, 1] – thermal stress at each time step
        Returns:
            crack_risk: [batch] – predicted crack probability (0~1)
            sigma_residual: [batch] – predicted residual stress
        """
        batch_size, seq_len, _ = I_seq.shape
        # Initial state: zero (or could be learned embedding of static_params)
        S = torch.zeros(batch_size, self.hidden_dim, device=I_seq.device)

        for t in range(seq_len):
            I_t = I_seq[:, t, :]               # [batch, input_dim]
            sigma_t = sigma_thermal_seq[:, t, :]  # [batch, 1]
            S = self.liquid(S, I_t, sigma_t)

        sigma_residual = self.sigma_head(S).squeeze(-1)        # [batch]
        crack_risk = self.crack_head(sigma_residual / 500.0)   # normalize by typical max stress
        return crack_risk, sigma_residual


# ----------------------------------------------------------------------
# Physics loss function (Eq. 3) – requires spatial derivatives of temperature field.
# This is a placeholder; in a real setting you would need to output a spatial temperature map.
def physics_loss(T: torch.Tensor, sigma_residual: torch.Tensor,
                 kappa: float, E: float, alpha_therm: float, delta_T: float,
                 lambda1: float = 1.0, lambda2: float = 1.0) -> torch.Tensor:
    """
    Compute physics-informed loss terms.
    Note: This function assumes we have access to temperature gradient and Laplacian.
          Here we use random placeholders – replace with actual derivatives.
    """
    # Placeholders: these should be computed from a spatial temperature field
    grad_T = torch.rand_like(T) * 0.01
    laplacian_T = torch.rand_like(T) * 0.01

    term1 = torch.mean(torch.abs(grad_T - kappa * laplacian_T))
    term2 = torch.mean(torch.abs(sigma_residual - E * alpha_therm * delta_T * laplacian_T))
    return lambda1 * term1 + lambda2 * term2