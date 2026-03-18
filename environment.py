"""
Laser cladding environment for RL.
State: [T_max, f_osc, crack_risk, rate_dep]   (4-dim)
Action: [ΔP, Δv, ΔF]   (continuous adjustments)
Uses LNN to predict crack risk.
"""

import numpy as np
import torch
import torch.nn as nn

from lnn_model import PhysicsInformedLNN


class LaserCladdingEnv:
    """
    Simulated environment for laser cladding process.
    """
    def __init__(self, lnn_model: PhysicsInformedLNN, dt: float = 0.1, target_height: float = 0.5):
        self.lnn = lnn_model
        self.dt = dt
        self.target_height = target_height
        self.device = next(lnn_model.parameters()).device

        # Initial process parameters
        self.P = 1.5          # laser power (kW)
        self.v = 8.0          # scanning speed (mm/s)
        self.F = 9.0          # powder feed rate (g/min)
        self.k = 0.6          # powder capture efficiency

        # Process variables
        self.layer_height = 0.0   # built layer height (mm)
        self.T_max = 1200.0        # peak temperature (°C)
        self.f_osc = 50.0          # melt pool oscillation frequency (Hz)

        # Action bounds
        self.action_bounds = {
            'P': (0.5, 3.0),      # kW
            'v': (4.0, 15.0),     # mm/s
            'F': (5.0, 15.0)      # g/min
        }

    def reset(self):
        """Reset to initial state."""
        self.P = 1.5
        self.v = 8.0
        self.F = 9.0
        self.layer_height = 0.0
        self.T_max = 1200.0
        self.f_osc = 50.0
        return self._get_state()

    def _get_state(self):
        """Construct state vector [T_max, f_osc, crack_risk, rate_dep]."""
        # Deposition rate (Eq. 5)
        rate_dep = self.k * self.F   # mm³/s (simplified)

        # Get crack risk from LNN
        with torch.no_grad():
            # Prepare inputs: static parameters (P, v, F) as a batch of 1
            static = torch.tensor([[self.P, self.v, self.F]], device=self.device).float()
            # Simulate a short input sequence (e.g., 5 time steps with constant values)
            I_seq = static.unsqueeze(1).repeat(1, 5, 1)          # [1, 5, 3]
            # Thermal stress sequence: proportional to power
            sigma_thermal = 0.1 * self.P * torch.ones(1, 5, 1, device=self.device)
            crack_risk, _ = self.lnn(static, I_seq, sigma_thermal)
            crack_risk = crack_risk.item()

        state = np.array([self.T_max, self.f_osc, crack_risk, rate_dep], dtype=np.float32)
        return state

    def step(self, action):
        """
        Apply action (ΔP, Δv, ΔF) and update state.
        action: numpy array of shape (3,) with values in [-1, 1] (normalized adjustments).
        Returns:
            next_state, reward, done, info
        """
        # Convert normalized action to actual changes (±20% of current value)
        delta_P = action[0] * 0.2 * self.P
        delta_v = action[1] * 0.2 * self.v
        delta_F = action[2] * 0.2 * self.F

        # Update and clip parameters
        self.P = np.clip(self.P + delta_P, *self.action_bounds['P'])
        self.v = np.clip(self.v + delta_v, *self.action_bounds['v'])
        self.F = np.clip(self.F + delta_F, *self.action_bounds['F'])

        # Simplified physics updates
        self.T_max = 1000 + 500 * (self.P / 1.5) * (8.0 / self.v)
        self.f_osc = 30 + 5 * self.v
        rate_dep = self.k * self.F
        self.layer_height += rate_dep * self.dt * 0.01   # small scaling factor

        # Get new crack risk
        with torch.no_grad():
            static = torch.tensor([[self.P, self.v, self.F]], device=self.device).float()
            I_seq = static.unsqueeze(1).repeat(1, 5, 1)
            sigma_thermal = 0.1 * self.P * torch.ones(1, 5, 1, device=self.device)
            crack_risk, _ = self.lnn(static, I_seq, sigma_thermal)
            crack_risk = crack_risk.item()

        # Compute height deviation
        delta_h = self.layer_height - self.target_height

        # Reward will be computed externally (by reward.py)
        # But we need to return something; we'll return state and let caller compute reward
        next_state = self._get_state()
        done = self.layer_height > 5.0   # episode ends after building ~5 mm
        info = {
            'delta_h': delta_h,
            'crack_risk': crack_risk,
            'P': self.P,
            'v': self.v,
            'F': self.F
        }
        return next_state, info, done