"""
Utility functions: device management, plotting, etc.
"""

import torch
import matplotlib.pyplot as plt


def get_device():
    """Return torch.device (cuda if available else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_learning_curve(rewards, save_path=None):
    """Plot episode rewards."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("RL Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)