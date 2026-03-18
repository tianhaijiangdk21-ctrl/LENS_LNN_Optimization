# Physics-Informed Liquid Neural Network + PPO for Laser Cladding Optimization

This repository implements a reinforcement learning framework for optimizing laser cladding (LENS) process parameters. It combines a **Physics-Informed Liquid Neural Network (LNN)** with **Proximal Policy Optimization (PPO)**.

## Features

- LNN with thermal stress modulated weights (Eqs. 1-2 in the paper)
- Physics-informed loss (Fourier heat conduction + thermo-mechanical constraint)
- Custom reward function balancing height accuracy, crack risk, and energy consumption (Eq. 6)
- Continuous action space: adjustments to laser power, scanning speed, powder feed rate
- PPO training with GAE and clipped surrogate loss

## Project Structure
