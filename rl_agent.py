"""
Actor-Critic network and PPO trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Shared backbone with separate actor (mean, log_std) and critic heads.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        value = self.critic(x)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        """
        Returns: action, log_prob, value
        """
        mean, std, value = self.forward(state)
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value


class PPO:
    """
    Proximal Policy Optimization trainer with GAE and clipped surrogate.
    """
    def __init__(self,
                 actor_critic: ActorCritic,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 64):
        self.model = actor_critic
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = next(actor_critic.parameters()).device

    def compute_gae(self, rewards, dones, values, next_value):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, trajectories):
        """
        trajectories: dict with keys:
            'states': list of tensors [1, state_dim]
            'actions': list of tensors [1, action_dim]
            'log_probs': list of tensors [1, 1]
            'rewards': list of float
            'dones': list of float (0 or 1)
            'next_value': float (value of last state)
        """
        states = torch.cat(trajectories['states']).to(self.device)
        actions = torch.cat(trajectories['actions']).to(self.device)
        old_log_probs = torch.cat(trajectories['log_probs']).to(self.device)
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        next_value = trajectories['next_value']

        # Compute values for all states
        with torch.no_grad():
            _, _, values = self.model(states)
            values = values.squeeze(-1).cpu().numpy().tolist()
        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        advantages = torch.tensor(advantages, device=self.device).float()
        returns = torch.tensor(returns, device=self.device).float()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch in loader:
                s, a, old_log_p, ret, adv = batch
                mean, std, value = self.model(s)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_prob - old_log_p)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(value.squeeze(-1), ret)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()