"""
Main training loop: RL with PPO + Physics-Informed LNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from lnn_model import PhysicsInformedLNN
from environment import LaserCladdingEnv
from reward import compute_reward
from rl_agent import ActorCritic, PPO
import config
import utils


def train():
    # Set device
    device = utils.get_device()

    # Initialize LNN (load pretrained if available)
    lnn = PhysicsInformedLNN(input_dim=3, hidden_dim=32).to(device)
    # Optional: load pretrained weights
    # lnn.load_state_dict(torch.load('pretrained_lnn.pth'))

    # Create environment
    env = LaserCladdingEnv(lnn, dt=config.DT, target_height=config.TARGET_HEIGHT)

    # Create actor-critic and PPO
    ac = ActorCritic(state_dim=4, action_dim=3, hidden_dim=128).to(device)
    ppo = PPO(ac,
              lr=config.LR,
              gamma=config.GAMMA,
              lam=config.LAM,
              clip_eps=config.CLIP_EPS,
              epochs=config.PPO_EPOCHS,
              batch_size=config.BATCH_SIZE)

    # Logging
    episode_rewards = []
    best_reward = -np.inf

    for ep in range(config.NUM_EPISODES):
        state = env.reset()
        ep_reward = 0
        trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

        for step in range(config.MAX_STEPS):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action, log_prob, _ = ac.get_action(state_tensor)
            action_np = action.squeeze().cpu().detach().numpy()

            next_state, info, done = env.step(action_np)

            # Compute reward using reward.py
            delta_h = info['delta_h']
            crack_risk = info['crack_risk']
            P = info['P']
            # Convert to tensors for reward function
            delta_h_t = torch.tensor(delta_h, device=device)
            crack_risk_t = torch.tensor(crack_risk, device=device)
            P_t = torch.tensor(P, device=device)
            reward = compute_reward(delta_h_t, crack_risk_t, P_t, dt=config.DT).item()

            # Store trajectory
            trajectories['states'].append(state_tensor)
            trajectories['actions'].append(action)
            trajectories['log_probs'].append(log_prob.detach())
            trajectories['rewards'].append(reward)
            trajectories['dones'].append(float(done))

            state = next_state
            ep_reward += reward

            if done:
                break

        # Get value of final state
        last_state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        _, _, last_value = ac.get_action(last_state_tensor)
        trajectories['next_value'] = last_value.squeeze().cpu().item()

        # Update policy
        ppo.update(trajectories)

        episode_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(ac.state_dict(), 'best_actor_critic.pth')

        if (ep+1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {ep+1:4d} | Avg Reward (last 50): {avg_reward:.2f} | Best: {best_reward:.2f}")

    # Plot learning curve
    utils.plot_learning_curve(episode_rewards, save_path='reward_curve.png')
    print("Training completed.")


if __name__ == "__main__":
    train()