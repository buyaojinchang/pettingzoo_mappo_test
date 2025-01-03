# PPO for Pendulum-v1 control problem: test loop
# Born time: 2024-09-01
# Dylan

import gym
import torch
import torch.nn as nn
import os
import pygame
import numpy as np
from torch.distributions import Normal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Computing device: ", device)

env = gym.make('Pendulum-v1', render_mode='rgb_array')

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3

        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-2.0, 2.0)

        return action


# Function to convert Gym's image to a format Pygame can display
def process_frame(frame):
    frame = np.transpose(frame, (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame, (width, height))


# Test phase
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ppo_actor_20240902223343.pth'

actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))

# Initialize Pygame
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

num_episodes = 100
for episode in range(num_episodes):
    state, others = env.reset()
    episode_reward = 0
    done = False
    count = 0

    for step_i in range(200):
        action = actor.select_action(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
        next_state, reward, done, truncation, _ = env.step(action)
        episode_reward += reward
        state = next_state
        count += 1
        print(f"{count}:{action}")

        frame = env.render()  # Get the frame for rendering in rgb_array mode
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(60)  # FPS
        if done:
            state, others = env.reset()
            print(f"Test Episode: {episode + 1}, Reward: {episode_reward}")
            break

pygame.quit()
env.close()
