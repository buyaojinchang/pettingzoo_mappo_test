import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, hidden_dims, output_dims):
        super(Actor, self).__init__()
        self.fc = nn.Linear(input_dims, hidden_dims)
        self.rnn = nn.GRUCell(hidden_dims, hidden_dims)
        self.mean_head = nn.Linear(hidden_dims, output_dims)
        self.log_std_head = nn.Linear(hidden_dims, output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, obs, h_in):
        x = torch.relu(self.fc(obs))
        h_out = self.rnn(x, h_in)
        mean = self.mean_head(h_out)
        log_std_out = self.log_std_head(h_out)
        std = torch.exp(log_std_out)
        return mean, std, h_out

class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, hidden_dims):
        super(Critic, self).__init__()
        self.fc = nn.Linear(input_dims, hidden_dims)
        self.rnn = nn.GRUCell(hidden_dims, hidden_dims)
        self.pi = nn.Linear(hidden_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, h_in):
        x = torch.relu(self.fc(state))
        h_out = self.rnn(x, h_in)
        value = self.pi(h_out)
        return value, h_out


class Agent(object):
    def __init__(self, env, agent_name,lr_actor,
                 gamma, lam, clip_range, hidden_dim):
        self.env = env
        self.agent_name = agent_name
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.obs_dim = self.get_obs_dim()
        self.action_dim = self.get_action_dim()
        self.actor = Actor(lr_actor=lr_actor, input_dims=self.obs_dim,
                           hidden_dims=self.hidden_dim, output_dims=self.action_dim)

    def get_obs_dim(self):
        obs_dim = self.env.observation_spaces[self.agent_name].shape[0]
        return obs_dim
    def get_action_dim(self):
        action_dim = self.env.action_spaces[self.agent_name].shape[0]
        return action_dim
    def get_pi_h(self, obs, h_in):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0)
        single_h_in = torch.tensor(data=h_in, dtype=torch.float).unsqueeze(0)
        mean,std, h_out = self.actor.forward(obs=single_obs, h_in=single_h_in)
        pi = Normal(mean, std)
        return pi, h_out

class Agents(object):
    def __init__(self,agents_list, lr_critic, hidden_dim):
        self.agents_list = agents_list
        self.lr_critic = lr_critic
        self.state_dim = self.get_state_dim()
        self.critics_list = [Critic(lr_critic, self.state_dim, hidden_dim) for _ in range(len(self.agents_list))]

    def get_state_dim(self):
        state_dim = 0
        for agent in self.agents_list:
            state_dim += agent.obs_dim
        return state_dim




