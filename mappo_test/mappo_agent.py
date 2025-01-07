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
    def __init__(self, env, agent_name,lr_actor, hidden_dim):
        self.env = env
        self.agent_name = agent_name
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
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
        obs = torch.tensor(data=obs, dtype=torch.float)
        h_in = torch.tensor(data=h_in, dtype=torch.float)
        mean,std, h_out = self.actor.forward(obs=obs, h_in=h_in)
        pi = Normal(mean, std)
        return pi, h_out

class Agents(object):
    def __init__(self,agents_list, lr_critic, hidden_dim):
        self.agents_list = agents_list
        self.agents_name_list = self.get_agents_name()
        self.lr_critic = lr_critic
        self.state_dim = self.get_state_dim()
        self.critics_list = [Critic(lr_critic, self.state_dim, hidden_dim) for _ in range(len(self.agents_list))]

    def get_agents_name(self):
        agent_name_list = []
        for agent_i in self.agents_list:
            agent_name_list.append(agent_i.agent_name)
        return agent_name_list

    def get_state_dim(self):
        state_dim = 0
        for agent in self.agents_list:
            state_dim += agent.obs_dim
        return state_dim

class DataBuffer(object):
    def __init__(self):
        self.buffer = []
        self.shuffle_index = []

    def shuffle(self):
        self.shuffle_index = np.random.permutation(len(self.buffer))

    def sample_minibatch(self, mini_batch_size, start_idx=0):
        end_idx = min(start_idx + mini_batch_size, len(self.shuffle_index))
        sampled_indices = self.shuffle_index[start_idx:end_idx]
        return [self.buffer[i] for i in sampled_indices]

    def add_chunk(self, trajectory, chunk_size):
        for i in range(0, len(trajectory), chunk_size):
            self.buffer.append(trajectory[i:i + chunk_size])







