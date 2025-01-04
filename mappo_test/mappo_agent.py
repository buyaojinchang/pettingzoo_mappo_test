import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims,
                 n_agent, action_dim):
        super(Critic, self).__init__()


class Agent(object):
    def __init__(self, env, agent_name,lr_actor, lr_critic,
                 gamma, lam, clip_range, hidden_dim):
        self.env = env
        self.agent_name = agent_name
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.obs_dim = self.get_obs_dim()
        self.action_dim = self.get_action_dim()

    def get_obs_dim(self):
        obs_dim = self.env.observation_spaces[self.agent_name].shape[0]
        return obs_dim
    def get_action_dim(self):
        action_dim = self.env.action_spaces[self.agent_name].shape[0]
        return action_dim

class Agents(Agent):
    def __init__(self, ):
        super(Agents, self).__init__()
        self.agents = []

    def initialize_agents(self, multi_obs_dict, ):
        # 1.1 Get obs_dim, state_dim
        obs_dim_list = []
        for agent_obs in multi_obs_dict.values():
            obs_dim_list.append(agent_obs.shape[0])
        state_dim = sum(obs_dim)

        # 1.2 Get action_dim
        action_dim_list = []
        for agent_name in agent_names_list:
            action_dim_list.append(env.action_space(agent_name).sample().shape[0])

        # 1.3 init all agents
        agents = []
        for agent_i in range(agent_num):
            print(f"Initializing agent {agent_i}")
            agent = Agent()    # TODO
            agents.append(agent)

        return agents, obs_dim_list, action_dim_list, state_dim

    def multi_obs_to_state(multi_obs):
        state = np.array([])
        for agent_obs in multi_obs.values():
            state = np.concatenate([state, agent_obs])
        return state


