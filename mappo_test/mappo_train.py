import numpy as np
import torch
from torch.distributions import Categorical

def multi_obs_to_state(multi_obs_dict):
    state = np.array([])
    for agent_obs in multi_obs_dict.values():
        state = np.concatenate([state, agent_obs])
    return state


def multi_obs_to_obs(multi_obs_dict, agent_name):
    obs = multi_obs_dict[agent_name]
    return obs


class MappoTrain:
    def __init__(self, env, agents, num_episodes, batch_size, num_steps, ):
        self.env = env
        self.agents = agents  # agents类
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_agents = self.env.num_agents

    def training_loop(self):  #训练循环。
        for episode_i in range(self.num_episodes):
            data_buffer = {}
            for batch_i in range(self.batch_size):
                trajectory = []

                h_pi = [torch.zeros(1, self.agents.agents_list[i].hidden_dim) for i in range(self.num_agents)]
                h_v = [torch.zeros(1, self.agents.agents_list[i].hidden_dim) for i in range(self.num_agents)]

                multi_obs, info = self.env.reset()
                state = multi_obs_to_state(multi_obs)

                for step_i in range(self.num_steps):

                    state_list = []

                    obs_list = []
                    h_pi_list = []
                    h_v_list = []
                    log_action_list = []

                    reward_list = []
                    next_state_list = []
                    next_obs_list = []

                    action_dict = {}    # 用于和环境交互。

                    with torch.no_grad():
                        for agent_i in range(self.num_agents):

                            obs_i = multi_obs_to_obs(multi_obs, self.env.agents[agent_i])
                            obs_i_tensor = torch.tensor(obs_i).float()
                            h_pi_i_tensor = torch.tensor(h_pi[agent_i]).float()
                            pi_i, h_pi_out = self.agents.agents_list[agent_i].get_pi_h(obs_i_tensor, h_pi_i_tensor)
                            raw_action_i = pi_i.sample()
                            action_i = torch.sigmoid(raw_action_i)
                            action_dict[self.agents.agents_list[agent_i].agent_name] = action_i
                            value_i, h_v_out = self.agents.critics_list[agent_i].forward(state, h_v[agent_i])

                            obs_list.append(obs_i)
                            h_pi_list.append(h_pi_out)
                            h_v_list.append(h_v_out)

