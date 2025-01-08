import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_adversary_v3

from mappo_test.mappo_agent import DataBuffer


def multi_obs_to_state(multi_obs_dict):
    state = np.array([])
    for agent_obs in multi_obs_dict.values():
        state = np.concatenate([state, agent_obs])
    return state


def multi_obs_to_obs(multi_obs_dict, agent_name):
    obs = multi_obs_dict[agent_name]
    return obs

def reshape_list_to_tensor(list_to_reshape):
    tensor_to_reshape = torch.stack(list_to_reshape)
    tensor_reshape = tensor_to_reshape.view(-1, tensor_to_reshape.shape[-1])
    return tensor_reshape

class MappoTrain:
    def __init__(self, env, agents, num_episodes, batch_size, chunk_size, mini_batch_size, num_steps, gamma, lam, clip_range):
        self.env = env
        self.agents = agents  # agents类
        self.num_episodes = num_episodes
        self.batch_size = batch_size * (num_steps // chunk_size)
        self.chunk_size = chunk_size
        self.mini_batch_size = mini_batch_size * (num_steps // chunk_size)
        self.num_steps = num_steps
        self.num_agents = self.env.num_agents
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range

        self.scenario = "simple_adversary_v3"
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.agent_path = self.current_path + '/models/' + self.scenario + '/'
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.highest_reward = 0

        self.episode_rewards = []

    def decide_save_model(self,episode_i, episode_reward):
        if episode_i == 0:
            self.highest_reward = episode_reward
        if episode_reward > self.highest_reward:
            self.highest_reward = episode_reward
            print(f"--------------Highest reward updated: {round(self.highest_reward, 4)}--------------")
            print(f"Saving model at episode {episode_i}")
            for agent_i in range(self.num_agents):
                agent = self.agents.agents_list[agent_i]
                flag = os.path.exists(self.agent_path)
                if not flag:
                    os.makedirs(self.agent_path)
                torch.save(agent.actor.state_dict(),
                           f'{self.agent_path}' + f'agent_{agent_i}_actor_{self.scenario}_{self.timestamp}.pth')
            print(f"Model saved at episode {episode_i}")

    def test_model(self, episode_i):
        if (episode_i + 1) % 10 == 0:
            print("Testing!")
            env = simple_adversary_v3.parallel_env(N=2,
                                                   max_cycles=self.num_steps,  # max steps per episode
                                                   continuous_actions=True,
                                                   render_mode="human")
            for test_epi_i in range(5):
                multi_obs, others = env.reset()
                h_pi = [torch.zeros(self.agents.agents_list[i].hidden_dim) for i in range(self.num_agents)]
                for step_i in range(self.num_steps):
                    multi_action = {}
                    for agent_i, agent_name in enumerate(self.agents.agents_name_list):
                        agent = self.agents.agents_list[agent_i]
                        single_obs = multi_obs[agent_name]
                        single_pi, h_pi_update = agent.get_pi_h(torch.tensor(single_obs), torch.tensor(h_pi[agent_i]))
                        single_action = torch.sigmoid(single_pi.sample())   # take action based on obs
                        h_pi[agent_i] = h_pi_update
                        multi_action[agent_name] = single_action
                    multi_next_obs, multi_reward, multi_done, multi_truncation, info = env.step(multi_action)
                    multi_obs = multi_next_obs

    def save_plt_reward(self):
        np.savetxt(self.current_path + f'/epi_reward_{self.scenario}_{self.timestamp}.txt', self.episode_rewards)

        plt.plot(self.episode_rewards, color='purple', alpha=1)

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Pettingzoo: {self.scenario}')
        plt.legend()
        plt.savefig(f"reward_{self.scenario}_{self.timestamp}.png", format='png')
        plt.grid()
        plt.show()

    def training_loop(self):  #训练循环。
        for episode_i in range(self.num_episodes):
            print(f'Episode {episode_i + 1}')
            data_buffer = DataBuffer()
            for batch_i in range(self.batch_size):
                trajectory = []

                h_pi = [torch.zeros(self.agents.agents_list[i].hidden_dim) for i in range(self.num_agents)]
                h_v = [torch.zeros(self.agents.agents_list[i].hidden_dim) for i in range(self.num_agents)]

                multi_obs, info = self.env.reset()
                state = multi_obs_to_state(multi_obs)

                for step_i in range(self.num_steps):

                    state_record = state
                    obs_dict = {}
                    h_pi_dict = {}
                    h_v_dict = {}
                    value_dict = {}
                    log_action_dict = {}

                    reward_dict = {}
                    next_obs_dict = {}

                    action_dict = {}    # 用于和环境交互。

                    with torch.no_grad():



                        for agent_i in range(self.num_agents):
                            obs_i = multi_obs_to_obs(multi_obs, self.env.agents[agent_i])
                            obs_i_tensor = torch.tensor(obs_i).float()
                            h_pi_i_tensor = torch.tensor(h_pi[agent_i]).float()
                            pi_i, h_pi_out = self.agents.agents_list[agent_i].get_pi_h(obs_i_tensor, h_pi_i_tensor)
                            raw_action_i = pi_i.sample()
                            action_i = torch.sigmoid(raw_action_i)
                            clamped_action_i = torch.clamp(action_i, min=1e-8)
                            log_action_i = torch.log(clamped_action_i)
                            h_v_i_tensor = torch.tensor(h_v[agent_i]).float()
                            state_tensor = torch.tensor(state).float()
                            value_i, h_v_out = self.agents.critics_list[agent_i].forward(state_tensor, h_v_i_tensor)

                            h_pi_dict[self.agents.agents_name_list[agent_i]] = h_pi_out.numpy()
                            h_v_dict[self.agents.agents_name_list[agent_i]] = h_v_out.numpy()
                            value_dict[self.agents.agents_name_list[agent_i]] = value_i.numpy()
                            log_action_dict[self.agents.agents_name_list[agent_i]] = log_action_i.numpy()
                            action_dict[self.agents.agents_name_list[agent_i]] = action_i.numpy()

                        obs_dict.update(multi_obs)

                        multi_obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
                        dones = any(a or b for a, b in zip(terminations.values(), truncations.values()))
                        state = multi_obs_to_state(multi_obs)

                        reward_dict.update(rewards)
                        next_state_record = state
                        next_obs_dict.update(multi_obs)

                    trajectory.append({
                        'state' : state_record,
                        'obs_dict' : obs_dict,
                        'h_pi_dict' : h_pi_dict,
                        'h_v_dict' : h_v_dict,
                        'value_dict' : value_dict,
                        'log_action_dict' : log_action_dict,
                        'reward_dict' : reward_dict,
                        'next_state_dict' : next_state_record,
                        'next_obs_dict' : next_obs_dict,
                        'dones' : dones,
                    })

                    if dones:
                        break

                # 1. 初始化返回和优势函数
                advantage_dicts = [{agent: 0.0 for agent in self.agents.agents_name_list} for _ in range(len(trajectory))]
                return_dicts = [{agent: 0.0 for agent in self.agents.agents_name_list} for _ in range(len(trajectory))]
                last_advantage = {agent: 0.0 for agent in self.agents.agents_name_list}
                last_return = {agent: 0.0 for agent in self.agents.agents_name_list}

                # 2. 反向遍历 trajectory
                for t in reversed(range(len(trajectory))):
                    reward_dict = trajectory[t]['reward_dict']
                    value_dict = trajectory[t]['value_dict']
                    dones = trajectory[t]['dones']

                    next_value_dict = trajectory[t+1]['value_dict'] if t < len(trajectory) - 1 and not dones \
                        else {agent: 0.0 for agent in self.agents.agents_name_list}

                    for agent in self.agents.agents_name_list:
                        delta = reward_dict[agent] + self.gamma * next_value_dict[agent] - value_dict[agent]

                        if dones:
                            last_return[agent] = reward_dict[agent]
                            last_advantage[agent] = delta
                        else:
                            last_return[agent] = reward_dict[agent] + self.gamma * last_return[agent]
                            last_advantage[agent] = delta + self.gamma * self.lam * last_advantage[agent]
                        return_dicts[t][agent] = last_return[agent]
                        advantage_dicts[t][agent] = last_advantage[agent]

                start_idx = 0
                while start_idx < len(trajectory):
                    end_idx = start_idx
                    while end_idx < len(trajectory) and not trajectory[end_idx]['dones']:
                        end_idx = end_idx + 1

                    for agent in self.agents.agents_name_list:
                        returns = [return_dicts[t][agent] for t in range(start_idx, end_idx + 1)]
                        advantages = [advantage_dicts[t][agent] for t in range(start_idx, end_idx + 1)]
                        mean_return = np.mean(returns)
                        mean_advantage = np.mean(advantages)
                        std_return = np.std(returns) + 1e-8
                        std_advantage = np.std(advantages) + 1e-8

                        for t in range(start_idx, end_idx + 1):
                            return_dicts[t][agent] = (returns[t] - mean_return) / std_return
                            advantage_dicts[t][agent] = (advantages[t] - mean_advantage) / std_advantage

                    start_idx = end_idx + 1

                for t in range(len(trajectory)):
                    trajectory[t]['advantage_dict'] = advantage_dicts[t]
                    trajectory[t]['return_dict'] = return_dicts[t]

                data_buffer.add_chunk(trajectory, self.chunk_size)

            data_buffer.shuffle()

            reward_buffer = []
            for k in range(self.batch_size//self.mini_batch_size):

                start_idx = k * self.mini_batch_size
                mini_batch_buffer = data_buffer.sample_minibatch(self.mini_batch_size, start_idx)

                for agent_i in range(self.num_agents):

                    ratio_batch = []
                    advantage_batch = []
                    log_action_batch = []

                    value_old_batch = []
                    value_new_batch = []
                    return_batch = []


                    for chunk in mini_batch_buffer:
                        obs_chunk = []
                        h_pi_chunk = []

                        state_chunk = []
                        h_v_chunk = []
                        value_old_chunk = []
                        return_chunk = []

                        advantage_chunk = []
                        log_action_chunk = []
                        for tao in chunk:
                            obs_train = torch.tensor(tao['obs_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            obs_chunk.append(obs_train)

                            h_pi_train = torch.tensor(tao['h_pi_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            h_pi_chunk.append(h_pi_train)

                            log_action = torch.tensor(tao['log_action_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            log_action_chunk.append(log_action)

                            advantage_train = torch.tensor(tao['advantage_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            advantage_train_expanded = advantage_train.unsqueeze(0).expand(1, self.agents.agents_list[agent_i].action_dim)
                            advantage_chunk.append(advantage_train_expanded)

                            state_train = torch.tensor(tao['state'], dtype=torch.float, requires_grad=True)
                            state_chunk.append(state_train)

                            value_train = torch.tensor(tao['value_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            value_old_chunk.append(value_train)

                            return_train = torch.tensor(tao['return_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            return_chunk.append(return_train)

                            h_v_train = torch.tensor(tao['h_v_dict'][self.agents.agents_name_list[agent_i]], dtype=torch.float, requires_grad=True)
                            h_v_chunk.append(h_v_train)

                        obs_chunk_tensor = torch.stack(obs_chunk)
                        h_pi_chunk_tensor = torch.stack(h_pi_chunk)

                        log_action_chunk_tensor = torch.stack(log_action_chunk)
                        pi_out, _ = self.agents.agents_list[agent_i].get_pi_h(obs_chunk_tensor, h_pi_chunk_tensor)
                        raw_action_out = pi_out.sample()
                        action_out = torch.sigmoid(raw_action_out)
                        clamped_action_out = torch.clamp(action_out, min=1e-8)
                        log_action_new_chunk_tensor = torch.log(clamped_action_out)
                        ratio_chunk = torch.exp(log_action_new_chunk_tensor - log_action_chunk_tensor)
                        ratio_batch.append(ratio_chunk)

                        advantage_chunk_tensor = torch.stack(advantage_chunk)
                        advantage_batch.append(advantage_chunk_tensor)

                        log_action_batch.append(log_action_new_chunk_tensor)

                        state_chunk_tensor = torch.stack(state_chunk)
                        h_v_chunk_tensor = torch.stack(h_v_chunk)

                        value_new_chunk,_ = self.agents.critics_list[agent_i].forward(state_chunk_tensor, h_v_chunk_tensor)
                        value_new_batch.append(value_new_chunk)

                        value_old_chunk_tensor = torch.stack(value_old_chunk)
                        value_old_batch.append(value_old_chunk_tensor)

                        return_chunk_tensor = torch.stack(return_chunk)
                        return_batch.append(return_chunk_tensor)

                    ratio_batch_tensor_reshape = reshape_list_to_tensor(ratio_batch).float()
                    clipped_ratio_batch_tensor_reshape = torch.clamp(ratio_batch_tensor_reshape, min=1-self.clip_range, max=1+self.clip_range)
                    advantage_batch_tensor_reshape = reshape_list_to_tensor(advantage_batch).float()
                    log_action_batch_tensor_reshape = reshape_list_to_tensor(log_action_batch).float()

                    value_old_batch_tensor_reshape = reshape_list_to_tensor(value_old_batch).float()
                    value_old_clipped_batch_tensor_reshape = torch.clamp(value_old_batch_tensor_reshape, min= value_old_batch_tensor_reshape -self.clip_range, max= value_old_batch_tensor_reshape + self.clip_range)
                    value_new_batch_tensor_reshape = reshape_list_to_tensor(value_new_batch).float()
                    return_batch_tensor_reshape = torch.stack(return_batch).float().view(-1, 1)

                    loss_actor = (-torch.mean(torch.sum(torch.min(ratio_batch_tensor_reshape*advantage_batch_tensor_reshape,
                                                                  clipped_ratio_batch_tensor_reshape*advantage_batch_tensor_reshape*advantage_batch_tensor_reshape)),dim= -1)
                                  + 0.01 * torch.mean(torch.sum(log_action_batch_tensor_reshape*torch.exp(log_action_batch_tensor_reshape), dim=-1)))
                    self.agents.agents_list[agent_i].actor.optimizer.zero_grad()
                    loss_actor.backward()
                    self.agents.agents_list[agent_i].actor.optimizer.step()

                    loss_critic = torch.max(nn.MSELoss()(return_batch_tensor_reshape, value_new_batch_tensor_reshape), nn.MSELoss()(value_old_clipped_batch_tensor_reshape, return_batch_tensor_reshape))
                    self.agents.critics_list[agent_i].optimizer.zero_grad()
                    loss_critic.backward()
                    self.agents.critics_list[agent_i].optimizer.step()

                    reward_buffer.append(torch.mean(return_batch_tensor_reshape).item())

            episode_reward = np.sum(reward_buffer) / (self.batch_size//self.mini_batch_size) * 1e9
            self.episode_rewards.append(episode_reward)
            print(f'episode_reward:{episode_reward}')

            self.decide_save_model(episode_i, episode_reward)
            self.test_model(episode_i)

        self.env.close()
        self.save_plt_reward()















