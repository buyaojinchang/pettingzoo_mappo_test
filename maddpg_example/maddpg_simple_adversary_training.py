# Pettingzoo simple adversary using MADDPG: training
# Dylan
# 2024.2.28

from pettingzoo.mpe import simple_adversary_v3
import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt

from maddpg_simple_adversary_agent import Agent

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

NUM_EPISODE = 100
NUM_STEP = 25

LR_ACTOR = 0.01
LR_CRITIC = 0.01
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.01
MEMORY_SIZE = 100000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL = 100

PRINT_INTERVAL = 10
highest_reward = 0


# Write a function to convert multi_obs in dict type to state in dict type
def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_obs in multi_obs.values():
        state = np.concatenate([state, agent_obs])
    return state


env = simple_adversary_v3.parallel_env(N=2,
                                       max_cycles=NUM_STEP,  # max steps per episode
                                       continuous_actions=True)

multi_obs, others = env.reset()
NUM_AGENT = env.num_agents
agent_name_list = env.agents

# 1 Initialize agents
# 1.1 Get obs_dim, state_dim
obs_dim = []
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])
state_dim = sum(obs_dim)

# 1.2 Get action_dim
action_dim = []
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])

# 1.3 init all agents
agents = []
for agent_i in range(NUM_AGENT):
    print(f"Initializing agent {agent_i}")
    agent = Agent(memo_size=MEMORY_SIZE, obs_dim=obs_dim[agent_i], state_dim=state_dim, n_agent=NUM_AGENT,
                  action_dim=action_dim[agent_i], alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM,
                  fc2_dims=HIDDEN_DIM, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)
    agents.append(agent)

# Save models
scenario = "simple_adversary_v3"
print(f"Scenario: {scenario}")
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + '/models/' + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# 2 Training loop
EPISODE_REWARD_BUFFER = []
for episode_i in range(NUM_EPISODE):
    multi_obs, info = env.reset()
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}
    for step_i in range(NUM_STEP):
        total_step = episode_i * NUM_STEP + step_i

        # 2.1 Collect actions from all agents
        multi_action = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_action = agent.get_action(single_obs)  # take action based on obs
            multi_action[agent_name] = single_action

        # 2.2 Execute action at and observe reward rt and new state st+1
        multi_next_obs, multi_reward, multi_done, multi_truncation, info = env.step(multi_action)

        state = multi_obs_to_state(multi_obs)
        next_state = multi_obs_to_state(multi_next_obs)  # why the same as state?

        if step_i >= NUM_STEP - 1:
            multi_done = {agent_name: True for agent_name in agent_name_list}

        # 2.3 Add memory (obs, next_obs, state, next_state, action, reward, done)
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_next_obs = multi_next_obs[agent_name]
            single_action = multi_action[agent_name]  # 5 continuous actions
            single_reward = multi_reward[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memo(single_obs, single_next_obs, state, next_state, single_action, single_reward,
                                         single_done)

        # 2.4 Update target networks every TARGET_UPDATE_INTERVAL

        # Start learning
        # Collect next actions of all agents
        multi_batch_obses = []
        multi_batch_next_obses = []
        multi_batch_states = []
        multi_batch_next_states = []
        multi_batch_actions = []
        multi_batch_next_actions = []
        multi_batch_online_actions = []
        multi_batch_rewards = []
        multi_batch_dones = []

        # 2.4.1 Sample a batch of memories
        current_memo_size = min(MEMORY_SIZE, total_step + 1)
        if current_memo_size < BATCH_SIZE:
            batch_idx = range(0, current_memo_size)
        else:
            batch_idx = np.random.choice(current_memo_size,
                                         BATCH_SIZE)  # Pick up the same batch indexes for all agents
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses, batch_next_obses, batch_states, batch_next_states, \
                batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample(
                batch_idx)

            # Single + batch
            batch_obses_tensor = torch.tensor(batch_obses, dtype=torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses, dtype=torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float).to(device)
            batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float).to(device)

            # Multiple + batch
            multi_batch_obses.append(batch_obses_tensor)
            multi_batch_next_obses.append(batch_next_obses_tensor)
            multi_batch_states.append(batch_states_tensor)
            multi_batch_next_states.append(batch_next_states_tensor)
            multi_batch_actions.append(batch_actions_tensor)

            single_batch_next_action = agent.target_actor.forward(batch_next_obses_tensor)  # a' = target(o')
            multi_batch_next_actions.append(single_batch_next_action)

            single_batch_online_action = agent.actor.forward(batch_obses_tensor)  # a = online(o)
            multi_batch_online_actions.append(single_batch_online_action)

            multi_batch_rewards.append(batch_rewards_tensor)
            multi_batch_dones.append(batch_dones_tensor)

        multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim=1).to(device)
        multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim=1).to(device)
        multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim=1).to(device)

        # 2.4.2 Update critic and actor
        if (total_step + 1) % TARGET_UPDATE_INTERVAL == 0:

            for agent_i in range(NUM_AGENT):
                agent = agents[agent_i]

                # # Update actor and critic learning rates
                # for param_group in agent.actor.optimizer.param_groups:
                #     param_group['lr'] = lr_actor
                # for param_group in agent.critic.optimizer.param_groups:
                #     param_group['lr'] = lr_critic

                batch_obses_tensor = multi_batch_obses[agent_i]
                batch_states_tensor = multi_batch_states[agent_i]
                batch_next_states_tensor = multi_batch_next_states[agent_i]
                batch_rewards_tensor = multi_batch_rewards[agent_i]
                batch_dones_tensor = multi_batch_dones[agent_i]
                batch_actions_tensor = multi_batch_actions[agent_i]

                # 2.4.2.1 Calculate target Q using target critic
                critic_target_q = agent.target_critic.forward(batch_next_states_tensor,
                                                              multi_batch_next_actions_tensor.detach())
                y = (batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q).flatten()

                # 2.4.2.2 Calculate current Q using critic
                critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor).flatten()

                # 2.4.2.3 Update critic
                critic_loss = nn.MSELoss()(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                agent.critic.optimizer.step()

                # 2.4.2.4 Update actor
                actor_loss = agent.critic.forward(batch_states_tensor,
                                                  multi_batch_online_actions_tensor.detach()).flatten()
                actor_loss = -torch.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                # 2.4.2.5 Update target critic
                for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                # 2.4.2.6 Update target actor
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum([single_reward for single_reward in multi_reward.values()])

    EPISODE_REWARD_BUFFER.append(episode_reward)
    # print(f"Episode: {episode_i} Reward: {round(episode_reward, 4)}")

    # 3 Render the environment
    if (episode_i + 1) % 100 == 0:
        print("Testing!")
        env = simple_adversary_v3.parallel_env(N=2,
                                               max_cycles=NUM_STEP,  # max steps per episode
                                               continuous_actions=True,
                                               render_mode="human")
        for test_epi_i in range(10):
            multi_obs, others = env.reset()
            for step_i in range(NUM_STEP):
                multi_action = {}
                for agent_i, agent_name in enumerate(agent_name_list):
                    agent = agents[agent_i]
                    single_obs = multi_obs[agent_name]
                    single_action = agent.get_action(single_obs)  # take action based on obs
                    multi_action[agent_name] = single_action
                multi_next_obs, multi_reward, multi_done, multi_truncation, info = env.step(multi_action)
                multi_obs = multi_next_obs

    # 4 Save agents' models
    if episode_i == 0:
        highest_reward = episode_reward
    if episode_reward > highest_reward:
        highest_reward = episode_reward
        print(f"--------------Highest reward updated: {round(highest_reward, 4)}--------------")
        print(f"Saving model at episode {episode_i}")
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(agent.actor.state_dict(), f'{agent_path}' + f'agent_{agent_i}_actor_{scenario}_{timestamp}.pth')
        print(f"Model saved at episode {episode_i}")


env.close()

# 5 Save the rewards
reward_path = os.path.join(current_path, "model/" + scenario + f'/reward_{timestamp}.csv')
np.savetxt(current_path + f'/epi_reward_{scenario}_{timestamp}.txt', EPISODE_REWARD_BUFFER)

# 6 Plot the rewards
plt.plot(EPISODE_REWARD_BUFFER, color='purple', alpha=1)


plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Pettingzoo: {scenario}')
plt.legend()
plt.savefig(f"reward_{scenario}_{timestamp}.png", format='png')
plt.grid()
plt.show()
