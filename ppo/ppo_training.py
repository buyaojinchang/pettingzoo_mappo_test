import gym
import numpy as np
from tensorboardX import SummaryWriter
import os
import time
import torch
from ppo_agent_good import PPOAgent

# Initialize environment
scenario = "Pendulum-v1"
env = gym.make(id=scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Directory for saving models
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# Hyperparameters
NUM_EPISODE = 1000
NUM_STEP = 200  # Each episode involves 200 steps in Pendulum-v1
UPDATE_INTERVAL = 50
BATCH_SIZE = 25

# Initialize agent
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_reward = -2000
writer = SummaryWriter('ppo_logs')  # TODO:writer

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # total_step = episode_i * NUM_STEP + step_i + 1
        action, value = agent.get_action(state)
        next_state, reward, done, truncation, info = env.step(action)
        episode_reward += reward
        # reward = (reward + 8.1) / 8.1
        done = True if step_i == NUM_STEP - 1 else False
        agent.replay_buffer.add_memo(state, action, reward, value, done)

        state = next_state

        if (step_i + 1) % UPDATE_INTERVAL == 0 or step_i == NUM_STEP - 1:
            agent.update()

    if episode_reward >= -100 and episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model + f'ppo_actor_{timestamp}.pth')
        print(f"Best reward: {best_reward}!")

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}")

    writer.add_scalar(tag="Episode reward", scalar_value=episode_reward, global_step=episode_i)  # TODO:writer


env.close()

# Save the rewards as txt file
np.savetxt(current_path + f'/ppo_reward_{timestamp}.txt', REWARD_BUFFER)
