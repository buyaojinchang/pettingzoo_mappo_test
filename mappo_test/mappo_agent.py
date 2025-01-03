from pettingzoo.mpe import simple_adversary_v3
import torch

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Write a function to convert multi_obs in dict type to state in dict type

env = simple_adversary_v3.parallel_env()
multi_obs, infos = env.reset()

# 1 Initialize agents
# 1.1 Get obs_dim, state_dim
obs_dim = []
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])
# 1.2 Get action_dim
# 1.3 init all agents

# Save models

# 2 Training loop
# 2.1 Collect actions from all agents
# 2.2 Execute action at and observe reward rt and new state st+1
# 2.3 Add memory (obs, next_obs, state, next_state, action, reward, done)
# 2.4 Update target networks every TARGET_UPDATE_INTERVAL

# Start learning
# Collect next actions of all agents
# 2.4.1 Sample a batch of memories
# Single + batch
# Multiple + batch

# 2.4.2 Update critic and actor
# # Update actor and critic learning rates
# 2.4.2.1 Calculate target Q using target critic
# 2.4.2.2 Calculate current Q using critic
# 2.4.2.3 Update critic
# 2.4.2.4 Update actor
# 2.4.2.5 Update target critic
# 2.4.2.6 Update target actor

# 3 Render the environment

# 4 Save agents' models

# 5 Save the rewards

# 6 Plot the rewards