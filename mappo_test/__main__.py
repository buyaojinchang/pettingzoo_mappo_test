from pettingzoo.mpe import simple_adversary_v3
from mappo_agent import Agent
from mappo_agent import Agents
from mappo_test.mappo_train import MappoTrain
import torch



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Hyper parameters
NUM_EPISODES = 100
NUM_STEPS = 50
BATCH_SIZE = 512
CHUNK_SIZE = 25
MINI_BATCH_SIZE = 64
K = 8

LR_CRITIC = 0.01
LR_ACTOR = 0.01
GAMMA = 0.99
LAM = 0.95
HIDDEN_DIM = 128
CLIP_RANGE = 0.2


if __name__ == "__main__":
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    multi_obs, info = env.reset()
    agents_list = []
    for agent_i in range(env.num_agents):
        agent = Agent(env=env, agent_name=env.agents[agent_i], lr_actor=LR_ACTOR, hidden_dim=HIDDEN_DIM)
        print(f"Initializing agent {agent.agent_name}")
        agents_list.append(agent)    # TODO
    agents = Agents(agents_list, LR_CRITIC, HIDDEN_DIM)
    train = MappoTrain(env=env, agents=agents, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE, mini_batch_size=MINI_BATCH_SIZE,
                       num_steps=NUM_STEPS, gamma=GAMMA, lam=LAM, clip_range=CLIP_RANGE)  # TODO
    train.training_loop()