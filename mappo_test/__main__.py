from pettingzoo.mpe import simple_adversary_v3
from mappo_agent import Agent

from mappo_test.mappo_train import MappoTrain
import torch



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Hyper parameters
NUM_EPISODES = 100
NUM_STEPS = 25
BATCH_SIZE = 512
K = 4

LR_CRITIC = 0.01
LR_ACTOR = 0.01
GAMMA = 0.99
LAM = 0.95
HIDDEN_DIM = 128
CLIP_RANGE = 0.2


if __name__ == "__main__":
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    multi_obs, info = env.reset()
    agents = []
    for agent_i in range(env.num_agents):
        Agent(env=env, agent_name=env.agents[agent_i], lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
              gamma=GAMMA, lam=LAM, clip_range=CLIP_RANGE, hidden_dim=HIDDEN_DIM)
        agents.append(Agent)    # TODO
    train = MappoTrain(env, agents, NUM_EPISODES, BATCH_SIZE, NUM_STEPS)  # TODO
    train.training_loop()