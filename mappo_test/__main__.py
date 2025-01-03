from pettingzoo.mpe import simple_adversary_v3
from mappo_agent import Agents
from mappo_test.mappo_train import MappoTrain
import torch



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Hyper parameters
NUM_EPISODES = 100
NUM_STEPS = 25
BATCH_SIZE = 512
LR_CRITIC = 0.01
LR_ACTOR = 0.01

if __name__ == "__main__":
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    agents = Agents()  # TODO
    train = MappoTrain(env, agents, NUM_EPISODES, BATCH_SIZE, NUM_STEPS)  # TODO
    train.training_loop()