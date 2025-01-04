import numpy as np
import torch


class MappoTrain:
    def __init__(self, env, agents, num_episodes, batch_size, num_steps, ):
        self.env = env
        self.agents = agents  # 储存agents类的列表。
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_agents = self.env.num_agents
        self.agents_name_list = self.env.agents

    # 2 Training loop
    def training_loop(self):  #训练循环。


        for episode_i in range(self.num_episodes):
            data_buffer = {}
            for batch_i in range(self.batch_size):
                trajectory = []
                h_pi = [torch.zeros(1, self.agents.get_hidden_dim()) for _ in range(num_agents)]
                h_v = [torch.zeros(1, self.agents.get_hidden_dim()) for _ in range(num_agents)]
                for step_i in range(self.num_steps):
                    for agent_i in range(self.num_agents):