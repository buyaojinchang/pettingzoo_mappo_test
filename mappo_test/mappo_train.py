import numpy as np

class MappoTrain:
    def __init__(self, env, agents, num_episodes, batch_size, num_steps, ):
        self.env = env
        self.num_episodes = num_episodes


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

    # 2 Training loop
    def training_loop(env, num_episodes, batch_size, ):  #训练循环。
        multi_obs, info = env.reset()
        agent_name_list = env.agents
        NUM_AGENT = env.num_agents
        agents_list, obs_dim, action_dim, state_dim = initialize_agents(env,multi_obs, agent_name_list, NUM_AGENT, )  # TODO
        for episode_i in range(num_episodes):
            data_buffer = []
            for batch_i in range(batch_size):
                trajectory = []
