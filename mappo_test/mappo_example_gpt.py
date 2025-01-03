import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


##########################
# 1. 定义网络结构
##########################

class RecurrentActor(nn.Module):
    """
    每个智能体的策略网络：输入是观测和上一个隐藏状态
    输出是动作分布（和新的隐藏状态）
    """

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(RecurrentActor, self).__init__()
        # 用于提取特征的全连接层
        self.fc = nn.Linear(obs_dim, hidden_dim)
        # RNN 用于捕捉时间序列依赖
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        # 动作分布输出层
        self.pi = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, h_in):
        """
        obs: (batch_size, obs_dim)
        h_in: (batch_size, hidden_dim)   作为 RNN 的隐藏状态输入
        """
        x = torch.relu(self.fc(obs))  # 提取特征
        h_out = self.rnn(x, h_in)  # 更新隐藏状态
        logits = self.pi(h_out)  # 输出动作的 logits
        return logits, h_out


class RecurrentCritic(nn.Module):
    """
    值函数网络：输入是全局状态(或观测)和隐藏状态
    输出是 state value
    """

    def __init__(self, obs_dim, hidden_dim):
        super(RecurrentCritic, self).__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, h_in):
        x = torch.relu(self.fc(obs))
        h_out = self.rnn(x, h_in)
        value = self.value_head(h_out)
        return value, h_out


##########################
# 2. 多智能体训练超参数 & 初始化
##########################

# 假设有 N 个智能体，每个智能体可以有相同或不同的网络；这里演示相同的网络
N = 2  # 例如 2 个智能体
obs_dim = 10  # 每个智能体观测维度
act_dim = 5  # 动作空间维度
hidden_dim = 64  # RNN 隐藏层大小

# 初始化 actor/critic
actors = [RecurrentActor(obs_dim, act_dim, hidden_dim) for _ in range(N)]
critics = [RecurrentCritic(obs_dim, hidden_dim) for _ in range(N)]

# 优化器（示例统一用一个，也可为每个智能体分别定义）
actor_params = []
critic_params = []
for i in range(N):
    actor_params += list(actors[i].parameters())
    critic_params += list(critics[i].parameters())

actor_optimizer = optim.Adam(actor_params, lr=1e-3)
critic_optimizer = optim.Adam(critic_params, lr=1e-3)

# 其他超参数（这里仅示例）
T = 100  # 每次 roll out 的最大时间步
batch_size = 10  # 批量数
num_epochs = 3  # 每次更新时的 epoch 数
gamma = 0.99  # 折扣因子
lam = 0.95  # GAE 参数
clip_range = 0.2  # PPO 中截断范围
K = 4  # 一次迭代要进行的 mini-batch 轮次(外层伪代码中的 K)


##########################
# 3. 采样阶段 (rollout)
##########################

def rollout_once(env, actors, critics, T, N):
    """
    进行一次 roll out 的示例函数，收集 T 步的数据。
    这里示例只返回一个列表；实际中还需要并行环境等。
    """
    # 记录存储：每个 time step 都记录 [o, a, r, log_pi, v, h_pi_in, h_v_in]
    trajectory = []

    # 初始化隐藏状态 (N, hidden_dim)，假设是 0
    h_pi = [torch.zeros(1, hidden_dim) for _ in range(N)]
    h_v = [torch.zeros(1, hidden_dim) for _ in range(N)]

    # 假设我们可以从 env 重置得到所有智能体的初始观测
    obs = env.reset()  # shape: (N, obs_dim)

    for t in range(T):
        actions = []
        values = []
        log_pis = []

        with torch.no_grad():
            for i in range(N):
                obs_i = torch.FloatTensor(obs[i]).unsqueeze(0)  # (1, obs_dim)

                # 策略网络前向
                logits, h_pi_out = actors[i](obs_i, h_pi[i])
                dist = Categorical(logits=logits)

                # 动作采样
                action_i = dist.sample()

                # Critic 网络前向
                value_i, h_v_out = critics[i](obs_i, h_v[i])

                actions.append(action_i.item())
                values.append(value_i.item())
                log_pis.append(dist.log_prob(action_i).item())

                # 更新隐藏状态
                h_pi[i] = h_pi_out
                h_v[i] = h_v_out

        # 将所有智能体动作合并后执行到环境中，返回下一时刻的观测、奖励和 done
        next_obs, rewards, done, info = env.step(actions)

        # 存储当前时刻信息
        step_data = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_pis': log_pis,
            'h_pi': [h.detach().clone() for h in h_pi],
            'h_v': [h.detach().clone() for h in h_v],
        }
        trajectory.append(step_data)

        obs = next_obs
        if done:
            # 如果环境结束，这里仅做演示，break
            break

    return trajectory


##########################
# 4. GAE 优势估计
##########################

def compute_gae(trajectory, critics, gamma=0.99, lam=0.95):
    """
    对 roll out 的数据进行 GAE 计算，得到 advantage & return
    这里只演示单条轨迹的情况，实际需要对 N 个智能体分别处理
    """
    # trajectory: list of dict. 每一项如：
    # {
    #   'obs': [...],
    #   'actions': [...],
    #   'rewards': [...],
    #   'values': [...],
    #   ...
    # }

    # 注意：Recurrent 版本的 value 需要下一个时刻隐藏状态才能计算 bootstrap
    # 这里只演示简单方式

    advantages = []
    returns = []

    # 对所有 agent 的 advantage 进行存储
    for i in range(N):
        advantages_i = []
        returns_i = []
        gae = 0.0
        last_value = 0.0

        # 从后往前计算
        for t in reversed(range(len(trajectory))):
            reward_i = trajectory[t]['rewards'][i]
            value_i = trajectory[t]['values'][i]

            # delta = r + γ * V(s_{t+1}) - V(s_t)
            # 这里为了简化，假设 rollout 结束最后一项 next_value=0
            next_value_i = 0.0 if t == len(trajectory) - 1 else trajectory[t + 1]['values'][i]
            delta = reward_i + gamma * next_value_i - value_i

            # GAE 递推
            gae = delta + gamma * lam * gae
            advantages_i.append(gae)

        # 现在 advantages_i 是从后往前的，所以翻转
        advantages_i.reverse()

        # compute returns
        returns_i = [a + v for a, v in zip(advantages_i, [step['values'][i] for step in trajectory])]

        advantages.append(advantages_i)
        returns.append(returns_i)

    return advantages, returns


##########################
# 5. 主训练循环
##########################

def train_mappo(env, actors, critics, actor_optimizer, critic_optimizer,
                T, batch_size, num_epochs, K, gamma, lam, clip_range):
    for iteration in range(batch_size):
        # (1) Rollout 收集数据
        trajectory = rollout_once(env, actors, critics, T, N)

        # (2) 计算优势 & returns
        advantages, returns = compute_gae(trajectory, critics, gamma, lam)

        # 整理收集的数据，构建训练所需的批次
        # 示例只展示结构；实际上需要将每个 time step、每个 agent 的数据都展开
        obs_batch = []
        action_batch = []
        old_log_pi_batch = []
        advantage_batch = []
        return_batch = []
        # ...

        # 这里只是演示如何将 trajectory 的每一步、每个 agent 的数据取出来
        for t in range(len(trajectory)):
            for i in range(N):
                obs_batch.append(trajectory[t]['obs'][i])
                action_batch.append(trajectory[t]['actions'][i])
                old_log_pi_batch.append(trajectory[t]['log_pis'][i])
                advantage_batch.append(advantages[i][t])
                return_batch.append(returns[i][t])

        # 转为张量
        obs_batch = torch.FloatTensor(obs_batch)
        action_batch = torch.LongTensor(action_batch)
        old_log_pi_batch = torch.FloatTensor(old_log_pi_batch)
        advantage_batch = torch.FloatTensor(advantage_batch)
        return_batch = torch.FloatTensor(return_batch)

        # (3) Mini-batch 更新（这里很简单地把所有数据一次性扔进去，真实中需要分片+多轮 K）
        for epoch in range(num_epochs):

            # forward policy
            # 简化：不区分不同 agent，不使用 RNN 隐藏状态，示例说明 PPO 流程
            logits = []
            values_out = []
            idx = 0
            for i in range(len(obs_batch)):
                # 这里只是为了演示 forward，实际 Recurrent 会逐时间步处理隐藏状态
                # idx -> agent 也要注意，这里忽略
                logits_i, _ = actors[0](obs_batch[i].unsqueeze(0), torch.zeros(1, hidden_dim))
                dist_i = Categorical(logits=logits_i)
                values_i, _ = critics[0](obs_batch[i].unsqueeze(0), torch.zeros(1, hidden_dim))

                logits.append(dist_i.log_prob(action_batch[i]))
                values_out.append(values_i)

            log_pi = torch.cat(logits)
            values_out = torch.cat(values_out).squeeze(-1)  # shape [batch_size,]

            # PPO 损失：policy loss
            ratio = torch.exp(log_pi - old_log_pi_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = nn.MSELoss()(values_out, return_batch)

            # 总损失
            loss = policy_loss + value_loss

            # 反向传播
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

        print(f"Iter {iteration} done. Loss: {loss.item():.4f}")


##########################
# 6. 主调函数 (示例)
##########################

class DummyEnv:
    """
    一个假的环境类，只是演示用
    """

    def reset(self):
        # 返回 N=2 个观测，每个观测维度 obs_dim=10
        return [[0.0] * obs_dim for _ in range(N)]

    def step(self, actions):
        # 返回 next_obs, rewards, done, info
        # 假装观察不变，奖励只看动作之和
        rewards = [sum(actions) * 0.01 for _ in range(N)]
        next_obs = [[0.0] * obs_dim for _ in range(N)]
        done = False
        info = {}
        return next_obs, rewards, done, info


if __name__ == "__main__":
    # 创建假的环境
    env = DummyEnv()

    # 开始训练
    train_mappo(env,
                actors, critics,
                actor_optimizer, critic_optimizer,
                T, batch_size, num_epochs,
                K, gamma, lam, clip_range)