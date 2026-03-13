from typing import Optional
import numpy as np
import numba
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)

# 创建索引，使用padding来确保序列长度一致
def create_indices(
    episode_ends:np.ndarray,
    episode_mask: np.ndarray,) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        

    indices = list()
    for i in range(len(episode_ends)):#遍历每个episode
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx #计算当前episode的长度
        indices.append([start_idx, end_idx, episode_length])

    indices = np.array(indices)#即每段episode的索引
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer,
        obs_keys,
        max_length: int,
        fail_rate: float=0.1,
        episode_mask: Optional[np.ndarray]=None,
        ):
        super().__init__()

        fail_reward = -max_length * fail_rate
        episode_ends = replay_buffer.episode_ends[:]#获取episode结束的索引位置
        if hasattr(replay_buffer, 'success') and replay_buffer.success is not None:
            success = replay_buffer.success[:] #长度和episode_ends相同，1标志该episode成功，0标志失败
        else:
            success = np.ones_like(episode_ends, dtype=bool)
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends,
                episode_mask=episode_mask
                )
            
        rewards_list = []
        for idx in range(len(indices)):
            __, __, episode_length = indices[idx]#序列的索引
            rewards = self.progess_reward(episode_length)#计算每个观察的累积回报
            if success[idx]:#如果该episode失败
                rewards = rewards + fail_reward #该片段的每步都得到一个失败reward 设为最大长度的1/10
            rewards_list.append(rewards)

        self.rewards = np.concatenate(rewards_list)
        self.length = len(self.rewards)
        #将所有观察的return拼接起来，方便后续值函数的二元组 观察-return对应

        self.keys = list(obs_keys)
        self.replay_buffer = replay_buffer
    
    def __len__(self):
        return self.length
    
    def progess_reward(self, episode_length):

        # 计算每步的累积回报（从当前步到最后一步的奖励总和）
        returns = np.arange(1 - episode_length, 1, step=1, dtype=np.float32) #np.array [length]
        
        return returns
        

    def sample_sequence(self, idx):
        result = dict()

        for key in self.keys:
            result[key] = self.replay_buffer[key][idx]#取出二元组：观测-奖励
        result['reward'] = np.array([self.rewards[idx]]) 
        
        return result