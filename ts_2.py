import numpy as np
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# 创建一个简单的测试用例
def test_replay_buffer_reward_assignment():
    # 创建一个空的replay buffer
    buffer = ReplayBuffer.create_empty_numpy()
    
    # 模拟一些基本数据
    obs_data = np.random.rand(10, 4)  # 10个时间步，每个观察4维
    action_data = np.random.rand(10, 2)  # 10个时间步，每个动作2维
    episode_ends = np.array([5, 10])  # 两个episode，分别在第5和第10步结束
    
    # 添加数据到buffer
    data_dict = {
        'obs': obs_data,
        'action': action_data
    }
    buffer.update_meta({'episode_ends': episode_ends})
    buffer.data.update(data_dict)
    
    # 测试reward赋值
    rewards_list = [np.array([-4, -3, -2, -1, 0]), np.array([-4, -3, -2, -1, 0])]  # 两个episode的reward
    reward_data = np.concatenate(rewards_list, axis=0)
    
    print("Original buffer keys:", list(buffer.keys()))
    print("Reward data shape:", reward_data.shape)
    print("Expected shape based on observations:", obs_data.shape[0])
    
    # 执行赋值操作
    buffer['reward'] = reward_data
    
    print("After assignment, buffer keys:", list(buffer.keys()))
    print("Reward data retrieved:", buffer['reward'])
    print("Assignment successful!")

if __name__ == "__main__":
    test_replay_buffer_reward_assignment()