import numpy as np
from diffusion_policy_3d.common.sampler import SequenceSampler
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

def test_sampler():
    print("测试修改后的SequenceSampler...")
    
    # 创建一个简单的replay buffer进行测试
    episode_lengths = [45, 10, 100]  # 3个episode，长度分别为10, 15, 8
    total_length = sum(episode_lengths)
    
    # 计算episode结束索引
    episode_ends = np.cumsum(episode_lengths)
    print(f"Episode结束索引: {episode_ends}")
    
    # 使用ReplayBuffer的正确方法创建实例
    replay_buffer = ReplayBuffer.create_empty_numpy()
    
    # 添加数据
    data_dict = {}
    data_dict['obs'] = np.arange(total_length, dtype=np.float32).reshape(-1, 1)
    data_dict['action'] = np.arange(total_length, dtype=np.float32).reshape(-1, 1) * 2  # 动作数据是观测数据的2倍
    
    # 使用add_episode方法添加数据
    replay_buffer.add_episode(data_dict)
    
    # 创建sampler，指定action为动作数据键
    sequence_length = 12
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=sequence_length,
        pad_after=sequence_length-1,
        keys=['obs', 'action'], # 指定action为动作数据
    )
    
    print(f"Sampler长度: {len(sampler)}")
    print(f"序列长度: {sequence_length}")
    
    # 测试不同情况
    print("\n=== 测试正常情况 ===")
    # 测试中间episode的正常情况
    idx = 10  # 选择一个中间的索引
    if idx < len(sampler):
        result = sampler.sample_sequence(idx)
        print(f"索引 {idx} 的结果:")
        print(f"  obs shape: {result['obs'].shape}")
        print(f"  action shape: {result['action'].shape}")  # 应该是 (sequence_length+1, 1) = (6, 1)
        print(f"  obs data: {result['obs'].flatten()}")
        print(f"  action data: {result['action'].flatten()}")
        print(f"  mask: {result['mask']}")
    
    print("\n=== 测试边界情况 ===")
    
    # 测试第一个episode开始的情况 (buffer_start_idx = 0)
    print("\n--- 测试第一个episode开始 (buffer_start_idx = 0) ---")
    # 找到第一个episode开始的索引
    for i in range(len(sampler)):
        buffer_start_idx, buffer_end_idx, padflag, sample_start_idx, sample_end_idx = sampler.indices[i]
        if buffer_start_idx == 0:
            print(f"找到索引 {i}: buffer_start_idx={buffer_start_idx}, buffer_end_idx={buffer_end_idx}, "
                  f"sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}")
            result = sampler.sample_sequence(i)
            print(f"  obs shape: {result['obs'].shape}")
            print(f"  action shape: {result['action'].shape}")  # 应该是 (sequence_length+1, 1) = (6, 1)
            print(f"  obs data: {result['obs'].flatten()}")
            print(f"  action data: {result['action'].flatten()}")
            print(f"  mask: {result['mask']}")
    
    # 测试episode结束的情况
    print("\n--- 测试episode结束情况 ---")
    # 找到episode结束的索引
    for i in range(len(sampler)-1, -1, -1):
        buffer_start_idx, buffer_end_idx, pad_flag, sample_start_idx, sample_end_idx = sampler.indices[i]
        # 如果sample_end_idx < sequence_length，说明需要padding
        if sample_end_idx < sequence_length:
            print(f"找到索引 {i}: buffer_start_idx={buffer_start_idx}, buffer_end_idx={buffer_end_idx}, "
                  f"sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}")
            result = sampler.sample_sequence(i)
            print(f"  obs shape: {result['obs'].shape}")
            print(f"  action shape: {result['action'].shape}")
            print(f"  obs data: {result['obs'].flatten()}")
            print(f"  action data: {result['action'].flatten()}")
            print(f"  mask: {result['mask']}")
            
    
    # 测试需要前后padding的情况
    print("\n--- 测试需要前后padding的情况 ---")
    for i in range(len(sampler)):
        buffer_start_idx, buffer_end_idx, pad_flag, sample_start_idx, sample_end_idx = sampler.indices[i]
        if sample_start_idx > 0 and sample_end_idx < sequence_length:
            print(f"找到索引 {i}: buffer_start_idx={buffer_start_idx}, buffer_end_idx={buffer_end_idx}, "
                  f"sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}")
            result = sampler.sample_sequence(i)
            print(f"  obs shape: {result['obs'].shape}")
            print(f"  action shape: {result['action'].shape}")
            print(f"  obs data: {result['obs'].flatten()}")
            print(f"  action data: {result['action'].flatten()}")
            print(f"  mask: {result['mask']}")
            break
    
    print("\n=== 验证动作数据的特殊处理 ===")
    # 重点验证动作数据是否确实使用了 sequence_length+1 的长度
    for i in range(min(5, len(sampler))):  # 测试前5个样本
        result = sampler.sample_sequence(i)
        obs_shape = result['obs'].shape[0]
        action_shape = result['action'].shape[0]
        print(f"索引 {i}: obs长度={obs_shape}, action长度={action_shape} (期望: obs={sequence_length}, action={sequence_length+1})")
        
        # 检查动作数据是否确实是从buffer_start_idx-1开始采样的
        buffer_start_idx, buffer_end_idx, pad_flag, sample_start_idx, sample_end_idx = sampler.indices[i]
        print(f"  原始索引: buffer_start_idx={buffer_start_idx}, buffer_end_idx={buffer_end_idx}")
        
        # 检查动作数据的值是否正确（从buffer_start_idx-1开始）
        expected_action_start_idx = max(buffer_start_idx - 1, 0)
        expected_action_values = np.arange(expected_action_start_idx, 
                                          min(expected_action_start_idx + sequence_length + 1, total_length), 
                                          dtype=np.float32) * 2
        actual_action_values = result['action'].flatten()
        print(f"  期望动作值: {expected_action_values}")
        print(f"  实际动作值: {actual_action_values}")
        print(f"  动作数据是否匹配: {np.allclose(expected_action_values[:len(actual_action_values)], actual_action_values)}")
        print()
    
    print("测试完成!")

if __name__ == "__main__":
    test_sampler()