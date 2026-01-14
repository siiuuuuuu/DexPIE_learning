import torch
import numpy as np
from diffusion_policy_3d.model.vision.Dino_obs_encoder import DinoObsEncoder

def test_dino_encoder():
    print("开始测试 DinoObsEncoder...")
    
    # 定义形状元数据
    shape_meta = {
        'obs': {
            'img': {
                'shape': [3, 224, 224],  # RGB图像形状 (C, H, W)
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': [7],  # 机器人关节位置
                'type': 'low_dim'
            }
        }
    }
    
    # 创建编码器实例
    encoder = DinoObsEncoder(
        shape_meta=shape_meta,
        model_name="vit_base_patch14_dinov2",  # 使用较小的模型以节省资源
        out_indices=(2, 5, 8, 11),  # 减少层数以加快测试
        transforms=None
    )
    encoder.to('cuda')
    
    print("编码器创建成功!")
    print(f"RGB keys: {encoder.rgb_keys}")
    print(f"Low-dim keys: {encoder.low_dim_keys}")
    print(f"Embedding dimension: {encoder.embed_dim}")
    
    # 准备测试数据
    batch_size = 2
    seq_len = 1  # 时间步数
    
    obs_dict = {
        'img': torch.randn(batch_size, seq_len, 3, 224, 224).to('cuda'),  # RGB图像
        'agent_pos': torch.randn(batch_size, seq_len, 7).to('cuda')      # 机器人状态
    }
    
    print(f"\n输入数据形状:")
    for key, tensor in obs_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # 执行前向传播
    print("\n执行前向传播...")
    try:
        with torch.no_grad():  # 不需要梯度计算
            output = encoder(obs_dict)
        
        print(f"输出形状: {output.shape}")
        print("测试成功! DinoObsEncoder工作正常。")
        
        # 验证输出的一些属性
        print(f"输出张量的最大值: {output.max().item():.4f}")
        print(f"输出张量的最小值: {output.min().item():.4f}")
        print(f"输出张量的平均值: {output.mean().item():.4f}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dino_encoder()