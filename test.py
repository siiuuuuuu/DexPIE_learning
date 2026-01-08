#!/usr/bin/env python3
import numpy as np
from diffusion_policy_3d.common.tools import MATHTOOLS

def test_relative_action_computation():
    # 创建 MATHTOOLS 实例
    tools = MATHTOOLS()
    
    # 测试数据：创建一些动作序列
    print("测试相对动作计算")
    
    # 创建一个包含3个动作的序列，每个动作是9维的 [x, y, z, rot_6d]
    seq_length = 3
    actions = np.random.rand(seq_length, 9)
    # 使旋转部分更合理（接近正交矩阵的6D表示）
    actions[:, 3:6] = np.random.rand(seq_length, 3) * 2 - 1  # 第一个3D向量
    actions[:, 6:9] = np.random.rand(seq_length, 3) * 2 - 1  # 第二个3D向量
    
    print(f"输入动作形状: {actions.shape}")
    print(f"第一个动作: {actions[0]}")
    
    # 将动作转换为变换矩阵
    poses = tools.xyz_6drot_to_mat(actions)
    print(f"变换矩阵形状: {poses.shape}")
    print(f"第一个变换矩阵:\n{poses[0]}")
    
    # 提取参考位姿
    pose_0 = poses[0]
    print(f"参考位姿 (pose_0):\n{pose_0}")
    
    # 计算参考位姿的逆
    inv_pose_0 = tools.se3_inverse(pose_0)
    print(f"参考位姿的逆 (inv_pose_0):\n{inv_pose_0}")
    
    # 验证逆矩阵的正确性
    identity_check = np.dot(inv_pose_0, pose_0)
    print(f"验证 inv_pose_0 * pose_0 = I:\n{identity_check}")
    print(f"单位矩阵误差: {np.max(np.abs(identity_check - np.eye(4)))}")
    
    # 计算相对动作
    relative_poses = np.einsum("ij,njk->nik", inv_pose_0, poses)
    print(f"相对变换矩阵形状: {relative_poses.shape}")
    print(f"第一个相对变换矩阵 (应该是单位矩阵):\n{relative_poses[0]}")
    print(f"第二个相对变换矩阵:\n{relative_poses[1]}")
    
    # 验证第一个相对变换矩阵是否接近单位矩阵
    is_identity = np.allclose(relative_poses[0], np.eye(4), atol=1e-6)
    print(f"第一个相对变换矩阵是否为单位矩阵: {is_identity}")
    

    
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_relative_action_computation()