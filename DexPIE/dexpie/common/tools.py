#!/usr/bin/env python3
import math
import numpy as np
import cv2

class  MATHTOOLS:
    
    # 从xyzrpy创建旋转矩阵
    def xyzrpy2Mat(self,x, y, z, roll, pitch, yaw):
        transformation_matrix = np.eye(4)
        A = np.cos(yaw)
        B = np.sin(yaw)
        C = np.cos(pitch)
        D = np.sin(pitch)
        E = np.cos(roll)
        F = np.sin(roll)
        DE = D * E
        DF = D * F
        transformation_matrix[0, 0] = A * C
        transformation_matrix[0, 1] = A * DF - B * E
        transformation_matrix[0, 2] = B * F + A * DE
        transformation_matrix[0, 3] = x
        transformation_matrix[1, 0] = B * C
        transformation_matrix[1, 1] = A * E + B * DF
        transformation_matrix[1, 2] = B * DE - A * F
        transformation_matrix[1, 3] = y
        transformation_matrix[2, 0] = -D
        transformation_matrix[2, 1] = C * F
        transformation_matrix[2, 2] = C * E
        transformation_matrix[2, 3] = z
        transformation_matrix[3, 0] = 0
        transformation_matrix[3, 1] = 0
        transformation_matrix[3, 2] = 0
        transformation_matrix[3, 3] = 1
        return transformation_matrix
    
    
    def mat2xyzrpy(self,matrix):
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.asin(-matrix[2, 0])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        return [x, y, z, roll, pitch, yaw]
    
    def xyzQuaternion2matrix(x, y, z, qx, qy, qz, qw):
        # 四元数到旋转矩阵：根据标准公式，将四元数分量转换为一个 3x3 的旋转矩阵。
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
        
        # 构造齐次变换矩阵：在右下角补上 1，在第四列填入位置向量 [x, y, z]。
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T

    def rpy_to_rotvec(self,roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        将RPY角度(绕X/Y/Z轴旋转)转换为旋转矢量
        
        参数:
            roll: 绕X轴的旋转角度(弧度)
            pitch: 绕Y轴的旋转角度(弧度)
            yaw: 绕Z轴的旋转角度(弧度)
        
        返回:
            旋转矢量 [rx, ry, rz], 方向表示旋转轴, 模长表示旋转角度(弧度)
        """
        # 构建绕各轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵 (顺序: Z-Y-X)
        R = np.dot(R_z, np.dot(R_y, R_x))
        
        # 从旋转矩阵计算旋转矢量
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if np.abs(theta) < 1e-10:  # 零旋转
            return np.array([0, 0, 0])
        else:
            # 旋转轴
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * np.sin(theta))
            
            # 旋转矢量 = 旋转轴 * 旋转角度
            return axis * theta


    def rotvec_to_rpy(self,rotvec: np.ndarray) -> tuple:
        """
        将旋转矢量转换为RPY角度(绕X/Y/Z轴旋转)
        
        参数:
            rotvec: 旋转矢量 [rx, ry, rz]
        
        返回:
            (roll, pitch, yaw): 绕X/Y/Z轴的旋转角度(弧度)
        """
        theta = np.linalg.norm(rotvec)
        
        if np.abs(theta) < 1e-10:  # 零旋转
            return (0, 0, 0)
        
        # 旋转轴
        axis = rotvec / theta
        
        # 构建旋转矩阵 (Rodrigues公式)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
        # 从旋转矩阵提取RPY (采用Z-Y-X顺序)
        # 参考: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return (roll, pitch, yaw)
    
    def rotvec2mat(self,rotvec):
        #旋转矢量到列惯例旋转矩阵
        theta = np.linalg.norm(rotvec)
        eps = 1e-6
        if theta < eps:
            return np.eye(3)
        
        axis = rotvec / theta
        K = np.array([
            [0,    -axis[2],  axis[1]],
            [axis[2], 0,      -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    def xyz_rotvec_to_mat(self,xyz_rotvec):
        pos=xyz_rotvec[:3]
        rotvec=xyz_rotvec[3:]
        T=np.eye(4)
        T[:3,3]=pos
        R=self.rotvec2mat(rotvec)
        T[:3,:3]=R
        return T
    def mat2xyz_rotvec(self,matrix, eps=1e-6):
        """
        从 列惯例4×4 齐次矩阵中提取 xyz + 旋转矢量
        
        参数:
            matrix: 4×4 齐次变换矩阵 (numpy数组)
            eps: 奇点检测阈值
        
        返回:
            [x, y, z, rx, ry, rz] 
            其中 [rx, ry, rz] 是旋转矢量 = 旋转轴 × 旋转角(弧度)
        """
        # 提取平移分量
        x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        
        # 提取旋转矩阵
        R = matrix[:3, :3]
        
        # 计算旋转角 θ
        trace = np.trace(R)
        cos_theta = max(min((trace - 1) / 2.0, 1.0), -1.0)  # 数值安全
        theta = math.acos(cos_theta)
        
        # 计算旋转矢量
        if theta < eps:
            # 接近零旋转
            rx, ry, rz = 0.0, 0.0, 0.0
            
        elif theta > math.pi - eps:
            # θ ≈ π，sin(θ) ≈ 0，需要特殊处理
            # 旋转轴是 (R + I) 的任意非零列
            diag = np.diag(R)
            idx = np.argmax(diag)  # 选择最稳定的列
            
            axis = np.array([R[0, idx], R[1, idx], R[2, idx]])
            axis[idx] += 1  # (R + I) 的第idx列
            
            # 归一化
            axis_norm = np.linalg.norm(axis)
            axis = axis / axis_norm if axis_norm > eps else np.array([1.0, 0.0, 0.0])
            
            rx, ry, rz = axis * theta
            
        else:
            # 正常情况：使用罗德里格斯公式
            sin_theta = math.sin(theta)
            factor = theta / (2 * sin_theta)
            
            rx = factor * (R[2, 1] - R[1, 2])
            ry = factor * (R[0, 2] - R[2, 0])
            rz = factor * (R[1, 0] - R[0, 1])
        
        return [x, y, z, rx, ry, rz]
    
    def se3_inverse(self,T):
        """
        Compute the inverse of an SE(3) transformation matrix.
        For T = [R | t; 0 | 1], the inverse is T⁻¹ = [Rᵀ | -Rᵀt; 0 | 1]
        
        Args:
            T: 4x4 SE(3) transformation matrix
            
        Returns:
            4x4 inverse transformation matrix
        """
        R = T[:3, :3]  # 3x3 rotation matrix
        t = T[:3, 3]   # 3x1 translation vector
        
        # Compute R transpose
        R_transpose = R.T
        
        # Compute -Rᵀt
        t_inv = -np.dot(R_transpose, t)
        
        # Build the inverse matrix
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_transpose
        T_inv[:3, 3] = t_inv
        
        return T_inv
  
    def normalize(self,x, axis=-1, eps=1e-8):
        """沿指定轴做 L2 归一化"""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    def rot6d_to_mat(self,d6:np.ndarray):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = self.normalize(a1)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = self.normalize(b2)
        b3 = np.cross(b1, b2, axis=-1)
        out = np.stack((b1, b2, b3), axis=-2)
        return out
    def xyz_6drot_to_mat(self,xyz_6drot:np.ndarray):
        batch_dims = xyz_6drot.shape[:-1]
        pos=xyz_6drot[...,:3]
        rot_6d=xyz_6drot[...,3:]
        mats = np.tile(np.eye(4), batch_dims + (1, 1))
        rot_mat= self.rot6d_to_mat(rot_6d)
        mats[..., :3, :3] = rot_mat
        mats[..., :3, 3] = pos
        return mats
    def mat_to_rot6d(self,mat:np.ndarray):
        batch_dim = mat.shape[:-2]
        out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
        return out
    def mat2xyz_6drot(self, matrix: np.ndarray, eps=1e-6):
        # 检查输入维度，处理批量操作
        if len(matrix.shape) == 3:  # [T, 4, 4] 批量处理
            batch_size = matrix.shape[0]
            # 提取平移分量 [T, 3]
            pos = matrix[:, :3, 3]
            # 提取旋转矩阵 [T, 3, 3]
            R = matrix[:, :3, :3]
            # 转换为6D旋转表示 [T, 6]
            rot_6d = self.mat_to_rot6d(R)
            # 组合位置和旋转 [T, 9]
            result = np.concatenate([pos, rot_6d], axis=-1)
            return result
        else:  # 单个 4x4 矩阵
            x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
            # 提取旋转矩阵
            R = matrix[:3, :3]
            rot_6d = self.mat_to_rot6d(R)
            return np.array([x, y, z, *rot_6d])
