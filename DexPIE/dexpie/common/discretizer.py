import torch
import numpy as np

class UniformDiscretizer:
    """将连续值均匀离散化到固定数量的bins"""
    
    def __init__(self, num_bins=201, v_min=-10.0, v_max=10.0):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        
        # 计算每个bin的宽度
        self.bin_width = (v_max - v_min) / (num_bins - 1)
        
        # 预计算bin的中心点值（用于后续连续值估计）
        self.bin_centers = torch.linspace(v_min, v_max, num_bins)
    
    def discretize(self, values):
        """
        将连续值转换为bin索引
        :param values: 连续值张量 [batch_size] 或 [batch_size, 1]
        :return: bin索引 (long类型)
        """
        # 限制值在范围内
        clamped_values = torch.clamp(values, self.v_min, self.v_max)
        
        # 计算所属的bin索引
        # 公式: index = floor((value - v_min) / bin_width)
        bin_indices = ((clamped_values - self.v_min) / self.bin_width).long()
        
        # 确保索引在有效范围内
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices
    
    def discretize_to_gs_distribution(self, values, gaussian_std=1.0):
        """
        将连续值转换为高斯分布
        :param values: 连续值 [batch_size]
        :param gaussian_std: 高斯分布标准差
        :return: 概率分布 [batch_size, num_bins]
        """
        batch_size = values.shape[0]
        
        # 获取bin索引
        bin_indices = self.discretize(values)
        
        # 创建高斯分布
        distribution = torch.zeros(batch_size, self.num_bins, device=values.device)
        
        # 创建一个表示所有bins位置的tensor
        bins_tensor = torch.arange(self.num_bins, device=values.device, dtype=values.dtype).unsqueeze(0)  # [1, num_bins]
        
        # 扩展bin_indices以匹配bins_tensor的形状
        bin_indices_expanded = bin_indices.unsqueeze(1).float()  # [batch_size, 1]
        
        # 计算每个样本的每个bin与其中心bin的距离的平方
        distances_sq = (bins_tensor - bin_indices_expanded) ** 2  # [batch_size, num_bins]
        
        # 应用高斯函数
        gaussian_values = torch.exp(-distances_sq / (2 * gaussian_std ** 2))  # [batch_size, num_bins]
        
        # 归一化每个样本的分布
        distribution = gaussian_values / gaussian_values.sum(dim=1, keepdim=True)  # [batch_size, num_bins]
        
        return distribution

    def discretize_to_distribution(self, values, epsilon=0.0):
        """
        将连续值转换为one-hot分布
        :param values: 连续值 [batch_size]
        :param epsilon: 标签平滑参数
        :return: 概率分布 [batch_size, num_bins]
        """
        batch_size = values.shape[0]
        
        # 获取bin索引
        bin_indices = self.discretize(values)
        
        # 创建one-hot编码
        distribution = torch.zeros(batch_size, self.num_bins, device=values.device)
        distribution.scatter_(1, bin_indices.unsqueeze(1), 1.0)
        
        # 可选：添加标签平滑
        if epsilon > 0:
            distribution = (1 - epsilon) * distribution + epsilon / self.num_bins
        
        return distribution
    
    def undiscretize(self, bin_indices):
        """
        从bin索引恢复连续值（取bin中心）
        :param bin_indices: [batch_size]
        :return: 连续值 [batch_size]
        """
        return self.bin_centers[bin_indices]
    
    def undiscretize_distribution(self, distribution):
        """
        从概率分布计算期望值
        :param distribution: 概率分布 [batch_size, num_bins]
        :return: 连续值 [batch_size]
        """
        return torch.sum(distribution * self.bin_centers.unsqueeze(0).to(distribution.device), dim=-1)