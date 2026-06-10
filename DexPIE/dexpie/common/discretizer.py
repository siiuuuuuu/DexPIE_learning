import torch
import numpy as np

class UniformDiscretizer:
    """Uniformly discretize continuous values into a fixed number of bins."""
    
    def __init__(self, num_bins=201, v_min=-10.0, v_max=10.0):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        
        # Compute each bin width.
        self.bin_width = (v_max - v_min) / (num_bins - 1)
        
        # Precompute bin centers for later continuous-value estimates.
        self.bin_centers = torch.linspace(v_min, v_max, num_bins)
    
    def discretize(self, values):
        """
        Convert continuous values to bin indices.
        :param values: Continuous tensor [batch_size] or [batch_size, 1].
        :return: Bin indices with long dtype.
        """
        # Clamp values into range.
        clamped_values = torch.clamp(values, self.v_min, self.v_max)
        
        # Compute the owning bin index.
        # Formula: index = floor((value - v_min) / bin_width).
        bin_indices = ((clamped_values - self.v_min) / self.bin_width).long()
        
        # Keep indices inside the valid range.
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices
    
    def discretize_to_gs_distribution(self, values, gaussian_std=1.0):
        """
        Convert continuous values to Gaussian distributions.
        :param values: Continuous values [batch_size].
        :param gaussian_std: Gaussian standard deviation.
        :return: Probability distribution [batch_size, num_bins].
        """
        batch_size = values.shape[0]
        
        # Get bin indices.
        bin_indices = self.discretize(values)
        
        # Create Gaussian distribution.
        distribution = torch.zeros(batch_size, self.num_bins, device=values.device)
        
        # Create a tensor representing all bin positions.
        bins_tensor = torch.arange(self.num_bins, device=values.device, dtype=values.dtype).unsqueeze(0)  # [1, num_bins]
        
        # Expand bin_indices to match bins_tensor.
        bin_indices_expanded = bin_indices.unsqueeze(1).float()  # [batch_size, 1]
        
        # Compute squared distance from each bin to each sample's center bin.
        distances_sq = (bins_tensor - bin_indices_expanded) ** 2  # [batch_size, num_bins]
        
        # Apply Gaussian function.
        gaussian_values = torch.exp(-distances_sq / (2 * gaussian_std ** 2))  # [batch_size, num_bins]
        
        # Normalize each sample distribution.
        distribution = gaussian_values / gaussian_values.sum(dim=1, keepdim=True)  # [batch_size, num_bins]
        
        return distribution

    def discretize_to_distribution(self, values, epsilon=0.0):
        """
        Convert continuous values to one-hot distributions.
        :param values: Continuous values [batch_size].
        :param epsilon: Label smoothing parameter.
        :return: Probability distribution [batch_size, num_bins].
        """
        batch_size = values.shape[0]
        
        # Get bin indices.
        bin_indices = self.discretize(values)
        
        # Create one-hot encoding.
        distribution = torch.zeros(batch_size, self.num_bins, device=values.device)
        distribution.scatter_(1, bin_indices.unsqueeze(1), 1.0)
        
        # Optionally add label smoothing.
        if epsilon > 0:
            distribution = (1 - epsilon) * distribution + epsilon / self.num_bins
        
        return distribution
    
    def undiscretize(self, bin_indices):
        """
        Recover continuous values from bin indices by using bin centers.
        :param bin_indices: [batch_size]
        :return: Continuous values [batch_size].
        """
        return self.bin_centers[bin_indices]
    
    def undiscretize_distribution(self, distribution):
        """
        Compute expected values from probability distributions.
        :param distribution: Probability distribution [batch_size, num_bins].
        :return: Continuous values [batch_size].
        """
        return torch.sum(distribution * self.bin_centers.unsqueeze(0).to(distribution.device), dim=-1)
