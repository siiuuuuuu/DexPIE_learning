import math
import torch
import torch.nn as nn


class AdvantageSinusoidalPosEmb(nn.Module):
    """
    Sinusoidal embedding for continuous advantage score in [0, 1].
    Kept separate from timestep positional embedding to avoid coupling.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] != 1:
                raise ValueError(
                    f"AdvantageSinusoidalPosEmb expects scalar input per sample, got {tuple(x.shape)}"
                )
            x = x[:, 0]

        x = x.float()
        device = x.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return x[:, None]

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * (-emb_scale))
        emb = x[:, None] * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1)
        return emb
