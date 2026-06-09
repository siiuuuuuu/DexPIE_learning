from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
from dexpie.model.value_fun.value_net import ValueNet
import numpy as np
from dexpie.model.common.module_attr_mixin import ModuleAttrMixin
#实现loss计算，forward方法（求期望得到价值）

class ValueCritic(ModuleAttrMixin):
    def __init__(self,
                shape_meta: dict,
                model_name: str,
                pretrained: bool,
                frozen: bool,
                v_min: float,
                v_max: float,
                num_bins: int,
                transforms: list,
                # use single rgb model for all rgb inputs
                share_rgb_model: bool=True,):
        super().__init__()
        self.bin_centers = torch.linspace(v_min, v_max, num_bins)
        value_critic = ValueNet(
            shape_meta=shape_meta,
            model_name=model_name,
            pretrained=pretrained,
            frozen=frozen,
            num_bins=num_bins,
            transforms=transforms,
            share_rgb_model=share_rgb_model,
        )
        self.value_critic = value_critic

    def _prepare_obs(self, obs_dict: Dict) -> Dict:
        # shallow-copy dict only; avoid cloning all tensors
        nobs = obs_dict.copy()
        image = nobs['image'] / 255.0
        if image.shape[-1] == 3:
            if len(image.shape) == 5:
                image = image.permute(0, 1, 4, 2, 3)
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
        nobs['image'] = image

        if "wrist_img" in nobs:
            wrist_img = nobs["wrist_img"] / 255.0
            if wrist_img.shape[-1] == 3:
                if len(wrist_img.shape) == 5:
                    wrist_img = wrist_img.permute(0, 1, 4, 2, 3)
                if len(wrist_img.shape) == 4:
                    wrist_img = wrist_img.permute(0, 3, 1, 2)
            nobs["wrist_img"] = wrist_img
        return nobs

    @torch.no_grad()
    def forward(self, obs_dict: Dict, obs_preprocessed: bool = False) -> torch.Tensor:
        #归一化处理
        nobs = obs_dict if obs_preprocessed else self._prepare_obs(obs_dict)
        B=nobs['image'].shape[0]

        pred_logits=self.value_critic(nobs) #里面会把B,T合并起来
        pred_prob = torch.softmax(pred_logits, dim=-1)
        pred_prob = torch.clamp(pred_prob, min=1e-8, max=1.0)
        value = torch.sum(pred_prob * self.bin_centers.unsqueeze(0).to(pred_prob.device), dim=-1)
        return value.reshape(B,-1,1) #shape [B,T,1] 后续一般T为2
    
    def compute_loss(self, batch) -> torch.Tensor:
        nobs = self._prepare_obs(batch['obs'])
        target = batch['target']
        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() != 3:
            raise ValueError(f"Expected target to have shape [B, num_bins] or [B, T, num_bins], got {target.shape}")

        pred_logits = self.value_critic(nobs)#shape [B, num_bins]
        pred_prob = torch.softmax(pred_logits, dim=-1).unsqueeze(1)
        pred_prob = torch.clamp(pred_prob, min=1e-8, max=1.0)
        cross_entropy = -torch.sum(target * torch.log(pred_prob), dim=-1)#交叉熵
        loss = cross_entropy.mean()#对B取平均
        return loss
        
