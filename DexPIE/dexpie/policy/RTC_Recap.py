from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dexpie.model.common.normalizer import LinearNormalizer
from dexpie.policy.base_policy import BasePolicy
from dexpie.model.diffusion.conditional_unet1d import ConditionalUnet1D
from dexpie.model.diffusion.mask_generator import LowdimMaskGenerator
from dexpie.common.model_util import print_params
from dexpie.common.pytorch_util import dict_apply
from termcolor import cprint
from dexpie.model.vision.timm_obs_encoder import TimmObsEncoder
import numpy as np


class RTCRecapPolicy(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            max_latency_steps=3,
            cond_dropout_prob=0.75,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type='film',
            positive_cond_drop_prob=0.3,
            positive_cfg_scale=2,
            use_depth=False,
            use_depth_only=False,
            obs_encoder: TimmObsEncoder = None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.max_latency_steps = max_latency_steps
        assert 0.0 <= cond_dropout_prob <= 1.0
        self.cond_dropout_prob = cond_dropout_prob
        self.use_depth = use_depth
        self.use_depth_only = use_depth_only
        cprint(f"use_depth: {use_depth}, use_depth_only: {use_depth_only}", 'red')
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }

        if use_depth and not use_depth_only:
            obs_shape_meta['image']['shape'][0] = 4  # 3,H,W -> 4,H,W

        if use_depth and use_depth_only:
            obs_shape_meta['image']['shape'][0] = 1

        obs_feature_dim = np.prod(obs_encoder.output_shape())
        obs_feature_dim += diffusion_step_embed_dim  # 加上 positive_embedding 的维度

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )
        self.is_positive_embedding = nn.Embedding(2, diffusion_step_embed_dim)
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.positive_cond_drop_prob = positive_cond_drop_prob
        self.positive_cfg_scale = positive_cfg_scale
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = obs_dict.copy()
        exc_action = obs_dict.pop("exc_action", None)  # 推理时输入执行动作前缀
        if exc_action is None:
            exc_horizon = 0
        else:
            exc_horizon = exc_action.shape[1]
            assert exc_horizon <= self.max_latency_steps, "exc_horizon must be less than max_latency_steps"

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
        if "wrist_img" in nobs:
            nobs["wrist_img"] /= 255.0
            if nobs["wrist_img"].shape[-1] == 3:
                if len(nobs["wrist_img"].shape) == 5:
                    nobs["wrist_img"] = nobs["wrist_img"].permute(0, 1, 4, 2, 3)
                if len(nobs["wrist_img"].shape) == 4:
                    nobs["wrist_img"] = nobs["wrist_img"].permute(0, 3, 1, 2)

        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...])
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do and append positive condition embedding for inference
        base_global_cond = nobs_features.reshape(B, -1)
        positive_embedding = self.is_positive_embedding(
            torch.ones(B, device=base_global_cond.device, dtype=torch.long)
        ).reshape(B, -1)  # 使用恒为1的positive_embedding作为条件（好的动作）
        global_cond = torch.cat([base_global_cond, positive_embedding], dim=-1)
        global_cond_uncond = torch.cat(
            [base_global_cond, torch.zeros_like(positive_embedding)], dim=-1
        )  # 用0作为空条件，当作为无条件输入

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        if exc_horizon > 0:
            cond_data[:, :exc_horizon, ...] = exc_action
            cond_mask[:, :exc_horizon, ...] = True  # 注入前缀条件

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            cfg_scale=self.positive_cfg_scale,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1 + exc_horizon
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return action

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            global_cond_uncond=None,
            cfg_scale=1.0,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            if global_cond_uncond is not None and cfg_scale != 1.0:
                # Parallel CFG: concatenate unconditional and conditional branches in one batch.
                model_input = torch.cat([trajectory, trajectory], dim=0)
                if local_cond is not None:
                    local_cond_input = torch.cat([local_cond, local_cond], dim=0)
                else:
                    local_cond_input = None
                global_cond_input = torch.cat([global_cond_uncond, global_cond], dim=0)
                model_output_pair = model(
                    model_input, t, local_cond=local_cond_input, global_cond=global_cond_input
                )
                model_output_uncond, model_output_cond = torch.chunk(model_output_pair, 2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = model(trajectory, t,
                    local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                # **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        obs_dict = obs_dict.copy()
        exc_action = obs_dict.pop("exc_action", None)
        if exc_action is None:
            exc_horizon = 0
        else:
            exc_horizon = exc_action.shape[1]
            assert exc_horizon <= self.max_latency_steps, "exc_horizon must be less than max_latency_steps"

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # normalize image by hand
        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
        if "wrist_img" in nobs:
            nobs["wrist_img"] /= 255.0
            if nobs["wrist_img"].shape[-1] == 3:
                if len(nobs["wrist_img"].shape) == 5:
                    nobs["wrist_img"] = nobs["wrist_img"].permute(0, 1, 4, 2, 3)
                if len(nobs["wrist_img"].shape) == 4:
                    nobs["wrist_img"] = nobs["wrist_img"].permute(0, 3, 1, 2)

        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do and append positive condition embedding for inference
            base_global_cond = nobs_features.reshape(B, -1)
            positive_embedding = self.is_positive_embedding(
                torch.ones(B, device=base_global_cond.device, dtype=torch.long)
            ).reshape(B, -1)
            global_cond = torch.cat([base_global_cond, positive_embedding], dim=-1)
            global_cond_uncond = torch.cat(
                [base_global_cond, torch.zeros_like(positive_embedding)], dim=-1
            )
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            if exc_horizon > 0:
                cond_data[:, :exc_horizon, ...] = exc_action
                cond_mask[:, :exc_horizon, ...] = True
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
            if exc_horizon > 0:
                cond_data[:, :exc_horizon, :Da] = exc_action
                cond_mask[:, :exc_horizon, :Da] = True
            global_cond_uncond = None

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            cfg_scale=self.positive_cfg_scale,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1 + exc_horizon
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, is_positive: torch.BoolTensor, obs_preprocessed: bool = False):
        # is_positive: shape [B]
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # normalize image by hand (skip when workspace already prepared obs)
        if not obs_preprocessed:
            nobs['image'] /= 255.0
            if nobs['image'].shape[-1] == 3:
                if len(nobs['image'].shape) == 5:
                    nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
                if len(nobs['image'].shape) == 4:
                    nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
            if "wrist_img" in nobs:
                nobs["wrist_img"] /= 255.0
                if nobs["wrist_img"].shape[-1] == 3:
                    if len(nobs["wrist_img"].shape) == 5:
                        nobs["wrist_img"] = nobs["wrist_img"].permute(0, 1, 4, 2, 3)
                    if len(nobs["wrist_img"].shape) == 4:
                        nobs["wrist_img"] = nobs["wrist_img"].permute(0, 3, 1, 2)
        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)

        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        intervention = batch["intervention"]
        is_positive = is_positive.bool() | intervention.to(is_positive.device)
        positive_embedding = self.is_positive_embedding(is_positive.long()).reshape(batch_size, -1)  # [B, emb]

        # Classifier-Free Guidance training: randomly drop positive condition.
        if self.training and self.positive_cond_drop_prob > 0:
            drop_mask = (torch.rand(batch_size, 1, device=positive_embedding.device)
                         < self.positive_cond_drop_prob)
            positive_embedding = positive_embedding.masked_fill(drop_mask, 0.0)  # 用全0作为空条件

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...]  # 只用第一个观测
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = torch.cat([nobs_features.reshape(batch_size, -1), positive_embedding], dim=-1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # RTC前缀训练：随机选择真实动作前缀并直接注入，且前缀位置不计算loss
        delay = torch.randint(
            0, self.max_latency_steps + 1, (bsz,),
            device=trajectory.device, dtype=torch.int64
        )
        keep_cond = torch.rand(bsz, device=trajectory.device) >= self.cond_dropout_prob
        prefix_mask = torch.arange(horizon, device=trajectory.device).unsqueeze(0) < delay.unsqueeze(1)  # [B,T]
        prefix_mask = prefix_mask & keep_cond.unsqueeze(1)
        noisy_trajectory[prefix_mask] = cond_data[prefix_mask]

        # compute loss mask
        loss_mask = ~condition_mask  # 作为条件的位置为0，让该位置loss为0
        if "mask" in batch:
            padding_mask = batch['mask'].unsqueeze(-1).to(trajectory.device)  # [B,T,1]
            loss_mask = loss_mask & padding_mask
        loss_mask = loss_mask & ~prefix_mask.unsqueeze(-1)

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
