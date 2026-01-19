from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
#from diffusion_policy_3d.model.vision.transformer_obs_encoder import TransformerObsEncoder
from diffusion_policy_3d.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy_3d.model.diffusion.DiTx import DiTX
from diffusion_policy_3d.model.common.sample_util import *

class Flow_DIT_ImagePolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=128,
            visual_cond_len=1024,
            n_layer=6,
            n_head=8,
            n_emb=768,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            obs_encoder: TimmObsEncoder = None,
            language_conditioned=False,
            sample_t_mode="beta", 
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        # create Flow_DITx model
        #obs_feature_dim = obs_encoder.embed_dim DINO
        #obs_feature_dim = n_emb # CLIP
        obs_feature_dim = int((obs_encoder.output_shape()[-1]))# R3M shape [B,512+agent_pose.shape]
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None

     
        cprint(f"[Flow_DIT_ImagePolicy] Using DiTX model", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            visual_cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            language_conditioned=language_conditioned,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.sample_t_mode = sample_t_mode
   
        cprint(f"[Flow_DIT_ImagePolicy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        print_params(self)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = obs_dict.copy()
            
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2) 
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
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...])
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        vis_cond = nobs_features.reshape(B, -1, Do)
        lang_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"
            
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        return action

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs['image'] /= 255.0 #归一化到[0,1]相当于transforms.ToTensor()
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim#视觉编码器token维度
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # condition through visual feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device) 
        vis_cond = nobs_features.reshape(B, -1, Do) # B, self.n_obs_steps*L, Do
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95),
            eps: float = 1.0e-8
        ) -> torch.optim.Optimizer:

        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[Flow_DIT_ImagePolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map.'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        else:
            cprint(f"[Flow_DIT_ImagePolicy] Use same lr for obs_encoder: {lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                    backbone_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": weight_decay,#用一样的weight_decay
            })

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, eps=eps
        )
        return optimizer
    
    def sample_t(self, batch_size, mode):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', 'mode', 'cosmap', 'beta', 'discrete'.")
        return t

    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        """
        Linear interpolation between noise and target data with optional noise preservation.
        
        Args:
            noise (Tensor): Initial noise at t=0
            target (Tensor): Target data point at t=1  
            timestep (float): Interpolation parameter in [0, 1]
                            t=0 returns pure noise, t=1 returns target + epsilon*noise
            epsilon (float): Noise preservation factor. Controls minimum noise retained.
                            Default 0.0 for standard linear interpolation.
                            
        Returns:
            Tensor: Interpolated data point at given timestep
            
        Examples:
            >>> # Standard linear interpolation (epsilon=0)
            >>> result = linear_interpolate(noise, data, 0.5)  # 50% noise + 50% data
            
            >>> # With noise preservation (epsilon=0.01) 
            >>> result = linear_interpolate(noise, data, 1.0, epsilon=0.01)  # data + 1% noise
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        
        return interpolated_data_point

    
    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training.
        Flow training is used to train the model to predict instantaneous velocity given a timestep.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device) 
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            pred = self.model(x, ti, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, **kwargs):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nobs['image'] /= 255.0 #在obsencoder进行归一化
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        if "mask" in batch:
            padding_mask=batch['mask'].unsqueeze(-1).to(nactions.device)#[B,T,1] 为1的位置为有效数据
        # handle different ways of passing observation
        local_cond = None
        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            # we assume language condition is passed as 'task_name'
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
        flow_target_dict=self.get_flow_velocity(nactions, 
                                                    vis_cond=vis_cond,
                                                    lang_cond=lang_cond if lang_cond is not None else None)
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'], 
            timestep=flow_target_dict['t'].squeeze(),#shape [B]
            vis_cond=vis_cond,
            lang_cond=flow_target_dict['lang_cond'] if lang_cond is not None else None)
        """Compute flow matching loss"""
        v_flow_target = flow_target_dict['v_target']
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow = loss_flow * padding_mask.to(dtype=loss_flow.dtype)
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean')
        loss = loss_flow.mean()
        
        return loss
        