from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.common.tools import MATHTOOLS
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
from scipy.ndimage import zoom

class GR1DexDatasetImage(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            n_action_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_act_normal=False,
            use_img=True,
            use_wrist_img=False,
            use_depth=False,
            use_relative_action=True,
            ):
        super().__init__()
        cprint(f'Loading GR1DexDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_act_normal = use_act_normal
        self.use_img = use_img
        self.use_wrist_img = use_wrist_img
        self.use_depth = use_depth
        self.use_relative_action = use_relative_action
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tools=MATHTOOLS()

        buffer_keys = [
            'state', 
            'action',] 
        if self.use_img:
            buffer_keys.append('img') #对齐数据采集的img
        if self.use_wrist_img:
            buffer_keys.append('wrist_img')
        if self.use_depth:
            buffer_keys.append('depth')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        if self.use_act_normal:
            data = {'action': self.replay_buffer['action']}
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)#进行归一化
        else:
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()#恒等变换为了占位

        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_wrist_img:
            normalizer['wrist_img'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        #agent_pos = sample['state'][:,].astype(np.float32)#只取所需观察即前n_obs_steps个
        agent_pos = sample['state'][:self.n_obs_steps,:6].astype(np.float32)
        current_agent_pose = sample['state'][:self.n_obs_steps, 6:].astype(np.float32)
        #当前的位姿，给动作作为基准，比前一个动作作为基准简单直观多了
        if self.use_img:
            image = sample['img'][:self.n_obs_steps,].astype(np.float32)
        if self.use_wrist_img:
            wrist_img = sample['wrist_img'][:self.n_obs_steps,].astype(np.float32)
        if self.use_depth:
            depth = sample['depth'][:self.n_obs_steps,].astype(np.float32)
        if self.use_relative_action:
            arm_action=sample['action'][:,:9]
            pose=self.tools.xyz_6drot_to_mat(arm_action)
            if current_agent_pose.shape[-1]<6: #如果没有提供当前位姿，就用第一个动作位姿作为参考位姿
                pose_0=pose[0]#每个序列开始时的当前动作位姿为参考位姿
                inv_pose_0=self.tools.se3_inverse(pose_0)
            else:
                pose_0=self.tools.xyz_rotvec_to_mat(current_agent_pose[0])#其是xyz+rotvec格式
                inv_pose_0=self.tools.se3_inverse(pose_0)
            Relative_pose=np.einsum("ij,njk->nik",inv_pose_0, pose)#相对于参考位姿的相对位姿
            Relative_act=self.tools.mat2xyz_6drot(Relative_pose)
            action=np.concatenate([Relative_act,sample['action'][:,9:]],axis=-1)#相对位姿和绝对关节角
            

        data = {
            'obs': {
                'agent_pos': agent_pos,
                },}
        
        if self.use_img:
            data['obs']['image'] = image
        if self.use_wrist_img:
            data['obs']['wrist_img'] = wrist_img
        if self.use_depth:
            data['obs']['depth'] = depth
        if self.use_relative_action:
            data['action']=action.astype(np.float32)
        
        data['mask']=sample["mask"]

        return data #四元组 这里就应该obs只使用前n_obs个不然内存占用
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data #返回torch tensor

