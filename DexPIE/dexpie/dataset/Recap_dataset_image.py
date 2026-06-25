from typing import Dict
import torch
import numpy as np
import copy
from dexpie.common.pytorch_util import dict_apply
from dexpie.common.replay_buffer import ReplayBuffer
from dexpie.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from dexpie.common.tools import MATHTOOLS
from dexpie.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from dexpie.dataset.base_dataset import BaseDataset
from termcolor import cprint
from scipy.ndimage import zoom

class RecapDatasetImage(BaseDataset):
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
            use_precomputed_advantage=True,
            advantage_key='advantage',
            ):
        super().__init__()
        cprint(f'Loading RecapDatasetImage from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_act_normal = use_act_normal
        self.use_img = use_img
        self.use_wrist_img = use_wrist_img
        self.use_depth = use_depth
        self.use_relative_action = use_relative_action
        self.use_precomputed_advantage = use_precomputed_advantage
        self.advantage_key = advantage_key
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tools=MATHTOOLS()

        buffer_keys = [
            'state', 
            'action',
        ] 
        disk_replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        self.has_intervention = 'intervention' in disk_replay_buffer.keys()
        self.has_advantage = self.use_precomputed_advantage and (self.advantage_key in disk_replay_buffer.keys())
        if self.has_intervention:
            buffer_keys.append('intervention')
        else:
            cprint("No 'intervention' in dataset, mark expert samples as intervention-positive.", 'yellow')
        if self.has_advantage:
            buffer_keys.append(self.advantage_key)
            cprint(f"Using precomputed advantage from data/{self.advantage_key}.", 'cyan')
        elif self.use_precomputed_advantage:
            cprint(
                f"No data/{self.advantage_key} in dataset; Recap/DexPIE training requires offline advantage labels.",
                'yellow'
            )
        if self.use_img:
            buffer_keys.append('img') # Match the image key used during data collection.
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
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)# Fit normalizer.
        else:
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()# Identity transform as a placeholder.

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
        valid_mask = sample['mask']
        valid_steps = int(valid_mask.sum())
        if 'intervention' in sample:
            if valid_steps > 0:
                intervention_num = int(sample['intervention'][valid_mask].sum())
            else:
                intervention_num = 0
            intervent_positive = np.array(
                intervention_num > (valid_steps / 3.0), dtype=np.bool_
            )  # True only when intervention steps exceed one third of valid steps; nearby autonomous steps are usually negatives.
        else:
            intervent_positive = np.array(True, dtype=np.bool_)  # Mark pure expert data as intervention-positive when this field is missing.
        agent_pos = sample['state'][[0,-1],:6].astype(np.float32)
        current_agent_pose = sample['state'][:self.n_obs_steps, 6:].astype(np.float32)
        # Use the current pose as the action reference, which is simpler than using the previous action.
        if self.use_img:
            image = sample['img'][[0,-1],].astype(np.float32) # Use first and last images to compute advantage from two observations.
        if self.use_wrist_img:
            wrist_img = sample['wrist_img'][[0,-1],].astype(np.float32)
        if self.use_depth:
            depth = sample['depth'][[0,-1],].astype(np.float32)
        if self.use_relative_action:
            arm_action=sample['action'][:,:9]
            pose=self.tools.xyz_6drot_to_mat(arm_action)
            if current_agent_pose.shape[-1]<6: # If current pose is missing, use the first action pose as reference.
                pose_0=pose[0]# Current action pose at sequence start is the reference pose.
                inv_pose_0=self.tools.se3_inverse(pose_0)
            else:
                pose_0=self.tools.xyz_rotvec_to_mat(current_agent_pose[0])# xyz+rotvec format.
                inv_pose_0=self.tools.se3_inverse(pose_0)
            Relative_pose=np.einsum("ij,njk->nik",inv_pose_0, pose)# Relative pose with respect to the reference pose.
            Relative_act=self.tools.mat2xyz_6drot(Relative_pose)
            action=np.concatenate([Relative_act,sample['action'][:,9:]],axis=-1)# Relative pose plus absolute joint angles.
            

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
        if self.has_advantage:
            valid_indices = np.flatnonzero(sample["mask"])
            first_valid_idx = int(valid_indices[0]) if len(valid_indices) > 0 else 0
            data['advantage'] = np.array(sample[self.advantage_key][first_valid_idx], dtype=np.float32)
        
        data['mask']=sample["mask"]
        data['intervention']=intervent_positive
        return data 
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data # Return torch tensor.
