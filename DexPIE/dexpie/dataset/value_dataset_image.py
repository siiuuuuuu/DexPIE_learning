from typing import Dict
import torch
import numpy as np
import copy
from dexpie.common.pytorch_util import dict_apply
from dexpie.common.replay_buffer import ReplayBuffer
from dexpie.common.value_sampler import (SequenceSampler, get_val_mask, downsample_mask)
from dexpie.common.memmap_dataset import (
    resolve_dataset_storage,
    validate_visual_lengths,
)
from dexpie.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from dexpie.dataset.base_dataset import BaseDataset
from dexpie.common.discretizer import UniformDiscretizer
from termcolor import cprint
from scipy.ndimage import zoom

class ValueDatasetImage(BaseDataset):
    def __init__(self,
            zarr_path, 
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_img=True,
            use_wrist_img=False,
            max_length=1000,
            fail_rate=0.1,
            fail_gamma=1.0,
            num_bins=201,
            v_min=-1,
            v_max=0,
            storage_mode='auto',
            ):
        super().__init__()
        cprint(f'Loading Dataset from {zarr_path}', 'green')
        self.use_img = use_img
        self.use_wrist_img = use_wrist_img
        self.max_length = max_length
        self.fail_rate = fail_rate
        self.fail_gamma = fail_gamma

        self.discretizer = UniformDiscretizer(num_bins=num_bins, v_min=v_min, v_max=v_max)
        visual_keys = []
        if self.use_img:
            visual_keys.append('img')
        if self.use_wrist_img:
            visual_keys.append('wrist_img')
        self.storage_mode, replay_zarr_path, self.visual_readers = \
            resolve_dataset_storage(
                zarr_path=zarr_path,
                storage_mode=storage_mode,
                visual_keys=visual_keys,
            )

        self.buffer_keys = [
            'state', 
            ]
        if self.storage_mode == 'memory':
            self.buffer_keys.extend(visual_keys)
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            replay_zarr_path, keys=self.buffer_keys)
        validate_visual_lengths(
            self.visual_readers, self.replay_buffer.n_steps
        )
        
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
            obs_keys=self.buffer_keys,
            max_length=self.max_length,
            fail_rate=self.fail_rate,
            fail_gamma=self.fail_gamma,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_wrist_img:
            normalizer['wrist_img'] = SingleFieldLinearNormalizer.create_identity()
        
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)[None, :6]
        reward = sample['reward'].astype(np.float32)
        # Normalize reward.
        norm_reward = torch.from_numpy(np.clip(reward/self.max_length, -1, 0)) # Normalize to [-1, 0].
        target_dist = self.discretizer.discretize_to_gs_distribution(norm_reward, gaussian_std=1)# Gaussian probability, shape [1, num_bins].
        if self.use_img:
            # Keep RGB observations as uint8 until they reach the target device.
            image = sample['img'][:][None, :]
        if self.use_wrist_img:
            wrist_img = sample['wrist_img'][:][None, :]
            
        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'target': target_dist # Target distribution for cross entropy.
                } # Shape is (1, C); add a time dimension here.
        if self.use_img:
            data['obs']['image'] = image
        if self.use_wrist_img:
            data['obs']['wrist_img'] = wrist_img

        return data 
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx) #
        if self.storage_mode == 'memmap':
            frame_idx = self.sampler.get_frame_index(idx)
            for key, reader in self.visual_readers.items():
                sample[key] = reader.read_index(frame_idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data # Return torch tensor.
