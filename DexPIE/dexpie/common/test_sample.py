from typing import Optional
import numpy as np
import numba
from dexpie.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)

# Create indices and use padding to keep sequence lengths consistent.
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):# Iterate over each episode.
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx # Current episode length.
        
        min_start = -pad_before # Minimum start index; negative values require past observations/actions.
        max_start = episode_length - sequence_length + pad_after # Maximum start index; equals episode_length-1 when H=n_action.
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):# Iterate over all sequence start positions in this episode.
            if idx==0:
                pad_flag = True
            else:
                pad_flag = False
            buffer_start_idx = max(idx, 0) + start_idx# Clamp to this episode; minimum is episode start.
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx# Clamp to this episode; maximum is episode end.
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            # Sample start offset; nonzero means leading padding is needed.
            sample_end_idx = sequence_length - end_offset
            # Sample end offset; values below sequence_length mean trailing padding is needed.
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, pad_flag, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)# Full index array.
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]# Episode end indices.
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, pad_flag, sample_start_idx, sample_end_idx \
            = self.indices[idx]# Sample sequence index.
        result = dict()

        mask =np.zeros(self.sequence_length, dtype=bool)
        mask[sample_start_idx:sample_end_idx] = True
        result['mask'] = mask# Mask out padded positions from loss computation.

        for key in self.keys:
            input_arr = self.replay_buffer[key]

            if key == 'action':
                # Action data uses length sequence_length+1 and samples from buffer_start_idx-1 to buffer_end.
                if pad_flag:
                    actual_buffer_start_idx=buffer_start_idx
                    # Sequence start; there is no previous action, so use the next value as the baseline.
                else:
                    actual_buffer_start_idx = buffer_start_idx - 1 
                actual_buffer_end_idx = buffer_end_idx

                # Get sample data.
                if key not in self.key_first_k:
                    sample = input_arr[actual_buffer_start_idx:actual_buffer_end_idx]
                
                # Create sequence_length+1 output for action data.
                data = np.zeros(
                    shape=(self.sequence_length + 1,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if pad_flag:
                    data[sample_start_idx+1:sample_end_idx+1] = sample
                    data[:sample_start_idx+1] = sample[0]# Pad the front with the first value.
                else:
                    data[sample_start_idx:sample_end_idx+1] = sample

                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx+1:] = sample[-1]# Pad the back with the last value.
            else:
            # performance optimization, avoid small allocation if possible
                if key not in self.key_first_k:
                    sample = input_arr[buffer_start_idx:buffer_end_idx]# Directly slice required data.
                else:
                    # performance optimization, only load used obs steps
                    n_data = buffer_end_idx - buffer_start_idx
                    k_data = min(self.key_first_k[key], n_data)
                    # fill value with Nan to catch bugs
                    # the non-loaded region should never be used
                    sample = np.full((n_data,) + input_arr.shape[1:], 
                        fill_value=np.nan, dtype=input_arr.dtype)
                    try:
                        sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                    except Exception as e:
                        import pdb; pdb.set_trace()
                data = sample
                if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):# Padding is needed.
                    data = np.zeros(
                        shape=(self.sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype)
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]# Pad the front with the first value.
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]# Pad the back with the last value.
                    data[sample_start_idx:sample_end_idx] = sample# Copy the middle region directly.

            result[key] = data
        return result
