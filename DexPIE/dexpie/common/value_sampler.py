from typing import Optional
import numpy as np
import numba
from dexpie.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
# Create indices for selected episodes.
def create_indices(
    episode_ends: np.ndarray,
    episode_mask: np.ndarray,
) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape

    n_selected = np.sum(episode_mask)
    indices = np.zeros((n_selected, 4), dtype=np.int64)
    write_idx = 0
    for i in range(len(episode_ends)):  # Iterate over each episode.
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx  # Current episode length.

        indices[write_idx, 0] = i
        indices[write_idx, 1] = start_idx
        indices[write_idx, 2] = end_idx
        indices[write_idx, 3] = episode_length
        write_idx += 1

    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
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
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        obs_keys,
        max_length: int,
        fail_rate: float = 0.1,
        fail_gamma: float = 1.0,
        episode_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()

        if not 0.0 <= fail_gamma <= 1.0:
            raise ValueError(f"fail_gamma must be in [0, 1], got {fail_gamma}")

        fail_reward = -max_length * fail_rate
        episode_ends = replay_buffer.episode_ends[:]  # Episode end indices.
        if ('success' in replay_buffer.meta) and (replay_buffer.meta['success'] is not None):
            # Same length as episode_ends; 1 means success and 0 means failure.
            success = replay_buffer.meta['success'][:].astype(np.bool_)
            if success.shape[0] != episode_ends.shape[0]:
                raise ValueError(
                    f"meta['success'] length {success.shape[0]} does not match n_episodes {episode_ends.shape[0]}"
                )
        else:
            success = np.ones_like(episode_ends, dtype=bool)

        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        rewards_list = []
        frame_indices_list = []
        for idx in range(len(indices)):
            episode_idx, start_idx, end_idx, episode_length = indices[idx]
            rewards = self.progess_reward(episode_length)  # Compute cumulative return for each observation.
            if not success[episode_idx]:  # Failed episode.
                # Treat failure as a terminal penalty and discount it backward in time.
                steps_to_failure = np.arange(
                    episode_length - 1, -1, -1, dtype=np.float32
                )
                failure_discount = np.power(
                    np.float32(fail_gamma), steps_to_failure
                )
                rewards = rewards + fail_reward * failure_discount
            rewards_list.append(rewards)
            frame_indices_list.append(
                np.arange(start_idx, end_idx, dtype=np.int64)
            )

        if len(rewards_list) == 0:
            self.rewards = np.zeros((0,), dtype=np.float32)
            self.frame_indices = np.zeros((0,), dtype=np.int64)
        else:
            self.rewards = np.concatenate(rewards_list)
            self.frame_indices = np.concatenate(frame_indices_list)
        self.length = len(self.rewards)
        assert self.length == len(self.frame_indices)
        # Concatenate all observation returns for later observation-return pairing.

        self.keys = list(obs_keys)
        self.replay_buffer = replay_buffer

    def __len__(self):
        return self.length

    def progess_reward(self, episode_length):

        # Compute per-step cumulative return from the current step to the final step.
        returns = np.arange(1 - episode_length, 1, step=1, dtype=np.float32)  # np.array [length]

        return returns

    def sample_sequence(self, idx):
        result = dict()
        frame_idx = self.get_frame_index(idx)

        for key in self.keys:
            result[key] = self.replay_buffer[key][frame_idx]
        result['reward'] = np.array([self.rewards[idx]])

        return result

    def get_frame_index(self, idx):
        return int(self.frame_indices[idx])
