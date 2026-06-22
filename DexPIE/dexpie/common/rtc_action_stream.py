import numpy as np
import ray


class RTCActionStream:
    """Maintains async RTC policy chunks and exposes only valid future actions."""

    def __init__(self, policy_actor, tool, use_wrist_img=False, max_latency_step=3):
        self.policy_actor = policy_actor
        self.tool = tool
        self.use_wrist_img = bool(use_wrist_img)
        self.max_latency_step = int(max_latency_step)
        self.reset()

    def reset(self):
        self.initialized = False
        self.action_horizon = 0
        self.action_cursor = 0
        self.np_action = None
        self.arm_mats = None
        self.policy_ref_mat = None
        self.pending_ref = None

    def step(self, qpos, image, wrist_image, current_arm_mat):
        """Advance the 25Hz RTC stream by one slot.

        The returned command is the current 25Hz slot for compatibility. When a
        fresh policy chunk becomes available, command["valid_sequence"] contains
        only future actions that are not RTC overlap/padding from the previous
        chunk.
        """
        obs = self._make_obs_dict(qpos, image, wrist_image, exc_action=None)
        valid_sequence = None
        if not self.initialized:
            self._initialize(obs, current_arm_mat)
            valid_sequence = self._make_valid_sequence(
                valid_start=0,
                start_delay_steps=0,
            )

        if self._should_prefetch():
            self._prefetch(obs, current_arm_mat)

        command = self._command_at_cursor()
        self.action_cursor += 1
        if self.action_cursor >= self.action_horizon:
            next_sequence = self._advance_segment()
            if next_sequence is not None:
                valid_sequence = next_sequence

        command["valid_sequence"] = valid_sequence
        return command

    def _make_obs_dict(self, qpos, image, wrist_image, exc_action):
        obs_dict = {
            "agent_pos": np.stack([qpos], axis=0)[None, ...],
            "image": np.stack([image], axis=0)[None, ...],
            "exc_action": exc_action,
        }
        if self.use_wrist_img:
            obs_dict["wrist_img"] = np.stack([wrist_image], axis=0)[None, ...]
        return obs_dict

    def _initialize(self, obs_dict, current_arm_mat):
        action = ray.get(self.policy_actor.inference.remote(obs_dict))
        self.np_action = action.numpy()
        self.action_horizon = self.np_action.shape[0]
        self.action_cursor = 0
        self.policy_ref_mat = current_arm_mat
        self.arm_mats = self._absolute_arm_mats(self.policy_ref_mat, self.np_action)
        self.pending_ref = None
        self.initialized = True

    def _should_prefetch(self):
        return (
            self.action_horizon > self.max_latency_step
            and self.action_cursor == self.action_horizon - self.max_latency_step
        )

    def _prefetch(self, obs_dict, current_arm_mat):
        self.policy_ref_mat = current_arm_mat
        exc_arm_mats = np.einsum(
            "ij,tjk->tik",
            self.tool.se3_inverse(self.policy_ref_mat),
            self.arm_mats[-self.max_latency_step :],
        )
        exc_arm_action = self.tool.mat2xyz_6drot(exc_arm_mats).astype(np.float32)
        exc_hand_action = self.np_action[-self.max_latency_step :, 9:].astype(np.float32)
        exc_action = np.concatenate([exc_arm_action, exc_hand_action], axis=-1)[
            None,
            ...,
        ]
        prefetch_obs = dict(obs_dict)
        prefetch_obs["exc_action"] = exc_action
        self.pending_ref = self.policy_actor.inference.remote(prefetch_obs)

    def _advance_segment(self):
        if self.pending_ref is None:
            self.initialized = False
            return None

        action = ray.get(self.pending_ref)
        self.np_action = action.numpy()
        self.action_horizon = self.np_action.shape[0]
        if self.action_horizon <= 0:
            self.initialized = False
            self.pending_ref = None
            return None

        valid_start = min(self.max_latency_step, self.action_horizon - 1)
        self.action_cursor = valid_start
        self.arm_mats = self._absolute_arm_mats(self.policy_ref_mat, self.np_action)
        self.pending_ref = None
        self.initialized = True
        return self._make_valid_sequence(
            valid_start=valid_start,
            start_delay_steps=1,
        )

    def _command_at_cursor(self):
        if self.action_cursor >= self.action_horizon:
            raise RuntimeError(
                f"action cursor out of range: {self.action_cursor} >= {self.action_horizon}"
            )
        arm_mat = self.arm_mats[self.action_cursor]
        hand_action = self.np_action[self.action_cursor, 9:].astype(np.float32)
        action = np.concatenate(
            [self.tool.mat2xyz_6drot(arm_mat), hand_action],
            axis=-1,
        ).astype(np.float32)
        return {
            "arm_mat": arm_mat,
            "target_pose": self.tool.mat2xyz_rotvec(arm_mat),
            "hand_action": hand_action,
            "action": action,
        }

    def _make_valid_sequence(self, valid_start, start_delay_steps=0):
        valid_start = int(valid_start)
        if self.np_action is None or valid_start >= self.action_horizon:
            return None

        arm_mats = self.arm_mats[valid_start:].copy()
        hand_actions = self.np_action[valid_start:, 9:].astype(np.float32).copy()
        arm_actions = self.tool.mat2xyz_6drot(arm_mats).astype(np.float32)
        target_poses = np.asarray(
            [self.tool.mat2xyz_rotvec(arm_mat) for arm_mat in arm_mats],
            dtype=np.float32,
        )
        actions = np.concatenate([arm_actions, hand_actions], axis=-1).astype(np.float32)
        return {
            "valid_start": valid_start,
            "horizon": self.action_horizon,
            "start_delay_steps": int(start_delay_steps),
            "arm_mats": arm_mats,
            "target_poses": target_poses,
            "arm_actions": arm_actions,
            "hand_actions": hand_actions,
            "actions": actions,
        }

    def _absolute_arm_mats(self, policy_ref_mat, np_action):
        relative_arm_mat = self.tool.xyz_6drot_to_mat(np_action[:, :9])
        return np.einsum("ij,tjk->tik", policy_ref_mat, relative_arm_mat)
