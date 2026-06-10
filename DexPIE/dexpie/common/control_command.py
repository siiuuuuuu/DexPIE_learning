from dataclasses import dataclass

import numpy as np


@dataclass
class ControlCommand:
    target_pose: object
    hand_command: object
    action: np.ndarray
    intervention: int

    @classmethod
    def from_intervention(cls, pose_command, hand_action_raw):
        hand_command = np.asarray(hand_action_raw, dtype=np.float32)
        hand_action_array = np.clip(hand_command, 0, 1000).astype(np.float32) / 1000.0
        action = np.concatenate((pose_command["arm_action"], hand_action_array))
        return cls(
            target_pose=pose_command["target_pose"],
            hand_command=hand_command,
            action=action,
            intervention=1,
        )

    @classmethod
    def from_policy(cls, policy_command, hand_action_converter):
        return cls(
            target_pose=policy_command["target_pose"],
            hand_command=hand_action_converter(policy_command["hand_action"]),
            action=policy_command["action"],
            intervention=0,
        )
