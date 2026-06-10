import numpy as np


def default_tracker_to_tcp_mat():
    return np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


class InterventionPoseProcessor:
    """Converts expert tracker poses into target arm actions during intervention."""

    def __init__(self, tool, tracker_to_tcp_mat=None):
        self.tool = tool
        self.tracker_to_tcp_mat = (
            default_tracker_to_tcp_mat()
            if tracker_to_tcp_mat is None
            else np.asarray(tracker_to_tcp_mat, dtype=float)
        )
        self.init_tracker_mat = None
        self.init_arm_mat = None

    def clear_reference(self):
        self.init_tracker_mat = None
        self.init_arm_mat = None

    def has_reference(self):
        return self.init_tracker_mat is not None and self.init_arm_mat is not None

    def tracker_to_tcp(self, tracker_mat):
        return np.dot(np.asarray(tracker_mat, dtype=float), self.tracker_to_tcp_mat)

    def reset_reference(self, tracker_mat, arm_mat):
        self.init_tracker_mat = self.tracker_to_tcp(tracker_mat)
        self.init_arm_mat = np.asarray(arm_mat, dtype=float).copy()

    def compute(self, tracker_mat):
        if not self.has_reference():
            raise RuntimeError("intervention pose reference is not initialized")

        current_tracker_mat = self.tracker_to_tcp(tracker_mat)
        increment_mat = np.dot(
            self.tool.se3_inverse(self.init_tracker_mat),
            current_tracker_mat,
        )
        target_arm_mat = np.dot(self.init_arm_mat, increment_mat)
        return {
            "increment_matrix": increment_mat,
            "target_arm_mat": target_arm_mat,
            "arm_action": self.tool.mat2xyz_6drot(target_arm_mat),
            "target_pose": self.tool.mat2xyz_rotvec(target_arm_mat),
        }
