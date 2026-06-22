import numpy as np
from human_intervention.SM_inspire_manus_zmq import inspire_Manus
from human_intervention import triad_openvr


# Wrap teleoperation as a background expert-intervention source.
class ExpertIntervention(object):
    def __init__(
        self,
        use_right_hand=True,
        use_left_hand=False,
        manus_zmq_endpoint="tcp://127.0.0.1:2044",
        manus_control_threshold=10,
        manus_scale_factor=15,
        manus_zmq_rcvhwm=1,
        manus_zmq_conflate=True,
        manus_zmq_poll_timeout_ms=100,
    ):
        self.human_intervention = False
        self.hand_action = inspire_Manus(
            use_right_hand=use_right_hand,
            use_left_hand=use_left_hand,
            zmq_endpoint=manus_zmq_endpoint,
            control_threshold=manus_control_threshold,
            scale_factor=manus_scale_factor,
            rcvhwm=manus_zmq_rcvhwm,
            conflate=manus_zmq_conflate,
            poll_timeout_ms=manus_zmq_poll_timeout_ms,
        )
        self.tracker = triad_openvr.triad_openvr()
        self.tracker.print_discovered_objects()

    def start(self):
        self.hand_action.start()

    def latest_hand_sample(self):
        return self.hand_action.latest_right_sample()

    def read_tracker_mat(self):
        tracker_data = np.eye(4)
        pose_mat = self.tracker.devices["tracker_1"].get_pose_matrix()

        if pose_mat is None:
            return None

        pose_np = np.array([
            [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2], pose_mat[0][3]],
            [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2], pose_mat[1][3]],
            [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2], pose_mat[2][3]],
        ])
        tracker_data[:3, :4] = pose_np
        return tracker_data

    def read_hand_action(self):
        sample = self.latest_hand_sample()
        if sample is None:
            return np.zeros(6, dtype=np.float32)
        return sample["action"]

    def __call__(self):
        tracker_data = self.read_tracker_mat()
        if tracker_data is None:
            return None, None
        hand_action = self.read_hand_action()
        return tracker_data, hand_action

    def finalize(self):
        if hasattr(self, "hand_action"):
            self.hand_action.finalize()
        if hasattr(self, "tracker"):
            del self.tracker
