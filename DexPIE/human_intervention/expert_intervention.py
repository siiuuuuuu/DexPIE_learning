import numpy as np
from human_intervention.shared_manus_inspire import inspire_Manus
from human_intervention import triad_openvr
import time
# Wrap teleoperation as a background expert-intervention subprocess.
class ExpertIntervention(object):
    def __init__(self, use_right_hand=True, use_left_hand=False):
        # Initialize.
        self.human_intervention = False
        self.hand_action = inspire_Manus(use_right_hand=use_right_hand, use_left_hand=use_left_hand)
        self.tracker = triad_openvr.triad_openvr()# Connect tracker; it is already running in its own process.
        self.tracker.print_discovered_objects()

    def start(self):
        self.hand_action.start()# Start hand subprocess.

    # Callback that returns the current human-intervention action.
    def __call__(self):

        tracker_data=np.eye(4)
        pose_mat=self.tracker.devices["tracker_1"].get_pose_matrix()# Get current tracker pose.

        if pose_mat is None:
            return None, None # Invalid callback data; skip this frame.
        
        pose_np = np.array([
            [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2], pose_mat[0][3]],
            [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2], pose_mat[1][3]],
            [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2], pose_mat[2][3]]
        ])
        tracker_data[:3,:4] = pose_np 
        hand_action=self.hand_action()["right"] # Get latest glove action.

        return tracker_data, hand_action
        
    def finalize(self):
        if hasattr(self, "hand_action"):
            self.hand_action.finalize()
        if hasattr(self, "tracker"):
            del self.tracker
