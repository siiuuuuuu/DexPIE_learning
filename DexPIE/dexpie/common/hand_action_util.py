import numpy as np

def hand_action_util(hand_action: np.ndarray):
    """
    hand_action: shape [6]
    """
    hand_action=hand_action*1000 # Unnormalize to 0-1000.
    hand_action=np.clip(hand_action, 0, 1000)
    pinky_angle = int(hand_action[0])
    ring_angle = int(hand_action[1])
    middle_angle = int(hand_action[2])
    index_angle = int(hand_action[3])
    thumb_angle_2 = int(hand_action[4])
    thumb_angle = int(hand_action[5])
    return [pinky_angle, ring_angle, middle_angle, index_angle, thumb_angle_2, thumb_angle]
