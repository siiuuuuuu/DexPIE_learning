from dataclasses import dataclass

import numpy as np


@dataclass
class CollectionObservation:
    cam_dict: dict
    image: np.ndarray
    wrist_image: np.ndarray
    qpos: np.ndarray
    tcp_pose: np.ndarray
    robot_state: np.ndarray
    arm_mat: np.ndarray


class ObservationBuilder:
    """Reads synchronized collection inputs needed by policy/intervention code."""

    def __init__(self, camera, robot, tool, use_wrist_img=False):
        self.camera = camera
        self.robot = robot
        self.tool = tool
        self.use_wrist_img = bool(use_wrist_img)

    def read(self):
        cam_dict = self.camera()
        image = cam_dict["front_color"]
        wrist_image = cam_dict["right_color"] if self.use_wrist_img else None

        robot_obs = self.robot.get_obs()
        if robot_obs is None:
            return None

        tcp_pose = robot_obs["tcp_pose"]
        return CollectionObservation(
            cam_dict=cam_dict,
            image=image,
            wrist_image=wrist_image,
            qpos=robot_obs["joint"],
            tcp_pose=tcp_pose,
            robot_state=robot_obs["state"],
            arm_mat=self.tool.xyz_rotvec_to_mat(tcp_pose),
        )
