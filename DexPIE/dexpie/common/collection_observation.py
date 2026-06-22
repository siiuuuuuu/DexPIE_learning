from dataclasses import dataclass
import time

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
    timestamp_dict: dict


class ObservationBuilder:
    """Reads synchronized collection inputs needed by policy/intervention code."""

    def __init__(
        self,
        camera,
        robot,
        tool,
        use_wrist_img=False,
        robot_state_reader=None,
        alignment_tolerance_ms=25.0,
    ):
        self.camera = camera
        self.robot = robot
        self.tool = tool
        self.use_wrist_img = bool(use_wrist_img)
        self.robot_state_reader = robot_state_reader
        self.alignment_tolerance_ms = float(alignment_tolerance_ms)

    def read(self):
        t_record_start_ns = time.monotonic_ns()
        cam_dict = self.camera()
        t_camera_read_ns = time.monotonic_ns()

        image = cam_dict["front_color"]
        wrist_image = cam_dict["right_color"] if self.use_wrist_img else None

        front_meta = cam_dict.get("front_meta", {})
        wrist_meta = cam_dict.get("right_meta", {}) if self.use_wrist_img else {}
        anchor_ns = front_meta.get("t_host_ns")
        if anchor_ns is None:
            anchor_ns = t_camera_read_ns

        robot_obs = self._read_robot_obs(anchor_ns)
        if robot_obs is None:
            return None

        robot_obs_ns = robot_obs.get("t_robot_obs_host_ns")
        robot_delta_ms = _delta_ms(robot_obs_ns, anchor_ns)
        if (
            robot_delta_ms is not None
            and abs(robot_delta_ms) > self.alignment_tolerance_ms
        ):
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
            timestamp_dict={
                "t_record_start_ns": t_record_start_ns,
                "t_camera_read_ns": t_camera_read_ns,
                "t_anchor_ns": anchor_ns,
                "t_front_camera_host_ns": front_meta.get("t_host_ns"),
                "front_camera_seq": front_meta.get("seq"),
                "front_camera_frame_no": front_meta.get("frame_no"),
                "front_camera_dev_ts": front_meta.get("t_dev_ts"),
                "t_wrist_camera_host_ns": wrist_meta.get("t_host_ns"),
                "wrist_camera_seq": wrist_meta.get("seq"),
                "wrist_camera_frame_no": wrist_meta.get("frame_no"),
                "wrist_camera_dev_ts": wrist_meta.get("t_dev_ts"),
                "t_robot_obs_host_ns": robot_obs_ns,
                "t_aligned_robot_obs_ns": robot_obs_ns,
                "sync_delta_robot_obs_ms": robot_delta_ms,
                "sync_delta_wrist_camera_ms": _delta_ms(
                    wrist_meta.get("t_host_ns"),
                    anchor_ns,
                ),
            },
        )

    def _read_robot_obs(self, anchor_ns):
        if self.robot_state_reader is not None:
            return self.robot_state_reader.obs_at_time_ns(anchor_ns)

        robot_obs = self.robot.get_obs()
        if robot_obs is None:
            return None

        robot_obs = dict(robot_obs)
        robot_obs["t_robot_obs_host_ns"] = time.monotonic_ns()
        return robot_obs


def _delta_ms(t_ns, ref_ns):
    if t_ns is None or ref_ns is None:
        return None
    return (int(t_ns) - int(ref_ns)) / 1e6
