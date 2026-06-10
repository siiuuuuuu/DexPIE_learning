import h5py
import numpy as np


class InterventionEpisodeBuffer:
    """Accumulates autonomous/intervention collection data for one episode."""

    def __init__(self, use_wrist_img=False):
        self.use_wrist_img = bool(use_wrist_img)
        self.robot_states = []
        self.front_images = []
        self.wrist_images = []
        self.actions = []
        self.interventions = []

    def __len__(self):
        return len(self.actions)

    def append(self, robot_state, cam_dict, action, intervention):
        self.robot_states.append(np.asarray(robot_state).copy())
        self.front_images.append(np.asarray(cam_dict["front_color"]).copy())
        if self.use_wrist_img:
            self.wrist_images.append(np.asarray(cam_dict["right_color"]).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.interventions.append(int(intervention))

    def to_arrays(self):
        arrays = {
            "color": np.asarray(self.front_images),
            "env_qpos_proprioception": np.asarray(self.robot_states),
            "action": np.asarray(self.actions, dtype=np.float32),
            "intervention": np.asarray(self.interventions, dtype=np.uint8),
        }
        if self.use_wrist_img:
            arrays["wrist_color"] = np.asarray(self.wrist_images)
        return arrays

    def save_h5(self, record_file_name, success=None):
        arrays = self.to_arrays()
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=arrays["color"])
            if self.use_wrist_img:
                f.create_dataset("wrist_color", data=arrays["wrist_color"])
            f.create_dataset(
                "env_qpos_proprioception",
                data=arrays["env_qpos_proprioception"],
            )
            f.create_dataset("action", data=arrays["action"])
            f.create_dataset("intervention", data=arrays["intervention"])
            if success is not None:
                f.attrs["success"] = bool(success)

        return {
            "seq_length": len(self),
            "color_shape": arrays["color"].shape,
            "wrist_color_shape": arrays.get("wrist_color", None).shape
            if self.use_wrist_img
            else None,
            "action_shape": arrays["action"].shape,
            "intervention_shape": arrays["intervention"].shape,
            "env_qpos_shape": arrays["env_qpos_proprioception"].shape,
            "record_file_name": record_file_name,
        }
