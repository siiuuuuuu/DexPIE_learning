"""High-rate smoothing for RTC policy action sequences."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PolicyTrajectorySmootherConfig:
    enabled: bool = True
    smooth_hz: float = 120.0
    w_track: float = 1.0
    w_anchor: float = 80.0
    w_vel: float = 0.01
    w_acc: float = 20.0
    max_velocity: float = 0.60
    max_acceleration: float = 6.0
    max_rot_speed_deg: float = 120.0
    smooth_hand: bool = True


class PolicyTrajectorySmoother:
    """Upsamples policy-rate waypoints and smooths task-space translation."""

    def __init__(self, tool, policy_dt, config=None, workspace_limits=None):
        self.tool = tool
        self.policy_dt = float(policy_dt)
        self.config = config or PolicyTrajectorySmootherConfig()
        self.workspace_limits = workspace_limits or {}

        if self.policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")
        if self.config.smooth_hz <= 0.0:
            raise ValueError("smooth_hz must be positive")

    def smooth(self, sequence, current_arm_mat=None):
        """Return a smoothed sequence and its timestamp period."""
        if sequence is None:
            return None, self.policy_dt

        arm_mats = np.asarray(sequence.get("arm_mats"), dtype=np.float64)
        if arm_mats.ndim != 3 or arm_mats.shape[0] == 0:
            return sequence, self.policy_dt

        smooth_dt = 1.0 / float(self.config.smooth_hz)
        ratio = max(1, int(round(self.policy_dt / smooth_dt)))
        if not self.config.enabled or ratio <= 1 or arm_mats.shape[0] <= 1:
            return dict(sequence), self.policy_dt

        smooth_dt = self.policy_dt / ratio
        policy_count = arm_mats.shape[0]
        smooth_count = (policy_count - 1) * ratio + 1
        policy_times = np.arange(policy_count, dtype=np.float64) * self.policy_dt
        smooth_times = np.arange(smooth_count, dtype=np.float64) * smooth_dt

        linear_positions = self._resample_array(
            arm_mats[:, :3, 3],
            policy_times,
            smooth_times,
        )
        positions = self._smooth_positions(
            linear_positions,
            arm_mats[:, :3, 3],
            ratio,
        )
        positions = self._limit_positions(
            positions,
            None if current_arm_mat is None else current_arm_mat[:3, 3],
            int(sequence.get("start_delay_steps", 0)),
        )

        rotations = self._resample_rotations(
            arm_mats[:, :3, :3],
            policy_times,
            smooth_times,
        )
        rotations = self._limit_rotations(
            rotations,
            None if current_arm_mat is None else current_arm_mat[:3, :3],
            int(sequence.get("start_delay_steps", 0)),
            smooth_dt,
        )

        smooth_mats = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], smooth_count, axis=0)
        smooth_mats[:, :3, :3] = rotations
        smooth_mats[:, :3, 3] = positions

        target_poses = np.asarray(
            [self.tool.mat2xyz_rotvec(arm_mat) for arm_mat in smooth_mats],
            dtype=np.float32,
        )
        arm_actions = self.tool.mat2xyz_6drot(smooth_mats).astype(np.float32)

        hand_actions = sequence.get("hand_actions")
        if hand_actions is not None:
            hand_actions = np.asarray(hand_actions, dtype=np.float32)
            if self.config.smooth_hand and hand_actions.shape[0] == policy_count:
                hand_actions = self._resample_array(
                    hand_actions,
                    policy_times,
                    smooth_times,
                ).astype(np.float32)

        smoothed = dict(sequence)
        smoothed["arm_mats"] = smooth_mats
        smoothed["target_poses"] = target_poses
        smoothed["arm_actions"] = arm_actions
        if hand_actions is not None:
            smoothed["hand_actions"] = hand_actions
            smoothed["actions"] = np.concatenate(
                [arm_actions, hand_actions],
                axis=-1,
            ).astype(np.float32)
        smoothed["policy_sequence_length"] = int(policy_count)
        smoothed["smooth_hz"] = float(1.0 / smooth_dt)
        smoothed["smooth_ratio"] = int(ratio)
        return smoothed, smooth_dt

    def _smooth_positions(self, linear_positions, anchor_positions, ratio):
        n_steps = linear_positions.shape[0]
        if n_steps <= 2:
            return linear_positions.copy()

        cfg = self.config
        hessian = max(0.0, float(cfg.w_track)) * np.eye(n_steps, dtype=np.float64)
        rhs = max(0.0, float(cfg.w_track)) * linear_positions

        anchor_weight = max(0.0, float(cfg.w_anchor))
        if anchor_weight > 0.0:
            for anchor_index, anchor in enumerate(anchor_positions):
                smooth_index = min(anchor_index * ratio, n_steps - 1)
                hessian[smooth_index, smooth_index] += anchor_weight
                rhs[smooth_index] += anchor_weight * anchor

        vel_weight = max(0.0, float(cfg.w_vel))
        if vel_weight > 0.0:
            diff = np.eye(n_steps, dtype=np.float64)
            diff = diff[1:] - diff[:-1]
            hessian += vel_weight * diff.T @ diff

        acc_weight = max(0.0, float(cfg.w_acc))
        if acc_weight > 0.0 and n_steps > 2:
            diff2 = np.eye(n_steps, dtype=np.float64)
            diff2 = diff2[2:] - 2.0 * diff2[1:-1] + diff2[:-2]
            hessian += acc_weight * diff2.T @ diff2

        hessian += 1e-9 * np.eye(n_steps, dtype=np.float64)
        try:
            return np.linalg.solve(hessian, rhs)
        except np.linalg.LinAlgError:
            return linear_positions.copy()

    def _limit_positions(self, positions, current_position, start_delay_steps):
        out = np.asarray(positions, dtype=np.float64).copy()
        out = self._clip_workspace(out)
        if out.shape[0] == 0:
            return out

        vmax = max(0.0, float(self.config.max_velocity))
        amax = max(0.0, float(self.config.max_acceleration))
        if vmax <= 0.0 and amax <= 0.0:
            return out

        dt = 1.0 / float(self.config.smooth_hz)
        if current_position is None:
            start_index = 1
            prev = out[0].copy()
            prev2 = prev.copy()
        else:
            start_index = 0
            prev = np.asarray(current_position, dtype=np.float64).reshape(3)
            prev2 = prev.copy()

        for index in range(start_index, out.shape[0]):
            command = out[index]
            step_dt = dt
            if index == 0 and start_delay_steps > 0:
                step_dt = max(dt, float(start_delay_steps) * self.policy_dt)
            if vmax > 0.0:
                command = self._limit_norm(command, prev, vmax * step_dt)
            if amax > 0.0 and index > 0:
                command = self._limit_norm(command, 2.0 * prev - prev2, amax * dt * dt)
            command = self._clip_workspace(command)
            out[index] = command
            prev2 = prev
            prev = command
        return out

    def _limit_rotations(self, rotations, current_rotation, start_delay_steps, smooth_dt):
        max_speed = math.radians(max(0.0, float(self.config.max_rot_speed_deg)))
        if max_speed <= 0.0 or rotations.shape[0] == 0:
            return rotations

        out = np.asarray(rotations, dtype=np.float64).copy()
        if current_rotation is None:
            start_index = 1
            prev = self._project_rotation(out[0])
        else:
            start_index = 0
            prev = self._project_rotation(current_rotation)

        for index in range(start_index, out.shape[0]):
            step_dt = smooth_dt
            if index == 0 and start_delay_steps > 0:
                step_dt = max(smooth_dt, float(start_delay_steps) * self.policy_dt)
            command = self._step_rotation(prev, out[index], max_speed * step_dt)
            out[index] = command
            prev = command
        return out

    def _resample_rotations(self, rotations, source_times, target_times):
        out = []
        for target_time in target_times:
            if target_time <= source_times[0]:
                out.append(self._project_rotation(rotations[0]))
                continue
            if target_time >= source_times[-1]:
                out.append(self._project_rotation(rotations[-1]))
                continue
            upper = int(np.searchsorted(source_times, target_time, side="right"))
            lower = upper - 1
            dt = source_times[upper] - source_times[lower]
            alpha = 0.0 if dt <= 0.0 else (target_time - source_times[lower]) / dt
            out.append(
                self._interpolate_rotation(
                    rotations[lower],
                    rotations[upper],
                    min(max(alpha, 0.0), 1.0),
                )
            )
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _resample_array(values, source_times, target_times):
        values = np.asarray(values)
        if values.ndim == 1:
            return np.interp(target_times, source_times, values)

        flat = values.reshape(values.shape[0], -1)
        out = np.empty((target_times.shape[0], flat.shape[1]), dtype=np.float64)
        for dim in range(flat.shape[1]):
            out[:, dim] = np.interp(target_times, source_times, flat[:, dim])
        return out.reshape((target_times.shape[0],) + values.shape[1:])

    def _clip_workspace(self, values):
        out = np.asarray(values, dtype=np.float64).copy()
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            limit = self.workspace_limits.get(axis)
            if limit is not None and len(limit) == 2:
                out[..., index] = np.clip(out[..., index], float(limit[0]), float(limit[1]))
        return out

    @staticmethod
    def _limit_norm(value, center, radius):
        radius = max(0.0, float(radius))
        value = np.asarray(value, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        delta = value - center
        norm = float(np.linalg.norm(delta))
        if norm <= radius or norm <= 1e-12:
            return value.copy()
        return center + delta * (radius / norm)

    @classmethod
    def _step_rotation(cls, from_rot, to_rot, max_angle):
        from_rot = cls._project_rotation(from_rot)
        to_rot = cls._project_rotation(to_rot)
        delta = cls._project_rotation(np.dot(from_rot.T, to_rot))
        delta_rotvec = cls._rotmat_to_rotvec(delta)
        angle = np.linalg.norm(delta_rotvec)
        if angle <= max_angle or angle <= 1e-9:
            return to_rot
        return cls._project_rotation(
            np.dot(
                from_rot,
                cls._rotvec_to_rotmat(delta_rotvec * (max_angle / angle)),
            )
        )

    @classmethod
    def _interpolate_rotation(cls, rotation0, rotation1, alpha):
        rotation0 = cls._project_rotation(rotation0)
        rotation1 = cls._project_rotation(rotation1)
        delta_rotation = cls._project_rotation(np.dot(rotation0.T, rotation1))
        delta_rotvec = cls._rotmat_to_rotvec(delta_rotation)
        return cls._project_rotation(
            np.dot(rotation0, cls._rotvec_to_rotmat(alpha * delta_rotvec))
        )

    @staticmethod
    def _project_rotation(rotation):
        u, _, vh = np.linalg.svd(rotation)
        projected = np.dot(u, vh)
        if np.linalg.det(projected) < 0:
            u[:, -1] *= -1
            projected = np.dot(u, vh)
        return projected

    @staticmethod
    def _rotmat_to_rotvec(rotation, eps=1e-9):
        cos_theta = (np.trace(rotation) - 1.0) / 2.0
        cos_theta = min(max(cos_theta, -1.0), 1.0)
        theta = math.acos(cos_theta)
        if theta < eps:
            return np.zeros(3)

        if math.pi - theta < 1e-6:
            diag = np.diag(rotation)
            axis_index = int(np.argmax(diag))
            axis = np.asarray(rotation[:, axis_index], dtype=np.float64).copy()
            axis[axis_index] += 1.0
            axis_norm = np.linalg.norm(axis)
            if axis_norm < eps:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / axis_norm
            return axis * theta

        factor = theta / (2.0 * math.sin(theta))
        return factor * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )

    @staticmethod
    def _rotvec_to_rotmat(rotvec, eps=1e-9):
        theta = np.linalg.norm(rotvec)
        if theta < eps:
            return np.eye(3)

        axis = rotvec / theta
        axis_cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return (
            np.eye(3)
            + math.sin(theta) * axis_cross
            + (1.0 - math.cos(theta)) * np.dot(axis_cross, axis_cross)
        )
