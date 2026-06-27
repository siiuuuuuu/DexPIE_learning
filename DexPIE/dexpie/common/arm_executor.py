"""High-rate UR arm executor for policy and human intervention commands."""

import math
import threading
import time
from collections import deque

import numpy as np

from dexpie.common.intervention_pose_processor import InterventionPoseProcessor


ARM_MODE_IDLE = 0
ARM_MODE_POLICY = 1
ARM_MODE_INTERVENTION = 2


class HighRateArmExecutor:
    """Owns robot.servo calls and runs them at a fixed high frequency."""

    def __init__(
        self,
        robot,
        tool,
        tracker_reader=None,
        servo_frequency=120.0,
        tracker_frequency=60.0,
        policy_frequency=25.0,
        tracker_timeout=0.25,
        policy_timeout=0.5,
        tracker_interpolation_delay=None,
        policy_interpolation_delay=None,
        action_history_size=128,
    ):
        self.robot = robot
        self.tool = tool
        self.tracker_reader = tracker_reader
        self.servo_frequency = float(servo_frequency)
        self.tracker_frequency = float(tracker_frequency)
        self.policy_frequency = float(policy_frequency)
        self.tracker_timeout = float(tracker_timeout)
        self.policy_timeout = float(policy_timeout)
        self.action_history_size = int(action_history_size)

        if self.servo_frequency <= 0:
            raise ValueError("servo_frequency must be positive")
        if self.tracker_frequency <= 0:
            raise ValueError("tracker_frequency must be positive")
        if self.policy_frequency <= 0:
            raise ValueError("policy_frequency must be positive")

        self.tracker_interpolation_delay = (
            1.0 / self.tracker_frequency
            if tracker_interpolation_delay is None
            else float(tracker_interpolation_delay)
        )
        self.policy_interpolation_delay = (
            1.0 / self.policy_frequency
            if policy_interpolation_delay is None
            else float(policy_interpolation_delay)
        )

        self._mode = ARM_MODE_IDLE
        self._mode_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._robot_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._target_ready = threading.Event()
        self._stop_event = threading.Event()

        self._policy_samples = deque(maxlen=max(8, self.action_history_size))
        self._tracker_samples = deque(maxlen=8)
        self._action_history = deque(maxlen=self.action_history_size)
        self._latest_motion = None
        self._latest_tracker_time = None
        self._latest_tracker_time_ns = None
        self._intervention_reference_arm_mat = None
        self._pose_processor = InterventionPoseProcessor(tool)
        self._error = None
        self._threads = []

    def start(self):
        if self._threads:
            return
        self._stop_event.clear()
        self._target_ready.clear()
        self._error = None
        self._threads = [
            threading.Thread(
                target=self._run_guarded,
                args=("tracker loop", self._tracker_loop),
                name="dexpie-tracker-loop",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_guarded,
                args=("servo loop", self._servo_loop),
                name="dexpie-arm-servo-loop",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self):
        self._stop_event.set()
        self._target_ready.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        self.set_idle(stop_servo=True)

    def reset_episode(self):
        self.set_idle(stop_servo=False)
        with self._motion_lock:
            self._policy_samples.clear()
            self._tracker_samples.clear()
            self._action_history.clear()
            self._latest_motion = None
            self._latest_tracker_time = None
            self._latest_tracker_time_ns = None
        self._pose_processor.clear_reference()
        self._intervention_reference_arm_mat = None

    def start_policy(self):
        with self._mode_lock:
            self._mode = ARM_MODE_POLICY
        with self._motion_lock:
            self._policy_samples.clear()
        self._target_ready.clear()

    def start_intervention(self, reference_arm_mat):
        with self._mode_lock:
            self._mode = ARM_MODE_INTERVENTION
        with self._motion_lock:
            self._tracker_samples.clear()
            self._latest_tracker_time = None
            self._latest_tracker_time_ns = None
        self._pose_processor.clear_reference()
        self._intervention_reference_arm_mat = np.asarray(reference_arm_mat).copy()
        self._target_ready.clear()

    def set_idle(self, stop_servo=False):
        with self._mode_lock:
            self._mode = ARM_MODE_IDLE
        self._target_ready.clear()
        if stop_servo:
            with self._robot_lock:
                self.robot.stop_servo()

    def submit_policy_command(self, command):
        self.raise_if_failed()
        if command.target_arm_mat is None:
            raise ValueError("policy command must include target_arm_mat")

        sequence = {
            "arm_mats": np.asarray([command.target_arm_mat], dtype=np.float64),
            "target_poses": np.asarray([command.target_pose], dtype=np.float64),
            "arm_actions": np.asarray([command.action[:9]], dtype=np.float32),
        }
        return self.submit_policy_sequence(sequence)

    def submit_policy_sequence(self, sequence, start_time_ns=None, step_period_s=None):
        self.raise_if_failed()
        if sequence is None:
            return None

        arm_mats = np.asarray(sequence.get("arm_mats"), dtype=np.float64)
        if arm_mats.ndim != 3 or arm_mats.shape[0] == 0:
            return None

        if start_time_ns is None:
            start_time_ns = time.monotonic_ns()
        start_time_ns = int(start_time_ns)
        if step_period_s is None:
            step_period_s = 1.0 / self.policy_frequency
        step_period_ns = int(round(float(step_period_s) * 1e9))

        target_poses = sequence.get("target_poses")
        if target_poses is not None:
            target_poses = np.asarray(target_poses, dtype=np.float64)
        arm_actions = sequence.get("arm_actions")
        if arm_actions is not None:
            arm_actions = np.asarray(arm_actions, dtype=np.float32)

        samples = []
        for index, arm_mat in enumerate(arm_mats):
            target_ns = start_time_ns + index * step_period_ns
            target_pose = (
                target_poses[index]
                if target_poses is not None
                else np.asarray(self.tool.mat2xyz_rotvec(arm_mat), dtype=np.float64)
            )
            arm_action = (
                arm_actions[index]
                if arm_actions is not None
                else np.asarray(self.tool.mat2xyz_6drot(arm_mat), dtype=np.float32)
            )
            motion = {
                "target_arm_mat": arm_mat,
                "result_matrix": arm_mat,
                "target_pose": np.asarray(target_pose, dtype=np.float64),
                "arm_action": np.asarray(arm_action, dtype=np.float32),
                "t_policy_command_host_ns": np.asarray(target_ns, dtype=np.int64),
            }
            samples.append((target_ns / 1e9, self._copy_motion(motion)))

        with self._mode_lock:
            self._mode = ARM_MODE_POLICY
        with self._motion_lock:
            start_time_s = start_time_ns / 1e9
            preserved_samples = [
                sample
                for sample in self._policy_samples
                if sample[0] < start_time_s
            ][-2:]
            self._policy_samples.clear()
            self._policy_samples.extend(preserved_samples)
            self._policy_samples.extend(samples)
        self._target_ready.set()
        return start_time_ns

    def latest_motion(self):
        self.raise_if_failed()
        with self._motion_lock:
            if self._latest_motion is None:
                return None
            return self._copy_motion(self._latest_motion)

    def latest_action_time_ns(self):
        self.raise_if_failed()
        with self._motion_lock:
            if self._latest_motion is None:
                return None
            return int(
                np.asarray(
                    self._latest_motion["t_arm_action_host_ns"],
                    dtype=np.int64,
                ).item()
            )

    def motion_at_time_ns(self, target_time_ns):
        self.raise_if_failed()
        with self._motion_lock:
            if not self._action_history:
                return None
            motion = self._nearest_by_time(
                self._action_history,
                int(target_time_ns),
                "t_arm_action_host_ns",
            )
        return self._copy_motion(motion)

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _run_guarded(self, loop_name, loop_fn):
        try:
            loop_fn()
        except Exception as exc:
            self._record_error(loop_name, exc)
            self._stop_event.set()
            self._target_ready.set()

    def _record_error(self, operation, exc):
        with self._error_lock:
            if self._error is None:
                self._error = RuntimeError(f"{operation} failed: {exc}")

    def _get_mode(self):
        with self._mode_lock:
            return self._mode

    def _tracker_loop(self):
        period = 1.0 / self.tracker_frequency
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            if self._get_mode() == ARM_MODE_INTERVENTION and self.tracker_reader is not None:
                tracker_mat = self.tracker_reader()
                if tracker_mat is not None:
                    self._handle_tracker_sample(tracker_mat)
            next_tick = self._wait_until_next_tick(next_tick, period)

    def _handle_tracker_sample(self, tracker_mat):
        reference_arm_mat = self._intervention_reference_arm_mat
        if reference_arm_mat is None:
            return

        if not self._pose_processor.has_reference():
            self._pose_processor.reset_reference(tracker_mat, reference_arm_mat)

        pose_command = self._pose_processor.compute(tracker_mat)
        now = time.monotonic()
        now_ns = time.monotonic_ns()
        target_arm_mat = np.asarray(pose_command["target_arm_mat"], dtype=np.float64)
        motion = {
            "target_arm_mat": target_arm_mat,
            "result_matrix": target_arm_mat,
            "target_pose": np.asarray(pose_command["target_pose"], dtype=np.float64),
            "arm_action": np.asarray(pose_command["arm_action"], dtype=np.float32),
            "increment_matrix": np.asarray(pose_command["increment_matrix"], dtype=np.float64),
            "t_tracker_host_ns": np.asarray(now_ns, dtype=np.int64),
        }
        with self._motion_lock:
            self._tracker_samples.append((now, self._copy_motion(motion)))
            self._latest_tracker_time = now
            self._latest_tracker_time_ns = now_ns
        self._target_ready.set()

    def _servo_loop(self):
        period = 1.0 / self.servo_frequency
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            mode = self._get_mode()
            if mode == ARM_MODE_IDLE:
                next_tick = self._wait_until_next_tick(next_tick, period)
                continue

            if not self._target_ready.wait(timeout=0.001):
                next_tick = self._wait_until_next_tick(next_tick, period)
                continue

            now = time.monotonic()
            now_ns = time.monotonic_ns()
            if mode == ARM_MODE_POLICY:
                motion = self._policy_motion(now)
            elif mode == ARM_MODE_INTERVENTION:
                motion = self._intervention_motion(now)
            else:
                motion = None

            if motion is not None:
                motion["t_arm_action_host_ns"] = np.asarray(now_ns, dtype=np.int64)
                motion["t_arm_servo_host_ns"] = np.asarray(now_ns, dtype=np.int64)
                motion["arm_executor_mode"] = np.asarray(mode, dtype=np.int64)
                if mode == ARM_MODE_INTERVENTION:
                    with self._motion_lock:
                        latest_tracker_time_ns = self._latest_tracker_time_ns
                    motion["t_tracker_latest_host_ns"] = np.asarray(
                        -1 if latest_tracker_time_ns is None else latest_tracker_time_ns,
                        dtype=np.int64,
                    )

                with self._robot_lock:
                    self.robot.servo(np.asarray(motion["target_pose"]).tolist())
                with self._motion_lock:
                    copied_motion = self._copy_motion(motion)
                    self._latest_motion = copied_motion
                    self._action_history.append(copied_motion)

            next_tick = self._wait_until_next_tick(next_tick, period)

    def _policy_motion(self, now):
        with self._motion_lock:
            samples = list(self._policy_samples)
        if not samples:
            return None

        latest_sample_time = samples[-1][0]
        if now - latest_sample_time > self.policy_timeout:
            raise RuntimeError(
                f"policy arm command is stale ({now - latest_sample_time:.3f}s > "
                f"{self.policy_timeout:.3f}s)"
            )

        target_time = now - self.policy_interpolation_delay
        return self._motion_at(samples, target_time, "t_policy_command_host_ns")

    def _intervention_motion(self, now):
        with self._motion_lock:
            samples = list(self._tracker_samples)
            latest_tracker_time = self._latest_tracker_time
        if not samples or latest_tracker_time is None:
            return None

        target_age = now - latest_tracker_time
        if target_age > self.tracker_timeout:
            raise RuntimeError(
                f"tracker target is stale ({target_age:.3f}s > "
                f"{self.tracker_timeout:.3f}s)"
            )

        target_time = now - self.tracker_interpolation_delay
        return self._motion_at(samples, target_time, "t_tracker_host_ns")

    def _wait_until_next_tick(self, previous_tick, period):
        next_tick = previous_tick + period
        delay = next_tick - time.monotonic()
        if delay > 0:
            self._stop_event.wait(delay)
            return next_tick

        missed_periods = int(-delay // period) + 1
        return next_tick + missed_periods * period

    def _motion_at(self, samples, target_time, source_time_key):
        if len(samples) == 1 or target_time <= samples[0][0]:
            return self._with_hold_metadata(samples[0][1], source_time_key)
        if target_time >= samples[-1][0]:
            return self._with_hold_metadata(samples[-1][1], source_time_key)

        for sample_index in range(1, len(samples)):
            t1, motion1 = samples[sample_index]
            if target_time <= t1:
                t0, motion0 = samples[sample_index - 1]
                dt = t1 - t0
                if dt <= 0:
                    return self._with_hold_metadata(motion1, source_time_key)
                alpha = (target_time - t0) / dt
                alpha = min(max(alpha, 0.0), 1.0)
                return self._interpolate_motion(motion0, motion1, alpha, source_time_key)

        return self._with_hold_metadata(samples[-1][1], source_time_key)

    def _with_hold_metadata(self, motion, source_time_key):
        motion = self._copy_motion(motion)
        source_ns = np.asarray(motion[source_time_key], dtype=np.int64)
        motion["t_arm_target_host_ns"] = source_ns.copy()
        motion["interpolation_alpha"] = np.asarray(0.0, dtype=np.float64)
        self._set_source_pair_metadata(motion, source_time_key, source_ns, source_ns)
        return motion

    def _interpolate_motion(self, motion0, motion1, alpha, source_time_key):
        result_matrix = self._interpolate_matrix(
            motion0["result_matrix"],
            motion1["result_matrix"],
            alpha,
        )
        motion = {
            "target_arm_mat": result_matrix,
            "result_matrix": result_matrix,
            "target_pose": np.asarray(self.tool.mat2xyz_rotvec(result_matrix)),
            "arm_action": np.asarray(self.tool.mat2xyz_6drot(result_matrix)),
            "interpolation_alpha": np.asarray(alpha, dtype=np.float64),
        }
        t0_ns = int(np.asarray(motion0[source_time_key]).item())
        t1_ns = int(np.asarray(motion1[source_time_key]).item())
        target_ns = round((1.0 - alpha) * t0_ns + alpha * t1_ns)
        motion["t_arm_target_host_ns"] = np.asarray(target_ns, dtype=np.int64)
        self._set_source_pair_metadata(
            motion,
            source_time_key,
            np.asarray(t0_ns, dtype=np.int64),
            np.asarray(t1_ns, dtype=np.int64),
        )
        return motion

    @staticmethod
    def _set_source_pair_metadata(motion, source_time_key, t0_ns, t1_ns):
        if source_time_key == "t_tracker_host_ns":
            motion["t_tracker0_host_ns"] = np.asarray(t0_ns, dtype=np.int64)
            motion["t_tracker1_host_ns"] = np.asarray(t1_ns, dtype=np.int64)
        elif source_time_key == "t_policy_command_host_ns":
            motion["t_policy0_host_ns"] = np.asarray(t0_ns, dtype=np.int64)
            motion["t_policy1_host_ns"] = np.asarray(t1_ns, dtype=np.int64)

    def _interpolate_matrix(self, matrix0, matrix1, alpha):
        matrix0 = np.asarray(matrix0)
        matrix1 = np.asarray(matrix1)
        result_matrix = np.eye(4)
        result_matrix[:3, 3] = (
            (1.0 - alpha) * matrix0[:3, 3] + alpha * matrix1[:3, 3]
        )
        result_matrix[:3, :3] = self._interpolate_rotation(
            matrix0[:3, :3],
            matrix1[:3, :3],
            alpha,
        )
        return result_matrix

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

    @staticmethod
    def _copy_motion(motion):
        return {key: np.asarray(value).copy() for key, value in motion.items()}

    @staticmethod
    def _nearest_by_time(samples, target_time_ns, time_key):
        samples = list(samples)
        best_sample = samples[-1]
        best_delta = abs(int(np.asarray(best_sample[time_key]).item()) - target_time_ns)
        for sample in reversed(samples[:-1]):
            delta = abs(int(np.asarray(sample[time_key]).item()) - target_time_ns)
            if delta < best_delta:
                best_sample = sample
                best_delta = delta
            else:
                break
        return best_sample
