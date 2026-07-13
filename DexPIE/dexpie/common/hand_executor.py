import logging
import threading
import time
from collections import deque

import numpy as np


logger = logging.getLogger(__name__)

HAND_MODE_IDLE = 0
HAND_MODE_POLICY = 1
HAND_MODE_INTERVENTION = 2


class HighRateHandExecutor:
    """Hand executor with policy-time interpolation and intervention smoothing."""

    def __init__(
        self,
        send_callback,
        hz=120.0,
        w=30.0,
        z=0.85,
        dim=6,
        data_timeout=0.5,
        command_history_size=128,
        policy_schedule_size=64,
        input_alpha=0.84,
    ):
        self.send_callback = send_callback
        self.hz = float(hz)
        self.dim = int(dim)
        self.command_history_size = int(command_history_size)
        self.policy_schedule_size = int(policy_schedule_size)
        self.input_alpha = float(input_alpha)
        natural_frequency = float(w)
        damping_ratio = float(z)

        if self.hz <= 0:
            raise ValueError("hz must be positive")
        if natural_frequency <= 0:
            raise ValueError("w must be positive")
        if damping_ratio <= 0:
            raise ValueError("z must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.command_history_size <= 0:
            raise ValueError("command_history_size must be positive")
        if self.policy_schedule_size <= 0:
            raise ValueError("policy_schedule_size must be positive")
        if not 0 < self.input_alpha <= 1:
            raise ValueError("input_alpha must be in the range (0, 1]")

        normalized_frequency = natural_frequency / self.hz
        stability_margin = (
            normalized_frequency * normalized_frequency
            + 4.0 * damping_ratio * normalized_frequency
        )
        if stability_margin >= 4.0:
            raise ValueError(
                "unstable smoother configuration; increase hz or reduce w"
            )

        self.k_p = natural_frequency * natural_frequency
        self.k_d = 2.0 * damping_ratio * natural_frequency

        self.current_angles = np.zeros(self.dim, dtype=np.float32)
        self.current_velocity = np.zeros(self.dim, dtype=np.float32)
        self.target_angles = None
        self.smooth_target = None
        self.initialized = False

        self.lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.data_event = threading.Event()
        self.is_running = False
        self.worker_thread = None

        self.last_update_time = 0.0
        self.last_update_time_ns = None
        self.last_manus_sample_time_ns = None
        self.last_manus_seq = None
        self.data_timeout = float(data_timeout)
        self.target_mode = HAND_MODE_IDLE
        self.command_history = deque(maxlen=self.command_history_size)
        self.policy_schedule = deque(maxlen=self.policy_schedule_size)
        self.latest_command_sample_value = None
        self.error = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        with self.error_lock:
            self.error = None
        with self.lock:
            self.command_history.clear()
            self.policy_schedule.clear()
            self.latest_command_sample_value = None
        self.worker_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.worker_thread.start()
        logger.info("HighRateHandExecutor started (freq=%sHz)", self.hz)

    def stop(self):
        self.is_running = False
        self.data_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None
        self.data_event.clear()

    def reset_state(self, initial_angles=None):
        with self.lock:
            if initial_angles is None:
                self.current_angles[:] = 0
                self.initialized = False
            else:
                initial = self._as_target_array(initial_angles)
                self.current_angles = initial.copy()
                self.smooth_target = initial.copy()
                self.initialized = True
            self.current_velocity[:] = 0
            self.target_angles = None
            self.smooth_target = None if initial_angles is None else self.current_angles.copy()
            self.last_update_time = 0.0
            self.last_update_time_ns = None
            self.last_manus_sample_time_ns = None
            self.last_manus_seq = None
            self.target_mode = HAND_MODE_IDLE
            self.command_history.clear()
            self.policy_schedule.clear()
            self.latest_command_sample_value = None
        with self.error_lock:
            self.error = None
        self.data_event.clear()

    def clear_history(self):
        with self.lock:
            self.command_history.clear()
            self.policy_schedule.clear()
            self.latest_command_sample_value = None

    def clear_policy_sequence(self):
        with self.lock:
            self.policy_schedule.clear()

    def submit_policy_sequence(self, targets, start_time_ns=None, step_period_s=0.04):
        """Submit policy hand waypoints as actual-time targets.

        Policy commands are executed by timestamped linear interpolation, not by
        the intervention low-pass smoother. This keeps the policy action meaning
        aligned with training data: action is the command that should be sent at
        that timestamp.
        """
        if not self.is_running:
            return None

        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != self.dim:
            raise ValueError(
                f"expected policy hand sequence shape [T,{self.dim}], got {targets.shape}"
            )
        if targets.shape[0] == 0:
            return None
        if targets.shape[0] > self.policy_schedule_size:
            raise ValueError(
                "policy hand sequence length "
                f"{targets.shape[0]} exceeds policy_schedule_size "
                f"{self.policy_schedule_size}"
            )
        if start_time_ns is None:
            start_time_ns = time.monotonic_ns()
        start_time_ns = int(start_time_ns)
        step_period_ns = int(round(float(step_period_s) * 1e9))

        samples = []
        for index, target in enumerate(targets):
            target_ns = start_time_ns + index * step_period_ns
            samples.append((target_ns, self._as_target_array(target)))

        with self.lock:
            preserved_count = max(0, self.policy_schedule_size - len(samples))
            preserved_samples = [
                sample
                for sample in self.policy_schedule
                if sample[0] < start_time_ns
            ][-min(2, preserved_count):]
            self.policy_schedule.clear()
            self.policy_schedule.extend(preserved_samples)
            self.policy_schedule.extend(samples)
            self.target_mode = HAND_MODE_POLICY
        self.data_event.set()
        return start_time_ns

    def update(
        self,
        new_target,
        source_mode=HAND_MODE_POLICY,
        target_time_ns=None,
        manus_seq=None,
        manus_sample_time_ns=None,
    ):
        if not self.is_running:
            return None

        if target_time_ns is None:
            target_time_ns = time.monotonic_ns()
        target = self._as_target_array(new_target)
        now = time.time()
        with self.lock:
            self.target_angles = target
            self.last_update_time = now
            self.last_update_time_ns = int(target_time_ns)
            self.target_mode = int(source_mode)
            if self.target_mode == HAND_MODE_INTERVENTION:
                self.policy_schedule.clear()
                self.last_manus_sample_time_ns = (
                    None
                    if manus_sample_time_ns is None
                    else int(manus_sample_time_ns)
                )
                self.last_manus_seq = None if manus_seq is None else int(manus_seq)
            else:
                self.last_manus_sample_time_ns = None
                self.last_manus_seq = None

            if not self.initialized:
                self.current_angles = target.copy()
                self.current_velocity[:] = 0
                self.smooth_target = target.copy()
                self.initialized = True
            elif self.smooth_target is None:
                self.smooth_target = target.copy()

        self.data_event.set()
        return int(target_time_ns)

    def latest_command_sample(self):
        self.raise_if_failed()
        with self.lock:
            if self.latest_command_sample_value is None:
                return None
            return self._copy_command_sample(self.latest_command_sample_value)

    def latest_command_time_ns(self):
        self.raise_if_failed()
        with self.lock:
            if self.latest_command_sample_value is None:
                return None
            return int(self.latest_command_sample_value["t_hand_action_host_ns"])

    def command_at_time_ns(self, target_time_ns):
        self.raise_if_failed()
        with self.lock:
            if not self.command_history or target_time_ns is None:
                return None
            sample = self._nearest_by_time(
                self.command_history,
                int(target_time_ns),
                "t_hand_action_host_ns",
            )
            return self._copy_command_sample(sample)

    def raise_if_failed(self):
        with self.error_lock:
            error = self.error
        if error is not None:
            raise error

    def _as_target_array(self, target):
        array = np.asarray(target, dtype=np.float32).reshape(-1)
        if array.size != self.dim:
            raise ValueError(
                f"joint target dim mismatch: expected {self.dim}, got {array.size}"
            )
        return np.clip(array, 0, 1000)

    def _control_loop(self):
        dt = 1.0 / self.hz

        while self.is_running:
            if not self.data_event.wait(timeout=0.1):
                continue
            if not self.is_running:
                break

            loop_start = time.time()
            now_ns = time.monotonic_ns()
            angles_to_send = None
            target_time_ns = None
            target_mode = HAND_MODE_IDLE
            manus_sample_time_ns = None
            manus_seq = None
            pending_future_policy = False

            with self.lock:
                policy_target = self._policy_target_at_time_locked(now_ns)
                if policy_target is not None:
                    angles_to_send, target_time_ns = policy_target
                    target_mode = HAND_MODE_POLICY
                    self.target_mode = HAND_MODE_POLICY
                    self.target_angles = angles_to_send.copy()
                    self.smooth_target = angles_to_send.copy()
                    self.current_angles = angles_to_send.copy()
                    self.current_velocity[:] = 0
                    self.last_update_time = time.time()
                    self.last_update_time_ns = int(target_time_ns)
                    self.last_manus_sample_time_ns = None
                    self.last_manus_seq = None
                    self.initialized = True
                elif self.policy_schedule:
                    pending_future_policy = True

                if angles_to_send is None and not pending_future_policy:
                    smooth_result = self._smooth_intervention_locked(dt)
                    if smooth_result is not None:
                        (
                            angles_to_send,
                            target_time_ns,
                            target_mode,
                            manus_sample_time_ns,
                            manus_seq,
                        ) = smooth_result

            if angles_to_send is None:
                if pending_future_policy:
                    time.sleep(dt)
                    continue
                self.data_event.clear()
                continue

            try:
                command = np.asarray(
                    self.send_callback(np.asarray(angles_to_send).tolist()),
                    dtype=np.int32,
                ).reshape(-1)
                if command.size != self.dim:
                    raise RuntimeError(
                        f"expected {self.dim}D hand command, got {command.size} values"
                    )
                action_time_ns = time.monotonic_ns()
                sample = {
                    "command": command.copy(),
                    "t_hand_action_host_ns": action_time_ns,
                    "t_hand_command_host_ns": action_time_ns,
                    "t_hand_target_host_ns": target_time_ns,
                    "t_manus_sample_host_ns": manus_sample_time_ns,
                    "manus_seq": manus_seq,
                    "hand_executor_mode": int(target_mode),
                }
                with self.lock:
                    self.command_history.append(sample)
                    self.latest_command_sample_value = self._copy_command_sample(sample)
                    if target_mode == HAND_MODE_POLICY:
                        self.current_angles = command.astype(np.float32)
                        self.smooth_target = self.current_angles.copy()
                        self.current_velocity[:] = 0
            except Exception as exc:
                logger.error("HighRateHandExecutor callback error: %s", exc)
                with self.error_lock:
                    if self.error is None:
                        self.error = RuntimeError(
                            f"hand control loop failed: {exc}"
                        )
                self.is_running = False
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    def _policy_target_at_time_locked(self, now_ns):
        if not self.policy_schedule:
            return None

        timeout_ns = int(round(self.data_timeout * 1e9))
        latest_target_ns = int(self.policy_schedule[-1][0])
        if now_ns - latest_target_ns > timeout_ns:
            self.policy_schedule.clear()
            if self.target_mode == HAND_MODE_POLICY:
                self.target_mode = HAND_MODE_IDLE
                self.target_angles = None
                self.smooth_target = None
            return None

        while len(self.policy_schedule) > 2 and self.policy_schedule[1][0] <= now_ns:
            self.policy_schedule.popleft()

        samples = list(self.policy_schedule)
        if now_ns < samples[0][0]:
            return None
        if len(samples) == 1 or now_ns >= samples[-1][0]:
            return samples[-1][1].copy(), int(samples[-1][0])

        for sample_index in range(1, len(samples)):
            t1, target1 = samples[sample_index]
            if now_ns <= t1:
                t0, target0 = samples[sample_index - 1]
                if t1 <= t0:
                    return target1.copy(), int(t1)
                alpha = (now_ns - t0) / (t1 - t0)
                alpha = min(max(alpha, 0.0), 1.0)
                target = (1.0 - alpha) * target0 + alpha * target1
                target_ns = round((1.0 - alpha) * t0 + alpha * t1)
                return np.clip(target, 0, 1000).astype(np.float32), int(target_ns)

        return samples[-1][1].copy(), int(samples[-1][0])

    def _smooth_intervention_locked(self, dt):
        if self.target_angles is None:
            return None
        if self.target_mode == HAND_MODE_POLICY:
            return None

        if time.time() - self.last_update_time > self.data_timeout:
            self.current_velocity[:] = 0
            self.smooth_target = None
            self.target_mode = HAND_MODE_IDLE
            return None

        if self.smooth_target is None:
            self.smooth_target = self.target_angles.copy()

        self.smooth_target = (
            (1.0 - self.input_alpha) * self.smooth_target
            + self.input_alpha * self.target_angles
        )

        error = self.smooth_target - self.current_angles
        accel = self.k_p * error - self.k_d * self.current_velocity
        self.current_velocity += accel * dt
        self.current_angles += self.current_velocity * dt
        clipped_angles = np.clip(self.current_angles, 0, 1000)
        outward_velocity = (
            ((clipped_angles <= 0) & (self.current_velocity < 0))
            | ((clipped_angles >= 1000) & (self.current_velocity > 0))
        )
        self.current_velocity[outward_velocity] = 0.0
        self.current_angles = clipped_angles
        return (
            self.current_angles.copy(),
            self.last_update_time_ns,
            int(self.target_mode),
            self.last_manus_sample_time_ns,
            self.last_manus_seq,
        )

    @staticmethod
    def _copy_command_sample(sample):
        return {
            "command": np.asarray(sample["command"]).copy(),
            "t_hand_action_host_ns": sample.get("t_hand_action_host_ns"),
            "t_hand_command_host_ns": sample.get("t_hand_command_host_ns"),
            "t_hand_target_host_ns": sample.get("t_hand_target_host_ns"),
            "t_manus_sample_host_ns": sample.get("t_manus_sample_host_ns"),
            "manus_seq": sample.get("manus_seq"),
            "hand_executor_mode": sample.get("hand_executor_mode", HAND_MODE_IDLE),
        }

    @staticmethod
    def _nearest_by_time(samples, target_time_ns, time_key):
        samples = list(samples)
        best_sample = samples[-1]
        best_delta = abs(int(best_sample[time_key]) - target_time_ns)
        for sample in reversed(samples[:-1]):
            delta = abs(int(sample[time_key]) - target_time_ns)
            if delta < best_delta:
                best_sample = sample
                best_delta = delta
            else:
                break
        return best_sample
