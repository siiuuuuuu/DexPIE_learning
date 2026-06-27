"""High-rate UR robot state reader with timestamped history."""

import threading
import time
from collections import deque


class HighRateRobotStateReader:
    """Polls UR state independently so collection can align by timestamp."""

    def __init__(self, robot, read_frequency=125.0, history_size=128):
        self.robot = robot
        self.read_frequency = float(read_frequency)
        self.history_size = int(history_size)

        self._samples = deque(maxlen=self.history_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest_obs(self):
        with self._lock:
            if not self._samples:
                return None
            return dict(self._samples[-1])

    def latest_obs_time_ns(self):
        with self._lock:
            if not self._samples:
                return None
            return int(self._samples[-1].get("t_robot_obs_host_ns"))

    def obs_at_time_ns(self, target_time_ns):
        with self._lock:
            samples = list(self._samples)
        return self._nearest_by_time(samples, target_time_ns)

    def raise_if_failed(self):
        if self._last_error is not None:
            raise RuntimeError("HighRateRobotStateReader failed") from self._last_error

    def _loop(self):
        period = 1.0 / self.read_frequency
        next_time = time.monotonic()

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                obs = self.robot.get_obs()
                if obs is not None:
                    sample = dict(obs)
                    sample["t_robot_obs_host_ns"] = time.monotonic_ns()
                    with self._lock:
                        self._samples.append(sample)
            except Exception as exc:
                self._last_error = exc
                break

            next_time += period
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = loop_start

    @staticmethod
    def _nearest_by_time(samples, target_time_ns):
        if not samples or target_time_ns is None:
            return None

        target_time_ns = int(target_time_ns)
        best_sample = samples[-1]
        best_abs_delta = abs(
            int(best_sample.get("t_robot_obs_host_ns", 0)) - target_time_ns
        )

        for sample in reversed(samples[:-1]):
            sample_time = int(sample.get("t_robot_obs_host_ns", 0))
            abs_delta = abs(sample_time - target_time_ns)
            if abs_delta > best_abs_delta:
                break
            best_sample = sample
            best_abs_delta = abs_delta

        return dict(best_sample)
