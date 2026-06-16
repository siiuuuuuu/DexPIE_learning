import logging
import threading
import time

import numpy as np


logger = logging.getLogger(__name__)


class JointSmoother:
    """2nd-order low-pass joint smoother with a high-frequency sender thread."""

    def __init__(
        self,
        send_callback,
        hz=120.0,
        w=25.0,
        z=0.8,
        dim=6,
        data_timeout=0.5,
    ):
        self.send_callback = send_callback
        self.hz = float(hz)
        self.dim = int(dim)

        self.k_p = float(w) * float(w)
        self.k_d = 2.0 * float(z) * float(w)

        self.current_angles = np.zeros(self.dim, dtype=np.float32)
        self.current_velocity = np.zeros(self.dim, dtype=np.float32)
        self.target_angles = None
        self.smooth_target = None
        self.initialized = False

        self.lock = threading.Lock()
        self.data_event = threading.Event()
        self.is_running = False
        self.worker_thread = None

        self.last_update_time = 0.0
        self.data_timeout = float(data_timeout)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.worker_thread.start()
        logger.info("JointSmoother started (freq=%sHz)", self.hz)

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
                self.initialized = True
            self.current_velocity[:] = 0
            self.target_angles = None
            self.smooth_target = None
            self.last_update_time = 0.0
        self.data_event.clear()

    def update(self, new_target):
        if not self.is_running:
            return False

        target = self._as_target_array(new_target)
        now = time.time()
        with self.lock:
            self.target_angles = target
            self.last_update_time = now

            if not self.initialized:
                self.current_angles = target.copy()
                self.current_velocity[:] = 0
                self.smooth_target = target.copy()
                self.initialized = True
            elif self.smooth_target is None:
                self.smooth_target = target.copy()

        self.data_event.set()
        return True

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

            if time.time() - self.last_update_time > self.data_timeout:
                self.data_event.clear()
                with self.lock:
                    self.current_velocity[:] = 0
                    self.smooth_target = None
                continue

            loop_start = time.time()
            angles_to_send = None

            with self.lock:
                if self.target_angles is not None:
                    if self.smooth_target is None:
                        self.smooth_target = self.target_angles.copy()

                    alpha = 0.6
                    self.smooth_target = (
                        (1.0 - alpha) * self.smooth_target
                        + alpha * self.target_angles
                    )

                    error = self.smooth_target - self.current_angles
                    accel = self.k_p * error - self.k_d * self.current_velocity
                    self.current_velocity += accel * dt
                    self.current_angles += self.current_velocity * dt
                    self.current_angles = np.clip(self.current_angles, 0, 1000)
                    angles_to_send = self.current_angles.tolist()

            if angles_to_send is not None:
                try:
                    self.send_callback(angles_to_send)
                except Exception as e:
                    logger.error("JointSmoother callback error: %s", e)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))
