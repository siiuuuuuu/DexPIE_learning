import ctypes
import numpy as np
from multiprocessing import Process, Event, Condition, Lock, Value, Array
import time
import socket


class SharedLatestActionBuffer:
    """Single-producer / single-consumer latest-action buffer.

    Stores only the latest raw 0-1000 hand command in shared memory and tracks
    sequence/time.
    Consumer can block until a newer seq appears.
    """

    def __init__(self, dim=6):
        self.dim = int(dim)
        self._data = Array(ctypes.c_float, self.dim, lock=False)

        self._lock = Lock()
        self._cond = Condition(self._lock)

        self._seq = Value(ctypes.c_ulonglong, 0)
        self._t_host_ns = Value(ctypes.c_ulonglong, 0)
        self._valid = Value(ctypes.c_byte, 0)
        self._closed = Value(ctypes.c_byte, 0)

    def _as_array(self):
        return np.ctypeslib.as_array(self._data)

    def _snapshot(self, copy_data=True):
        data = self._as_array()

        return {
            "seq": int(self._seq.value),
            "t_host_ns": int(self._t_host_ns.value),
            "action": data.copy() if copy_data else data,
        }

    def publish(self, action, t_host_ns=None):
        if action is None:
            return

        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size != self.dim:
            raise ValueError(f"action dim mismatch: expected {self.dim}, got {vec.size}")

        if t_host_ns is None:
            t_host_ns = time.monotonic_ns()

        with self._cond:
            if self._closed.value:
                return

            self._as_array()[:] = vec
            self._seq.value = int(self._seq.value) + 1
            self._t_host_ns.value = int(t_host_ns)
            self._valid.value = 1
            self._cond.notify_all()

    def latest(self, copy_data=True):
        with self._cond:
            if not self._valid.value:
                return None

            return self._snapshot(copy_data=copy_data)

    def wait_next(self, last_seq=0, timeout=None, copy_data=True):
        deadline = None if timeout is None else (time.monotonic() + float(timeout))

        with self._cond:
            while True:
                if self._closed.value:
                    return None

                if self._valid.value and int(self._seq.value) > int(last_seq):
                    return self._snapshot(copy_data=copy_data)

                if timeout is None:
                    self._cond.wait()
                else:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        return None
                    self._cond.wait(remain)

    def close(self):
        with self._cond:
            self._closed.value = 1
            self._cond.notify_all()


class ManusTeleoperationProcess(Process):
    def __init__(
        self,
        action_buffer=None,
        manus_port=8888,
        control_threshold=10,
        scale_factor=15,
    ):
        super().__init__()
        self.daemon = True
        self.stop_event = Event()

        self.action_buffer = action_buffer

        self.manus_port = manus_port
        self.control_threshold = control_threshold
        self.scale_factor = scale_factor

        self.left_hand_connect = False
        self.right_hand_connect = False

        self.right_hand_data0 = None
        self.set_data = None

        self.server = None
        self.connection = None
        self.address = None
        self._last_published_action = np.zeros(6, dtype=np.float32)

        print("MANUS action subprocess initialized")
        print(f"manus port: {manus_port}")

    def initialize_manus_connection(self):
        try:
            print("Building MANUS socket connection...")
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind(("localhost", self.manus_port))
            self.server.listen(0)
            print("Listening MANUS stream...")

            self.connection, self.address = self.server.accept()
            print(f"MANUS client connected: {self.address}")

        except Exception as e:
            print(f"MANUS connection failed: {e}")
            raise

    def receive_manus_data(self):
        try:
            data = []
            for _ in range(9):
                recv_bytes = self.connection.recv(8)
                if not recv_bytes:
                    return None
                float_str = recv_bytes.decode("ascii")
                value = float(float_str)
                data.append(value)
            return data
        except Exception as e:
            print(f"MANUS recv failed: {e}")
            return None

    def process_hand_calibration(self, data):
        if not self.right_hand_connect and data[0] == 1002 and data[1] != 0:
            self.right_hand_connect = True
            self.right_hand_data0 = np.array(data)
            self.set_data = self.right_hand_data0 - self.right_hand_data0
            print("Right hand calibrated")
            return True
        return False

    def calculate_finger_angles(self, data):
        if not self.right_hand_connect or data[0] != 1002:
            return None

        diff = np.array(data) - self.set_data
        scaled_data = np.array(data) * self.scale_factor
        self.set_data = np.array(scaled_data) - self.right_hand_data0

        if np.max(np.abs(diff)) < self.control_threshold:
            return None

        return self.set_data

    def angle_process(self, angles):
        if angles is None:
            return None

        pinky_angle = int(max(0, min(1000, 1000 - angles[8])))
        ring_angle = int(max(0, min(1000, 1000 - angles[7])))
        middle_angle = int(max(0, min(1000, 1000 - angles[6])))
        index_angle = int(max(0, min(1000, 1000 - angles[4])))
        thumb_angle_2 = int(min(1000, max(0, 0 + 1.5 * angles[2])))
        thumb_angle = int(min(1000, max(0, 1000 - 1.7 * angles[1])))

        action = np.array(
            [
                pinky_angle,
                ring_angle,
                middle_angle,
                index_angle,
                thumb_angle_2,
                thumb_angle,
            ],
            dtype=np.float32,
        )
        return action

    def run(self):
        try:
            print("Preparing MANUS action subprocess...")
            self.initialize_manus_connection()
        except Exception as e:
            print(f"prepare MANUS action subprocess failed: {e}")
            return

        try:
            while not self.stop_event.is_set():
                t_start = time.time()

                data = self.receive_manus_data()
                if data is None:
                    raise ConnectionError("MANUS data interrupted")

                self.process_hand_calibration(data)

                angles = self.calculate_finger_angles(data)
                action = self.angle_process(angles)

                if action is not None:
                    self._last_published_action = action
                if self.action_buffer is not None:
                    # Publish at steady rate; if no new gesture this cycle, repeat last action.
                    self.action_buffer.publish(
                        self._last_published_action,
                        t_host_ns=time.monotonic_ns(),
                    )

                t_end = time.time()
                time.sleep(max(0.0, 1.0 / 50.0 - (t_end - t_start)))

        except ConnectionError as e:
            print(f"connection error: {e}")
        except Exception as e:
            print(f"teleoperation exception: {e}")
        finally:
            self.cleanup()

    def finalize(self):
        self.stop_event.set()

    def terminate(self):
        return super().terminate()

    def cleanup(self):
        print("Cleaning subprocess resources...")
        try:
            if self.connection:
                self.connection.close()
                print("socket connection closed")

            if self.server:
                self.server.close()
                print("server socket closed")

        except Exception as e:
            print(f"cleanup error: {e}")

        print("cleanup done")


class inspire_Manus:
    """Top-level MANUS action reader.

    Starts one MANUS socket subprocess per enabled hand and returns the latest
    raw 0-1000 Inspire hand command from shared memory. If no newer sample is
    available before the wait timeout, __call__ returns the last cached action
    for that hand, initialized as zeros. 
    """

    def __init__(
        self,
        use_right_hand=True,
        use_left_hand=False,
        manus_port=8888,
        control_threshold=10,
        scale_factor=15,
        wait_timeout_s=0.03,
    ):
        self.use_right_hand = use_right_hand
        self.use_left_hand = use_left_hand

        self.right_buffer = SharedLatestActionBuffer(dim=6)
        self.left_buffer = SharedLatestActionBuffer(dim=6)

        self._last_right_seq = 0
        self._last_left_seq = 0
        self._last_right_action = np.zeros(6, dtype=np.float32)
        self._last_left_action = np.zeros(6, dtype=np.float32)
        self.wait_timeout_s = None if wait_timeout_s is None else float(wait_timeout_s)

        if use_right_hand:
            self.righthand_process = ManusTeleoperationProcess(
                action_buffer=self.right_buffer,
                manus_port=manus_port,
                control_threshold=control_threshold,
                scale_factor=scale_factor,
            )

        if use_left_hand:
            self.lefthand_process = ManusTeleoperationProcess(
                action_buffer=self.left_buffer,
                manus_port=manus_port,
                control_threshold=control_threshold,
                scale_factor=scale_factor,
            )

    def start(self):
        if self.use_right_hand:
            self.righthand_process.start()
        if self.use_left_hand:
            self.lefthand_process.start()

    def __call__(self):
        action_dict = {}

        if self.use_right_hand:
            sample = self.right_buffer.wait_next(
                last_seq=self._last_right_seq,
                timeout=self.wait_timeout_s,
                copy_data=True,
            )
            if sample is not None:
                self._last_right_seq = sample["seq"]
                self._last_right_action = sample["action"]
            action_dict["right"] = self._last_right_action.copy()

        if self.use_left_hand:
            sample = self.left_buffer.wait_next(
                last_seq=self._last_left_seq,
                timeout=self.wait_timeout_s,
                copy_data=True,
            )
            if sample is not None:
                self._last_left_seq = sample["seq"]
                self._last_left_action = sample["action"]
            action_dict["left"] = self._last_left_action.copy()

        return action_dict

    def finalize(self):
        if self.use_right_hand:
            self.righthand_process.finalize()
            self.righthand_process.join()
        if self.use_left_hand:
            self.lefthand_process.finalize()
            self.lefthand_process.join()

        # wake up any potential waiters
        self.right_buffer.close()
        self.left_buffer.close()

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass


if __name__ == "__main__":
    inspire_manus = inspire_Manus(use_right_hand=True, use_left_hand=False)

    inspire_manus.start()
    time.sleep(1)

    for _ in range(100):
        start_time = time.time()
        action_dict = inspire_manus()
        end_time = time.time()
        print(f"Action time: {end_time - start_time}")

        if "right" in action_dict:
            print("Right Hand Action:", action_dict["right"])
        if "left" in action_dict:
            print("Left Hand Action:", action_dict["left"])
        time.sleep(max(0.0, 1.0 / 25.0 - (time.time() - start_time)))

    inspire_manus.finalize()
