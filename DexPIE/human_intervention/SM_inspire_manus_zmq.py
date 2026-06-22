"""ZMQ MANUS-to-Inspire source.

This module is the in-process replacement for the legacy TCP/multiprocessing
MANUS reader. It subscribes to main_zmq_protobuf.cpp, keeps only the latest
right-hand command, and exposes the same source methods used by
HighRateHandController: start(), latest_right_sample(), __call__(), finalize().
"""

import struct
import threading
import time

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None


WIRE_VARINT = 0
WIRE_FIXED32 = 5
WIRE_LENGTH_DELIMITED = 2
HAND_VALUE_COUNT = 8


def _read_varint(buf, pos):
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")


def _read_field(buf, pos):
    key, pos = _read_varint(buf, pos)
    field_number = key >> 3
    wire_type = key & 0x07

    if wire_type == WIRE_VARINT:
        value, pos = _read_varint(buf, pos)
        return field_number, wire_type, value, pos

    if wire_type == WIRE_LENGTH_DELIMITED:
        length, pos = _read_varint(buf, pos)
        end = pos + length
        if end > len(buf):
            raise ValueError("truncated length-delimited field")
        return field_number, wire_type, buf[pos:end], end

    if wire_type == WIRE_FIXED32:
        end = pos + 4
        if end > len(buf):
            raise ValueError("truncated fixed32 field")
        return field_number, wire_type, buf[pos:end], end

    raise ValueError(f"unsupported wire type {wire_type}")


def _parse_packed_float32(buf):
    if len(buf) % 4 != 0:
        raise ValueError(f"packed float length is not divisible by 4: {len(buf)}")
    if not buf:
        return []
    return list(struct.unpack("<" + "f" * (len(buf) // 4), buf))


def _parse_stamp_message(buf):
    stamp = {"sec": 0, "nanosec": 0}
    pos = 0
    while pos < len(buf):
        field_number, wire_type, value, pos = _read_field(buf, pos)
        if wire_type != WIRE_VARINT:
            continue
        if field_number == 1:
            stamp["sec"] = int(value)
        elif field_number == 2:
            stamp["nanosec"] = int(value)
    return stamp


def _parse_header_message(buf):
    header = {"frame_id": 0, "stamp": {"sec": 0, "nanosec": 0}, "data_seq": 0}
    pos = 0
    while pos < len(buf):
        field_number, wire_type, value, pos = _read_field(buf, pos)
        if field_number == 1 and wire_type == WIRE_VARINT:
            header["frame_id"] = int(value)
        elif field_number == 2 and wire_type == WIRE_LENGTH_DELIMITED:
            header["stamp"] = _parse_stamp_message(value)
        elif field_number == 3 and wire_type == WIRE_VARINT:
            header["data_seq"] = int(value)
    return header


def _parse_hand_ergo_message(buf):
    hand = {"valid": False, "glove_id": 0, "angle_deg": []}
    pos = 0
    while pos < len(buf):
        field_number, wire_type, value, pos = _read_field(buf, pos)
        if field_number == 1 and wire_type == WIRE_VARINT:
            hand["valid"] = bool(value)
        elif field_number == 2 and wire_type == WIRE_VARINT:
            hand["glove_id"] = int(value)
        elif field_number == 3 and wire_type == WIRE_LENGTH_DELIMITED:
            hand["angle_deg"] = _parse_packed_float32(value)
    return hand


def parse_manus_zmq_frame(buf):
    frame = {
        "header": {"frame_id": 0, "stamp": {"sec": 0, "nanosec": 0}, "data_seq": 0},
        "left": {"valid": False, "glove_id": 0, "angle_deg": []},
        "right": {"valid": False, "glove_id": 0, "angle_deg": []},
    }
    pos = 0
    while pos < len(buf):
        field_number, wire_type, value, pos = _read_field(buf, pos)
        if field_number == 1 and wire_type == WIRE_LENGTH_DELIMITED:
            frame["header"] = _parse_header_message(value)
        elif field_number == 2 and wire_type == WIRE_LENGTH_DELIMITED:
            frame["left"] = _parse_hand_ergo_message(value)
        elif field_number == 3 and wire_type == WIRE_LENGTH_DELIMITED:
            frame["right"] = _parse_hand_ergo_message(value)
    return frame


class LatestActionBuffer:
    """Thread-safe latest-only action buffer."""

    def __init__(self, dim=6):
        self.dim = int(dim)
        self._action = np.zeros(self.dim, dtype=np.float32)
        self._seq = 0
        self._t_host_ns = 0
        self._valid = False
        self._closed = False
        self._cond = threading.Condition()

    def _snapshot(self):
        return {
            "seq": int(self._seq),
            "t_host_ns": int(self._t_host_ns),
            "action": self._action.copy(),
        }

    def publish(self, action, t_host_ns=None, seq=None):
        if action is None:
            return

        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size != self.dim:
            raise ValueError(f"action dim mismatch: expected {self.dim}, got {vec.size}")

        if t_host_ns is None:
            t_host_ns = time.monotonic_ns()

        with self._cond:
            if self._closed:
                return
            self._action[:] = vec
            if seq is None:
                self._seq += 1
            else:
                self._seq = int(seq)
            self._t_host_ns = int(t_host_ns)
            self._valid = True
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            if not self._valid:
                return None
            return self._snapshot()

    def wait_next(self, last_seq=0, timeout=None):
        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        with self._cond:
            while True:
                if self._closed:
                    return None
                if self._valid and int(self._seq) > int(last_seq):
                    return self._snapshot()
                if timeout is None:
                    self._cond.wait()
                else:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        return None
                    self._cond.wait(remain)

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class ManusZmqSource:
    """Receives MANUS ZMQ frames and publishes latest Inspire actions."""

    def __init__(
        self,
        use_right_hand=True,
        use_left_hand=False,
        zmq_endpoint="tcp://127.0.0.1:2044",
        control_threshold=10,
        scale_factor=15,
        wait_timeout_s=0.03,
        rcvhwm=1,
        conflate=True,
        poll_timeout_ms=100,
        stats_every_s=0.0,
    ):
        if use_left_hand:
            raise NotImplementedError("ZMQ source currently maps the right hand only")
        self.use_right_hand = bool(use_right_hand)
        self.use_left_hand = bool(use_left_hand)
        self.zmq_endpoint = zmq_endpoint
        self.control_threshold = control_threshold
        self.scale_factor = scale_factor
        self.wait_timeout_s = None if wait_timeout_s is None else float(wait_timeout_s)
        self.rcvhwm = int(rcvhwm)
        self.conflate = bool(conflate)
        self.poll_timeout_ms = int(poll_timeout_ms)
        self.stats_every_s = float(stats_every_s)

        self.right_buffer = LatestActionBuffer(dim=6)
        self.left_buffer = LatestActionBuffer(dim=6)
        self._last_right_seq = 0
        self._last_left_seq = 0
        self._last_right_action = np.zeros(6, dtype=np.float32)
        self._last_left_action = np.zeros(6, dtype=np.float32)

        self.right_hand_connect = False
        self.right_hand_data0 = None
        self.set_data = None
        self._last_published_action = np.zeros(6, dtype=np.float32)

        self._stop_event = threading.Event()
        self._thread = None
        self._context = None
        self._socket = None
        self._poller = None
        self._recv_errors = 0
        self._received = 0
        self._published = 0
        self._last_frame_seq = None
        self._dropped_seq = 0
        self._last_stats_time = time.monotonic()
        self._last_stats_received = 0

        print("MANUS ZMQ source initialized")
        print(f"manus zmq endpoint: {zmq_endpoint}")

    def start(self):
        if self._thread is not None:
            raise RuntimeError("MANUS ZMQ source is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="manus-zmq-source",
            daemon=True,
        )
        self._thread.start()

    def latest_right_sample(self):
        if not self.use_right_hand:
            return None
        return self.right_buffer.latest()

    def __call__(self):
        action_dict = {}
        if self.use_right_hand:
            sample = self.right_buffer.wait_next(
                last_seq=self._last_right_seq,
                timeout=self.wait_timeout_s,
            )
            if sample is not None:
                self._last_right_seq = sample["seq"]
                self._last_right_action = sample["action"]
            action_dict["right"] = self._last_right_action.copy()

        if self.use_left_hand:
            sample = self.left_buffer.wait_next(
                last_seq=self._last_left_seq,
                timeout=self.wait_timeout_s,
            )
            if sample is not None:
                self._last_left_seq = sample["seq"]
                self._last_left_action = sample["action"]
            action_dict["left"] = self._last_left_action.copy()
        return action_dict

    def finalize(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.right_buffer.close()
        self.left_buffer.close()

    def _run_guarded(self):
        try:
            self._run()
        except Exception as exc:
            print(f"MANUS ZMQ source failed: {exc}")
        finally:
            self._cleanup()

    def _connect(self):
        if zmq is None:
            raise ImportError("pyzmq is required for MANUS ZMQ source")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, self.rcvhwm)
        if self.conflate:
            self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(self.zmq_endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        print(
            "Listening MANUS ZMQ stream "
            f"(endpoint={self.zmq_endpoint}, rcvhwm={self.rcvhwm}, "
            f"conflate={self.conflate})..."
        )

    def _run(self):
        self._connect()
        while not self._stop_event.is_set():
            events = dict(self._poller.poll(self.poll_timeout_ms))
            if self._socket not in events:
                continue
            try:
                payload = self._socket.recv()
                frame = parse_manus_zmq_frame(payload)
                self._received += 1
                self._handle_frame(frame)
                self._maybe_print_stats()
            except Exception as exc:
                self._recv_errors += 1
                if self._recv_errors <= 5 or self._recv_errors % 100 == 0:
                    print(f"MANUS ZMQ recv failed: {exc}")

    def _handle_frame(self, frame):
        header = frame["header"]
        source_seq = int(header.get("data_seq", 0))
        if self._last_frame_seq is not None and source_seq > self._last_frame_seq + 1:
            self._dropped_seq += source_seq - self._last_frame_seq - 1
        self._last_frame_seq = source_seq

        right = frame["right"]
        if not right["valid"]:
            return
        if len(right["angle_deg"]) != HAND_VALUE_COUNT:
            raise ValueError(
                f"expected {HAND_VALUE_COUNT} right-hand values, "
                f"got {len(right['angle_deg'])}"
            )

        data = np.asarray([1002.0] + right["angle_deg"], dtype=np.float32)
        self._process_hand_calibration(data)
        angles = self._calculate_finger_angles(data)
        action = self._angle_process(angles)

        if action is not None:
            self._last_published_action = action

        self.right_buffer.publish(
            self._last_published_action,
            t_host_ns=time.monotonic_ns(),
            seq=source_seq if source_seq > 0 else None,
        )
        self._published += 1

    def _process_hand_calibration(self, data):
        if not self.right_hand_connect and data[0] == 1002 and data[1] != 0:
            self.right_hand_connect = True
            self.right_hand_data0 = np.array(data)
            self.set_data = self.right_hand_data0 - self.right_hand_data0
            print("Right hand calibrated from ZMQ")
            return True
        return False

    def _calculate_finger_angles(self, data):
        if not self.right_hand_connect or data[0] != 1002:
            return None
        diff = np.array(data) - self.set_data
        scaled_data = np.array(data) * self.scale_factor
        self.set_data = np.array(scaled_data) - self.right_hand_data0
        if np.max(np.abs(diff)) < self.control_threshold:
            return None
        return self.set_data

    def _angle_process(self, angles):
        if angles is None:
            return None
        pinky_angle = int(max(0, min(1000, 1000 - angles[8])))
        ring_angle = int(max(0, min(1000, 1000 - angles[7])))
        middle_angle = int(max(0, min(1000, 1000 - angles[6])))
        index_angle = int(max(0, min(1000, 1000 - angles[4])))
        thumb_angle_2 = int(min(1000, max(0, 0 + 1.5 * angles[2])))
        thumb_angle = int(min(1000, max(0, 1000 - 1.7 * angles[1])))
        return np.array(
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

    def _maybe_print_stats(self):
        if self.stats_every_s <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_stats_time
        if elapsed < self.stats_every_s:
            return
        received_delta = self._received - self._last_stats_received
        hz = received_delta / elapsed if elapsed > 0 else 0.0
        print(
            f"MANUS ZMQ stats: recv_hz={hz:.1f} received={self._received} "
            f"published={self._published} drop_seq={self._dropped_seq} "
            f"recv_errors={self._recv_errors}"
        )
        self._last_stats_time = now
        self._last_stats_received = self._received

    def _cleanup(self):
        print("Cleaning MANUS ZMQ source resources...")
        try:
            if self._socket is not None:
                self._socket.close(0)
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
        except Exception as exc:
            print(f"MANUS ZMQ cleanup error: {exc}")
        print("MANUS ZMQ cleanup done")

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass


# Keep the same class name used by servoL.py / hand_control_controller.py.
inspire_Manus = ManusZmqSource


if __name__ == "__main__":
    source = inspire_Manus(stats_every_s=1.0)
    source.start()
    try:
        while True:
            sample = source.latest_right_sample()
            if sample is not None:
                print(f"seq={sample['seq']} action={sample['action']}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        source.finalize()
