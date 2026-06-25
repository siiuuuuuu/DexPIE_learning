#!/usr/bin/env python3
"""
Shared-memory version of MultiRealSense.

This module keeps the external API close to multi_realsense.py:
- MultiRealSense(...)
- start()
- __call__() -> latest frames dict
- finalize()

Key change:
- camera process writes frames into shared-memory ring buffers
- main process reads the newest committed frame without pickling large numpy arrays
"""

import ctypes
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import cv2
import numpy as np
import pyrealsense2 as rs

np.printoptions(3, suppress=True)


# -------- RealSense helpers (kept from original file style) --------
def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [
        devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))
    ]
    devices.sort()
    print("Found {} devices: {}".format(len(devices), devices), flush=True)
    return devices


def init_given_realsense_D415(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
    color_fps=30,
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing D415 camera {}".format(device), flush=True)

    if enable_depth:
        w, h = 1280, 720
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)

    if enable_rgb:
        w, h = 640, 480
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, int(color_fps))

    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )
        print("D415 camera {} init done.".format(device), flush=True)
        return pipeline, align, depth_scale, camera_info

    print("D415 camera {} init done.".format(device), flush=True)
    return pipeline, None, None, None


def grid_sample_pcd(point_cloud, grid_size=0.005):
    coords = point_cloud[:, :3]
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)

    keys = (
        grid_coords[:, 0]
        + grid_coords[:, 1] * 10000
        + grid_coords[:, 2] * 100000000
    )
    _, indices = np.unique(keys, return_index=True)
    return point_cloud[indices]


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale=1):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


# -------- Shared-memory ring buffer --------
def _attach_shm(name):
    # Python 3.13+ supports track=..., older versions do not.
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


class SharedFrameRing:
    """Shared-memory ring for fixed-shape numpy frames."""

    def __init__(self, slots, shape, dtype=np.uint8):
        self.slots = int(slots)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

        item_nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        total_nbytes = self.slots * item_nbytes

        self.shm = shared_memory.SharedMemory(create=True, size=total_nbytes)
        self.arr = np.ndarray(
            (self.slots, *self.shape), dtype=self.dtype, buffer=self.shm.buf
        )
        self.arr.fill(0)

        # Metadata shared via multiprocessing primitives (small payload).
        self.lock = mp.Lock()
        self.latest_seq = mp.Value(ctypes.c_ulonglong, 0)
        self.latest_idx = mp.Value(ctypes.c_longlong, -1)

        self.slot_seq = mp.Array(ctypes.c_ulonglong, self.slots, lock=False)
        self.slot_host_ns = mp.Array(ctypes.c_ulonglong, self.slots, lock=False)
        self.slot_dev_ts = mp.Array(ctypes.c_double, self.slots, lock=False)
        self.slot_frame_no = mp.Array(ctypes.c_ulonglong, self.slots, lock=False)
        self.slot_valid = mp.Array(ctypes.c_byte, self.slots, lock=False)

    def descriptor(self):
        return {
            "shm_name": self.shm.name,
            "slots": self.slots,
            "shape": self.shape,
            "dtype": self.dtype.str,
            "lock": self.lock,
            "latest_seq": self.latest_seq,
            "latest_idx": self.latest_idx,
            "slot_seq": self.slot_seq,
            "slot_host_ns": self.slot_host_ns,
            "slot_dev_ts": self.slot_dev_ts,
            "slot_frame_no": self.slot_frame_no,
            "slot_valid": self.slot_valid,
        }

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


class SharedFrameRingAccessor:
    """Attach to a SharedFrameRing from parent or child process."""

    def __init__(self, desc):
        self.desc = desc
        self.slots = int(desc["slots"])
        self.shape = tuple(desc["shape"])
        self.dtype = np.dtype(desc["dtype"])

        self.lock = desc["lock"]
        self.latest_seq = desc["latest_seq"]
        self.latest_idx = desc["latest_idx"]
        self.slot_seq = desc["slot_seq"]
        self.slot_host_ns = desc["slot_host_ns"]
        self.slot_dev_ts = desc["slot_dev_ts"]
        self.slot_frame_no = desc["slot_frame_no"]
        self.slot_valid = desc["slot_valid"]

        self.shm = _attach_shm(desc["shm_name"])
        self.arr = np.ndarray(
            (self.slots, *self.shape), dtype=self.dtype, buffer=self.shm.buf
        )

    def publish(self, frame, t_host_ns, t_dev_ts, frame_no):
        if frame is None:
            return

        if frame.dtype != self.dtype:
            frame = frame.astype(self.dtype, copy=False)
        if frame.shape != self.shape:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Ensure contiguous write source.
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        with self.lock:
            seq = int(self.latest_seq.value) + 1
            idx = (seq - 1) % self.slots
            self.arr[idx] = frame

            self.slot_seq[idx] = seq
            self.slot_host_ns[idx] = int(t_host_ns)
            self.slot_dev_ts[idx] = float(t_dev_ts)
            self.slot_frame_no[idx] = int(frame_no)
            self.slot_valid[idx] = 1

            self.latest_idx.value = idx
            self.latest_seq.value = seq

    def latest(self, copy_frame=False):
        with self.lock:
            idx = int(self.latest_idx.value)
            if idx < 0 or self.slot_valid[idx] == 0:
                return None

            sample = {
                "seq": int(self.slot_seq[idx]),
                "t_host_ns": int(self.slot_host_ns[idx]),
                "t_dev_ts": float(self.slot_dev_ts[idx]),
                "frame_no": int(self.slot_frame_no[idx]),
                "slot_idx": idx,
                "frame": self.arr[idx].copy() if copy_frame else self.arr[idx],
            }
            return sample

    def nearest_by_host_time(self, target_host_ns, copy_frame=False):
        target_host_ns = int(target_host_ns)
        with self.lock:
            best_idx = -1
            best_delta_ns = None
            for idx in range(self.slots):
                if self.slot_valid[idx] == 0:
                    continue
                delta_ns = int(self.slot_host_ns[idx]) - target_host_ns
                abs_delta_ns = abs(delta_ns)
                if best_delta_ns is None or abs_delta_ns < best_delta_ns:
                    best_idx = idx
                    best_delta_ns = abs_delta_ns

            if best_idx < 0:
                return None

            delta_ns = int(self.slot_host_ns[best_idx]) - target_host_ns
            sample = {
                "seq": int(self.slot_seq[best_idx]),
                "t_host_ns": int(self.slot_host_ns[best_idx]),
                "t_dev_ts": float(self.slot_dev_ts[best_idx]),
                "frame_no": int(self.slot_frame_no[best_idx]),
                "slot_idx": best_idx,
                "sync_delta_to_target_ms": delta_ns / 1e6,
                "frame": self.arr[best_idx].copy()
                if copy_frame
                else self.arr[best_idx],
            }
            return sample

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass


class SingleVisionProcess(mp.Process):
    def __init__(
        self,
        device,
        ring_desc,
        enable_rgb=True,
        enable_depth=False,
        enable_pointcloud=False,
        sync_mode=0,
        num_points=2048,
        z_far=1.0,
        z_near=0.1,
        use_grid_sampling=True,
        use_crop=False,
        img_size=384,
        color_fps=30,
    ):
        super().__init__()
        self.daemon = True

        self.device = device
        self.ring_desc = ring_desc

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode

        self.use_grid_sampling = use_grid_sampling
        self.use_crop = use_crop

        self.resize = True
        self.height, self.width = img_size, img_size
        self.color_fps = int(color_fps)

        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points

    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            color_frame_obj = aligned_frames.get_color_frame()
            frame_ts = color_frame_obj.get_timestamp()
            frame_no = color_frame_obj.get_frame_number()
            color_frame = np.asanyarray(color_frame_obj.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())

            clip_lower = 0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high

            if self.enable_pointcloud:
                point_cloud_frame = self.create_colored_point_cloud(
                    color_frame,
                    depth_frame,
                    far=self.z_far,
                    near=self.z_near,
                    num_points=self.num_points,
                    use_crop=self.use_crop,
                )
            else:
                point_cloud_frame = None
        else:
            color_frame_obj = frame.get_color_frame()
            frame_ts = color_frame_obj.get_timestamp()
            frame_no = color_frame_obj.get_frame_number()
            color_frame = np.asanyarray(color_frame_obj.get_data(), dtype=np.uint8)
            depth_frame = None
            point_cloud_frame = None

        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(
                    color_frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            if self.enable_depth and depth_frame is not None:
                depth_frame = cv2.resize(
                    depth_frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )

        return color_frame, depth_frame, point_cloud_frame, frame_ts, frame_no

    def run(self):
        ring = SharedFrameRingAccessor(self.ring_desc)

        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense_D415(
            self.device,
            enable_rgb=self.enable_rgb,
            enable_depth=self.enable_depth,
            enable_point_cloud=self.enable_pointcloud,
            sync_mode=self.sync_mode,
            color_fps=self.color_fps,
        )

        try:
            while True:
                color_frame, _, _, frame_ts, frame_no = self.get_vision()
                t_host_ns = time.monotonic_ns()
                ring.publish(
                    frame=color_frame,
                    t_host_ns=t_host_ns,
                    t_dev_ts=frame_ts,
                    frame_no=frame_no,
                )
        finally:
            ring.close()

    def terminate(self):
        return super().terminate()

    def crop_point_cloud(self, point_cloud):
        return point_cloud

    def create_colored_point_cloud(self, color, depth, far=1.0, near=0.1, num_points=10000, use_crop=False):
        assert depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1]

        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])

        mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        cloud = cloud[mask]
        color = color.reshape([-1, 3])
        color = color[mask]

        colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        if self.use_grid_sampling:
            colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
        if use_crop:
            colored_cloud = self.crop_point_cloud(colored_cloud)

        if num_points > colored_cloud.shape[0]:
            num_pad = num_points - colored_cloud.shape[0]
            pad_points = np.zeros((num_pad, 6), dtype=np.float32)
            colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
        else:
            selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
            colored_cloud = colored_cloud[selected_idx]

        np.random.shuffle(colored_cloud)
        return colored_cloud


class MultiRealSense:
    def __init__(
        self,
        use_front_cam=True,
        use_right_cam=True,
        front_cam_idx=0,
        right_cam_idx=1,
        front_num_points=4096,
        right_num_points=1024,
        front_z_far=1.0,
        front_z_near=0.1,
        right_z_far=0.5,
        right_z_near=0.01,
        use_grid_sampling=True,
        use_crop=False,
        img_size=256,
        ring_slots=8,
        front_camera_fps=30,
        wrist_camera_fps=60,
        sync_right_to_front=True,
        sync_wait_timeout_ms=5.0,
    ):
        self.devices = get_realsense_id()

        self.use_front_cam = use_front_cam
        self.use_right_cam = use_right_cam
        self.front_camera_fps = int(front_camera_fps)
        self.wrist_camera_fps = int(wrist_camera_fps)
        self.sync_right_to_front = bool(sync_right_to_front)
        self.sync_wait_timeout_ms = float(sync_wait_timeout_ms)

        # Shared rings: only RGB is shared (matching current project usage).
        color_shape = (img_size, img_size, 3)
        self.front_ring = SharedFrameRing(ring_slots, color_shape, dtype=np.uint8) if use_front_cam else None
        self.right_ring = SharedFrameRing(ring_slots, color_shape, dtype=np.uint8) if use_right_cam else None

        if use_front_cam:
            self.front_process = SingleVisionProcess(
                self.devices[front_cam_idx],
                self.front_ring.descriptor(),
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=1,
                num_points=front_num_points,
                z_far=front_z_far,
                z_near=front_z_near,
                use_grid_sampling=use_grid_sampling,
                use_crop=use_crop,
                img_size=img_size,
                color_fps=self.front_camera_fps,
            )
            self.front_reader = SharedFrameRingAccessor(self.front_ring.descriptor())

        if use_right_cam:
            self.right_process = SingleVisionProcess(
                self.devices[right_cam_idx],
                self.right_ring.descriptor(),
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=1,
                num_points=right_num_points,
                z_far=right_z_far,
                z_near=right_z_near,
                use_grid_sampling=use_grid_sampling,
                use_crop=use_crop,
                img_size=img_size,
                color_fps=self.wrist_camera_fps,
            )
            self.right_reader = SharedFrameRingAccessor(self.right_ring.descriptor())

    def start(self):
        if self.use_front_cam:
            self.front_process.start()
            print("front camera start.", flush=True)

        if self.use_right_cam:
            self.right_process.start()
            print("right camera start.", flush=True)

    @staticmethod
    def _wait_latest(reader, sleep_s=0.001):
        sample = reader.latest(copy_frame=True)
        while sample is None:
            time.sleep(sleep_s)
            sample = reader.latest(copy_frame=True)
        return sample

    @staticmethod
    def _wait_nearest_by_host_time(
        reader,
        target_host_ns,
        timeout_ms=0.0,
        sleep_s=0.001,
    ):
        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        best = None
        while True:
            sample = reader.nearest_by_host_time(
                target_host_ns,
                copy_frame=True,
            )
            if sample is not None:
                best = sample
                if sample["t_host_ns"] >= target_host_ns:
                    return sample

            if time.monotonic() >= deadline:
                return best

            time.sleep(sleep_s)

    def __call__(self):
        cam_dict = {}
        front = None

        if self.use_front_cam:
            front = self._wait_latest(self.front_reader)
            cam_dict.update(
                {
                    "front_color": front["frame"],
                    "front_depth": None,
                    "front_point_cloud": None,
                    "front_meta": {
                        "seq": front["seq"],
                        "t_host_ns": front["t_host_ns"],
                        "t_dev_ts": front["t_dev_ts"],
                        "frame_no": front["frame_no"],
                    },
                }
            )

        if self.use_right_cam:
            if self.sync_right_to_front and front is not None:
                right = self._wait_nearest_by_host_time(
                    self.right_reader,
                    front["t_host_ns"],
                    timeout_ms=self.sync_wait_timeout_ms,
                )
                if right is None:
                    right = self._wait_latest(self.right_reader)
            else:
                right = self._wait_latest(self.right_reader)
            cam_dict.update(
                {
                    "right_color": right["frame"],
                    "right_depth": None,
                    "right_point_cloud": None,
                    "right_meta": {
                        "seq": right["seq"],
                        "t_host_ns": right["t_host_ns"],
                        "t_dev_ts": right["t_dev_ts"],
                        "frame_no": right["frame_no"],
                        "sync_delta_to_front_ms": right.get(
                            "sync_delta_to_target_ms"
                        ),
                    },
                }
            )

        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            if hasattr(self, "front_process") and self.front_process.is_alive():
                self.front_process.terminate()
                self.front_process.join()
            if hasattr(self, "front_reader"):
                self.front_reader.close()
            if self.front_ring is not None:
                self.front_ring.close()
                self.front_ring.unlink()

        if self.use_right_cam:
            if hasattr(self, "right_process") and self.right_process.is_alive():
                self.right_process.terminate()
                self.right_process.join()
            if hasattr(self, "right_reader"):
                self.right_reader.close()
            if self.right_ring is not None:
                self.right_ring.close()
                self.right_ring.unlink()

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass


if __name__ == "__main__":
    cam = MultiRealSense(
        use_right_cam=False,
        front_num_points=20000,
        use_grid_sampling=True,
        use_crop=False,
        img_size=256,
    )
    cam.start()
    time.sleep(1)

    for i in range(10):
        t0 = time.time()
        out = cam()
        dt = time.time() - t0
        front = out.get("front_color", None)
        shape = None if front is None else front.shape
        print("shared_read_time:", dt, "shape:", shape)
        time.sleep(max(0.0, 1.0 / 30.0 - dt))

    cam.finalize()
