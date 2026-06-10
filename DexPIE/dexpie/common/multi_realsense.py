#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque 
import imageio
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
import time
import multiprocessing
multiprocessing.set_start_method('fork')
from dexpie.common.LIFO_Queue import LIFOQueue

np.printoptions(3, suppress=True)

def get_realsense_id():
    ctx = rs.context()# RealSense API context object.
    devices = ctx.query_devices()# Query currently connected and visible RealSense devices.
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]# Get each device serial number.
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

def init_given_realsense_L515(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        # L515
        h, w = 768, 1024
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        # L515
        h, w = 540, 960
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        # for L515
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        
        # set min distance
        # for L515
        depth_sensor.set_option(rs.option.min_distance, 0.05)
        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None

def init_given_realsense_D455(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D455
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None
# Initialize D435 camera.
def init_given_realsense_D435(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device),flush=True)

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D455
        h, w = 720, 1280
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        w, h = 640, 480
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 60)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # Get depth scale and convert depth units to meters.
        align = rs.align(rs.stream.color)
        # Align depth frame to color frame.
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None
# Same structure as D435 initialization.
def init_given_realsense_D415(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    """
    Intel RealSense D415 initialization wrapper.
    Parameters have the same meaning as the D435 version.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing D415 camera {}".format(device),flush=True)

    if enable_depth:
        # D415 supports 1280x720@30 Z16, same as D435.
        w, h = 1280, 720
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)

    if enable_rgb:
        # Keep RGB settings consistent with D435.
        # For 1920x1080, change w,h to 1920,1080.
        w, h = 640, 480
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 60)

    # Resolve and start pipeline.
    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # Unit: meters.
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
        print("D415 camera {} init done.".format(device),flush=True)
        return pipeline, align, depth_scale, camera_info
    else:
        print("D415 camera {} init done.".format(device),flush=True)
        return pipeline, None, None, None


class CameraInfo():
    """Camera intrinsics."""
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        
class SingleVisionProcess(Process):
    def __init__(self, device, queue,
                enable_rgb=True,
                enable_depth=False,
                sync_mode=0,
                img_size=384) -> None:
        super(SingleVisionProcess, self).__init__()
        self.daemon = True# Daemon process exits automatically with the parent process.
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.sync_mode = sync_mode

  
        self.resize = True
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size
   
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()# Blocking API; waits for a frameset for later alignment.

        if self.enable_depth:
            aligned_frames = self.align.process(frame)# Align depth frame to color frame.
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
    
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            clip_lower =  0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high
            
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data(),dtype=np.uint8)
            depth_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)
        
        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.enable_depth:
                depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_frame, depth_frame


    def run(self):
        device_name = "D415"
        if device_name == "L515":
            init_given_realsense = init_given_realsense_L515
        elif device_name == "D435":
            init_given_realsense = init_given_realsense_D435
        elif device_name == "D455":
            init_given_realsense = init_given_realsense_D455
        elif device_name == "D415":
            init_given_realsense = init_given_realsense_D415
        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(self.device, 
                    enable_rgb=self.enable_rgb, enable_depth=self.enable_depth,
                    sync_mode=self.sync_mode)

        debug = False
        while True:
            color_frame, depth_frame = self.get_vision()
            self.queue.put([color_frame, depth_frame])
            #self.queue.put(color_frame)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

class MultiRealSense(object):
    def __init__(self, use_front_cam=True, use_right_cam=False,
                 front_cam_idx=0, right_cam_idx=1, 
                 img_size=1024):

        self.devices = get_realsense_id()
    
        self.front_queue = LIFOQueue(maxsize=5)# LIFO circular queue that stores only the latest 5 frames.
        self.right_queue = LIFOQueue(maxsize=5)# LIFO circular queue that stores only the latest 5 frames.

      
        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        if use_front_cam:
            self.front_process = SingleVisionProcess(self.devices[front_cam_idx], self.front_queue,
                            enable_rgb=True, enable_depth=False, sync_mode=1, img_size=img_size)
        if use_right_cam:
            self.right_process = SingleVisionProcess(self.devices[right_cam_idx], self.right_queue,
                    enable_rgb=True, enable_depth=False, sync_mode=1, img_size=img_size)

        self.use_front_cam = use_front_cam
        self.use_right_cam = use_right_cam
    def start(self):
        if self.use_front_cam:
            self.front_process.start()# Start the front camera subprocess and run loop.
            print("front camera start.",flush=True)

        if self.use_right_cam:
            self.right_process.start()
            print("right camera start.",flush=True)
         

    
    ## Callback that returns the latest frame.
    def __call__(self):  
        cam_dict = {}
        if self.use_front_cam:  
            front_color, front_depth = self.front_queue.get()# Get the latest frame from the circular queue.
            #front_color = self.front_queue.get()
            # External request frequency is usually higher than camera capture frequency.
            # The queue avoids blocking for capture; callbacks consume the latest completed frame.
            cam_dict.update({'front_color': front_color, 'front_depth': front_depth})
 
        if self.use_right_cam: 
            right_color, right_depth = self.right_queue.get()
            cam_dict.update({'right_color': right_color, 'right_depth': right_depth})
        
        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            self.front_process.terminate()
            self.front_process.join()
        if self.use_right_cam:
            self.right_process.terminate()
            self.right_process.join()
        if hasattr(self, 'front_queue'):
            self.front_queue.close()
        if hasattr(self, 'right_queue'):
            self.right_queue.close()    


    def __del__(self):
        self.finalize()
        

if __name__ == "__main__":
    cam = MultiRealSense(use_right_cam=False, img_size=512)# 1024 stays within 15 ms; 512 stays within 3 ms.
    import matplotlib.pyplot as plt
    cam.start()
    time.sleep(1)
    color_array=[]
    for i in range(100):
        o_time=time.time()
        out = cam()
        print("deque_time:", time.time()-o_time)
        time_1=time.time()
        color_array.append(out['front_color'])
        time.sleep(1/30-(time.time()-o_time))# Keep frame rate below 30 Hz to avoid missing the latest frame.
    cam.finalize()
    start_time=time.time()    
    imageio.mimsave('color_front.gif', color_array, duration=1/30)
    print("write_time:", time.time()-start_time)
