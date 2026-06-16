import numpy as np
from multiprocessing import Process, Event
import time
import socket
import os
import struct
from dexpie.common.LIFO_Queue import LIFOQueue
class ManusTele_intervention_Process(Process):
    """Human-intervention subprocess from MANUS glove to Inspire hand."""
    
    def __init__(self, queue=None, serial_port="/dev/ttyUSB0", baudrate=115200, 
                 manus_port=8888, control_threshold=10, scale_factor=15):
        """
        Initialize the teleoperation subprocess controller.
        
        Args:
            serial_port: Inspire hand serial port (default: /dev/ttyUSB0).
            baudrate: Serial baud rate (default: 115200).
            manus_port: MANUS data port (default: 8888).
            control_threshold: Control threshold (default: 10).
            scale_factor: Data scale factor (default: 15).
        """
        super().__init__()
        self.daemon = True  # Daemon process exits automatically with the parent process.
        self.stop_event = Event()#cwr is cs
        self.human_intervention = False # True when glove data controls the Inspire hand instead of policy actions.
        self.queue = queue

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.manus_port = manus_port
        self.control_threshold = control_threshold
        self.scale_factor = scale_factor
        
        # Initialize Inspire hand state.
        self.right_hand = None
        self.left_hand_connect = False
        self.right_hand_connect = False
        
        # Data cache.
        self.right_hand_data0 = None
        self.set_data = None
        
        # Network connection.
        self.server = None
        self.connection = None
        self.address = None
        
        print(f"🤖 MANUS遥操作子进程控制器初始化完成")
        print(f"📡 串口: {serial_port} @ {baudrate} bps")
        print(f"🌐 MANUS端口: {manus_port}")
    
    
    def initialize_manus_connection(self):
        """Initialize MANUS network connection."""
        try:
            print("🔄 正在建立MANUS连接...")
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind(("localhost", self.manus_port))
            print("✅ MANUS服务器绑定成功")
            
            self.server.listen(0)
            print("👂 正在监听MANUS数据...")
            
            self.connection, self.address = self.server.accept()
            print(f"🔗 MANUS客户端连接成功: {self.address}")
            
        except Exception as e:
            print(f"❌ MANUS连接失败: {e}")
            raise
    
    def receive_manus_data(self):
        """Receive MANUS glove data."""
        try:
            data = []
            for i in range(9):
                recv_bytes = self.connection.recv(8) # Blocking receive.
                if not recv_bytes:
                    return None
                float_str = recv_bytes.decode('ascii')
                value = float(float_str)
                data.append(value)
            return data
        except Exception as e:
            print(f"❌ 数据接收失败: {e}")
            return None
    
    def process_hand_calibration(self, data):
        """Handle hand calibration."""
        if not self.right_hand_connect and data[0] == 1002 and data[1] != 0:
            self.right_hand_connect = True
            self.right_hand_data0 = np.array(data)  # Record the initial pose.
            self.set_data = self.right_hand_data0 - self.right_hand_data0  # Initialize SET to zero.
            print("🎯 右手校准完成!")
            return True
        return False
    
    def calculate_finger_angles(self, data):
        """Compute finger angles."""
        if not self.right_hand_connect or data[0] != 1002:
            return None
        # Compute relative change.
        diff = np.array(data) - self.set_data
        scaled_data = np.array(data) * self.scale_factor
        self.set_data = np.array(scaled_data) - self.right_hand_data0  # Update SET to the current relative pose.
        
        # Check control threshold.
        if np.max(np.abs(diff)) < self.control_threshold:
            return None
        
        return self.set_data
    
    def angle_process(self,angles):
        if angles is None:
            return None,None
        pinky_angle = int(max(0, min(1000, 1000 - angles[8])))#Stretch
        ring_angle = int(max(0, min(1000, 1000 - angles[7])))#Stretch
        middle_angle = int(max(0, min(1000, 1000 - angles[6])))#Stretch
        index_angle = int(max(0, min(1000, 1000 - angles[4])))#Stretch
        thumb_angle_2 = int(min(1000, max(0, 0 + 1.5 * angles[2])))#Spread
        thumb_angle = int(min(1000, max(0, 1000 - 1.7 * angles[1])))#Stretch

        action=[pinky_angle,ring_angle,middle_angle,index_angle,thumb_angle_2,thumb_angle]
        normalized_action = np.array([a / 1000.0 for a in action])  # Normalize to the 0-1 range.
        return action,normalized_action


    def run(self):
        """Subprocess main loop."""
        # Initialize resources inside the subprocess.
        try:
            print("🔧 子进程准备遥操作系统...")
            self.initialize_manus_connection()
        except Exception as e:
            print(f"❌ 子进程准备遥操作系统失败: {e}")
            return
        
        try:
            while not self.stop_event.is_set():
                hand_dict=dict()
                time_start = time.time()
                # Receive MANUS data.
                data = self.receive_manus_data()# Blocking receive; consume quickly to avoid buffered stale data and FIFO latency.
                if data is None:
                    raise ConnectionError("MANUS数据接收中断")
                    
                # Handle hand calibration.
                self.process_hand_calibration(data)
                
                # Compute finger angles.
                angles = self.calculate_finger_angles(data)
                angles,normalized_action=self.angle_process(angles)
                hand_dict["action"]=angles
                hand_dict["normalized_action"]=normalized_action
                if normalized_action is not None:
                    self.queue.put(hand_dict)
                # Publish the latest hand control command for low-latency execution.
                end_time = time.time()
                time.sleep(max(0, 1/60 - (end_time - time_start)))  # Read at 60 Hz; stay above 33 Hz to avoid stale buffered data.

        except ConnectionError as e:
            print(f"❌ 连接错误: {e}")
        except Exception as e:
            print(f"❌ 遥操作异常: {e}")
        finally:
            self.cleanup()
    def finalize(self):
        self.stop_event.set()  # Signal the main loop to exit.

    def terminate(self) -> None:
        return super().terminate()

    def cleanup(self):
        """Clean up resources."""
        print("🧹 子进程正在清理资源...")
        
        try:
            if self.connection:
                self.connection.close()
                print("🔗 网络连接已关闭")
            
            if self.server:
                self.server.close()
                print("🌐 服务器已关闭")
            
            if self.right_hand:
                self.right_hand.reset()
                self.right_hand.close()
                print("🤖 灵巧手已重置并关闭")
                
        except Exception as e:
            print(f"⚠️  清理过程中出错: {e}")
        
        print("✅ 子进程清理完成")

class inspire_Manus(object):
    def __init__(self, use_right_hand=True, use_left_hand=False, baudrate=115200, 
                 manus_port=8888, control_threshold=10, scale_factor=15):
        
        self.right_queue=LIFOQueue(maxsize=5)
        self.left_queue=LIFOQueue(maxsize=5)
        # Default serial ports: right hand on /dev/ttyUSB0, left hand on /dev/ttyUSB1.
        self.right_serial_port="/dev/ttyUSB0"
        self.left_serial_port="/dev/ttyUSB1"

        if use_right_hand:
            self.righthand_process = ManusTele_intervention_Process(
                queue=self.right_queue,
                serial_port=self.right_serial_port,
                baudrate=baudrate,
                manus_port=manus_port,
                control_threshold=control_threshold,
                scale_factor=scale_factor
            )

        if use_left_hand:
            self.lefthand_process = ManusTele_intervention_Process(
                queue=self.left_queue,
                serial_port=self.left_serial_port,
                baudrate=baudrate,
                manus_port=manus_port,
                control_threshold=control_threshold,
                scale_factor=scale_factor
            )
        self.use_right_hand=use_right_hand
        self.use_left_hand=use_left_hand

    def start(self):
        if self.use_right_hand:
            self.righthand_process.start()
        if self.use_left_hand:
            self.lefthand_process.start()

    def __call__(self):
        action_dict={}
        if self.use_right_hand:
            right_action=self.right_queue.get()
            action_dict["right"]=right_action
        if self.use_left_hand:
            left_action=self.left_queue.get()
            action_dict["left"]=left_action
        return action_dict
    
    def finalize(self):
        if self.use_right_hand:
            self.righthand_process.finalize()
            self.righthand_process.join()
            self.right_queue.close()
        if self.use_left_hand:
            self.lefthand_process.finalize()
            self.lefthand_process.join()
            self.left_queue.close()

    def __del__(self):
        self.finalize()

if __name__ == '__main__':
    inspire_Manus_1=inspire_Manus(use_right_hand=True, use_left_hand=False)
    
    # Start subprocess.
    inspire_Manus_1.start()
    time.sleep(1)# Wait for initialization.
    for _ in range(500):

        start_time = time.time()
        action_dict=inspire_Manus_1()
        if "right" in action_dict:
            print("Right Hand Action:", action_dict["right"])
        if "left" in action_dict:
            print("Left Hand Action:", action_dict["left"])
        time.sleep(max(0, 1/25 - (time.time() - start_time)))

    inspire_Manus_1.finalize()
