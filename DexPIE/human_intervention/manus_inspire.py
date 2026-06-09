import numpy as np
from multiprocessing import Process, Event
import time
import socket
import os
import struct
from dexpie.common.LIFO_Queue import LIFOQueue
class ManusTele_intervention_Process(Process):
    """MANUS手套到Inspire灵巧手的人类干预子进程控制器"""
    
    def __init__(self, queue=None, serial_port="/dev/ttyUSB0", baudrate=115200, 
                 manus_port=8888, control_threshold=10, scale_factor=15):
        """
        初始化遥操作子进程控制器
        
        Args:
            serial_port: 灵巧手串口 (默认: /dev/ttyUSB0)
            baudrate: 串口波特率 (默认: 115200)
            manus_port: MANUS数据端口 (默认: 8888)
            control_threshold: 控制阈值 (默认: 10)
            scale_factor: 数据缩放因子 (默认: 15)
        """
        super().__init__()
        self.daemon = True  # 设为守护进程，主进程退出子进程自动退出
        self.stop_event = Event()
        self.human_intervention = False # 是否有人类干预,即由手套数据控制inspire hand，而不是推理得到的动作
        self.queue = queue

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.manus_port = manus_port
        self.control_threshold = control_threshold
        self.scale_factor = scale_factor
        
        # 初始化灵巧手
        self.right_hand = None
        self.left_hand_connect = False
        self.right_hand_connect = False
        
        # 数据缓存
        self.right_hand_data0 = None
        self.set_data = None
        
        # 网络连接
        self.server = None
        self.connection = None
        self.address = None
        
        print(f"🤖 MANUS遥操作子进程控制器初始化完成")
        print(f"📡 串口: {serial_port} @ {baudrate} bps")
        print(f"🌐 MANUS端口: {manus_port}")
    
    
    def initialize_manus_connection(self):
        """初始化MANUS网络连接"""
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
        """接收MANUS手套数据"""
        try:
            data = []
            for i in range(9):
                recv_bytes = self.connection.recv(8) #阻塞式 
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
        """处理手部校准"""
        if not self.right_hand_connect and data[0] == 1002 and data[1] != 0:
            self.right_hand_connect = True
            self.right_hand_data0 = np.array(data)  # 记录初始位置
            self.set_data = self.right_hand_data0 - self.right_hand_data0  # 初始化SET为0
            print("🎯 右手校准完成!")
            return True
        return False
    
    def calculate_finger_angles(self, data):
        """计算手指角度"""
        if not self.right_hand_connect or data[0] != 1002:
            return None
        # 计算相对变化
        diff = np.array(data) - self.set_data
        scaled_data = np.array(data) * self.scale_factor
        self.set_data = np.array(scaled_data) - self.right_hand_data0  # 更新SET为当前相对位置
        
        # 检查控制阈值
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
        normalized_action = np.array([a / 1000.0 for a in action])  # 归一化到0-1范围
        return action,normalized_action


    def run(self):
        """子进程主循环"""
        # 在子进程中初始化资源
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
                # 接收MANUS数据
                data = self.receive_manus_data()#阻塞接收数据，其有缓冲区不拿快点会拿到旧数据，表现为延迟（先进先出）
                if data is None:
                    raise ConnectionError("MANUS数据接收中断")
                    
                # 处理手部校准
                self.process_hand_calibration(data)
                
                # 计算手指角度
                angles = self.calculate_finger_angles(data)
                angles,normalized_action=self.angle_process(angles)
                hand_dict["action"]=angles
                hand_dict["normalized_action"]=normalized_action
                if normalized_action is not None:
                    self.queue.put(hand_dict)
                #通过回调获取最新的手部控制命令给手执行，应该延迟很小
                end_time = time.time()
                time.sleep(max(0, 1/60 - (end_time - time_start)))  # 60Hz读取频率，接收频率必须大于33hz不然缓冲区堆积会拿到旧数据

        except ConnectionError as e:
            print(f"❌ 连接错误: {e}")
        except Exception as e:
            print(f"❌ 遥操作异常: {e}")
        finally:
            self.cleanup()
    def finalize(self):
        self.stop_event.set()  # 设置主进程中断事件，通知主循环退出

    def terminate(self) -> None:
        return super().terminate()

    def cleanup(self):
        """清理资源"""
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
        #右手默认串口为/dev/ttyUSB0，左手为/dev/ttyUSB1
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
    
    # 启动子进程
    inspire_Manus_1.start()
    time.sleep(1)#等待初始化完成
    for _ in range(500):

        start_time = time.time()
        action_dict=inspire_Manus_1()
        if "right" in action_dict:
            print("Right Hand Action:", action_dict["right"])
        if "left" in action_dict:
            print("Left Hand Action:", action_dict["left"])
        time.sleep(max(0, 1/25 - (time.time() - start_time)))

    inspire_Manus_1.finalize()