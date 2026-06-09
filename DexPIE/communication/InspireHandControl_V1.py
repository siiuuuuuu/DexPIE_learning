"""
这个功能函数包含了手指的绝大部分功能，能够实现手指角度设置，手指力度的实时监测

使用示例
在外部文件中使用这个类：
from InspireHandContro_V1 import InspireHand

# 指定串口
hand = InspireHand("COM4", 115200)  # 这将自动连接到指定串口

# 或者，如果你不想自动连接：
# hand = InspireHand("COM4", 115200, auto_connect=False)
# hand.connect()  # 稍后手动连接

hand.reset()  # 重置到初始位置
import time
time.sleep(2)
hand.setangle(1000, 1000, 1000, 460, 560, 50)  # 设置新的角度
time.sleep(5)
hand.reset()  # 再次重置
hand.close()  # 关闭串口连接
"""

import serial
import time


class InspireHand:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, auto_connect=True):
        """
        初始化InspireHandR类
        :param port: 串口名称，必须指定
        :param baudrate: 波特率，必须指定
        :param auto_connect: 是否在初始化时自动连接，默认为True
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.hand_id = 1  # 手部ID

        if auto_connect:
            self.connect()

    def connect(self, port=None, baudrate=None):
        """
        连接到指定的串口
        :param port: 串口名称，如果为None则使用初始化时指定的端口
        :param baudrate: 波特率，如果为None则使用初始化时指定的波特率
        """
        if port is not None:
            self.port = port
        if baudrate is not None:
            self.baudrate = baudrate

        if not self.port:
            raise ValueError("串口未指定，请提供有效的串口名称")

        if self.ser and self.ser.is_open:
            self.ser.close()

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"已连接到串口 {self.port}, 波特率 {self.baudrate}")

            # 设置默认参数
            self.setpower(200, 200, 200, 200, 200, 200)  # 设置默认力度
            self.setspeed(1000, 1000, 1000, 1000, 1000, 1000)  # 设置默认速度
        except serial.SerialException as e:
            print(f"无法打开串口 {self.port}: {e}")
            raise

    def reset(self):
        """重置手指到初始位置"""
        self.setangle(1000, 1000, 1000, 1000, 1000, 1000)

    @staticmethod
    def data2bytes(data):
        """
        将数据转换为两个字节
        :param data: 输入数据
        :return: 两个字节的列表
        """
        return [data & 0xFF, (data >> 8) & 0xFF] if data != -1 else [0xFF, 0xFF]

    @staticmethod
    def num2str(num):
        """
        将数字转换为十六进制字符串
        :param num: 输入数字
        :return: 十六进制字符串的字节形式
        """
        return bytes.fromhex(f"{num:02x}")

    @staticmethod
    def checknum(data, leng):
        """
        计算校验和
        :param data: 数据列表
        :param leng: 长度
        :return: 校验和
        """
        return sum(data[2:leng]) & 0xFF

    def close(self):
        """关闭串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("运动结束，串口已关闭")
        else:
            print("串口未打开或已关闭")

    def send_command(self, address, data):
        """
        发送命令到机械手
        :param address: 命令地址
        :param data: 命令数据（六个手指的参数）
        """
        if self.ser is None or not self.ser.is_open:
            raise ConnectionError("串口未打开，请先调用connect方法")

        if any(not 0 <= d <= 1000 for d in data):
            print("数据超出正确范围：0-1000")
            return

        datanum = 0x0F
        # 构建命令包
        b = [0xEB, 0x90, self.hand_id, datanum, 0x12, address & 0xFF, address >> 8]
        for d in data:
            b.extend(self.data2bytes(d))
        b.append(self.checknum(b, datanum + 4))

        # 将命令转换为字节串并发送
        putdata = b"".join(map(self.num2str, b))
        self.ser.write(putdata)

    def setpower(self, *powers):
        """
        设置力度阈值
        :param powers: 六个手指的力度值（0-1000）
        """
        self.send_command(0x05DA, powers)

    def setspeed(self, *speeds):
        """
        设置速度
        :param speeds: 六个手指的速度值（0-1000）
        """
        self.send_command(0x05F2, speeds)

    def setangle(self, *angles):
        """
        设置角度
        :param angles: 六个手指的角度值（-1到1000）
        """
        if any(not -1 <= a <= 1000 for a in angles):
            print("数据超出正确范围：-1-1000")
            return
        self.send_command(0x05CE, angles)

    def gesture_force_clb(self):
        """力传感器校准"""
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("串口未打开")

        # 发送校准命令
        cmd = [0xEB, 0x90, self.hand_id, 0x04, 0x12, 0xF1, 0x03, 0x01]
        cmd.append(sum(cmd[2:]) & 0xFF)
        self.ser.write(bytes(cmd))

        print("力传感器校准进行中...")
        time.sleep(30)
        print("力传感器校准完成！")
        return True

    def get_actforce(self):
        """
        获取六个手指的实际力度值
        包含三个内部函数：
        {
        create_command(): 创建读取力度的命令
        read_frame(): 读取和验证数据帧
        parse_force_values(): 解析力度值
        }
        :return: 包含6个力度值的列表，如果获取失败则返回None
        """
        MAX_RETRIES = 3
        FRAME_SIZE = 20
        FORCE_COUNT = 6

        def create_command():
            """创建读取力度的命令"""
            datanum = 0x04
            command = [
                0xEB, 0x90,         # 包头
                self.hand_id,       # hand_id
                datanum,            # 数据个数
                0x11,               # 读操作
                0x2E, 0x06,         # 地址
                0x0C                # 读取长度
            ]
            command.append(self.checknum(command, datanum + 4))  # 校验和
            return b''.join(map(self.num2str, command))

        def read_frame():
            """读取并验证数据帧"""
            buffer = bytearray()
            timeout_count = 0
            MAX_TIMEOUT = 100

            # 读取完整数据帧
            while len(buffer) < FRAME_SIZE and timeout_count < MAX_TIMEOUT:
                # 查找帧头
                while len(buffer) < 2:
                    byte = self.ser.read(1)
                    if not byte:
                        timeout_count += 1
                        if timeout_count >= MAX_TIMEOUT:
                            return None
                        continue
                    
                    buffer.extend(byte)
                    if len(buffer) == 2 and (buffer[0] != 0x90 or buffer[1] != 0xEB):
                        buffer = buffer[1:]

                # 读取剩余数据
                if len(buffer) >= 2:
                    remaining_data = self.ser.read(FRAME_SIZE - len(buffer))
                    if remaining_data:
                        buffer.extend(remaining_data)

            return buffer if len(buffer) == FRAME_SIZE else None

        def parse_force_values(buffer):
            """解析力度值"""
            force_values = []
            last_valid = [0] * FORCE_COUNT

            for i in range(FORCE_COUNT):
                base_index = 7 + i * 2
                if base_index + 1 >= len(buffer):
                    return None

                # 组合字节为16位有符号整数
                value = (buffer[base_index + 1] << 8) | buffer[base_index]
                if value & 0x8000:  # 处理负数
                    value -= 65536

                # 异常值检查
                if abs(value) > 5000:
                    print(f"警告：第{i+1}个力度值异常 ({value})，使用上一次的值")
                    value = last_valid[i]
                else:
                    last_valid[i] = value

                force_values.append(value)

            return force_values

        # 主执行逻辑
        for retry in range(MAX_RETRIES):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(create_command())

                frame = read_frame()
                if not frame:
                    print(f"第{retry + 1}次尝试：读取数据帧失败")
                    continue

                if frame[0] != 0x90 or frame[1] != 0xEB:
                    print(f"第{retry + 1}次尝试：无效的帧头")
                    continue

                force_values = parse_force_values(frame)
                if force_values:
                    if __debug__:
                        print("实际力度值：", force_values)
                    return force_values

            except Exception as e:
                print(f"第{retry + 1}次尝试失败: {str(e)}")
            
            if retry < MAX_RETRIES - 1:
                print("正在重试...")
                time.sleep(0.1)
                
        print("达到最大重试次数，放弃")
        return None



