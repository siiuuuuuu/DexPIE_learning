"""
This helper covers most Inspire hand operations, including finger angle
commands and real-time force monitoring.

Usage example:
Import and use this class from another file:
from InspireHandContro_V1 import InspireHand

# Specify the serial port.
hand = InspireHand("COM4", 115200)  # Automatically connects to the port.

# Or disable automatic connection:
# hand = InspireHand("COM4", 115200, auto_connect=False)
# hand.connect()  # Connect manually later.

hand.reset()  # Reset to the initial pose.
import time
time.sleep(2)
hand.setangle(1000, 1000, 1000, 460, 560, 50)  # Set new angles.
time.sleep(5)
hand.reset()  # Reset again.
hand.close()  # Close the serial connection.
"""

import serial
import time


class InspireHand:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, auto_connect=True):
        """
        Initialize the InspireHand controller.
        :param port: Serial port name. Must be specified.
        :param baudrate: Serial baud rate. Must be specified.
        :param auto_connect: Whether to connect during initialization.
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.hand_id = 1  # Hand ID.

        if auto_connect:
            self.connect()

    def connect(self, port=None, baudrate=None):
        """
        Connect to the configured serial port.
        :param port: Serial port name. Uses the initial port when None.
        :param baudrate: Baud rate. Uses the initial baud rate when None.
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

            # Set default parameters.
            self.setpower(200, 200, 200, 200, 200, 200)  # Default force.
            self.setspeed(1000, 1000, 1000, 1000, 1000, 1000)  # Default speed.
        except serial.SerialException as e:
            print(f"无法打开串口 {self.port}: {e}")
            raise

    def reset(self):
        """Reset fingers to the initial pose."""
        self.setangle(1000, 1000, 1000, 1000, 1000, 1000)

    @staticmethod
    def data2bytes(data):
        """
        Convert an integer to two bytes.
        :param data: Input integer.
        :return: Two-byte list.
        """
        return [data & 0xFF, (data >> 8) & 0xFF] if data != -1 else [0xFF, 0xFF]

    @staticmethod
    def num2str(num):
        """
        Convert a number to a hexadecimal byte string.
        :param num: Input number.
        :return: Hexadecimal byte string.
        """
        return bytes.fromhex(f"{num:02x}")

    @staticmethod
    def checknum(data, leng):
        """
        Compute checksum.
        :param data: Data list.
        :param leng: Length.
        :return: Checksum.
        """
        return sum(data[2:leng]) & 0xFF

    def close(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("运动结束，串口已关闭")
        else:
            print("串口未打开或已关闭")

    def send_command(self, address, data):
        """
        Send a command to the robotic hand.
        :param address: Command address.
        :param data: Command payload for six fingers.
        """
        if self.ser is None or not self.ser.is_open:
            raise ConnectionError("串口未打开，请先调用connect方法")

        if any(not 0 <= d <= 1000 for d in data):
            print("数据超出正确范围：0-1000")
            return

        datanum = 0x0F
        # Build command packet.
        b = [0xEB, 0x90, self.hand_id, datanum, 0x12, address & 0xFF, address >> 8]
        for d in data:
            b.extend(self.data2bytes(d))
        b.append(self.checknum(b, datanum + 4))

        # Convert the command to bytes and send it.
        putdata = b"".join(map(self.num2str, b))
        self.ser.write(putdata)

    def setpower(self, *powers):
        """
        Set force thresholds.
        :param powers: Six finger force values (0-1000).
        """
        self.send_command(0x05DA, powers)

    def setspeed(self, *speeds):
        """
        Set finger speeds.
        :param speeds: Six finger speed values (0-1000).
        """
        self.send_command(0x05F2, speeds)

    def setangle(self, *angles):
        """
        Set finger angles.
        :param angles: Six finger angle values (-1 to 1000).
        """
        if any(not -1 <= a <= 1000 for a in angles):
            print("数据超出正确范围：-1-1000")
            return
        self.send_command(0x05CE, angles)

    def gesture_force_clb(self):
        """Calibrate force sensors."""
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("串口未打开")

        # Send calibration command.
        cmd = [0xEB, 0x90, self.hand_id, 0x04, 0x12, 0xF1, 0x03, 0x01]
        cmd.append(sum(cmd[2:]) & 0xFF)
        self.ser.write(bytes(cmd))

        print("力传感器校准进行中...")
        time.sleep(30)
        print("力传感器校准完成！")
        return True

    def get_actforce(self):
        """
        Read actual force values for all six fingers.
        Contains three nested helpers:
        {
        create_command(): Build the force-read command.
        read_frame(): Read and validate one data frame.
        parse_force_values(): Parse force values.
        }
        :return: List of six force values, or None on failure.
        """
        MAX_RETRIES = 3
        FRAME_SIZE = 20
        FORCE_COUNT = 6

        def create_command():
            """Build the force-read command."""
            datanum = 0x04
            command = [
                0xEB, 0x90,         # Packet header.
                self.hand_id,       # hand_id
                datanum,            # Data count.
                0x11,               # Read operation.
                0x2E, 0x06,         # Address.
                0x0C                # Read length.
            ]
            command.append(self.checknum(command, datanum + 4))  # Checksum.
            return b''.join(map(self.num2str, command))

        def read_frame():
            """Read and validate a data frame."""
            buffer = bytearray()
            timeout_count = 0
            MAX_TIMEOUT = 100

            # Read a complete data frame.
            while len(buffer) < FRAME_SIZE and timeout_count < MAX_TIMEOUT:
                # Find the frame header.
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

                # Read remaining bytes.
                if len(buffer) >= 2:
                    remaining_data = self.ser.read(FRAME_SIZE - len(buffer))
                    if remaining_data:
                        buffer.extend(remaining_data)

            return buffer if len(buffer) == FRAME_SIZE else None

        def parse_force_values(buffer):
            """Parse force values."""
            force_values = []
            last_valid = [0] * FORCE_COUNT

            for i in range(FORCE_COUNT):
                base_index = 7 + i * 2
                if base_index + 1 >= len(buffer):
                    return None

                # Combine bytes into a signed 16-bit integer.
                value = (buffer[base_index + 1] << 8) | buffer[base_index]
                if value & 0x8000:  # Handle negative values.
                    value -= 65536

                # Check outliers.
                if abs(value) > 5000:
                    print(f"警告：第{i+1}个力度值异常 ({value})，使用上一次的值")
                    value = last_valid[i]
                else:
                    last_valid[i] = value

                force_values.append(value)

            return force_values

        # Main retry loop.
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



