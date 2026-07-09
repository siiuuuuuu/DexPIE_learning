"""Hardware interfaces used by RTC expert intervention collection."""

import time

import numpy as np
import rtde_control
import rtde_receive

from communication.InspireHandControl_V1 import InspireHand


class URArmInterface:
    """RTDE-backed UR arm observation and actuation interface."""

    def __init__(
        self,
        host,
        workspace_limits,
        servo_speed=0.005,
        servo_acceleration=0.005,
        servo_dt=0.02,
        lookahead_time=0.2,
        gain=500,
        control_frequency=None,
    ):
        self.host = host
        self.workspace_limits = workspace_limits
        self.servo_speed = servo_speed
        self.servo_acceleration = servo_acceleration
        self.servo_dt = servo_dt
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.control_frequency = (
            None if control_frequency is None else float(control_frequency)
        )
        if self.control_frequency is not None and self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        self.rtde_c = None
        self.rtde_r = None
        self.connect()

    def connect(self):
        try:
            if self.control_frequency is None:
                self.rtde_c = rtde_control.RTDEControlInterface(self.host)
            else:
                self.rtde_c = rtde_control.RTDEControlInterface(
                    self.host,
                    self.control_frequency,
                )
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
            print("RTDE connected")
        except Exception as e:
            print(f"RTDE connection failed: {e}")
            raise

    def is_ready(self):
        return (
            not self.rtde_r.isProtectiveStopped()
            and not self.rtde_r.isEmergencyStopped()
        )

    def get_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_obs(self):
        """Return the robot proprioceptive observation used for recording."""
        try:
            joint_positions = np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)
            tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float32)
            return {
                "joint": joint_positions,
                "tcp_pose": tcp_pose,
                "state": np.concatenate((joint_positions, tcp_pose)),
            }
        except Exception as e:
            print(f"Failed to get robot state: {e}")
            return None

    def move_l(self, pose, speed=0.3, acceleration=0.3):
        self.rtde_c.moveL(pose, speed, acceleration)

    def _is_pose_safe(self, pose):
        x, y, z = pose[0], pose[1], pose[2]
        if not (self.workspace_limits["x"][0] <= x <= self.workspace_limits["x"][1]):
            print(f"X-axis out of bounds: {x:.3f}")
            return False
        if not (self.workspace_limits["y"][0] <= y <= self.workspace_limits["y"][1]):
            print(f"Y-axis out of bounds: {y:.3f}")
            return False
        if not (self.workspace_limits["z"][0] <= z <= self.workspace_limits["z"][1]):
            print(f"Z-axis out of bounds: {z:.3f}")
            return False
        return True

    def _clip_pose(self, pose):
        clipped = np.asarray(pose, dtype=np.float64).copy()
        clipped[0] = max(
            self.workspace_limits["x"][0],
            min(self.workspace_limits["x"][1], clipped[0]),
        )
        clipped[1] = max(
            self.workspace_limits["y"][0],
            min(self.workspace_limits["y"][1], clipped[1]),
        )
        clipped[2] = max(
            self.workspace_limits["z"][0],
            min(self.workspace_limits["z"][1], clipped[2]),
        )
        return clipped

    def servo(self, target_pose):
        servo_pose = target_pose
        if not self._is_pose_safe(target_pose):
            servo_pose = self._clip_pose(target_pose)
            print("Target out of workspace! Clipping servo target.")

        self.rtde_c.servoL(
            servo_pose,
            self.servo_speed,
            self.servo_acceleration,
            self.servo_dt,
            self.lookahead_time,
            self.gain,
        )

    def stop_servo(self):
        if self.rtde_c and self.rtde_c.isConnected():
            self.rtde_c.servoStop()

    def close(self, stop_script=False):
        if self.rtde_c and self.rtde_c.isConnected():
            try:
                self.rtde_c.servoStop()
                if stop_script:
                    self.rtde_c.stopScript()
            except Exception as e:
                print(f"Error during stopping robot: {e}")
            finally:
                self.rtde_c.disconnect()

        if self.rtde_r and self.rtde_r.isConnected():
            self.rtde_r.disconnect()


class InspireHandController:
    """Owns Inspire hand hardware communication in the main process."""

    def __init__(self, serial_port="/dev/ttyUSB0", baudrate=115200):
        print("Connecting Inspire hand...")
        self.hand = InspireHand(serial_port, baudrate)
        time.sleep(1)
        self.hand.reset()
        time.sleep(1)
        self.hand.setpower(500, 500, 500, 500, 500)
        self.hand.setspeed(300, 300, 300, 300, 300)
        print(f"Inspire hand connected: {self.hand.num2str(0x70)}")

    def apply(self, raw_action):
        command = np.clip(raw_action, 0, 1000).astype(np.int32)
        self.hand.setangle(*command.tolist())
        return command

    def reset(self):
        if self.hand is None:
            return
        try:
            self.hand.reset()
            print("Inspire hand reset")
        except Exception as e:
            print(f"Inspire hand reset failed: {e}")

    def close(self):
        if self.hand is None:
            return
        try:
            self.reset()
            self.hand.close()
            print("Inspire hand reset and closed")
        except Exception as e:
            print(f"Inspire hand cleanup failed: {e}")
        finally:
            self.hand = None
