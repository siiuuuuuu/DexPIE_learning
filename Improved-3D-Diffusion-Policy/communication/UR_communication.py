import rtde_control
import rtde_receive


class UR_Comm:
    def __init__(self, ip="192.168.3.6", workspace_limits={
        'x': [-1, 1],
        'y': [-1, 1], 
        'z': [0, 1]},
        servo_speed=0.005,
        servo_acceleration=0.005,
        dt=1.0/25,
        lookahead_time=0.2,
        gain=500):

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        except Exception as e:
            print(f"RTDE connection failed: {e}")
            self.rtde_c = None
            self.rtde_r = None

        self.workspace_limits = workspace_limits
        self.servo_speed = servo_speed          # 机器人最大TCP速度 (m/s)
        self.servo_acceleration = servo_acceleration   # 机器人最大TCP加速度 (m/s^2)
        self.dt = dt               # 控制频率 (25Hz)
        self.servo_dt=self.dt/2           # servoL内部控制频率 (即一次循环两次target_pose更新)
        self.lookahead_time = lookahead_time      # 平滑时间 (0.03-0.2之间)
        self.gain = gain                # 比例增益 (100-2000之间)

        self.initial_pose =[0.248,0.1212,0.3978,1.16,1.25,1.28] #210

    def cleanup(self):
        """Cleanup RTDE connections"""
        rtde_c=getattr(self, 'rtde_c', None)
        rtde_r=getattr(self, 'rtde_r', None)
        if rtde_c and self.rtde_c.isConnected():
            try:
                rtde_c.servoStop()
                #rtde_c.stopScript()
            except Exception as e:
                print(f"Failed to stop servo: {e}")
            finally:
                rtde_c.disconnect()

        if rtde_r and self.rtde_r.isConnected():
            rtde_r.disconnect()

    def check_robot_safety(self):
        """Check robot safety status"""
        try:

            safety_status = self.rtde_r.getSafetyStatus()
            robot_mode = self.rtde_r.getRobotMode()
            
            # 检查是否在安全状态
            if safety_status not in [1, 2]:
                print(f"Robot safety status error: {safety_status}")
                return False
            
            # 检查机器人模式
            if robot_mode != 7:  # 7=running mode
                print(f"Robot mode error, not in RUNNING mode: {robot_mode}")
                return False
                
            return True
        except Exception as e:
            print(f"Robot status check failed: {e}")
            self.cleanup()
            return False


    def is_pose_safe(self, pose, workspace_limits):
        """Check if target position is within safe workspace"""
        x, y, z = pose[0], pose[1], pose[2]
        
        if not (workspace_limits['x'][0] <= x <= workspace_limits['x'][1]):
            print(f"X-axis out of bounds: {x:.3f}")
            return False
        if not (workspace_limits['y'][0] <= y <= workspace_limits['y'][1]):
            print(f"Y-axis out of bounds: {y:.3f}")
            return False
        if not (workspace_limits['z'][0] <= z <= workspace_limits['z'][1]):
            print(f"Z-axis out of bounds: {z:.3f}")
            return False
        
        return True
    
    def get_robot_state(self):
        """Retrieve current robot state"""
        rob_dict={}
        try:
            joint_positions = self.rtde_r.getActualQ()
            mat=self.rtde_r.getActualTCPPose()
            rob_dict['joint_positions']=joint_positions
            rob_dict['mat']=mat
            return rob_dict
        except Exception as e:
            print(f"Failed to get robot state: {e}")
            self.cleanup()
            return None
        
    def get_robot_action(self):
        try:
            pose= self.rtde_r.getTargetTCPPose()
            joint_positions = self.rtde_r.getTargetQ()
            return pose, joint_positions
        except Exception as e:
            print(f"Failed to get robot action: {e}")
            self.cleanup()
            return None, None
        
    def set_arm_action(self, action):
        """Set robot target pose,参考位姿为基座位姿"""
        try:
            self.rtde_c.servoL(action, self.servo_speed, self.servo_acceleration, self.servo_dt, self.lookahead_time, self.gain)

        except Exception as e:
            print(f"Failed to set robot pose: {e}")
            self.cleanup()

    def reset_arm(self):
        """Reset robot to home position"""
        self.rtde_c.moveL(self.initial_pose, 0.2, 0.2)#阻塞式，等待到达初始位姿
