import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import MultiRealSense
from diffusion_policy_3d.common.tools import MATHTOOLS

from communication.UR_communication import UR_Comm
from communication.InspireHandControl_V1 import InspireHand


import numpy as np
import torch
from termcolor import cprint


class UR_Inspire_EnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=1, action_horizon=24, device="gpu",
                use_point_cloud=False, use_image=True, img_size=256,
                 num_points=4096,
                 use_waist=False):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.use_waist = use_waist
        self.dt=1/25 #与训练时一致
        self.tools=MATHTOOLS()

        # camera
        self.camera = MultiRealSense(use_front_cam=True, use_right_cam=False, # by default we use single cam. but we also support multi-cam
                            front_num_points=num_points,
                            img_size=img_size)
        print("camera init")

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # robot comm
        self.arm_comm = UR_Comm()
        print("arm_comm connected")
        self.hand_comm = InspireHand()
        print("hand_comm connected")

        time.sleep(1)
        self.hand_comm.reset()
        time.sleep(1)
        self.hand_comm.setpower(500, 500, 500, 500, 500)
        self.hand_comm.setspeed(300, 300, 300, 300, 300)

    
    
    def step(self, action_list: np.ndarray):

        current_arm_mat=self.tools.xyz_rotvec_to_mat(self.arm_comm.get_robot_state()['mat'])#当前TCP位姿
        print("current_arm_mat:",self.arm_comm.get_robot_state()['mat'])
        for action_id in range(self.action_horizon):
            start_time=time.time()

            act = action_list[action_id]
            relative_arm_mat=self.tools.xyz_6drot_to_mat(np.expand_dims(act[:9],axis=0))[0]
            arm_mat=np.dot(current_arm_mat,relative_arm_mat)#叠加相对位姿

            arm_action=self.tools.mat2xyz_rotvec(arm_mat)#UR格式
            hand_action=act[9:]*1000 #反归一化到0-1000

            hand_action=np.clip(hand_action, 0, 1000)

            pinky_angle = int(hand_action[0])
            ring_angle = int(hand_action[1])
            middle_angle = int(hand_action[2])
            index_angle = int(hand_action[3])
            thumb_angle_2 = int(hand_action[4])
            thumb_angle = int(hand_action[5])
            
            self.arm_comm.set_arm_action(arm_action)#执行动作，非阻塞
            self.hand_comm.setangle(pinky_angle, 
                                    ring_angle, 
                                    middle_angle, 
                                    index_angle, 
                                    thumb_angle_2, 
                                    thumb_angle)#执行动作，非阻塞
            
            if action_id != self.action_horizon-1:
                time.sleep(max(0, self.dt-(time.time()-start_time)))#保持25Hz频率

        #同步策略，执行完再获取观察
        cam_dict = self.camera()#回调获取最新一帧
        self.color_array.append(cam_dict['front_color'])
        env_qpos =self.arm_comm.get_robot_state()['joint_positions']
        self.env_qpos_array.append(env_qpos)
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)

        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict
    
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []

        if first_init:
            # ======== INIT ==========
            self.arm_comm.reset_arm()
            self.hand_comm.reset()
            self.camera.start()
        time.sleep(1)
        
        print("Robot ready!")
        
        # ======== INIT ==========
        cam_dict = self.camera()#回调获取最新一帧
        self.color_array.append(cam_dict['front_color'])

        env_qpos =self.arm_comm.get_robot_state()['joint_positions']
        self.env_qpos_array.append(env_qpos)
                    

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
            
        return obs_dict#获取起始观察
    
    def close(self):

        self.arm_comm.cleanup()
        self.hand_comm.close()
        self.camera.finalize()


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 3000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 256
    num_points = 4096
    first_init = True
    record_data = True

    env = UR_Inspire_EnvInference(obs_horizon=1,action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points,
                             )

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    try:
        while step_count < roll_out_length:
            with torch.no_grad():
                action = policy(obs_dict)[0]
                action_list = [act.numpy() for act in action]
            
            obs_dict = env.step(action_list)
            step_count += action_horizon
            print(f"step: {step_count}")
            
    except Exception as e:
        env.close()

    env.close()

    if record_data:
        import h5py
        root_dir = "/home/lrz/dp_data/rollout_data"
        save_dir = root_dir + "deploy_dir"
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.qpos_array)
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=np.array(color_array))
            f.create_dataset("depth", data=np.array(depth_array))
            f.create_dataset("cloud", data=np.array(cloud_array))
            f.create_dataset("qpos", data=np.array(qpos_array))
        
        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
            new_name = record_file_name.replace("demo.h5", renamed+'.h5')
            cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
        else:
            cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
