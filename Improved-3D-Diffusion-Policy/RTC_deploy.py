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
import ray
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import MultiRealSense
from diffusion_policy_3d.common.tools import MATHTOOLS
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.hand_action_util import hand_action_util



from communication.UR_communication import UR_Comm
from communication.InspireHandControl_V1 import InspireHand


import numpy as np
import torch
from termcolor import cprint


@ray.remote(num_gpus=1)
class async_policy:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy.to('cuda')
        self.action_horizon = self.policy.horizon - self.policy.n_obs_steps + 1
        self.action_dim = self.policy.action_dim

    #必须传np格式的字典不然ray非常慢
    def inference(self, obs_dict):
        obs_dict = {
            key: (None if value is None else torch.from_numpy(value.copy()).to('cuda'))
            for key, value in obs_dict.items()
        }
        with torch.no_grad():
            action_dict = self.policy(obs_dict)[0]
        
        return action_dict.detach().cpu()

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

    if workspace.__class__.__name__ == 'RTC_DPWorkspace':
        use_image = True
        use_point_cloud = False
        if cfg.task.dataset.use_wrist_img:
            use_wrist_img = True
        else:
            use_wrist_img = False
    else:
        use_image = False
        use_point_cloud = True
    
    use_jit_model = False

    policy = workspace.get_model()
    RTC_policy = async_policy.remote(policy)
    # fetch policy model
    #policy = workspace.get_model()
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
    tools=MATHTOOLS()
    dt=1/25
    img_size = 256
    num_points = 4096
    first_init = True
    record_data = True
    camera = MultiRealSense(use_front_cam=True, use_right_cam=use_wrist_img, # by default we use single cam. but we also support multi-cam
                            front_num_points=num_points,
                            img_size=img_size)
    arm_comm = UR_Comm()
    print("arm_comm connected")
    hand_comm = InspireHand()
    print("hand_comm connected")
    time.sleep(1)
    hand_comm.reset()
    time.sleep(1)
    hand_comm.setpower(500, 500, 500, 500, 500)
    hand_comm.setspeed(300, 300, 300, 300, 300)
    max_latency_step = 3 
    
    color_array = []
    if use_wrist_img:
        wrist_color_array = []
    robot_state_array = []
    action_array = []
    
    #reset robot
    if first_init:
        # ======== INIT ==========
        arm_comm.reset_arm()
        hand_comm.reset()
        camera.start()
        time.sleep(0.5)
        print("Robot reset!")
    time.sleep(1)
    cam_dict = camera()
    obs_img = cam_dict['front_color']
    agent_state_dict = arm_comm.get_robot_state()
    qpos =agent_state_dict['joint_positions']
    np_qpos = np.stack([qpos], axis=0)[None, ...]
    np_obs_img = np.stack([obs_img], axis=0)[None, ...]
    agent_mat = agent_state_dict["mat"]
    policy_ref_mat = tools.xyz_rotvec_to_mat(agent_mat)
    obs_dict = {
        'agent_pos': np_qpos,
        "image": np_obs_img,
        "exc_action": None,
    }#shape B,T,C,H,W

    step_count = 0
    action=ray.get(RTC_policy.inference.remote(obs_dict))#阻塞
    action_horizon = action.shape[0]
    np_action=action.numpy()#shape T,15
    relative_arm_mat=tools.xyz_6drot_to_mat(np_action[:, :9])#shape T,4,4
    arm_mats=np.einsum("ij,tjk->tik", policy_ref_mat, relative_arm_mat)#shape T,4,4
    current_arm_mat = policy_ref_mat
    while step_count < roll_out_length:
        try:
            ref = None
            for i in range(action_horizon):
                if step_count >= roll_out_length:
                    break
                start_time = time.time()

                if i != 0: #初始的就不用获取观测，直接用提供给policy的观测
                    cam_dict = camera()
                    obs_img = cam_dict['front_color']
                    agent_state_dict = arm_comm.get_robot_state()
                    qpos =agent_state_dict['joint_positions']
                    np_qpos = np.stack([qpos], axis=0)[None, ...]
                    np_obs_img = np.stack([obs_img], axis=0)[None, ...]
                    agent_mat = agent_state_dict["mat"]
                    current_arm_mat = tools.xyz_rotvec_to_mat(agent_mat) #当前观测的位姿
                    
                if i == action_horizon-max_latency_step: #执行到只剩最大延迟步数时，用当前观测更新policy
                    obs_dict['agent_pos'] = np_qpos
                    obs_dict['image'] = np_obs_img
                    policy_ref_mat = current_arm_mat #更新策略基于的位姿 policy_ref_mat，后续需要使用
                    exc_arm_mats = np.einsum(
                        "ij,tjk->tik",
                        tools.se3_inverse(policy_ref_mat),
                        arm_mats[-max_latency_step:]
                    ) #最后max_latency_step个动作换成相对于policy_ref_mat的位姿
                    exc_arm_action = tools.mat2xyz_6drot(exc_arm_mats).astype(np.float32) # [T, 9]
                    exc_hand_action = np_action[-max_latency_step:, 9:].astype(np.float32) # [T, 6]
                    exc_action = np.concatenate([exc_arm_action, exc_hand_action], axis=-1)[None, ...] # [B, T, 15]
                    obs_dict['exc_action'] = exc_action
                    ref = RTC_policy.inference.remote(obs_dict)#异步推理

                arm_mat=arm_mats[i]
                arm_action=tools.mat2xyz_rotvec(arm_mat) #UR格式
                arm_comm.set_arm_action(arm_action)
                hand_action = hand_action_util(np_action[i,9:])
                hand_comm.setangle(*hand_action)

                if i == action_horizon-1 and ref is not None:
                    action=ray.get(ref)#在最后一步的执行中获取异步推理结果，不阻塞动作执行（发送了动作后再获取）
                    action_horizon = action.shape[0] #换成horizon-max_latency_step
                    np_action=action.numpy()#替换后继续推理
                    relative_arm_mat=tools.xyz_6drot_to_mat(np_action[:, :9])#shape T,4,4
                    arm_mats=np.einsum("ij,tjk->tik", policy_ref_mat, relative_arm_mat)#shape T,4,4

                step_count += 1
                time.sleep(max(0, dt - (time.time() - start_time)))

        except Exception as e:
            arm_comm.cleanup()
            hand_comm.close()
            camera.finalize()
            
    print("deploy done")
    arm_comm.cleanup()
    hand_comm.close()
    camera.finalize()


if __name__ == "__main__":
    main()
