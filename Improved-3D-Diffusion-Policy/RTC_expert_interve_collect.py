import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import time
import pathlib
import threading
from datetime import datetime

import h5py
import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from pynput import keyboard
from termcolor import cprint

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.common.multi_realsense import MultiRealSense
from diffusion_policy_3d.common.tools import MATHTOOLS
from diffusion_policy_3d.common.hand_action_util import hand_action_util
from diffusion_policy_3d.common.keyboard_handler_util import create_on_press_handler

from communication.UR_communication import UR_Comm
from communication.InspireHandControl_V1 import InspireHand
from human_intervention.expert_intervention import ExpertIntervention

os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@ray.remote(num_gpus=1)
class async_policy:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy.to('cuda')
        self.action_horizon = self.policy.horizon - self.policy.n_obs_steps + 1
        self.action_dim = self.policy.action_dim

    # 必须传 np 格式字典，不然 ray 非常慢
    def inference(self, obs_dict):
        obs_dict = {
            key: (None if value is None else torch.from_numpy(value.copy()).to('cuda'))
            for key, value in obs_dict.items()
        }
        with torch.no_grad():
            action = self.policy(obs_dict)[0]
        return action.detach().cpu()


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ != 'RTC_DPWorkspace':
        cprint(
            f"[Warning] 当前 workspace={workspace.__class__.__name__}，建议使用 RTC_DPWorkspace 配置。",
            "yellow"
        )
    #cprint(f" 当前cfg强度={cfg.policy.positive_cfg_scale}")
    use_image = True
    use_point_cloud = False
    use_wrist_img = bool(cfg.task.dataset.use_wrist_img)

    tools = MATHTOOLS()

    # 初始化事件对象
    start_recording = threading.Event()
    human_intervention = threading.Event()
    complete_collect = threading.Event()
    flag = [0]  # 用列表包裹整数，使其可变（避免使用 global）
    flag_lock = threading.Lock()

    on_press_handler = create_on_press_handler(
        start_recording, human_intervention, complete_collect, flag_lock, flag
    )

    # 初始化专家进程
    expert = ExpertIntervention(use_right_hand=True, use_left_hand=False)
    expert.start()
    time.sleep(0.1)
    print("开启专家干预进程")

    listener = keyboard.Listener(on_press=on_press_handler)
    print("按下s键开始录制，再次按下s键结束录制，按下e键开启人工干预，再次按下e键结束干预，按下c键完成采集")
    listener.start()  # 开启监听线程

    time.sleep(0.1)

    policy = workspace.get_model()
    RTC_policy = async_policy.remote(policy)
    max_latency_step = int(getattr(policy, 'max_latency_steps', 3))

    data_dir = os.path.expanduser("~/dp_data/offlineRL_data/new_task1_iter1")
    os.makedirs(data_dir, exist_ok=True)

    trackertoTCPmat = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    dt = 1.0 / 25
    img_size = 256
    num_points = 4096
    first_init = True

    camera = MultiRealSense(
        use_front_cam=True,
        use_right_cam=use_wrist_img,
        front_num_points=num_points,
        img_size=img_size
    )
    arm_comm = UR_Comm()
    print("arm_comm connected")
    hand_comm = InspireHand()
    print("hand_comm connected")

    time.sleep(1)
    hand_comm.reset()
    time.sleep(1)
    hand_comm.setpower(500, 500, 500, 500, 500)
    hand_comm.setspeed(300, 300, 300, 300, 300)

    if first_init:
        arm_comm.reset_arm()
        hand_comm.reset()
        camera.start()
        time.sleep(0.5)
        print("Robot reset!")

    max_task_length = 1600  # 每个任务必须指定最大长度，其作为奖励计算归一化参数

    try:
        while not complete_collect.is_set():
            start_recording.wait()  # 等待开始录制
            if complete_collect.is_set():
                break

            step_count = 0
            color_array = []
            if use_wrist_img:
                wrist_color_array = []
            robot_state_array = []
            action_array = []
            intervention_array = []

            # RTC 自主执行状态
            rtc_initialized = False
            rtc_action_horizon = 0
            rtc_action_cursor = 0
            rtc_np_action = None
            rtc_arm_mats = None
            rtc_policy_ref_mat = None
            rtc_pending_ref = None

            # 人工干预参考系
            init_tracker_mat = np.eye(4)
            init_arm_mat = np.eye(4)

            while step_count < max_task_length and start_recording.is_set():
                obs_start_time = time.time()

                # 获取观察
                cam_dict = camera()  # 回调获取最新一帧
                obs_img = cam_dict['front_color']
                obs_wrist_img = cam_dict['right_color'] if use_wrist_img else None
                color_array.append(obs_img)
                if use_wrist_img:
                    wrist_color_array.append(obs_wrist_img)

                agent_state_dict = arm_comm.get_robot_state()
                qpos = agent_state_dict['joint_positions']
                agent_mat = agent_state_dict['mat']
                robot_state_array.append(np.concatenate((qpos, agent_mat)))

                mat = tools.xyz_rotvec_to_mat(agent_mat)
                np_qpos = np.stack([qpos], axis=0)[None, ...]  # B,T,C
                np_obs_img = np.stack([obs_img], axis=0)[None, ...]  # B,T,H,W,C
                np_obs_wrist_img = None
                if use_wrist_img:
                    np_obs_wrist_img = np.stack([obs_wrist_img], axis=0)[None, ...]

                obs_end_time = time.time()

                if human_intervention.is_set():
                    # 进入人工干预后，重置 RTC 流水线
                    rtc_initialized = False
                    rtc_pending_ref = None

                    start_time = time.time()
                    tracker_data, hand_action_dict = expert()  # 回调专家遥操数据
                    hand_action = hand_action_dict["action"]
                    hand_action_array = np.array(hand_action_dict["normalized_action"])  # 存储归一化动作

                    if tracker_data is None:
                        # 专家遥操数据为空时，pop掉当前观测（没有对应动作）
                        if len(color_array) > 0:
                            color_array.pop()
                        if use_wrist_img and len(wrist_color_array) > 0:
                            wrist_color_array.pop()
                        if len(robot_state_array) > 0:
                            robot_state_array.pop()
                        time.sleep(0.01)
                        continue

                    current_mat = np.dot(tracker_data, trackertoTCPmat)  # 转化为 TCP 位姿

                    if flag[0] == 0:
                        # 第一帧仅用于建立参考系，该帧观测无对应动作，pop掉
                        if len(color_array) > 0:
                            color_array.pop()
                        if use_wrist_img and len(wrist_color_array) > 0:
                            wrist_color_array.pop()
                        if len(robot_state_array) > 0:
                            robot_state_array.pop()
                        init_tracker_mat = current_mat
                        init_arm_mat = mat  # 干预阶段都以该位姿为参考
                        flag[0] = 1
                        continue

                    incre_mat = np.dot(tools.se3_inverse(init_tracker_mat), current_mat)  # 以 TCP 为参考系的相对位姿
                    target_arm_mat = np.dot(init_arm_mat, incre_mat)

                    action_array.append(np.concatenate((tools.mat2xyz_6drot(target_arm_mat), hand_action_array)))
                    intervention_array.append(1)

                    arm_action = tools.mat2xyz_rotvec(target_arm_mat)  # UR格式
                    arm_comm.set_arm_action(arm_action)
                    hand_comm.setangle(*hand_action)
                    step_count += 1

                    end_time = time.time()
                    time.sleep(max(0, dt - (end_time - start_time + obs_end_time - obs_start_time)))
                    continue

                # 自主策略（RTC 异步流水线）
                if not rtc_initialized:
                    init_obs_dict = {
                        'agent_pos': np_qpos,
                        'image': np_obs_img,
                        'exc_action': None,
                    }
                    if use_wrist_img:
                        init_obs_dict['wrist_img'] = np_obs_wrist_img

                    action = ray.get(RTC_policy.inference.remote(init_obs_dict))
                    rtc_np_action = action.numpy()  # T,15
                    rtc_action_horizon = rtc_np_action.shape[0]
                    rtc_action_cursor = 0
                    rtc_policy_ref_mat = mat

                    relative_arm_mat = tools.xyz_6drot_to_mat(rtc_np_action[:, :9])  # T,4,4
                    rtc_arm_mats = np.einsum("ij,tjk->tik", rtc_policy_ref_mat, relative_arm_mat)

                    rtc_pending_ref = None
                    rtc_initialized = True

                # 执行到只剩最大延迟步数时，用当前观测更新 policy
                if (
                    rtc_action_horizon > max_latency_step
                    and rtc_action_cursor == rtc_action_horizon - max_latency_step
                ):
                    rtc_policy_ref_mat = mat
                    exc_arm_mats = np.einsum(
                        "ij,tjk->tik",
                        tools.se3_inverse(rtc_policy_ref_mat),
                        rtc_arm_mats[-max_latency_step:]
                    )
                    exc_arm_action = tools.mat2xyz_6drot(exc_arm_mats).astype(np.float32)  # [T,9]
                    exc_hand_action = rtc_np_action[-max_latency_step:, 9:].astype(np.float32)  # [T,6]
                    exc_action = np.concatenate([exc_arm_action, exc_hand_action], axis=-1)[None, ...]  # [B,T,15]

                    next_obs_dict = {
                        'agent_pos': np_qpos,
                        'image': np_obs_img,
                        'exc_action': exc_action,
                    }
                    if use_wrist_img:
                        next_obs_dict['wrist_img'] = np_obs_wrist_img

                    rtc_pending_ref = RTC_policy.inference.remote(next_obs_dict)

                arm_mat = rtc_arm_mats[rtc_action_cursor]
                action_array.append(np.concatenate((tools.mat2xyz_6drot(arm_mat), rtc_np_action[rtc_action_cursor, 9:])))
                intervention_array.append(0)

                arm_action = tools.mat2xyz_rotvec(arm_mat)
                arm_comm.set_arm_action(arm_action)
                hand_action = hand_action_util(rtc_np_action[rtc_action_cursor, 9:])
                hand_comm.setangle(*hand_action)

                step_count += 1
                rtc_action_cursor += 1

                # 当前动作段执行结束后，切换到下一段异步推理结果
                if rtc_action_cursor >= rtc_action_horizon:
                    if rtc_pending_ref is not None:
                        action = ray.get(rtc_pending_ref)
                        rtc_np_action = action.numpy()
                        rtc_action_horizon = rtc_np_action.shape[0]
                        rtc_action_cursor = 0

                        relative_arm_mat = tools.xyz_6drot_to_mat(rtc_np_action[:, :9])
                        rtc_arm_mats = np.einsum("ij,tjk->tik", rtc_policy_ref_mat, relative_arm_mat)
                        rtc_pending_ref = None
                    else:
                        # horizon <= max_latency_step 时没有异步结果，下一帧重新阻塞推理
                        rtc_initialized = False

                policy_end_time = time.time()
                time.sleep(max(0, dt - (policy_end_time - obs_start_time)))

            # 录制结束后先停止伺服线程
            arm_comm.rtde_c.servoStop()
            time.sleep(0.3)

            # 保存数据（采用 RTC_deploy 记录格式，并保留 intervention 标注）
            if len(action_array) > 0:
                usr_success = input("是否成功完成任务？(y/n): ").lower().strip()
                success = (usr_success == 'y')
                user_choice = input("是否保存录制数据？(y/n): ").lower().strip()

                if user_choice == 'y':
                    record_file_name = os.path.join(
                        data_dir,
                        datetime.now().strftime("demo_%Y%m%d_%H%M%S") + ".h5"
                    )
                    with h5py.File(record_file_name, "w") as f:
                        print("Data recording")
                        seq_length = len(action_array)
                        color_array = np.array(color_array)
                        if use_wrist_img:
                            wrist_color_array = np.array(wrist_color_array)
                            f.create_dataset("wrist_color", data=wrist_color_array)
                        env_qpos_array = np.array(robot_state_array)
                        action_array = np.array(action_array)
                        intervention_array = np.array(intervention_array, dtype=np.uint8)

                        f.create_dataset("color", data=color_array)
                        f.create_dataset("env_qpos_proprioception", data=env_qpos_array)
                        f.create_dataset("action", data=action_array)
                        f.create_dataset("intervention", data=intervention_array)
                        f.attrs["success"] = success

                    print("Data recording done.")
                    if use_wrist_img:
                        cprint(f"wrist_color shape: {wrist_color_array.shape}", "yellow")
                    cprint(f"color shape: {color_array.shape}", "yellow")
                    cprint(f"action shape: {action_array.shape}", "yellow")
                    cprint(f"intervention shape: {intervention_array.shape}", "yellow")
                    cprint(f"env_qpos shape: {env_qpos_array.shape}", "yellow")
                    cprint(f"save data at step: {seq_length} in {record_file_name}", "yellow")
                else:
                    print("不保存数据")
            else:
                print("无数据，不保存")

            # reset robot
            hand_comm.reset()
            arm_comm.reset_arm()
            time.sleep(0.1)

    except Exception as e:
        cprint(f"[Error] {e}", "red")

    listener.stop()
    expert.finalize()

    arm_comm.cleanup()
    camera.finalize()
    hand_comm.close()

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
