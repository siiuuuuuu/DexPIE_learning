import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import time
import pathlib
from datetime import datetime

import hydra
import ray
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexpie.workspace.base_workspace import BaseWorkspace
from dexpie.common.shared_multi_realsense import MultiRealSense
from dexpie.common.tools import MATHTOOLS
from dexpie.common.intervention_episode_buffer import InterventionEpisodeBuffer
from dexpie.common.rtc_action_stream import RTCActionStream
from dexpie.common.intervention_keyboard_control import InterventionKeyboardControl
from dexpie.common.collection_observation import ObservationBuilder
from dexpie.common.rtc_collection import (
    AlignedActionProvider,
    InterventionModeManager,
    PolicyActionPlanner,
    RTCTimestampBuilder,
)
from dexpie.common.hand_executor import HighRateHandExecutor
from dexpie.common.robot_state_reader import HighRateRobotStateReader
from dexpie.common.human_hand_updater import HumanHandUpdater
from dexpie.common.arm_executor import HighRateArmExecutor

from communication.teleop_interfaces import InspireHandController, URArmInterface
from human_intervention.expert_intervention import ExpertIntervention

os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_UR_HOST = "192.168.3.6"
DEFAULT_WORKSPACE_LIMITS = {
    "x": [-1.5, 1.5],
    "y": [-1.5, 1.5],
    "z": [-0.5, 1.5],
}
DEFAULT_INITIAL_POSE = [0.248, 0.0812, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200
DEFAULT_HAND_RESET_COMMAND = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_HAND_EXECUTOR_HZ = 120.0
DEFAULT_HAND_INTERVENTION_W = 25.0  # Natural frequency; larger values track targets faster.
DEFAULT_HAND_INTERVENTION_Z = 0.8  # Damping ratio; larger values reduce overshoot and smooth motion.
DEFAULT_DATA_DIR = "~/dp_data/offlineRL_data/test_task3_iter1"
DEFAULT_CONTROL_DT = 1.0 / 25
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MAX_TASK_LENGTH = 1000  # Max task length used to normalize reward calculation.
DEFAULT_ROBOT_STATE_FREQUENCY = 125.0
DEFAULT_ARM_SERVO_FREQUENCY = 25
DEFAULT_TRACKER_FREQUENCY = 60.0
DEFAULT_MANUS_TIMEOUT = 0.25
DEFAULT_HUMAN_HAND_UPDATE_FREQUENCY = 120.0
DEFAULT_ALIGNMENT_TOLERANCE_MS = 25.0
DEFAULT_FRONT_CAMERA_FPS = 30
DEFAULT_WRIST_CAMERA_FPS = 60
DEFAULT_CAMERA_SYNC_WAIT_TIMEOUT_MS = 5.0


@ray.remote(num_gpus=1)
class async_policy:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy.to('cuda')
        self.action_horizon = self.policy.horizon - self.policy.n_obs_steps + 1
        self.action_dim = self.policy.action_dim

    def ready(self):
        return True

    # Pass a numpy-format dict; Ray is much slower with tensor payloads.
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
        'dexpie', 'config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace_name = workspace.__class__.__name__
    cfg_scale = OmegaConf.select(cfg, "policy.positive_cfg_scale")
    if workspace_name in ("DexPIEWorkspace", "RecapWorkspace") and cfg_scale is not None:
        cprint(f"Current workspace={workspace_name}, cfg scale={cfg_scale}", "yellow")
    use_image = True
    use_wrist_img = bool(cfg.task.dataset.use_wrist_img)
    tools = MATHTOOLS()

    # Initialize expert process.
    expert = ExpertIntervention(use_right_hand=True, use_left_hand=False)
    expert.start()
    time.sleep(0.1)
    print("Expert intervention process started.")

    keyboard_control = InterventionKeyboardControl()
    print("Press 's' to start/stop recording, 'e' to start/stop human intervention, and 'c' to finish collection.")
    keyboard_control.start()  # Start listener thread.

    time.sleep(0.1)

    try:
        policy = workspace.get_model()
        RTC_policy = async_policy.remote(policy)
        ray.get(RTC_policy.ready.remote())
    except Exception:
        keyboard_control.stop()
        expert.finalize()
        if ray.is_initialized():
            ray.shutdown()
        raise
    max_latency_step = int(getattr(policy, 'max_latency_steps', 3))
    rtc_stream = RTCActionStream(
        RTC_policy,
        tools,
        use_wrist_img=use_wrist_img,
        max_latency_step=max_latency_step,
    )
    data_dir = os.path.expanduser(DEFAULT_DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    dt = DEFAULT_CONTROL_DT
    img_size = DEFAULT_IMAGE_SIZE
    first_init = True

    camera = MultiRealSense(
        use_front_cam=True,
        use_right_cam=use_wrist_img,
        img_size=img_size,
        front_camera_fps=DEFAULT_FRONT_CAMERA_FPS,
        wrist_camera_fps=DEFAULT_WRIST_CAMERA_FPS,
        sync_right_to_front=True,
        sync_wait_timeout_ms=DEFAULT_CAMERA_SYNC_WAIT_TIMEOUT_MS,
    )
    robot = URArmInterface(
        DEFAULT_UR_HOST,
        DEFAULT_WORKSPACE_LIMITS,
        servo_speed=0.005,
        servo_acceleration=0.005,
        servo_dt=1.0 / DEFAULT_ARM_SERVO_FREQUENCY,
        lookahead_time=0.2,
        gain=500,
    )
    print("robot connected")
    hand_controller = InspireHandController(DEFAULT_HAND_PORT, DEFAULT_HAND_BAUDRATE)
    hand_executor = HighRateHandExecutor(
        send_callback=hand_controller.apply,
        hz=DEFAULT_HAND_EXECUTOR_HZ,
        w=DEFAULT_HAND_INTERVENTION_W,
        z=DEFAULT_HAND_INTERVENTION_Z,
        dim=len(DEFAULT_HAND_RESET_COMMAND),
    )
    hand_updater = HumanHandUpdater(
        sample_reader=expert.latest_hand_sample,
        hand_executor=hand_executor,
        is_active_fn=lambda: (
            keyboard_control.is_recording()
            and keyboard_control.is_intervening()
        ),
        update_frequency=DEFAULT_HUMAN_HAND_UPDATE_FREQUENCY,
        sample_timeout=DEFAULT_MANUS_TIMEOUT,
    )
    print("hand_controller connected")
    robot_state_reader = HighRateRobotStateReader(
        robot,
        read_frequency=DEFAULT_ROBOT_STATE_FREQUENCY,
    )
    arm_executor = HighRateArmExecutor(
        robot,
        tools,
        tracker_reader=expert.read_tracker_mat,
        servo_frequency=DEFAULT_ARM_SERVO_FREQUENCY,
        tracker_frequency=DEFAULT_TRACKER_FREQUENCY,
        policy_frequency=1.0 / dt,
    )
    obs_builder = ObservationBuilder(
        camera,
        robot,
        tools,
        use_wrist_img=use_wrist_img,
        robot_state_reader=robot_state_reader,
        alignment_tolerance_ms=DEFAULT_ALIGNMENT_TOLERANCE_MS,
    )
    timestamp_builder = RTCTimestampBuilder()
    intervention_manager = InterventionModeManager(
        rtc_stream,
        arm_executor,
        hand_executor,
    )
    policy_planner = PolicyActionPlanner(
        rtc_stream,
        arm_executor,
        hand_executor,
        dt=dt,
        timestamp_builder=timestamp_builder,
    )
    aligned_action_provider = AlignedActionProvider(
        arm_executor,
        hand_executor,
        alignment_tolerance_ms=DEFAULT_ALIGNMENT_TOLERANCE_MS,
        timestamp_builder=timestamp_builder,
    )

    if first_init:
        robot.move_l(DEFAULT_INITIAL_POSE, 0.2, 0.2)
        hand_controller.reset()
        hand_executor.reset_state(DEFAULT_HAND_RESET_COMMAND)
        camera.start()
        time.sleep(0.5)
        robot_state_reader.start()
        arm_executor.start()
        hand_executor.start()
        hand_updater.start()
        print("Robot reset!")
        print(
            "Robot state reader started "
            f"({DEFAULT_ROBOT_STATE_FREQUENCY:.1f}Hz)."
        )
        print(
            "Arm executor started "
            f"({DEFAULT_ARM_SERVO_FREQUENCY:.1f}Hz servo, "
            f"{DEFAULT_TRACKER_FREQUENCY:.1f}Hz tracker)."
        )
        print(
            "Human hand updater started "
            f"({DEFAULT_HUMAN_HAND_UPDATE_FREQUENCY:.1f}Hz)."
        )

    max_task_length = DEFAULT_MAX_TASK_LENGTH

    try:
        while not keyboard_control.should_stop_collection():
            keyboard_control.wait_recording()  # Wait for recording to start.
            if keyboard_control.should_stop_collection():
                break

            step_count = 0
            episode = InterventionEpisodeBuffer(use_wrist_img=use_wrist_img)
            intervention_manager.reset_episode()

            while step_count < max_task_length and keyboard_control.is_recording():
                obs_start_time = time.time()
                intervention_active = keyboard_control.is_intervening()

                if not robot.is_ready():
                    print("Robot is stopped (protective or emergency).")
                    break
                robot_state_reader.raise_if_failed()
                arm_executor.raise_if_failed()
                hand_executor.raise_if_failed()
                hand_updater.raise_if_failed()

                # Read observation.
                obs = obs_builder.read()
                if obs is None:
                    time.sleep(0.01)
                    continue

                intervention_manager.sync(intervention_active, obs)

                timestamps = timestamp_builder.begin_action_decision(
                    obs.timestamp_dict
                )
                if not intervention_active:
                    policy_planner.step(obs, timestamps)

                aligned_action = aligned_action_provider.read(timestamps)
                if aligned_action is None:
                    time.sleep(0.001)
                    continue

                episode.append(
                    obs.robot_state,
                    obs.cam_dict,
                    aligned_action.action,
                    aligned_action.intervention,
                    aligned_action.timestamps,
                )
                step_count += 1

                loop_end_time = time.time()
                time.sleep(max(0, dt - (loop_end_time - obs_start_time)))

            # Stop servo before ending the recording.
            arm_executor.set_idle(stop_servo=True)
            hand_executor.stop()
            time.sleep(0.3)

            # Save data in RTC_deploy format and keep intervention labels.
            if len(episode) > 0:
                usr_success = input("Was the task completed successfully? (y/n): ").lower().strip()
                success = (usr_success == 'y')
                user_choice = input("Save recorded data? (y/n): ").lower().strip()

                if user_choice == 'y':
                    record_file_name = os.path.join(
                        data_dir,
                        datetime.now().strftime("demo_%Y%m%d_%H%M%S") + ".h5"
                    )
                    print("Data recording")
                    summary = episode.save_h5(record_file_name, success=success)
                    print("Data recording done.")
                    if use_wrist_img:
                        cprint(f"wrist_color shape: {summary['wrist_color_shape']}", "yellow")
                    cprint(f"color shape: {summary['color_shape']}", "yellow")
                    cprint(f"action shape: {summary['action_shape']}", "yellow")
                    cprint(f"intervention shape: {summary['intervention_shape']}", "yellow")
                    cprint(f"env_qpos shape: {summary['env_qpos_shape']}", "yellow")
                    cprint(f"save data at step: {summary['seq_length']} in {record_file_name}", "yellow")
                else:
                    print("Recording discarded.")
            else:
                print("No data recorded; nothing to save.")

            # reset robot
            arm_executor.set_idle(stop_servo=True)
            hand_controller.reset()
            hand_executor.reset_state(DEFAULT_HAND_RESET_COMMAND)
            hand_executor.start()
            robot.move_l(DEFAULT_INITIAL_POSE, 0.2, 0.2)
            time.sleep(0.1)

    except Exception as e:
        cprint(f"[Error] {e}", "red")

    keyboard_control.stop()
    hand_updater.stop()
    arm_executor.stop()
    expert.finalize()

    robot_state_reader.stop()
    robot.close(stop_script=True)
    camera.finalize()
    hand_executor.stop()
    hand_controller.close()

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
