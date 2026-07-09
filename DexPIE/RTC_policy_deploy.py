import sys

# Use line-buffering for both stdout and stderr.
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import json
import threading
import time

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from pynput import keyboard
from termcolor import cprint

from dexpie.workspace.base_workspace import BaseWorkspace
from dexpie.common.arm_executor import HighRateArmExecutor
from dexpie.common.collection_observation import ObservationBuilder
from dexpie.common.hand_executor import HighRateHandExecutor
from dexpie.common.robot_state_reader import HighRateRobotStateReader
from dexpie.common.rtc_action_stream import RTCActionStream
from dexpie.common.rtc_collection import PolicyActionPlanner, RTCTimestampBuilder
from dexpie.common.shared_multi_realsense import MultiRealSense
from dexpie.common.tools import MATHTOOLS

from communication.teleop_interfaces import InspireHandController, URArmInterface


os.environ["WANDB_SILENT"] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_UR_HOST = "192.168.3.6"
DEFAULT_WORKSPACE_LIMITS = {
    "x": [-1.5, 1.5],
    "y": [-1.5, 1.5],
    "z": [-0.5, 1.5],
}
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200
DEFAULT_HAND_RESET_COMMAND = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_HAND_EXECUTOR_HZ = 120.0
DEFAULT_HAND_INTERVENTION_W = 25.0
DEFAULT_HAND_INTERVENTION_Z = 0.8
DEFAULT_CONTROL_DT = 1.0 / 30.0
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MAX_TASK_LENGTH = 1000
DEFAULT_ROBOT_STATE_FREQUENCY = 125.0
DEFAULT_ARM_SERVO_FREQUENCY = 120
DEFAULT_ALIGNMENT_TOLERANCE_MS = 25.0
DEFAULT_FRONT_CAMERA_FPS = 30
DEFAULT_WRIST_CAMERA_FPS = 60
DEFAULT_CAMERA_SYNC_WAIT_TIMEOUT_MS = 5.0
DEFAULT_HISTORY_WAIT_TIMEOUT_MS = 5.0
DEFAULT_RTC_OBS_LATENCY_STEPS = 1
DEFAULT_POLICY_STATE_TIMEOUT = 0.12
DEFAULT_POLICY_ROT_MAX_SPEED_DEG = 75.0
DEFAULT_POLICY_ROT_MAX_GAP_DEG = 10.0
DEFAULT_DIAG_INTERVAL_STEPS = 30


class PolicyKeyboardControl:
    """Keyboard control for pure policy deployment."""

    def __init__(self):
        self.enabled = threading.Event()
        self.stop_requested = threading.Event()
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        try:
            if not hasattr(key, "char") or key.char is None:
                return

            k = key.char.lower()
            if k == "s" and not self.enabled.is_set():
                self.enabled.set()
                print("\n[INFO] Key 's' pressed: start pure policy.")
            elif k == "s" and self.enabled.is_set():
                self.enabled.clear()
                print("\n[INFO] Key 's' pressed: stop pure policy.")
            elif k == "c":
                self.stop_requested.set()
                self.enabled.clear()
                print("\n[INFO] Key 'c' pressed: finish deployment.")
        except AttributeError:
            pass
        except Exception as exc:
            print(f"Error in key press handler: {exc}")

    def start(self):
        self.listener.start()

    def stop(self):
        if self.listener.running:
            self.listener.stop()

    def wait_enabled(self):
        while not self.stop_requested.is_set() and not self.enabled.is_set():
            self.enabled.wait(timeout=0.1)

    def is_enabled(self):
        return self.enabled.is_set()

    def should_stop(self):
        return self.stop_requested.is_set()


@ray.remote(num_gpus=1)
class AsyncPolicy:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy.to("cuda")
        self.action_horizon = self.policy.horizon - self.policy.n_obs_steps + 1
        self.action_dim = self.policy.action_dim

    def ready(self):
        return True

    def inference(self, obs_dict):
        obs_dict = {
            key: (None if value is None else torch.from_numpy(value.copy()).to("cuda"))
            for key, value in obs_dict.items()
        }
        with torch.no_grad():
            action = self.policy(obs_dict)[0]
        return action.detach().cpu()


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def dataset_action_offset_steps(cfg, default=0):
    zarr_path = OmegaConf.select(cfg, "task.dataset.zarr_path")
    if zarr_path is None:
        return int(default)
    attrs_path = pathlib.Path(str(zarr_path)).expanduser().joinpath(".zattrs")
    if not attrs_path.is_file():
        return int(default)
    try:
        with attrs_path.open("r", encoding="utf-8") as f:
            attrs = json.load(f)
        return int(attrs.get("action_offset_frames", default))
    except Exception:
        return int(default)


def fmt_ms(value):
    if value is None:
        return "None"
    return f"{float(value):.1f}"


def xyz_gap_cm(target_xyz, current_xyz):
    if target_xyz is None or current_xyz is None:
        return None
    return float(
        np.linalg.norm(
            np.asarray(target_xyz)[:3] - np.asarray(current_xyz)[:3]
        )
        * 100.0
    )


def print_policy_diagnostics(
    step_count,
    obs,
    timestamps,
    policy_command,
    arm_executor,
    force=False,
):
    valid_sequence = policy_command.get("valid_sequence")
    if (
        not force
        and valid_sequence is None
        and step_count % DEFAULT_DIAG_INTERVAL_STEPS != 0
    ):
        return

    action = policy_command.get("action")
    target_gap = None if action is None else xyz_gap_cm(action[:3], obs.tcp_pose[:3])
    latest_motion = arm_executor.latest_motion()
    servo_gap = None
    servo_cmd_age_ms = None
    if latest_motion is not None:
        servo_gap = xyz_gap_cm(latest_motion.get("target_pose"), obs.tcp_pose)
        cmd_ns = latest_motion.get("t_policy_command_host_ns")
        if cmd_ns is not None:
            servo_cmd_age_ms = (
                time.monotonic_ns() - int(np.asarray(cmd_ns).item())
            ) / 1e6

    seq_text = "seq=None"
    if valid_sequence is not None:
        seq_start_ns = timestamps.get("t_policy_sequence_start_ns")
        submit_delay_ms = None
        if seq_start_ns is not None:
            submit_delay_ms = (
                int(seq_start_ns) - int(timestamps["t_action_decision_ns"])
            ) / 1e6
        seq_gap = None
        target_poses = valid_sequence.get("target_poses")
        if target_poses is not None and len(target_poses) > 0:
            seq_gap = xyz_gap_cm(target_poses[0][:3], obs.tcp_pose[:3])
        seq_text = (
            f"seq_len={len(valid_sequence['hand_actions'])} "
            f"start_delay={valid_sequence.get('start_delay_steps')} "
            f"submit_delay_ms={fmt_ms(submit_delay_ms)} "
            f"first_gap_cm={fmt_ms(seq_gap)}"
        )

    print(
        "[policy_diag] "
        f"step={step_count} "
        f"obs_age_ms={fmt_ms(timestamps.get('obs_to_action_latency_ms'))} "
        f"front_rx_minus_img_ms={fmt_ms(obs.timestamp_dict.get('front_camera_receive_minus_image_ms'))} "
        f"robot_sync_ms={fmt_ms(obs.timestamp_dict.get('sync_delta_robot_obs_ms'))} "
        f"cursor_gap_cm={fmt_ms(target_gap)} "
        f"servo_gap_cm={fmt_ms(servo_gap)} "
        f"servo_cmd_age_ms={fmt_ms(servo_cmd_age_ms)} "
        f"{seq_text}"
    )


def clear_policy_outputs(arm_executor, hand_executor):
    arm_executor.set_idle(stop_servo=True)
    hand_executor.clear_policy_sequence()


def reset_to_initial(robot, hand_controller, hand_executor):
    hand_controller.reset()
    hand_executor.reset_state(DEFAULT_HAND_RESET_COMMAND)
    robot.move_l(DEFAULT_INITIAL_POSE, 0.2, 0.2)


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("dexpie", "config"))
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

    use_wrist_img = bool(cfg.task.dataset.use_wrist_img)
    tools = MATHTOOLS()
    dt = env_float("RTC_POLICY_DT", DEFAULT_CONTROL_DT)
    max_task_length = env_int("RTC_POLICY_MAX_TASK_LENGTH", DEFAULT_MAX_TASK_LENGTH)
    obs_latency_steps = env_int(
        "RTC_OBS_LATENCY_STEPS",
        DEFAULT_RTC_OBS_LATENCY_STEPS,
    )
    action_offset_steps = env_int(
        "RTC_ACTION_OFFSET_STEPS",
        dataset_action_offset_steps(cfg, default=0),
    )
    initial_valid_start = max(0, obs_latency_steps - action_offset_steps)
    initial_start_delay_steps = max(0, action_offset_steps - obs_latency_steps)

    keyboard_control = PolicyKeyboardControl()
    keyboard_control.start()
    print("Press 's' to start/stop pure policy, and 'c' to quit.")

    policy_actor = None
    camera = None
    robot = None
    hand_controller = None
    hand_executor = None
    robot_state_reader = None
    arm_executor = None

    try:
        policy = workspace.get_model()
        policy_actor = AsyncPolicy.remote(policy)
        ray.get(policy_actor.ready.remote())
        max_latency_step = int(getattr(policy, "max_latency_steps", 3))
        rtc_stream = RTCActionStream(
            policy_actor,
            tools,
            use_wrist_img=use_wrist_img,
            max_latency_step=max_latency_step,
            obs_latency_steps=obs_latency_steps,
            action_offset_steps=action_offset_steps,
        )

        cprint(
            "Policy deployment: no H5 saving, no human intervention.",
            "yellow",
        )
        cprint(
            f"RTC max_latency_steps={max_latency_step}, "
            f"obs_latency_steps={obs_latency_steps}, "
            f"action_offset_steps={action_offset_steps}, dt={dt:.4f}s",
            "yellow",
        )
        cprint(
            f"Initial action alignment: valid_start={initial_valid_start}, "
            f"start_delay_steps={initial_start_delay_steps}",
            "yellow",
        )
        cprint("Policy arm execution uses direct interpolation.", "yellow")

        camera = MultiRealSense(
            use_front_cam=True,
            use_right_cam=use_wrist_img,
            img_size=DEFAULT_IMAGE_SIZE,
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
            control_frequency=DEFAULT_ARM_SERVO_FREQUENCY,
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
        print("hand_controller connected")
        robot_state_reader = HighRateRobotStateReader(
            robot,
            read_frequency=DEFAULT_ROBOT_STATE_FREQUENCY,
        )
        arm_executor = HighRateArmExecutor(
            robot,
            tools,
            tracker_reader=None,
            servo_frequency=DEFAULT_ARM_SERVO_FREQUENCY,
            policy_frequency=1.0 / dt,
            policy_interpolation_delay=0.0,
            robot_state_reader=robot_state_reader,
            use_policy_mpc=False,
            policy_rot_max_speed_deg=DEFAULT_POLICY_ROT_MAX_SPEED_DEG,
            policy_rot_max_gap_deg=DEFAULT_POLICY_ROT_MAX_GAP_DEG,
            policy_state_timeout=DEFAULT_POLICY_STATE_TIMEOUT,
        )
        obs_builder = ObservationBuilder(
            camera,
            robot,
            tools,
            use_wrist_img=use_wrist_img,
            robot_state_reader=robot_state_reader,
            alignment_tolerance_ms=DEFAULT_ALIGNMENT_TOLERANCE_MS,
            history_wait_timeout_ms=DEFAULT_HISTORY_WAIT_TIMEOUT_MS,
        )
        timestamp_builder = RTCTimestampBuilder()
        policy_planner = PolicyActionPlanner(
            rtc_stream,
            arm_executor,
            hand_executor,
            dt=dt,
            timestamp_builder=timestamp_builder,
        )

        reset_to_initial(robot, hand_controller, hand_executor)
        camera.start()
        time.sleep(0.5)
        robot_state_reader.start()
        arm_executor.start()
        hand_executor.start()
        print("Robot reset!")
        print(
            "Robot state reader started "
            f"({DEFAULT_ROBOT_STATE_FREQUENCY:.1f}Hz)."
        )
        print(
            "Arm executor started "
            f"({DEFAULT_ARM_SERVO_FREQUENCY:.1f}Hz servo, "
            "direct interpolation)."
        )
        print(f"Hand executor started ({DEFAULT_HAND_EXECUTOR_HZ:.1f}Hz).")

        while not keyboard_control.should_stop():
            keyboard_control.wait_enabled()
            if keyboard_control.should_stop():
                break

            step_count = 0
            rtc_stream.reset()
            obs_builder.reset_episode()
            arm_executor.reset_episode()
            arm_executor.start_policy()
            hand_executor.clear_history()
            hand_executor.clear_policy_sequence()

            print("[INFO] Pure policy episode started.")
            while step_count < max_task_length and keyboard_control.is_enabled():
                if not robot.is_ready():
                    print("Robot is stopped (protective or emergency).")
                    break
                robot_state_reader.raise_if_failed()
                arm_executor.raise_if_failed()
                hand_executor.raise_if_failed()

                obs = obs_builder.read()
                if obs is None:
                    time.sleep(0.01)
                    continue

                timestamps = timestamp_builder.begin_action_decision(
                    obs.timestamp_dict
                )
                policy_command = policy_planner.step(obs, timestamps)
                print_policy_diagnostics(
                    step_count,
                    obs,
                    timestamps,
                    policy_command,
                    arm_executor,
                    force=policy_command.get("valid_sequence") is not None,
                )
                step_count += 1

            clear_policy_outputs(arm_executor, hand_executor)
            print(f"[INFO] Pure policy episode stopped at step {step_count}.")
            if keyboard_control.should_stop():
                break

            reset_to_initial(robot, hand_controller, hand_executor)
            time.sleep(0.1)

    except Exception as exc:
        cprint(f"[Error] {exc}", "red")
        raise
    finally:
        keyboard_control.stop()
        if arm_executor is not None:
            arm_executor.stop()
        if robot_state_reader is not None:
            robot_state_reader.stop()
        if camera is not None:
            camera.finalize()
        if robot is not None:
            robot.close(stop_script=True)
        if hand_executor is not None:
            hand_executor.stop()
        if hand_controller is not None:
            hand_controller.close()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
