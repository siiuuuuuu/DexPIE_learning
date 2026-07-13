from dataclasses import dataclass
import time

import numpy as np

from dexpie.common.arm_executor import ARM_MODE_INTERVENTION
from dexpie.common.hand_action_util import hand_action_util
from dexpie.common.hand_executor import HAND_MODE_INTERVENTION


def time_delta_ms(t_ns, ref_ns):
    if t_ns is None or ref_ns is None:
        return None
    return (int(t_ns) - int(ref_ns)) / 1e6


def scalar_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass
class AlignedAction:
    action: np.ndarray
    intervention: int
    timestamps: dict


class RTCTimestampBuilder:
    """Builds timestamp fields for RTC intervention collection."""

    def begin_action_decision(self, obs_timestamp_dict):
        timestamps = dict(obs_timestamp_dict)
        t_action_decision_ns = time.monotonic_ns()
        timestamps["t_action_decision_ns"] = t_action_decision_ns
        timestamps["obs_to_action_latency_ms"] = time_delta_ms(
            t_action_decision_ns,
            timestamps.get("t_anchor_ns"),
        )
        return timestamps

    def add_policy_sequence(self, timestamps, valid_sequence, sequence_start_ns):
        start_delay_steps = int(valid_sequence.get("start_delay_steps", 0))
        timestamps["policy_sequence_valid_start"] = int(
            valid_sequence.get("valid_start", 0)
        )
        timestamps["policy_sequence_length"] = int(
            len(valid_sequence["hand_actions"])
        )
        timestamps["policy_sequence_start_delay_steps"] = start_delay_steps
        timestamps["t_policy_sequence_start_ns"] = sequence_start_ns

    def add_policy_submit_times(
        self,
        timestamps,
        t_arm_command_host_ns,
        t_hand_target_host_ns,
    ):
        timestamps["t_arm_command_host_ns"] = t_arm_command_host_ns
        timestamps["t_hand_target_host_ns"] = t_hand_target_host_ns

    def add_arm_motion(self, timestamps, arm_motion):
        if arm_motion is None:
            return

        int_keys = (
            "t_arm_action_host_ns",
            "t_arm_servo_host_ns",
            "t_arm_target_host_ns",
            "t_policy_command_host_ns",
            "t_policy0_host_ns",
            "t_policy1_host_ns",
            "t_tracker_host_ns",
            "t_tracker0_host_ns",
            "t_tracker1_host_ns",
            "t_tracker_latest_host_ns",
            "arm_executor_mode",
        )
        for key in int_keys:
            if key in arm_motion:
                timestamps[key] = scalar_value(arm_motion[key])

        float_keys = (
            "interpolation_alpha",
        )
        for key in float_keys:
            if key in arm_motion:
                timestamps[key] = scalar_value(arm_motion[key])

        action_ns = timestamps.get("t_arm_action_host_ns")
        timestamps["t_aligned_arm_action_ns"] = action_ns
        timestamps["sync_delta_arm_action_ms"] = time_delta_ms(
            action_ns,
            timestamps.get("t_anchor_ns"),
        )
        timestamps["sync_delta_arm_servo_ms"] = time_delta_ms(
            timestamps.get("t_arm_servo_host_ns"),
            timestamps.get("t_anchor_ns"),
        )

    def add_hand_sample(self, timestamps, hand_sample):
        if hand_sample is None:
            return

        int_keys = (
            "t_hand_action_host_ns",
            "t_hand_command_host_ns",
            "t_hand_target_host_ns",
            "t_manus_sample_host_ns",
            "manus_seq",
            "hand_executor_mode",
        )
        for key in int_keys:
            if key in hand_sample:
                timestamps[key] = scalar_value(hand_sample[key])

        action_ns = timestamps.get("t_hand_action_host_ns")
        timestamps["t_aligned_hand_action_ns"] = action_ns
        timestamps["sync_delta_hand_action_ms"] = time_delta_ms(
            action_ns,
            timestamps.get("t_anchor_ns"),
        )

    def add_record_end(self, timestamps):
        t_record_end_ns = time.monotonic_ns()
        timestamps["t_record_end_ns"] = t_record_end_ns
        timestamps["record_loop_duration_ms"] = time_delta_ms(
            t_record_end_ns,
            timestamps.get("t_record_start_ns"),
        )

    def within_alignment_tolerance(self, timestamps, tolerance_ms):
        for key in ("sync_delta_arm_action_ms", "sync_delta_hand_action_ms"):
            value = timestamps.get(key)
            if value is None or abs(float(value)) > tolerance_ms:
                return False
        return True


class InterventionModeManager:
    """Keeps policy/intervention executor modes in sync with keyboard state."""

    def __init__(self, rtc_stream, arm_executor, hand_executor):
        self.rtc_stream = rtc_stream
        self.arm_executor = arm_executor
        self.hand_executor = hand_executor
        self.was_human_intervention = False

    def reset_episode(self):
        self.rtc_stream.reset()
        self.arm_executor.reset_episode()
        self.arm_executor.start_policy()
        self.hand_executor.clear_history()
        self.was_human_intervention = False

    def sync(self, intervention_active, obs):
        intervention_active = bool(intervention_active)
        if intervention_active == self.was_human_intervention:
            return

        self.rtc_stream.reset()
        if intervention_active:
            self.arm_executor.start_intervention(obs.arm_mat)
        else:
            self.arm_executor.start_policy()
        self.hand_executor.clear_policy_sequence()
        self.was_human_intervention = intervention_active


class PolicyActionPlanner:
    """Runs RTC policy steps and submits future-valid arm/hand sequences."""

    def __init__(
        self,
        rtc_stream,
        arm_executor,
        hand_executor,
        dt,
        timestamp_builder=None,
    ):
        self.rtc_stream = rtc_stream
        self.arm_executor = arm_executor
        self.hand_executor = hand_executor
        self.dt = float(dt)
        self.timestamp_builder = timestamp_builder or RTCTimestampBuilder()

    def step(self, obs, timestamps):
        policy_command = self.rtc_stream.step(
            obs.qpos,
            obs.image,
            obs.wrist_image,
            obs.arm_mat,
        )
        valid_sequence = policy_command.get("valid_sequence")
        if valid_sequence is None:
            return policy_command

        start_delay_steps = int(valid_sequence.get("start_delay_steps", 0))
        sequence_start_ns = time.monotonic_ns() + int(
            round(start_delay_steps * self.dt * 1e9)
        )
        self.timestamp_builder.add_policy_sequence(
            timestamps,
            valid_sequence,
            sequence_start_ns,
        )
        t_arm_command_host_ns = self.arm_executor.submit_policy_sequence(
            valid_sequence,
            start_time_ns=sequence_start_ns,
            step_period_s=self.dt,
        )
        hand_targets = np.asarray(
            [
                hand_action_util(hand_action)
                for hand_action in valid_sequence["hand_actions"]
            ],
            dtype=np.float32,
        )
        t_hand_target_host_ns = self.hand_executor.submit_policy_sequence(
            hand_targets,
            start_time_ns=sequence_start_ns,
            step_period_s=self.dt,
        )
        self.timestamp_builder.add_policy_submit_times(
            timestamps,
            t_arm_command_host_ns,
            t_hand_target_host_ns,
        )
        return policy_command


class AlignedActionProvider:
    """Reads arm/hand executor samples aligned to the observation anchor."""

    def __init__(
        self,
        arm_executor,
        hand_executor,
        alignment_tolerance_ms=25.0,
        timestamp_builder=None,
        history_wait_timeout_ms=3.0,
        history_wait_sleep_s=0.0005,
    ):
        self.arm_executor = arm_executor
        self.hand_executor = hand_executor
        self.alignment_tolerance_ms = float(alignment_tolerance_ms)
        self.timestamp_builder = timestamp_builder or RTCTimestampBuilder()
        self.history_wait_timeout_ms = max(0.0, float(history_wait_timeout_ms))
        self.history_wait_sleep_s = max(0.0, float(history_wait_sleep_s))

    def read(self, timestamps):
        anchor_ns = timestamps.get("t_anchor_ns")
        self._wait_histories_cover_anchor(anchor_ns)
        arm_motion = self.arm_executor.motion_at_time_ns(anchor_ns)
        hand_sample = self.hand_executor.command_at_time_ns(anchor_ns)
        if arm_motion is None or hand_sample is None:
            return None

        self.timestamp_builder.add_arm_motion(timestamps, arm_motion)
        self.timestamp_builder.add_hand_sample(timestamps, hand_sample)
        if not self.timestamp_builder.within_alignment_tolerance(
            timestamps,
            self.alignment_tolerance_ms,
        ):
            return None

        self.timestamp_builder.add_record_end(timestamps)
        return AlignedAction(
            action=self._action_from_samples(arm_motion, hand_sample),
            intervention=self._intervention_from_samples(
                arm_motion,
                hand_sample,
            ),
            timestamps=timestamps,
        )

    def _action_from_samples(self, arm_motion, hand_sample):
        arm_action = np.asarray(arm_motion["arm_action"], dtype=np.float32)
        hand_action = np.asarray(hand_sample["command"], dtype=np.float32) / 1000.0
        return np.concatenate((arm_action, hand_action)).astype(np.float32)

    def _intervention_from_samples(self, arm_motion, hand_sample):
        arm_mode = scalar_value(arm_motion.get("arm_executor_mode"))
        hand_mode = scalar_value(hand_sample.get("hand_executor_mode"))
        return int(
            arm_mode == ARM_MODE_INTERVENTION
            or hand_mode == HAND_MODE_INTERVENTION
        )

    def _wait_histories_cover_anchor(self, anchor_ns):
        if anchor_ns is None or self.history_wait_timeout_ms <= 0.0:
            return False

        anchor_ns = int(anchor_ns)
        deadline_ns = time.monotonic_ns() + int(
            self.history_wait_timeout_ms * 1e6
        )
        while time.monotonic_ns() < deadline_ns:
            if self._histories_cover_anchor(anchor_ns):
                return True
            time.sleep(self.history_wait_sleep_s)

        return self._histories_cover_anchor(anchor_ns)

    def _histories_cover_anchor(self, anchor_ns):
        return (
            self._covers_anchor(
                self.arm_executor.latest_action_time_ns(),
                anchor_ns,
            )
            and self._covers_anchor(
                self.hand_executor.latest_command_time_ns(),
                anchor_ns,
            )
        )

    @staticmethod
    def _covers_anchor(latest_time_ns, anchor_ns):
        return latest_time_ns is not None and int(latest_time_ns) >= int(anchor_ns)
