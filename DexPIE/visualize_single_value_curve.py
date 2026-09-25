import argparse
import os
import pathlib
import sys
from typing import Dict, List

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint

from dexpie.common.memmap_dataset import (
    resolve_dataset_storage,
    validate_visual_lengths,
)
from dexpie.common.replay_buffer import ReplayBuffer
from dexpie.workspace.critic_workspace import CriticWorkspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the critic value curve for one trajectory with optional "
            "colored frame ranges."
        )
    )
    parser.add_argument(
        "--critic_ckpt",
        type=str,
        required=True,
        help="Path to critic checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--dataset_path",
        "--zarr_path",
        dest="dataset_path",
        type=str,
        default=None,
        help=(
            "Optional dataset root (legacy Zarr or converted memmap layout). "
            "If not set, read from checkpoint cfg."
        ),
    )
    parser.add_argument(
        "--storage_mode",
        choices=("auto", "memory", "memmap"),
        default="auto",
        help=(
            "Dataset storage mode. The default auto-detects a converted "
            "memmap dataset."
        ),
    )
    parser.add_argument(
        "--traj_idx",
        type=int,
        required=True,
        help="Trajectory index (episode id, 0-based).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help=(
            "Optional inclusive trajectory-local end frame. For example, "
            "240 keeps frames 0 through 240."
        ),
    )
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        help=(
            "Highlight frame range with format start:end[:color[:label]]. "
            "Can be used multiple times, e.g. --highlight 120:180:red:mistake"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Max frames per forward pass to avoid CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Inference device, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/value_curve_single_traj.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure dpi.",
    )
    parser.add_argument(
        "--line_color",
        type=str,
        default="tab:blue",
        help="Main value curve color.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=1.8,
        help="Main value curve line width.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=1.0,
        help="Main value curve alpha (0~1). Smaller means lighter.",
    )
    parser.add_argument(
        "--highlight_alpha",
        type=float,
        default=0.18,
        help="Alpha for highlighted background span.",
    )
    parser.add_argument(
        "--recolor_highlight_segment",
        action="store_true",
        help=(
            "Recolor curve segments inside highlighted ranges using each "
            "highlight color."
        ),
    )
    parser.add_argument(
        "--x_as_frame",
        action="store_true",
        help="Use frame index on x-axis instead of seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate for frame-to-time conversion (default: 25Hz).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional custom figure title. Empty means no title.",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Show the legend box when enabled.",
    )
    parser.add_argument(
        "--label_fontsize",
        type=float,
        default=18,
        help="Font size for x/y axis labels.",
    )
    parser.add_argument(
        "--tick_fontsize",
        type=float,
        default=15,
        help="Font size for axis tick labels.",
    )
    parser.add_argument(
        "--enable_stochastic_aug",
        action="store_true",
        help=(
            "Keep stochastic augmentation during visualization. Disabled by "
            "default for stable curves."
        ),
    )
    return parser.parse_args()


def load_workspace_from_ckpt(ckpt_path: pathlib.Path) -> CriticWorkspace:
    try:
        import dill
    except ImportError as exc:
        raise ImportError(
            "dill is required to load .ckpt files. Please install dill first."
        ) from exc

    payload = torch.load(
        ckpt_path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
    )
    workspace = CriticWorkspace(payload["cfg"])
    workspace.load_payload(payload=payload)
    return workspace


def resolve_device(prefer_device: str, cfg_device: str) -> torch.device:
    if prefer_device is not None:
        return torch.device(prefer_device)
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        cprint(
            f"[Warn] cfg requires {cfg_device}, but CUDA is unavailable. "
            "Fallback to cpu.",
            "yellow",
        )
        return torch.device("cpu")
    return torch.device(cfg_device)


def _is_stochastic_augment(module: nn.Module) -> bool:
    name = module.__class__.__name__
    if name.startswith("Random"):
        return True
    return name in {
        "ColorJitter",
        "GaussianBlur",
        "RandAugment",
        "TrivialAugmentWide",
        "AutoAugment",
        "AugMix",
    }


def disable_stochastic_aug_for_visualization(critic) -> None:
    value_net = getattr(critic, "value_critic", None)
    if value_net is None:
        cprint(
            "[Warn] critic has no value_critic; skip disabling stochastic "
            "augmentations.",
            "yellow",
        )
        return

    key_transform_map = getattr(value_net, "key_transform_map", None)
    if key_transform_map is None:
        cprint(
            "[Warn] value_critic has no key_transform_map; skip disabling "
            "stochastic augmentations.",
            "yellow",
        )
        return

    total_removed = 0
    for key in list(key_transform_map.keys()):
        transform = key_transform_map[key]
        if isinstance(transform, nn.Sequential):
            kept_modules = []
            removed_names = []
            for module in transform:
                if _is_stochastic_augment(module):
                    removed_names.append(module.__class__.__name__)
                else:
                    kept_modules.append(module)

            if removed_names:
                total_removed += len(removed_names)
                key_transform_map[key] = (
                    nn.Sequential(*kept_modules)
                    if kept_modules
                    else nn.Identity()
                )
                cprint(
                    f"[Info] key '{key}' remove stochastic aug modules: "
                    f"{removed_names}",
                    "cyan",
                )
        elif _is_stochastic_augment(transform):
            total_removed += 1
            key_transform_map[key] = nn.Identity()
            cprint(
                f"[Info] key '{key}' replace stochastic aug "
                f"{transform.__class__.__name__} with Identity.",
                "cyan",
            )

    if total_removed == 0:
        cprint(
            "[Info] no stochastic augmentation modules found for "
            "visualization.",
            "cyan",
        )
    else:
        cprint(
            f"[Info] total stochastic augmentation modules removed: "
            f"{total_removed}",
            "cyan",
        )


def parse_highlight_specs(specs: List[str], traj_len: int) -> List[Dict]:
    highlights = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 2 or len(parts) > 4:
            raise ValueError(
                f"Invalid highlight '{spec}'. Expected "
                "start:end[:color[:label]]."
            )

        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid highlight '{spec}': start/end must be integers."
            ) from exc

        if start < 0 or end < 0:
            raise ValueError(
                f"Invalid highlight '{spec}': start/end must be non-negative."
            )
        if start > end:
            start, end = end, start

        color = parts[2] if len(parts) >= 3 and parts[2] else "red"
        label = parts[3] if len(parts) >= 4 and parts[3] else None
        if start >= traj_len:
            cprint(
                f"[Warn] skip highlight '{spec}' because it is outside "
                f"trajectory length {traj_len}.",
                "yellow",
            )
            continue

        highlights.append(
            {
                "start": start,
                "end": min(traj_len - 1, end),
                "color": color,
                "label": label,
            }
        )
    return highlights


def get_visual_key_map(obs_meta) -> Dict[str, str]:
    """Map critic observation names to dataset visual-array names."""
    visual_key_map = {}
    for obs_key, attr in obs_meta.items():
        if str(attr.get("type", "low_dim")) != "rgb":
            continue
        if obs_key == "image":
            visual_key_map[obs_key] = "img"
        elif obs_key == "wrist_img":
            visual_key_map[obs_key] = "wrist_img"
        else:
            raise ValueError(
                f"Unsupported critic RGB observation key: {obs_key}"
            )
    return visual_key_map


def _read_dataset_slice(
    replay_buffer: ReplayBuffer,
    visual_readers: Dict,
    data_key: str,
    start: int,
    stop: int,
) -> np.ndarray:
    if data_key in visual_readers:
        indices = np.arange(start, stop, dtype=np.int64)
        return visual_readers[data_key].read_indices(indices)
    if data_key not in replay_buffer:
        raise KeyError(f"dataset is missing data/{data_key}")
    return np.asarray(replay_buffer[data_key][start:stop])


def read_observation_chunk(
    replay_buffer: ReplayBuffer,
    visual_readers: Dict,
    obs_meta,
    visual_key_map: Dict[str, str],
    start: int,
    stop: int,
) -> Dict[str, np.ndarray]:
    obs_arrays = {}
    expected_len = stop - start

    for obs_key, attr in obs_meta.items():
        if obs_key == "agent_pos":
            if "agent_pos" in replay_buffer:
                array = _read_dataset_slice(
                    replay_buffer,
                    visual_readers,
                    "agent_pos",
                    start,
                    stop,
                )
            elif "state" in replay_buffer:
                state = _read_dataset_slice(
                    replay_buffer,
                    visual_readers,
                    "state",
                    start,
                    stop,
                )
                agent_pos_dim = int(attr.shape[0])
                if state.shape[-1] < agent_pos_dim:
                    raise ValueError(
                        f"state dim {state.shape[-1]} < required agent_pos "
                        f"dim {agent_pos_dim}"
                    )
                array = state[..., :agent_pos_dim]
            else:
                raise KeyError(
                    "Cannot build agent_pos: dataset has neither data/agent_pos "
                    "nor data/state."
                )
        elif obs_key in visual_key_map:
            data_key = visual_key_map[obs_key]
            if (
                data_key == "img"
                and data_key not in visual_readers
                and data_key not in replay_buffer
                and "image" in replay_buffer
            ):
                data_key = "image"
            array = _read_dataset_slice(
                replay_buffer,
                visual_readers,
                data_key,
                start,
                stop,
            )
        else:
            array = _read_dataset_slice(
                replay_buffer,
                visual_readers,
                obs_key,
                start,
                stop,
            )

        array = np.asarray(array, dtype=np.float32)
        if array.shape[0] != expected_len:
            raise ValueError(
                f"Observation '{obs_key}' has {array.shape[0]} frames in "
                f"slice [{start}, {stop}), expected {expected_len}."
            )
        obs_arrays[obs_key] = array

    return obs_arrays


def compute_values(
    critic,
    replay_buffer: ReplayBuffer,
    visual_readers: Dict,
    obs_meta,
    visual_key_map: Dict[str, str],
    episode_start: int,
    episode_stop: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    value_chunks = []
    with torch.inference_mode():
        for start in range(episode_start, episode_stop, chunk_size):
            stop = min(start + chunk_size, episode_stop)
            obs_arrays = read_observation_chunk(
                replay_buffer=replay_buffer,
                visual_readers=visual_readers,
                obs_meta=obs_meta,
                visual_key_map=visual_key_map,
                start=start,
                stop=stop,
            )
            obs_dict = {
                key: torch.from_numpy(array).unsqueeze(0).to(device)
                for key, array in obs_arrays.items()
            }

            pred_value = critic(obs_dict)  # [1, t, 1]
            value_chunks.append(pred_value[0, :, 0].detach().cpu())

            del obs_dict
            del pred_value

    return torch.cat(value_chunks, dim=0).numpy()


def plot_value_curve(
    args,
    values: np.ndarray,
    highlights: List[Dict],
):
    traj_len = values.shape[0]
    frame_ids = np.arange(traj_len)
    if args.x_as_frame:
        x = frame_ids
        x_label = "Frame"
    else:
        x = frame_ids / float(args.fps)
        x_label = "Time (s)"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x,
        values,
        color=args.line_color,
        linewidth=args.line_width,
        alpha=args.line_alpha,
    )

    used_labels = set()
    for highlight in highlights:
        start = highlight["start"]
        end = highlight["end"]
        color = highlight["color"]
        label = highlight["label"]

        span_label = None
        if args.show_legend and label and label not in used_labels:
            span_label = label
            used_labels.add(label)
        ax.axvspan(
            x[start],
            x[end],
            color=color,
            alpha=args.highlight_alpha,
            label=span_label,
        )

        if args.recolor_highlight_segment:
            mask = (frame_ids >= start) & (frame_ids <= end)
            if np.sum(mask) >= 2:
                ax.plot(
                    x[mask],
                    values[mask],
                    color=color,
                    linewidth=args.line_width + 0.6,
                )
            else:
                ax.scatter(x[mask], values[mask], color=color, s=18)

    ax.set_xlabel(x_label, fontsize=args.label_fontsize)
    ax.set_ylabel("Value", fontsize=args.label_fontsize)
    ax.tick_params(axis="both", labelsize=args.tick_fontsize)
    ax.grid(alpha=0.3)

    if args.title:
        ax.set_title(args.title)
    if args.show_legend:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {args.chunk_size}")
    if args.end_frame is not None and args.end_frame < 0:
        raise ValueError(
            f"end_frame must be non-negative, got {args.end_frame}"
        )
    if not args.x_as_frame and args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    ckpt_path = pathlib.Path(args.critic_ckpt).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"critic checkpoint not found: {ckpt_path}")

    workspace = load_workspace_from_ckpt(ckpt_path)
    cfg = workspace.cfg

    device = resolve_device(args.device, str(cfg.training.device))
    use_ema = bool(cfg.training.use_ema)
    critic = (
        workspace.ema_model
        if use_ema and workspace.ema_model is not None
        else workspace.model
    )
    critic.to(device)
    critic.eval()
    if args.enable_stochastic_aug:
        cprint(
            "[Info] keep stochastic augmentation enabled for visualization.",
            "cyan",
        )
    else:
        disable_stochastic_aug_for_visualization(critic)

    obs_meta = cfg.shape_meta.obs
    visual_key_map = get_visual_key_map(obs_meta)
    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else str(cfg.task.dataset.zarr_path)
    )
    storage_mode, replay_zarr_path, visual_readers = resolve_dataset_storage(
        zarr_path=dataset_path,
        storage_mode=args.storage_mode,
        visual_keys=list(visual_key_map.values()),
    )

    try:
        replay_buffer = ReplayBuffer.create_from_path(replay_zarr_path)
        validate_visual_lengths(visual_readers, int(replay_buffer.n_steps))
        cprint(
            f"[Info] dataset storage: {storage_mode} ({dataset_path})",
            "cyan",
        )

        n_episodes = replay_buffer.n_episodes
        if n_episodes <= 0:
            raise ValueError(f"empty dataset: {dataset_path}")
        if args.traj_idx < 0 or args.traj_idx >= n_episodes:
            raise ValueError(
                f"traj_idx {args.traj_idx} is out of bound for dataset with "
                f"{n_episodes} trajectories (max id: {n_episodes - 1})."
            )

        episode_ends = np.asarray(
            replay_buffer.episode_ends[:],
            dtype=np.int64,
        )
        episode_start = (
            0
            if args.traj_idx == 0
            else int(episode_ends[args.traj_idx - 1])
        )
        episode_stop = int(episode_ends[args.traj_idx])
        if episode_stop <= episode_start:
            raise ValueError(
                f"trajectory {args.traj_idx} is empty: "
                f"[{episode_start}, {episode_stop})"
            )
        original_traj_len = episode_stop - episode_start
        if args.end_frame is not None:
            requested_stop = episode_start + args.end_frame + 1
            if requested_stop < episode_stop:
                episode_stop = requested_stop
            elif args.end_frame >= original_traj_len:
                cprint(
                    f"[Warn] end_frame {args.end_frame} is outside "
                    f"trajectory length {original_traj_len}; use the full "
                    "trajectory.",
                    "yellow",
                )

        values = compute_values(
            critic=critic,
            replay_buffer=replay_buffer,
            visual_readers=visual_readers,
            obs_meta=obs_meta,
            visual_key_map=visual_key_map,
            episode_start=episode_start,
            episode_stop=episode_stop,
            chunk_size=args.chunk_size,
            device=device,
        )
    finally:
        for reader in visual_readers.values():
            reader.close()

    highlights = parse_highlight_specs(
        args.highlight,
        traj_len=len(values),
    )
    output_path = pathlib.Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_value_curve(
        args,
        values=values,
        highlights=highlights,
    )
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    cprint(f"[Done] figure saved to: {output_path}", "green")
    if len(values) < original_traj_len:
        cprint(
            f"[Info] trajectory {args.traj_idx}: using {len(values)} / "
            f"{original_traj_len} frames (end frame: {len(values) - 1})",
            "cyan",
        )
    else:
        cprint(
            f"[Info] trajectory {args.traj_idx} length: {len(values)} frames",
            "cyan",
        )


if __name__ == "__main__":
    main()
