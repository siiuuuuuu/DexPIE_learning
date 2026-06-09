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

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.workspace.critic_workspace import CriticWorkspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot critic value curve for one trajectory with optional colored frame ranges."
    )
    parser.add_argument(
        "--critic_ckpt",
        type=str,
        required=True,
        help="Path to critic checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Optional dataset zarr path. If not set, read from checkpoint cfg.",
    )
    parser.add_argument(
        "--traj_idx",
        type=int,
        required=True,
        help="Trajectory index (episode id, 0-based).",
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
        help="If set, recolor curve segments inside highlighted ranges using each highlight color.",
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
        help="Frame rate for frame->time conversion (default: 25Hz).",
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
        help="Show legend box when enabled.",
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
        help="Keep stochastic augmentation during visualization. Disabled by default for stable curves.",
    )
    return parser.parse_args()


def load_workspace_from_ckpt(ckpt_path: pathlib.Path) -> CriticWorkspace:
    try:
        import dill
    except ImportError as exc:
        raise ImportError("dill is required to load .ckpt files. Please install dill first.") from exc

    payload = torch.load(ckpt_path.open("rb"), map_location="cpu", pickle_module=dill)
    workspace = CriticWorkspace(payload["cfg"])
    workspace.load_payload(payload=payload)
    return workspace


def resolve_device(prefer_device: str, cfg_device: str) -> torch.device:
    if prefer_device is not None:
        return torch.device(prefer_device)
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        cprint(
            f"[Warn] cfg requires {cfg_device}, but CUDA is unavailable. Fallback to cpu.",
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
        cprint("[Warn] critic has no value_critic; skip disabling stochastic augmentations.", "yellow")
        return

    key_transform_map = getattr(value_net, "key_transform_map", None)
    if key_transform_map is None:
        cprint("[Warn] value_critic has no key_transform_map; skip disabling stochastic augmentations.", "yellow")
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
                if len(kept_modules) == 0:
                    key_transform_map[key] = nn.Identity()
                else:
                    key_transform_map[key] = nn.Sequential(*kept_modules)
                cprint(
                    f"[Info] key '{key}' remove stochastic aug modules: {removed_names}",
                    "cyan",
                )
        else:
            if _is_stochastic_augment(transform):
                total_removed += 1
                key_transform_map[key] = nn.Identity()
                cprint(
                    f"[Info] key '{key}' replace stochastic aug {transform.__class__.__name__} with Identity.",
                    "cyan",
                )

    if total_removed == 0:
        cprint("[Info] no stochastic augmentation modules found for visualization.", "cyan")
    else:
        cprint(f"[Info] total stochastic augmentation modules removed: {total_removed}", "cyan")


def parse_highlight_specs(specs: List[str], traj_len: int) -> List[Dict]:
    highlights = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 2 or len(parts) > 4:
            raise ValueError(
                f"Invalid highlight '{spec}'. Expected start:end[:color[:label]]."
            )

        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid highlight '{spec}': start/end must be integers.") from exc

        if start < 0 or end < 0:
            raise ValueError(f"Invalid highlight '{spec}': start/end must be non-negative.")

        if start > end:
            start, end = end, start

        color = parts[2] if len(parts) >= 3 and parts[2] else "red"
        label = parts[3] if len(parts) >= 4 and parts[3] else None

        if start >= traj_len or end < 0:
            cprint(
                f"[Warn] skip highlight '{spec}' because it is outside trajectory length {traj_len}.",
                "yellow",
            )
            continue

        start = max(0, start)
        end = min(traj_len - 1, end)
        if start > end:
            continue

        highlights.append(
            {
                "start": start,
                "end": end,
                "color": color,
                "label": label,
            }
        )
    return highlights


def load_obs_arrays(episode: Dict, obs_meta) -> Dict[str, np.ndarray]:
    obs_arrays: Dict[str, np.ndarray] = dict()
    traj_len = None

    state = episode["state"].astype(np.float32) if "state" in episode else None
    agent_pos_dim = int(obs_meta.agent_pos.shape[0]) if "agent_pos" in obs_meta else None

    for key in obs_meta.keys():
        if key == "agent_pos":
            if "agent_pos" in episode:
                arr = episode["agent_pos"].astype(np.float32)
            elif state is not None and agent_pos_dim is not None:
                if state.shape[-1] < agent_pos_dim:
                    raise ValueError(
                        f"state dim {state.shape[-1]} < required agent_pos dim {agent_pos_dim}"
                    )
                arr = state[:, :agent_pos_dim]
            else:
                raise KeyError("Cannot build agent_pos: neither 'agent_pos' nor compatible 'state'.")
        elif key == "image":
            if "img" in episode:
                arr = episode["img"].astype(np.float32)
            elif "image" in episode:
                arr = episode["image"].astype(np.float32)
            else:
                raise KeyError("shape_meta needs image, but dataset has no img/image key.")
        else:
            if key not in episode:
                raise KeyError(f"shape_meta needs '{key}', but dataset has no '{key}' key.")
            arr = episode[key].astype(np.float32)

        if traj_len is None:
            traj_len = int(arr.shape[0])
        elif int(arr.shape[0]) != traj_len:
            raise ValueError(
                f"Inconsistent trajectory length on key '{key}': {arr.shape[0]} vs {traj_len}."
            )

        obs_arrays[key] = arr

    if traj_len is None:
        raise ValueError("No observation keys found from shape_meta.")
    return obs_arrays


def compute_values(critic, obs_arrays: Dict[str, np.ndarray], chunk_size: int, device: torch.device):
    traj_len = next(iter(obs_arrays.values())).shape[0]
    value_chunks = []
    with torch.inference_mode():
        for start in range(0, traj_len, chunk_size):
            end = min(start + chunk_size, traj_len)
            obs_dict = dict()
            for key, arr in obs_arrays.items():
                obs_dict[key] = torch.from_numpy(arr[start:end]).unsqueeze(0).to(device)

            pred_value = critic(obs_dict)  # [1, t, 1]
            value_chunks.append(pred_value[0, :, 0].detach().cpu())

            del obs_dict
            del pred_value

        values = torch.cat(value_chunks, dim=0).numpy()

    return values


def plot_value_curve(args, values: np.ndarray, highlights: List[Dict], traj_idx: int):
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
    for h in highlights:
        start, end = h["start"], h["end"]
        color = h["color"]
        label = h["label"]

        x0 = x[start]
        x1 = x[end]
        span_label = None
        if args.show_legend and label and label not in used_labels:
            span_label = label
            used_labels.add(label)

        ax.axvspan(x0, x1, color=color, alpha=args.highlight_alpha, label=span_label)

        if args.recolor_highlight_segment:
            mask = (frame_ids >= start) & (frame_ids <= end)
            if np.sum(mask) >= 2:
                ax.plot(x[mask], values[mask], color=color, linewidth=args.line_width + 0.6)
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

    ckpt_path = pathlib.Path(args.critic_ckpt).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"critic checkpoint not found: {ckpt_path}")

    workspace = load_workspace_from_ckpt(ckpt_path)
    cfg = workspace.cfg

    device = resolve_device(args.device, str(cfg.training.device))
    use_ema = bool(cfg.training.use_ema)
    critic = workspace.ema_model if (use_ema and workspace.ema_model is not None) else workspace.model
    critic.to(device)
    critic.eval()
    if args.enable_stochastic_aug:
        cprint("[Info] keep stochastic augmentation enabled for visualization.", "cyan")
    else:
        disable_stochastic_aug_for_visualization(critic)

    zarr_path = args.zarr_path if args.zarr_path is not None else str(cfg.task.dataset.zarr_path)
    replay_buffer = ReplayBuffer.create_from_path(zarr_path)
    n_episodes = replay_buffer.n_episodes
    if n_episodes <= 0:
        raise ValueError(f"empty dataset: {zarr_path}")
    if args.traj_idx < 0 or args.traj_idx >= n_episodes:
        raise ValueError(
            f"traj_idx {args.traj_idx} is out of bound for dataset with {n_episodes} trajectories "
            f"(max id: {n_episodes - 1})."
        )

    episode = replay_buffer.get_episode(args.traj_idx, copy=False)
    obs_arrays = load_obs_arrays(episode, cfg.shape_meta.obs)

    chunk_size = max(1, int(args.chunk_size))
    values = compute_values(critic, obs_arrays, chunk_size=chunk_size, device=device)
    highlights = parse_highlight_specs(args.highlight, traj_len=len(values))

    output_path = pathlib.Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_value_curve(args, values=values, highlights=highlights, traj_idx=args.traj_idx)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    cprint(f"[Done] figure saved to: {output_path}", "green")
    cprint(f"[Info] trajectory {args.traj_idx} length: {len(values)} frames", "cyan")


if __name__ == "__main__":
    main()
