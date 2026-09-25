import argparse
import os
import pathlib
import random
import sys

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import matplotlib
import numpy as np
import torch
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
        description="Visualize frame-wise critic value on selected trajectories."
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
        help="Dataset storage mode. The default auto-detects converted memmap datasets.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=5,
        help="Number of random trajectories to visualize when episode range is not specified.",
    )
    parser.add_argument(
        "--episode_start",
        type=int,
        default=None,
        help="Inclusive start episode id (0-based).",
    )
    parser.add_argument(
        "--episode_end",
        type=int,
        default=None,
        help="Inclusive end episode id (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode sampling.",
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
        default="visualizations/critic_value_random_trajectories.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Max frames per forward pass to avoid CUDA OOM.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure dpi.",
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


def read_visual_frames(
    replay_buffer: ReplayBuffer,
    visual_readers: dict,
    key: str,
    start: int,
    stop: int,
) -> np.ndarray:
    """Read a contiguous visual slice from legacy Zarr or an NPY memmap."""
    if key in visual_readers:
        indices = np.arange(start, stop, dtype=np.int64)
        return visual_readers[key].read_indices(indices)
    if key not in replay_buffer:
        raise KeyError(f"dataset is missing visual key data/{key}")
    return np.asarray(replay_buffer[key][start:stop])


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

    obs_meta = cfg.shape_meta.obs
    use_image = "image" in obs_meta
    use_wrist = "wrist_img" in obs_meta
    if not use_image:
        raise ValueError("critic shape_meta does not contain image input.")

    visual_keys = ["img"]
    if use_wrist:
        visual_keys.append("wrist_img")

    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else str(cfg.task.dataset.zarr_path)
    )
    storage_mode, replay_zarr_path, visual_readers = resolve_dataset_storage(
        zarr_path=dataset_path,
        storage_mode=args.storage_mode,
        visual_keys=visual_keys,
    )
    replay_buffer = ReplayBuffer.create_from_path(replay_zarr_path)
    validate_visual_lengths(visual_readers, int(replay_buffer.n_steps))
    cprint(
        f"[Info] dataset storage: {storage_mode} ({dataset_path})",
        "cyan",
    )

    n_episodes = replay_buffer.n_episodes
    if n_episodes <= 0:
        raise ValueError(f"empty dataset: {dataset_path}")

    use_episode_range = args.episode_start is not None or args.episode_end is not None
    if use_episode_range:
        episode_start = 0 if args.episode_start is None else args.episode_start
        episode_end = (n_episodes - 1) if args.episode_end is None else args.episode_end
        if episode_start < 0 or episode_end < 0:
            raise ValueError("episode_start and episode_end must be non-negative.")
        if episode_start > episode_end:
            raise ValueError(
                f"episode_start ({episode_start}) must be <= episode_end ({episode_end})."
            )
        if episode_start >= n_episodes or episode_end >= n_episodes:
            raise ValueError(
                f"episode range [{episode_start}, {episode_end}] is out of bound for "
                f"dataset with {n_episodes} episodes (max id: {n_episodes - 1})."
            )
        sampled_episode_ids = list(range(episode_start, episode_end + 1))
        cprint(
            f"[Info] using contiguous episodes [{episode_start}, {episode_end}] "
            f"(count={len(sampled_episode_ids)}).",
            "cyan",
        )
    else:
        num_trajectories = min(args.num_trajectories, n_episodes)
        if num_trajectories < args.num_trajectories:
            cprint(
                f"[Info] dataset has only {n_episodes} episodes, using {num_trajectories}.",
                "yellow",
            )

        rng = random.Random(args.seed)
        sampled_episode_ids = rng.sample(range(n_episodes), k=num_trajectories)
        cprint(f"[Info] sampled episodes: {sampled_episode_ids}", "cyan")

    agent_pos_dim = int(obs_meta.agent_pos.shape[0])
    episode_ends = replay_buffer.episode_ends[:].astype(np.int64)

    values_by_episode = []
    with torch.inference_mode():
        for ep_idx in sampled_episode_ids:
            episode_start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            episode_end_idx = int(episode_ends[ep_idx])
            state = np.asarray(
                replay_buffer["state"][episode_start_idx:episode_end_idx],
                dtype=np.float32,
            )
            if state.shape[-1] < agent_pos_dim:
                raise ValueError(
                    f"state dim {state.shape[-1]} < required agent_pos dim {agent_pos_dim}"
                )
            agent_pos = state[:, :agent_pos_dim]

            traj_len = agent_pos.shape[0]
            chunk_size = max(1, int(args.chunk_size))
            value_chunks = []
            for start in range(0, traj_len, chunk_size):
                end = min(start + chunk_size, traj_len)
                global_start = episode_start_idx + start
                global_end = episode_start_idx + end
                img = read_visual_frames(
                    replay_buffer,
                    visual_readers,
                    "img",
                    global_start,
                    global_end,
                ).astype(np.float32, copy=False)
                obs_dict = {
                    "agent_pos": torch.from_numpy(agent_pos[start:end]).unsqueeze(0).to(device),
                    "image": torch.from_numpy(img).unsqueeze(0).to(device),
                }
                if use_wrist:
                    wrist_img = read_visual_frames(
                        replay_buffer,
                        visual_readers,
                        "wrist_img",
                        global_start,
                        global_end,
                    ).astype(np.float32, copy=False)
                    obs_dict["wrist_img"] = (
                        torch.from_numpy(wrist_img).unsqueeze(0).to(device)
                    )

                pred_value = critic(obs_dict)  # [1, t, 1]
                value_chunks.append(pred_value[0, :, 0].detach().cpu())

                del obs_dict
                del pred_value

            values = torch.cat(value_chunks, dim=0).numpy()
            values_by_episode.append((ep_idx, values))

            if device.type == "cuda":
                torch.cuda.empty_cache()

    for reader in visual_readers.values():
        reader.close()

    output_path = pathlib.Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for ep_idx, values in values_by_episode:
        frame_ids = np.arange(values.shape[0])
        plt.plot(frame_ids, values, linewidth=1.8, label=f"episode {ep_idx} (T={len(values)})")
    plt.xlabel("Frame")
    plt.ylabel("Predicted Value")
    if use_episode_range:
        plt.title(
            f"Critic Value Along Frames (Episodes {sampled_episode_ids[0]}-{sampled_episode_ids[-1]})"
        )
    else:
        plt.title("Critic Value Along Frames (Random Trajectories)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()

    cprint(f"[Done] figure saved to: {output_path}", "green")


if __name__ == "__main__":
    main()
