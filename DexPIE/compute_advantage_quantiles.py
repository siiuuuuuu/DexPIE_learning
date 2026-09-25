import json
import os
import pathlib
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import hydra
import numpy as np
import torch
import tqdm
import zarr
from omegaconf import OmegaConf, open_dict
from termcolor import cprint

from dexpie.common.memmap_dataset import (
    resolve_dataset_storage,
    validate_visual_lengths,
)
from dexpie.workspace.critic_workspace import CriticWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        cprint(
            f"[Warn] cfg requires {cfg_device}, but CUDA is unavailable. Fallback to cpu.",
            "yellow",
        )
        return torch.device("cpu")
    return torch.device(cfg_device)


def prepare_obs_once(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Keep exactly the same preprocessing as dexpie_workspace.
    nobs = obs_dict.copy()
    image = nobs["image"] / 255.0
    if image.shape[-1] == 3:
        if len(image.shape) == 5:
            image = image.permute(0, 1, 4, 2, 3)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
    nobs["image"] = image

    if "wrist_img" in nobs:
        wrist_img = nobs["wrist_img"] / 255.0
        if wrist_img.shape[-1] == 3:
            if len(wrist_img.shape) == 5:
                wrist_img = wrist_img.permute(0, 1, 4, 2, 3)
            if len(wrist_img.shape) == 4:
                wrist_img = wrist_img.permute(0, 3, 1, 2)
        nobs["wrist_img"] = wrist_img
    return nobs


def load_frozen_value_critic(
    ckpt_path: pathlib.Path, device: torch.device
) -> Tuple[torch.nn.Module, float, bool]:
    critic_workspace: CriticWorkspace = CriticWorkspace.create_from_checkpoint(str(ckpt_path))
    use_ema = bool(critic_workspace.cfg.training.use_ema)
    value_critic = (
        critic_workspace.ema_model
        if (use_ema and critic_workspace.ema_model is not None)
        else critic_workspace.model
    )
    value_critic.to(device)
    value_critic.eval()
    for p in value_critic.parameters():
        p.requires_grad = False

    critic_cfg = critic_workspace.cfg
    max_length = critic_cfg.get("max_length", critic_cfg.task.dataset.max_length)
    return value_critic, float(max_length), use_ema


def parse_top_percentages(cfg: OmegaConf) -> List[float]:
    raw = cfg.get("advantage_top_percentages", [30, 40, 10, 50, 20])
    if isinstance(raw, (int, float)):
        raw = [raw]
    top_percentages = [float(x) for x in raw]
    for p in top_percentages:
        if not (0.0 < p < 100.0):
            raise ValueError(f"Top percentage must be in (0, 100), got: {p}")
    return top_percentages


def get_dataset_flag(cfg: OmegaConf, key: str, default: bool) -> bool:
    value = cfg.task.dataset.get(key, default)
    return bool(value)


def read_visual_batch(
    data_group: zarr.Group,
    visual_readers: dict,
    key: str,
    start: int,
    stop: int,
) -> np.ndarray:
    """Read a contiguous visual batch from legacy Zarr or an NPY memmap."""
    if key in visual_readers:
        indices = np.arange(start, stop, dtype=np.int64)
        return visual_readers[key].read_indices(indices)
    if key not in data_group:
        raise KeyError(f"Dataset config requires `{key}`, but data/{key} is missing.")
    return np.asarray(data_group[key][start:stop])


def collect_values_from_zarr(
    zarr_root: zarr.Group,
    visual_readers: dict,
    value_critic: torch.nn.Module,
    cfg: OmegaConf,
    device: torch.device,
) -> np.ndarray:
    data_group = zarr_root["data"]
    state_arr = data_group["state"]
    n_steps = int(state_arr.shape[0])
    use_img = get_dataset_flag(cfg, "use_img", True)
    use_wrist_img = get_dataset_flag(cfg, "use_wrist_img", False)

    if use_img and "img" not in data_group and "img" not in visual_readers:
        raise KeyError("Dataset config requires `img`, but data/img and img.npy are missing.")
    if (
        use_wrist_img
        and "wrist_img" not in data_group
        and "wrist_img" not in visual_readers
    ):
        raise KeyError(
            "Dataset config requires `wrist_img`, but data/wrist_img and "
            "wrist_img.npy are missing."
        )

    batch_size = int(cfg.get("advantage_eval_batch_size", cfg.dataloader.batch_size))
    values = np.empty((n_steps,), dtype=np.float32)

    for start in tqdm.tqdm(range(0, n_steps, batch_size), desc="Evaluate critic values"):
        stop = min(start + batch_size, n_steps)
        obs = {
            "agent_pos": torch.from_numpy(
                state_arr[start:stop, :6].astype(np.float32)
            ).unsqueeze(1).to(device, non_blocking=True),
        }
        if use_img:
            obs["image"] = torch.from_numpy(
                read_visual_batch(data_group, visual_readers, "img", start, stop)
            ).unsqueeze(1).to(device, non_blocking=True)
        if use_wrist_img:
            obs["wrist_img"] = torch.from_numpy(
                read_visual_batch(
                    data_group,
                    visual_readers,
                    "wrist_img",
                    start,
                    stop,
                )
            ).unsqueeze(1).to(device, non_blocking=True)

        obs_for_value = prepare_obs_once(obs)
        batch_value = value_critic(obs_for_value, obs_preprocessed=True)  # [B,1,1]
        values[start:stop] = batch_value[:, 0, 0].detach().cpu().numpy().astype(np.float32)

        del obs
        del obs_for_value
        del batch_value

    if not np.all(np.isfinite(values)):
        raise RuntimeError("Critic produced non-finite values; refuse to write invalid advantages.")

    return values


def compute_horizon_window_advantages(
    values: np.ndarray,
    episode_ends: np.ndarray,
    horizon: int,
    max_length: float,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got: {horizon}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got: {max_length}")

    advantages = np.zeros_like(values, dtype=np.float32)
    episode_starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)

    for start, end in zip(episode_starts, episode_ends):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        idxs = np.arange(start, end, dtype=np.int64)
        end_idxs = np.minimum(idxs + horizon - 1, end - 1)
        valid_offsets = (end_idxs - idxs).astype(np.float32)
        advantages[idxs] = (
            -(valid_offsets / float(max_length))
            + values[end_idxs]
            - values[idxs]
        ).astype(np.float32)

    return advantages


def write_advantage_to_zarr(
    zarr_root: zarr.Group,
    advantage_key: str,
    advantages: np.ndarray,
    overwrite: bool,
) -> None:
    data_group = zarr_root["data"]
    if advantage_key in data_group:
        if not overwrite:
            raise FileExistsError(
                f"zarr data/{advantage_key} already exists. "
                "Set +overwrite_advantage=True to replace it."
            )
        del data_group[advantage_key]

    chunk_len = min(max(int(data_group["state"].chunks[0]), 1), int(advantages.shape[0]))
    data_group.array(
        name=advantage_key,
        data=advantages.astype(np.float32),
        chunks=(chunk_len,),
        dtype=np.float32,
        compressor=data_group["state"].compressor,
        overwrite=True,
    )


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("dexpie", "config")),
    config_name="DexPIE",
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_path = pathlib.Path(
        hydra.utils.to_absolute_path(str(cfg.task.dataset.zarr_path))
    ).expanduser()
    with open_dict(cfg):
        cfg.task.dataset.zarr_path = str(dataset_path)

    ckpt_cfg_path = cfg.get("value_critic_ckpt_path", None)
    if ckpt_cfg_path is None:
        raise ValueError("Missing `value_critic_ckpt_path` in config.")
    ckpt_path = pathlib.Path(hydra.utils.to_absolute_path(str(ckpt_cfg_path))).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Value critic checkpoint not found: {ckpt_path}")

    device = resolve_device(str(cfg.training.device))
    value_critic, max_length, use_ema = load_frozen_value_critic(ckpt_path, device)

    visual_keys = []
    if get_dataset_flag(cfg, "use_img", True):
        visual_keys.append("img")
    if get_dataset_flag(cfg, "use_wrist_img", False):
        visual_keys.append("wrist_img")
    requested_storage_mode = str(cfg.task.dataset.get("storage_mode", "auto"))
    storage_mode, replay_zarr_path, visual_readers = resolve_dataset_storage(
        zarr_path=str(dataset_path),
        storage_mode=requested_storage_mode,
        visual_keys=visual_keys,
    )

    zarr_root = zarr.open(replay_zarr_path, mode="r+")
    if "data" not in zarr_root or "meta" not in zarr_root:
        raise KeyError(f"Invalid replay dataset: missing data/meta group in {dataset_path}")
    if "episode_ends" not in zarr_root["meta"]:
        raise KeyError(
            f"Invalid replay dataset: missing meta/episode_ends in {dataset_path}"
        )

    episode_ends = zarr_root["meta"]["episode_ends"][:].astype(np.int64)
    n_steps = int(zarr_root["data"]["state"].shape[0])
    validate_visual_lengths(visual_readers, n_steps)
    if len(episode_ends) == 0 or int(episode_ends[-1]) != n_steps:
        raise ValueError(
            f"episode_ends[-1] must equal n_steps. got "
            f"episode_ends[-1]={episode_ends[-1] if len(episode_ends) else None}, n_steps={n_steps}"
        )

    advantage_key = str(cfg.get("advantage_key", "advantage"))
    overwrite_advantage = bool(cfg.get("overwrite_advantage", True))

    cprint(f"[Info] dataset steps: {n_steps}", "cyan")
    cprint(f"[Info] dataset episodes: {len(episode_ends)}", "cyan")
    cprint(f"[Info] dataset storage: {storage_mode} ({dataset_path})", "cyan")
    cprint(f"[Info] critic ckpt: {ckpt_path}", "cyan")
    cprint(f"[Info] use EMA critic: {use_ema}", "cyan")
    cprint(f"[Info] max_length: {max_length}", "cyan")
    cprint(f"[Info] advantage key: data/{advantage_key}", "cyan")

    try:
        with torch.inference_mode():
            values = collect_values_from_zarr(
                zarr_root=zarr_root,
                visual_readers=visual_readers,
                value_critic=value_critic,
                cfg=cfg,
                device=device,
            )
    finally:
        for reader in visual_readers.values():
            reader.close()

    advantages = compute_horizon_window_advantages(
        values=values,
        episode_ends=episode_ends,
        horizon=int(cfg.horizon),
        max_length=max_length,
    )
    write_advantage_to_zarr(
        zarr_root=zarr_root,
        advantage_key=advantage_key,
        advantages=advantages,
        overwrite=overwrite_advantage,
    )

    top_percentages = parse_top_percentages(cfg)
    top_stats = []
    for top_p in top_percentages:
        q = 1.0 - top_p / 100.0
        q_value = float(np.quantile(advantages, q))
        top_stats.append(
            {
                "top_percentage": top_p,
                "quantile": q,
                "advantage_threshold": q_value,
            }
        )

    dataset_out_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    output_filename = str(cfg.get("advantage_quantile_filename", "advantage_quantiles.json"))
    output_path = dataset_out_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_zarr_path": str(dataset_path),
        "dataset_storage_mode": storage_mode,
        "replay_zarr_path": str(replay_zarr_path),
        "value_critic_ckpt_path": str(ckpt_path),
        "critic_use_ema": bool(use_ema),
        "horizon": int(cfg.horizon),
        "max_length": max_length,
        "advantage_method": "horizon_window_nstep",
        "advantage_key": advantage_key,
        "overwrite_advantage": overwrite_advantage,
        "num_samples": int(advantages.shape[0]),
        "num_episodes": int(len(episode_ends)),
        "advantage_quantiles": top_stats,
    }

    label_info = {
        "generated_at": result["generated_at"],
        "dataset_zarr_path": str(dataset_path),
        "dataset_storage_mode": storage_mode,
        "replay_zarr_path": str(replay_zarr_path),
        "value_critic_ckpt_path": str(ckpt_path),
        "critic_use_ema": bool(use_ema),
        "method": "horizon_window_nstep",
        "formula": "advantage[t] = -(end_t - t) / max_length + V[end_t] - V[t]",
        "advantage_key": advantage_key,
        "horizon": int(cfg.horizon),
        "max_length": max_length,
        "num_steps": int(advantages.shape[0]),
        "num_episodes": int(len(episode_ends)),
    }
    zarr_root.attrs["advantage_label_info"] = label_info

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cprint(f"[Done] Saved advantage quantiles to: {output_path}", "green")
    cprint(f"[Done] Wrote zarr data/{advantage_key}, shape={advantages.shape}, dtype=float32", "green")
    for item in top_stats:
        cprint(
            f"top {item['top_percentage']:.0f}% -> q={item['quantile']:.3f}, thr={item['advantage_threshold']:.6f}",
            "green",
        )


if __name__ == "__main__":
    main()
