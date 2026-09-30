"""Shared I/O and critic inference helpers for offline advantage labeling."""

import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
import zarr
from omegaconf import OmegaConf
from termcolor import cprint

from dexpie.workspace.critic_workspace import CriticWorkspace


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        cprint(
            f"[Warn] cfg requires {cfg_device}, but CUDA is unavailable. Fallback to cpu.",
            "yellow",
        )
        return torch.device("cpu")
    return torch.device(cfg_device)


def prepare_obs_once(
    obs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
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
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, float, bool]:
    critic_workspace: CriticWorkspace = CriticWorkspace.create_from_checkpoint(
        str(ckpt_path)
    )
    use_ema = bool(critic_workspace.cfg.training.use_ema)
    value_critic = (
        critic_workspace.ema_model
        if (use_ema and critic_workspace.ema_model is not None)
        else critic_workspace.model
    )
    value_critic.to(device)
    value_critic.eval()
    for parameter in value_critic.parameters():
        parameter.requires_grad = False

    critic_cfg = critic_workspace.cfg
    max_length = critic_cfg.get("max_length", critic_cfg.task.dataset.max_length)
    return value_critic, float(max_length), use_ema


def parse_top_percentages(cfg: OmegaConf) -> List[float]:
    raw = cfg.get("advantage_top_percentages", [30, 40, 10, 50, 20])
    if isinstance(raw, (int, float)):
        raw = [raw]
    top_percentages = [float(value) for value in raw]
    for percentage in top_percentages:
        if not 0.0 < percentage < 100.0:
            raise ValueError(
                f"Top percentage must be in (0, 100), got: {percentage}"
            )
    return top_percentages


def get_dataset_flag(cfg: OmegaConf, key: str, default: bool) -> bool:
    return bool(cfg.task.dataset.get(key, default))


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
        raise KeyError(
            "Dataset config requires `img`, but data/img and img.npy are missing."
        )
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

    for start in tqdm.tqdm(
        range(0, n_steps, batch_size),
        desc="Evaluate critic values",
    ):
        stop = min(start + batch_size, n_steps)
        obs = {
            "agent_pos": torch.from_numpy(
                state_arr[start:stop, :6].astype(np.float32)
            )
            .unsqueeze(1)
            .to(device, non_blocking=True),
        }
        if use_img:
            obs["image"] = (
                torch.from_numpy(
                    read_visual_batch(
                        data_group,
                        visual_readers,
                        "img",
                        start,
                        stop,
                    )
                )
                .unsqueeze(1)
                .to(device, non_blocking=True)
            )
        if use_wrist_img:
            obs["wrist_img"] = (
                torch.from_numpy(
                    read_visual_batch(
                        data_group,
                        visual_readers,
                        "wrist_img",
                        start,
                        stop,
                    )
                )
                .unsqueeze(1)
                .to(device, non_blocking=True)
            )

        obs_for_value = prepare_obs_once(obs)
        batch_value = value_critic(obs_for_value, obs_preprocessed=True)
        values[start:stop] = (
            batch_value[:, 0, 0].detach().cpu().numpy().astype(np.float32)
        )

        del obs
        del obs_for_value
        del batch_value

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Critic produced non-finite values; refuse to write invalid advantages."
        )
    return values


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

    chunk_len = min(
        max(int(data_group["state"].chunks[0]), 1),
        int(advantages.shape[0]),
    )
    data_group.array(
        name=advantage_key,
        data=advantages.astype(np.float32),
        chunks=(chunk_len,),
        dtype=np.float32,
        compressor=data_group["state"].compressor,
        overwrite=True,
    )
