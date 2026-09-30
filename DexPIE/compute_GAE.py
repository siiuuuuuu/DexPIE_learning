import json
import math
import os
import pathlib
import random
import sys
from datetime import datetime
from typing import Dict, Tuple

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from termcolor import cprint

from dexpie.common.advantage_labeling import (
    collect_values_from_zarr,
    get_dataset_flag,
    load_frozen_value_critic,
    parse_top_percentages,
    resolve_device,
    write_advantage_to_zarr,
)
from dexpie.common.memmap_dataset import (
    resolve_dataset_storage,
    validate_visual_lengths,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_window_horizon(policy_horizon: int, window_multiplier: float) -> int:
    """Resolve the number of observation frames in one finite GAE window."""
    if policy_horizon < 1:
        raise ValueError(f"policy horizon must be >= 1, got: {policy_horizon}")
    if not np.isfinite(window_multiplier) or window_multiplier <= 0:
        raise ValueError(
            f"GAE window multiplier must be finite and > 0, got: {window_multiplier}"
        )
    return max(1, int(math.ceil(float(policy_horizon) * window_multiplier)))


def compute_window_gae_advantages(
    values: np.ndarray,
    episode_ends: np.ndarray,
    window_horizon: int,
    max_length: float,
    gae_lambda: float,
    gae_gamma: float = 1.0,
) -> np.ndarray:
    """Compute finite-window GAE without crossing episode boundaries.

    ``window_horizon`` follows the existing dataset convention: it is the
    number of observation frames in the window, so at most
    ``window_horizon - 1`` TD transitions contribute to one label.

    Terminal observations use the critic-predicted value as their bootstrap,
    matching the value target.
    """
    values = np.asarray(values, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    if window_horizon < 1:
        raise ValueError(f"window_horizon must be >= 1, got: {window_horizon}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got: {max_length}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got: {gae_lambda}")
    if not 0.0 <= gae_gamma <= 1.0:
        raise ValueError(f"gae_gamma must be in [0, 1], got: {gae_gamma}")
    if len(episode_ends) == 0:
        if len(values) != 0:
            raise ValueError("Non-empty values require non-empty episode_ends.")
        return np.zeros_like(values, dtype=np.float32)
    if int(episode_ends[-1]) != len(values):
        raise ValueError(
            f"episode_ends[-1] must equal len(values), got "
            f"{episode_ends[-1]} and {len(values)}."
        )

    advantages = np.zeros_like(values, dtype=np.float32)
    episode_starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)
    max_transitions = window_horizon - 1
    trace_decay = float(gae_gamma) * float(gae_lambda)
    trace_weights = np.power(
        trace_decay,
        np.arange(max_transitions, dtype=np.float64),
    )
    progress_reward = -1.0 / float(max_length)

    for start, end in zip(episode_starts, episode_ends):
        start = int(start)
        end = int(end)
        episode_length = end - start
        if episode_length <= 1 or max_transitions == 0:
            continue

        episode_values = values[start:end].astype(np.float64, copy=False)
        deltas = (
            progress_reward
            + float(gae_gamma) * episode_values[1:]
            - episode_values[:-1]
        )

        for local_t in range(episode_length - 1):
            n_transitions = min(max_transitions, episode_length - 1 - local_t)
            if n_transitions <= 0:
                continue
            advantages[start + local_t] = np.float32(
                np.dot(
                    trace_weights[:n_transitions],
                    deltas[local_t : local_t + n_transitions],
                )
            )

    if not np.all(np.isfinite(advantages)):
        raise RuntimeError("Computed non-finite GAE labels; refuse to write them.")
    return advantages


def build_gae_diagnostics(
    advantages: np.ndarray,
) -> Dict[str, float]:
    return {
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
    }


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
    ckpt_path = pathlib.Path(
        hydra.utils.to_absolute_path(str(ckpt_cfg_path))
    ).expanduser()
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
            f"episode_ends[-1]={episode_ends[-1] if len(episode_ends) else None}, "
            f"n_steps={n_steps}"
        )
    policy_horizon = int(cfg.horizon)
    gae_lambda = float(cfg.get("gae_lambda", 0.97))
    gae_gamma = float(cfg.get("gae_gamma", 1.0))
    window_multiplier = float(cfg.get("gae_window_multiplier", 1.0))
    window_horizon = resolve_window_horizon(policy_horizon, window_multiplier)
    max_transitions = window_horizon - 1
    advantage_key = str(cfg.get("advantage_key", "advantage"))
    overwrite_advantage = bool(cfg.get("overwrite_advantage", True))

    cprint(f"[Info] dataset steps: {n_steps}", "cyan")
    cprint(f"[Info] dataset episodes: {len(episode_ends)}", "cyan")
    cprint(f"[Info] dataset storage: {storage_mode} ({dataset_path})", "cyan")
    cprint(f"[Info] critic ckpt: {ckpt_path}", "cyan")
    cprint(f"[Info] use EMA critic: {use_ema}", "cyan")
    cprint(f"[Info] max_length: {max_length}", "cyan")
    cprint(
        f"[Info] GAE gamma={gae_gamma}, lambda={gae_lambda}, "
        f"policy_horizon={policy_horizon}, multiplier={window_multiplier}, "
        f"window_horizon={window_horizon}, max_transitions={max_transitions}",
        "cyan",
    )
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

    advantages = compute_window_gae_advantages(
        values=values,
        episode_ends=episode_ends,
        window_horizon=window_horizon,
        max_length=max_length,
        gae_lambda=gae_lambda,
        gae_gamma=gae_gamma,
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

    diagnostics = build_gae_diagnostics(
        advantages=advantages,
    )
    dataset_out_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    output_filename = str(
        cfg.get("advantage_quantile_filename", "advantage_quantiles.json")
    )
    output_path = dataset_out_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    result = {
        "generated_at": generated_at,
        "dataset_zarr_path": str(dataset_path),
        "dataset_storage_mode": storage_mode,
        "replay_zarr_path": str(replay_zarr_path),
        "value_critic_ckpt_path": str(ckpt_path),
        "critic_use_ema": bool(use_ema),
        "policy_horizon": policy_horizon,
        "gae_window_multiplier": window_multiplier,
        "gae_window_horizon": window_horizon,
        "gae_max_transitions": max_transitions,
        "gae_gamma": gae_gamma,
        "gae_lambda": gae_lambda,
        "max_length": max_length,
        "advantage_method": "finite_window_gae",
        "advantage_key": advantage_key,
        "overwrite_advantage": overwrite_advantage,
        "num_samples": int(advantages.shape[0]),
        "num_episodes": int(len(episode_ends)),
        "diagnostics": diagnostics,
        "advantage_quantiles": top_stats,
    }

    label_info = {
        "generated_at": generated_at,
        "dataset_zarr_path": str(dataset_path),
        "dataset_storage_mode": storage_mode,
        "replay_zarr_path": str(replay_zarr_path),
        "value_critic_ckpt_path": str(ckpt_path),
        "critic_use_ema": bool(use_ema),
        "method": "finite_window_gae",
        "formula": (
            "delta[t] = reward[t] + gamma * V[t+1] - V[t]; "
            "advantage[t] = sum_{k=0}^{d-1} "
            "(gamma * lambda)^k * delta[t+k]"
        ),
        "advantage_key": advantage_key,
        "policy_horizon": policy_horizon,
        "gae_window_multiplier": window_multiplier,
        "gae_window_horizon": window_horizon,
        "gae_max_transitions": max_transitions,
        "gae_gamma": gae_gamma,
        "gae_lambda": gae_lambda,
        "max_length": max_length,
        "num_steps": int(advantages.shape[0]),
        "num_episodes": int(len(episode_ends)),
    }
    zarr_root.attrs["advantage_label_info"] = label_info

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cprint(f"[Done] Saved GAE quantiles to: {output_path}", "green")
    cprint(
        f"[Done] Wrote zarr data/{advantage_key}, "
        f"shape={advantages.shape}, dtype=float32",
        "green",
    )
    for item in top_stats:
        cprint(
            f"top {item['top_percentage']:.0f}% -> "
            f"q={item['quantile']:.3f}, "
            f"thr={item['advantage_threshold']:.6f}",
            "green",
        )


if __name__ == "__main__":
    main()
