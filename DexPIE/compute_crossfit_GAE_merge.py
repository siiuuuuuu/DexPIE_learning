#!/usr/bin/env python3
"""Cross-label two replay datasets with two critics and merge the result.

Critic A is declared to have been trained on dataset A, so it labels dataset B.
Critic B labels dataset A.  The two source datasets are always opened read-only;
the result is written atomically as a new memmap-format dataset.
"""

import json
import math
import os
import pathlib
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

if __name__ == "__main__":
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import hydra
import numpy as np
import torch
import zarr
from numpy.lib.format import open_memmap
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from compute_GAE import (
    build_gae_diagnostics,
    compute_window_gae_advantages,
    resolve_window_horizon,
)
from compute_advantage_quantiles import (
    collect_values_from_zarr,
    get_dataset_flag,
    load_frozen_value_critic,
    parse_top_percentages,
    resolve_device,
)
from dexpie.common.memmap_dataset import LazyMemmapArray


OmegaConf.register_new_resolver("eval", eval, replace=True)


MEMMAP_ZARR_DIR = "data.zarr"
MEMMAP_FORMAT_ATTR = "_dexpie_memmap_format"
KNOWN_VISUAL_KEYS = ("img", "wrist_img", "depth")
SEMANTIC_ROOT_ATTRS = (
    "action_index_offset_frames",
    "action_offset_frames",
    "recorded_action_offset_frames",
)


def _json_equal(left, right) -> bool:
    try:
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    except TypeError:
        return left == right


def _common_attrs(left: Mapping, right: Mapping) -> Dict:
    result = {}
    for key in left.keys() & right.keys():
        if _json_equal(left[key], right[key]):
            result[key] = left[key]
    return result


def _array_keys(group: zarr.Group) -> List[str]:
    # Calling array_keys() on some older nested Zarr groups can assert while
    # iterating recursively.  Direct group keys are sufficient here.
    return sorted(key for key in group.keys() if isinstance(group[key], zarr.Array))


@dataclass
class DatasetView:
    path: pathlib.Path
    storage_mode: str
    replay_zarr_path: pathlib.Path
    root: zarr.Group
    external_visuals: Dict[str, LazyMemmapArray]
    episode_ends: np.ndarray
    n_steps: int
    n_episodes: int
    data_keys: Tuple[str, ...]
    visual_keys: Tuple[str, ...]

    def read_data(self, key: str, start: int, stop: int) -> np.ndarray:
        if key in self.external_visuals:
            indices = np.arange(start, stop, dtype=np.int64)
            return self.external_visuals[key].read_indices(indices)
        return np.asarray(self.root["data"][key][start:stop])

    def array_shape(self, key: str) -> Tuple[int, ...]:
        if key in self.external_visuals:
            return tuple(self.external_visuals[key].shape)
        return tuple(self.root["data"][key].shape)

    def array_dtype(self, key: str) -> np.dtype:
        if key in self.external_visuals:
            return np.dtype(self.external_visuals[key].dtype)
        return np.dtype(self.root["data"][key].dtype)

    def close(self) -> None:
        for reader in self.external_visuals.values():
            reader.close()


def open_dataset_readonly(path) -> DatasetView:
    dataset_path = pathlib.Path(os.path.expanduser(str(path))).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")

    converted_path = dataset_path / MEMMAP_ZARR_DIR
    if converted_path.is_dir():
        storage_mode = "memmap"
        replay_zarr_path = converted_path
    else:
        storage_mode = "memory"
        replay_zarr_path = dataset_path

    root = zarr.open(str(replay_zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError(f"dataset is missing data/meta groups: {dataset_path}")
    if "episode_ends" not in root["meta"]:
        raise ValueError(f"dataset is missing meta/episode_ends: {dataset_path}")

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_ends.size == 0:
        raise ValueError(f"dataset has no episodes: {dataset_path}")
    if np.any(np.diff(np.concatenate(([0], episode_ends))) <= 0):
        raise ValueError(f"dataset has non-positive episode lengths: {dataset_path}")
    n_steps = int(episode_ends[-1])

    external_visuals: Dict[str, LazyMemmapArray] = {}
    if storage_mode == "memmap":
        format_info = root.attrs.get(MEMMAP_FORMAT_ATTR, {})
        declared_visuals = format_info.get("visual_keys", []) if isinstance(format_info, dict) else []
        candidates = list(dict.fromkeys([*declared_visuals, *KNOWN_VISUAL_KEYS]))
        for key in candidates:
            npy_path = dataset_path / f"{key}.npy"
            if npy_path.is_file():
                external_visuals[key] = LazyMemmapArray(npy_path)

    zarr_data_keys = set(_array_keys(root["data"]))
    duplicate_keys = zarr_data_keys & external_visuals.keys()
    if duplicate_keys:
        raise ValueError(
            f"dataset stores visual keys in both Zarr and NPY: {sorted(duplicate_keys)}"
        )
    data_keys = tuple(sorted(zarr_data_keys | external_visuals.keys()))
    if "state" not in data_keys:
        raise ValueError(f"dataset is missing data/state: {dataset_path}")

    try:
        for key in data_keys:
            shape = (
                tuple(external_visuals[key].shape)
                if key in external_visuals
                else tuple(root["data"][key].shape)
            )
            if len(shape) < 1 or int(shape[0]) != n_steps:
                raise ValueError(
                    f"{dataset_path} data/{key} shape {shape} does not match "
                    f"the {n_steps} frames in meta/episode_ends"
                )
    except Exception:
        for reader in external_visuals.values():
            reader.close()
        raise

    visual_keys = tuple(
        sorted(
            set(external_visuals.keys())
            | (set(data_keys) & set(KNOWN_VISUAL_KEYS))
        )
    )
    return DatasetView(
        path=dataset_path,
        storage_mode=storage_mode,
        replay_zarr_path=replay_zarr_path,
        root=root,
        external_visuals=external_visuals,
        episode_ends=episode_ends,
        n_steps=n_steps,
        n_episodes=len(episode_ends),
        data_keys=data_keys,
        visual_keys=visual_keys,
    )


def validate_dataset_pair(
    dataset_a: DatasetView,
    dataset_b: DatasetView,
    advantage_key: str,
) -> Tuple[str, ...]:
    if not advantage_key or "/" in advantage_key:
        raise ValueError(f"invalid advantage key: {advantage_key!r}")
    if advantage_key in KNOWN_VISUAL_KEYS:
        raise ValueError(f"advantage key cannot be a visual key: {advantage_key}")

    keys_a = set(dataset_a.data_keys) - {advantage_key}
    keys_b = set(dataset_b.data_keys) - {advantage_key}
    if keys_a != keys_b:
        raise ValueError(
            "dataset data keys differ after excluding the generated advantage key; "
            f"only in A={sorted(keys_a - keys_b)}, only in B={sorted(keys_b - keys_a)}"
        )

    for key in sorted(keys_a):
        shape_a = dataset_a.array_shape(key)
        shape_b = dataset_b.array_shape(key)
        if shape_a[1:] != shape_b[1:]:
            raise ValueError(
                f"data/{key} trailing shapes differ: {shape_a[1:]} != {shape_b[1:]}"
            )
        dtype_a = dataset_a.array_dtype(key)
        dtype_b = dataset_b.array_dtype(key)
        if dtype_a != dtype_b:
            raise ValueError(f"data/{key} dtypes differ: {dtype_a} != {dtype_b}")

    if dataset_a.visual_keys != dataset_b.visual_keys:
        raise ValueError(
            f"visual keys differ: {dataset_a.visual_keys} != {dataset_b.visual_keys}"
        )

    meta_keys_a = set(_array_keys(dataset_a.root["meta"]))
    meta_keys_b = set(_array_keys(dataset_b.root["meta"]))
    if meta_keys_a != meta_keys_b:
        raise ValueError(
            "dataset meta keys differ; "
            f"only in A={sorted(meta_keys_a - meta_keys_b)}, "
            f"only in B={sorted(meta_keys_b - meta_keys_a)}"
        )

    for key in SEMANTIC_ROOT_ATTRS:
        has_a = key in dataset_a.root.attrs
        has_b = key in dataset_b.root.attrs
        if has_a != has_b:
            raise ValueError(f"root attribute {key!r} exists in only one dataset")
        if has_a and not _json_equal(dataset_a.root.attrs[key], dataset_b.root.attrs[key]):
            raise ValueError(
                f"root attribute {key!r} differs: "
                f"{dataset_a.root.attrs[key]!r} != {dataset_b.root.attrs[key]!r}"
            )

    return tuple(sorted(keys_a))


def _validate_meta_array_pair(
    key: str,
    array_a: zarr.Array,
    array_b: zarr.Array,
    n_episodes_a: int,
    n_episodes_b: int,
) -> str:
    if np.dtype(array_a.dtype) != np.dtype(array_b.dtype):
        raise ValueError(
            f"meta/{key} dtypes differ: {array_a.dtype} != {array_b.dtype}"
        )
    is_episode_a = array_a.ndim >= 1 and array_a.shape[0] == n_episodes_a
    is_episode_b = array_b.ndim >= 1 and array_b.shape[0] == n_episodes_b
    if is_episode_a != is_episode_b:
        raise ValueError(f"meta/{key} is episode-aligned in only one dataset")
    if is_episode_a:
        if array_a.shape[1:] != array_b.shape[1:]:
            raise ValueError(
                f"meta/{key} trailing shapes differ: "
                f"{array_a.shape[1:]} != {array_b.shape[1:]}"
            )
        return "episode"
    if array_a.shape != array_b.shape or not np.array_equal(array_a[...], array_b[...]):
        raise ValueError(
            f"meta/{key} is not episode-aligned and differs between datasets"
        )
    return "shared"


def _copy_data_in_batches(
    destination,
    destination_start: int,
    source: DatasetView,
    key: str,
    batch_size: int,
) -> None:
    for start in range(0, source.n_steps, batch_size):
        stop = min(start + batch_size, source.n_steps)
        destination[destination_start + start : destination_start + stop] = (
            source.read_data(key, start, stop)
        )


def _safe_chunks(array: zarr.Array, output_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    chunks = tuple(int(value) for value in array.chunks)
    if output_shape[0] < 1:
        raise ValueError("cannot create an empty output data array")
    return (min(chunks[0], output_shape[0]),) + chunks[1:]


def build_advantage_stats(
    advantages: np.ndarray,
    top_percentages: Sequence[float],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    advantages = np.asarray(advantages, dtype=np.float32)
    if advantages.ndim != 1 or advantages.size == 0:
        raise ValueError(f"advantages must be a non-empty vector, got {advantages.shape}")
    if not np.all(np.isfinite(advantages)):
        raise ValueError("advantages contain non-finite values")
    quantiles = []
    for top_percentage in top_percentages:
        q = 1.0 - float(top_percentage) / 100.0
        quantiles.append(
            {
                "top_percentage": float(top_percentage),
                "quantile": q,
                "advantage_threshold": float(np.quantile(advantages, q)),
            }
        )
    return build_gae_diagnostics(advantages), quantiles


def merge_labeled_datasets_to_memmap(
    dataset_a: DatasetView,
    advantages_a: np.ndarray,
    dataset_b: DatasetView,
    advantages_b: np.ndarray,
    output_path,
    advantage_key: str,
    output_filename: str,
    top_percentages: Sequence[float],
    label_info: Dict,
    copy_batch_size: int = 64,
) -> pathlib.Path:
    if copy_batch_size < 1:
        raise ValueError(f"copy_batch_size must be positive, got {copy_batch_size}")
    output_filename_path = pathlib.Path(output_filename)
    if (
        not output_filename
        or output_filename_path.is_absolute()
        or output_filename_path.name != output_filename
        or output_filename in {".", ".."}
    ):
        raise ValueError(
            f"output_filename must be a plain filename, got: {output_filename!r}"
        )
    data_keys = validate_dataset_pair(dataset_a, dataset_b, advantage_key)

    advantages_a = np.asarray(advantages_a, dtype=np.float32)
    advantages_b = np.asarray(advantages_b, dtype=np.float32)
    if advantages_a.shape != (dataset_a.n_steps,):
        raise ValueError(
            f"dataset A advantages have shape {advantages_a.shape}, "
            f"expected {(dataset_a.n_steps,)}"
        )
    if advantages_b.shape != (dataset_b.n_steps,):
        raise ValueError(
            f"dataset B advantages have shape {advantages_b.shape}, "
            f"expected {(dataset_b.n_steps,)}"
        )
    combined_advantages = np.concatenate((advantages_a, advantages_b))
    global_diagnostics, global_quantiles = build_advantage_stats(
        combined_advantages, top_percentages
    )
    diagnostics_a, _ = build_advantage_stats(advantages_a, top_percentages)
    diagnostics_b, _ = build_advantage_stats(advantages_b, top_percentages)

    output_path = pathlib.Path(os.path.expanduser(str(output_path))).resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output_path}")
    for source in (dataset_a.path, dataset_b.path):
        if output_path == source or source in output_path.parents:
            raise ValueError(f"output cannot be the source or a child of it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-",
            dir=str(output_path.parent),
        )
    )

    total_steps = dataset_a.n_steps + dataset_b.n_steps
    total_episodes = dataset_a.n_episodes + dataset_b.n_episodes
    try:
        output_root = zarr.open(str(temporary_path / MEMMAP_ZARR_DIR), mode="w")
        output_data = output_root.require_group("data")
        output_meta = output_root.require_group("meta")

        root_attrs = _common_attrs(dict(dataset_a.root.attrs), dict(dataset_b.root.attrs))
        root_attrs.pop("advantage_label_info", None)
        root_attrs.pop(MEMMAP_FORMAT_ATTR, None)
        output_root.attrs.update(root_attrs)
        output_root.attrs[MEMMAP_FORMAT_ATTR] = {
            "version": 1,
            "visual_keys": list(dataset_a.visual_keys),
        }
        output_root.attrs["advantage_label_info"] = label_info
        output_data.attrs.update(
            _common_attrs(
                dict(dataset_a.root["data"].attrs),
                dict(dataset_b.root["data"].attrs),
            )
        )
        output_meta.attrs.update(
            _common_attrs(
                dict(dataset_a.root["meta"].attrs),
                dict(dataset_b.root["meta"].attrs),
            )
        )

        merged_ends = np.concatenate(
            (
                dataset_a.episode_ends,
                dataset_b.episode_ends + dataset_a.n_steps,
            )
        ).astype(np.int64)
        output_meta.array(
            "episode_ends",
            data=merged_ends,
            chunks=(min(max(total_episodes, 1), 1024),),
            compressor=None,
            overwrite=True,
        )

        meta_keys = set(_array_keys(dataset_a.root["meta"])) - {"episode_ends"}
        for key in sorted(meta_keys):
            array_a = dataset_a.root["meta"][key]
            array_b = dataset_b.root["meta"][key]
            alignment = _validate_meta_array_pair(
                key,
                array_a,
                array_b,
                dataset_a.n_episodes,
                dataset_b.n_episodes,
            )
            if alignment == "episode":
                values = np.concatenate((array_a[:], array_b[:]), axis=0)
                chunks = (min(max(array_a.chunks[0], 1), total_episodes),) + tuple(
                    array_a.chunks[1:]
                )
            else:
                values = np.asarray(array_a[...])
                chunks = array_a.chunks
            output_array = output_meta.array(
                key,
                data=values,
                chunks=chunks,
                compressor=array_a.compressor,
                overwrite=True,
            )
            output_array.attrs.update(dict(array_a.attrs))

        fold_key = "advantage_crossfit_fold"
        if fold_key in output_meta:
            raise ValueError(f"reserved output meta key already exists: {fold_key}")
        output_meta.array(
            fold_key,
            data=np.concatenate(
                (
                    np.zeros(dataset_a.n_episodes, dtype=np.int8),
                    np.ones(dataset_b.n_episodes, dtype=np.int8),
                )
            ),
            chunks=(min(max(total_episodes, 1), 1024),),
            compressor=None,
            overwrite=True,
        )

        visual_key_set = set(dataset_a.visual_keys)
        for key in data_keys:
            if key in visual_key_set:
                continue
            source_array = dataset_a.root["data"][key]
            output_shape = (total_steps,) + source_array.shape[1:]
            output_array = output_data.empty(
                key,
                shape=output_shape,
                chunks=_safe_chunks(source_array, output_shape),
                dtype=source_array.dtype,
                compressor=source_array.compressor,
                overwrite=True,
            )
            output_array.attrs.update(dict(source_array.attrs))
            _copy_data_in_batches(
                output_array, 0, dataset_a, key, copy_batch_size
            )
            _copy_data_in_batches(
                output_array, dataset_a.n_steps, dataset_b, key, copy_batch_size
            )

        state_array = dataset_a.root["data"]["state"]
        advantage_chunks = (min(max(int(state_array.chunks[0]), 1), total_steps),)
        output_data.array(
            advantage_key,
            data=combined_advantages,
            chunks=advantage_chunks,
            dtype=np.float32,
            compressor=state_array.compressor,
            overwrite=True,
        )

        for key in dataset_a.visual_keys:
            source_shape = dataset_a.array_shape(key)
            destination = open_memmap(
                str(temporary_path / f"{key}.npy"),
                mode="w+",
                dtype=dataset_a.array_dtype(key),
                shape=(total_steps,) + source_shape[1:],
            )
            try:
                _copy_data_in_batches(
                    destination, 0, dataset_a, key, copy_batch_size
                )
                _copy_data_in_batches(
                    destination, dataset_a.n_steps, dataset_b, key, copy_batch_size
                )
                destination.flush()
            finally:
                mmap = getattr(destination, "_mmap", None)
                if mmap is not None:
                    mmap.close()

        result = dict(label_info)
        result.update(
            {
                "dataset_zarr_path": str(output_path),
                "dataset_storage_mode": "memmap",
                "replay_zarr_path": str(output_path / MEMMAP_ZARR_DIR),
                "num_samples": total_steps,
                "num_episodes": total_episodes,
                "partition_diagnostics": {
                    "dataset_a": diagnostics_a,
                    "dataset_b": diagnostics_b,
                },
                "diagnostics": global_diagnostics,
                "advantage_quantiles": global_quantiles,
            }
        )
        with (temporary_path / output_filename).open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        # Reopen the completed temporary output before publishing it.
        validation_root = zarr.open(
            str(temporary_path / MEMMAP_ZARR_DIR), mode="r"
        )
        validation_ends = np.asarray(
            validation_root["meta"]["episode_ends"][:], dtype=np.int64
        )
        if len(validation_ends) != total_episodes or int(validation_ends[-1]) != total_steps:
            raise RuntimeError("merged output has invalid episode boundaries")
        written_advantages = np.asarray(
            validation_root["data"][advantage_key][:], dtype=np.float32
        )
        if not np.array_equal(written_advantages, combined_advantages):
            raise RuntimeError("merged output advantage labels failed validation")
        for key in dataset_a.visual_keys:
            visual = np.load(
                str(temporary_path / f"{key}.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
            try:
                if visual.shape[0] != total_steps:
                    raise RuntimeError(
                        f"merged visual array {key} has invalid shape {visual.shape}"
                    )
            finally:
                mmap = getattr(visual, "_mmap", None)
                if mmap is not None:
                    mmap.close()

        if output_path.exists():
            raise FileExistsError(
                f"output appeared while processing; refusing to overwrite: {output_path}"
            )
        os.replace(str(temporary_path), str(output_path))
    except Exception:
        shutil.rmtree(str(temporary_path), ignore_errors=True)
        raise

    return output_path


def _required_crossfit_path(cfg: DictConfig, key: str) -> pathlib.Path:
    crossfit_cfg = cfg.get("crossfit", None)
    if crossfit_cfg is None or crossfit_cfg.get(key, None) is None:
        raise ValueError(f"missing crossfit.{key}")
    return pathlib.Path(
        hydra.utils.to_absolute_path(str(crossfit_cfg[key]))
    ).expanduser().resolve()


def _release_critic(critic, device: torch.device) -> None:
    del critic
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _evaluate_crossfit_partition(
    dataset: DatasetView,
    checkpoint_path: pathlib.Path,
    cfg: DictConfig,
    device: torch.device,
    window_horizon: int,
    gae_lambda: float,
    gae_gamma: float,
    expected_critic: Optional[Tuple[Dict, np.ndarray, float]] = None,
) -> Tuple[np.ndarray, Dict, np.ndarray, float]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"critic checkpoint not found: {checkpoint_path}")
    critic, max_length, use_ema = load_frozen_value_critic(checkpoint_path, device)
    bin_centers = critic.bin_centers.detach().cpu().numpy().astype(np.float32)
    checkpoint_info = {
        "path": str(checkpoint_path),
        "use_ema": bool(use_ema),
        "max_length": float(max_length),
        "v_min": float(bin_centers[0]),
        "v_max": float(bin_centers[-1]),
        "num_bins": int(len(bin_centers)),
    }
    try:
        if expected_critic is not None:
            expected_info, expected_bins, expected_max_length = expected_critic
            _validate_critic_pair(
                checkpoint_info,
                bin_centers,
                float(max_length),
                expected_info,
                expected_bins,
                expected_max_length,
            )
        with torch.inference_mode():
            values = collect_values_from_zarr(
                zarr_root=dataset.root,
                visual_readers=dataset.external_visuals,
                value_critic=critic,
                cfg=cfg,
                device=device,
            )
    finally:
        _release_critic(critic, device)

    advantages = compute_window_gae_advantages(
        values=values,
        episode_ends=dataset.episode_ends,
        window_horizon=window_horizon,
        max_length=max_length,
        gae_lambda=gae_lambda,
        gae_gamma=gae_gamma,
    )
    return advantages, checkpoint_info, bin_centers, float(max_length)


def _validate_critic_pair(
    info_a: Dict,
    bins_a: np.ndarray,
    max_length_a: float,
    info_b: Dict,
    bins_b: np.ndarray,
    max_length_b: float,
) -> None:
    if not math.isclose(max_length_a, max_length_b, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"critic max_length differs: {max_length_a} != {max_length_b}"
        )
    if bins_a.shape != bins_b.shape or not np.allclose(
        bins_a, bins_b, rtol=0.0, atol=1e-7
    ):
        raise ValueError(
            "critic value discretizations differ: "
            f"A={info_a['num_bins']} bins [{info_a['v_min']}, {info_a['v_max']}], "
            f"B={info_b['num_bins']} bins [{info_b['v_min']}, {info_b['v_max']}]"
        )


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("dexpie", "config")),
    config_name="DexPIE",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_a_path = _required_crossfit_path(cfg, "dataset_a_path")
    dataset_b_path = _required_crossfit_path(cfg, "dataset_b_path")
    critic_a_path = _required_crossfit_path(cfg, "critic_a_ckpt_path")
    critic_b_path = _required_crossfit_path(cfg, "critic_b_ckpt_path")
    output_path = _required_crossfit_path(cfg, "output_dataset_path")

    advantage_key = str(cfg.get("advantage_key", "advantage_gae"))
    output_filename = str(
        cfg.get("advantage_quantile_filename", "advantage_quantiles_gae.json")
    )
    gae_lambda = float(cfg.get("gae_lambda", 0.99))
    gae_gamma = float(cfg.get("gae_gamma", 1.0))
    window_multiplier = float(cfg.get("gae_window_multiplier", 1.2))
    policy_horizon = int(cfg.horizon)
    window_horizon = resolve_window_horizon(policy_horizon, window_multiplier)
    top_percentages = parse_top_percentages(cfg)
    copy_batch_size = int(cfg.get("crossfit_copy_batch_size", 64))
    device = resolve_device(str(cfg.training.device))

    requested_visual_keys = []
    if get_dataset_flag(cfg, "use_img", True):
        requested_visual_keys.append("img")
    if get_dataset_flag(cfg, "use_wrist_img", False):
        requested_visual_keys.append("wrist_img")

    dataset_a = open_dataset_readonly(dataset_a_path)
    dataset_b = open_dataset_readonly(dataset_b_path)
    try:
        validate_dataset_pair(dataset_a, dataset_b, advantage_key)
        for key in requested_visual_keys:
            if key not in dataset_a.data_keys:
                raise KeyError(f"task config requires data/{key}, but it is missing")

        cprint(
            f"[CrossFit] A: {dataset_a.path} "
            f"({dataset_a.n_episodes} episodes, {dataset_a.n_steps} frames)",
            "cyan",
        )
        cprint(
            f"[CrossFit] B: {dataset_b.path} "
            f"({dataset_b.n_episodes} episodes, {dataset_b.n_steps} frames)",
            "cyan",
        )
        cprint(f"[CrossFit] critic A (trained on A): {critic_a_path}", "cyan")
        cprint(f"[CrossFit] critic B (trained on B): {critic_b_path}", "cyan")
        cprint("[CrossFit] labeling A with critic B", "yellow")
        advantages_a, info_b, bins_b, max_length_b = _evaluate_crossfit_partition(
            dataset=dataset_a,
            checkpoint_path=critic_b_path,
            cfg=cfg,
            device=device,
            window_horizon=window_horizon,
            gae_lambda=gae_lambda,
            gae_gamma=gae_gamma,
        )
        cprint("[CrossFit] labeling B with critic A", "yellow")
        advantages_b, info_a, bins_a, max_length_a = _evaluate_crossfit_partition(
            dataset=dataset_b,
            checkpoint_path=critic_a_path,
            cfg=cfg,
            device=device,
            window_horizon=window_horizon,
            gae_lambda=gae_lambda,
            gae_gamma=gae_gamma,
            expected_critic=(info_b, bins_b, max_length_b),
        )
        _validate_critic_pair(
            info_a,
            bins_a,
            max_length_a,
            info_b,
            bins_b,
            max_length_b,
        )

        generated_at = datetime.now().isoformat(timespec="seconds")
        label_info = {
            "generated_at": generated_at,
            "method": "two_way_crossfit_finite_window_gae",
            "formula": (
                "delta[t] = reward[t] + gamma * V[t+1] - V[t]; "
                "advantage[t] = sum_{k=0}^{d-1} "
                "(gamma * lambda)^k * delta[t+k]"
            ),
            "advantage_key": advantage_key,
            "policy_horizon": policy_horizon,
            "gae_window_multiplier": window_multiplier,
            "gae_window_horizon": window_horizon,
            "gae_max_transitions": window_horizon - 1,
            "gae_gamma": gae_gamma,
            "gae_lambda": gae_lambda,
            "max_length": max_length_a,
            "crossfit_partitions": [
                {
                    "partition": "dataset_a",
                    "dataset_path": str(dataset_a.path),
                    "num_steps": dataset_a.n_steps,
                    "num_episodes": dataset_a.n_episodes,
                    "trained_critic": info_a,
                    "labeled_by_critic": info_b,
                },
                {
                    "partition": "dataset_b",
                    "dataset_path": str(dataset_b.path),
                    "num_steps": dataset_b.n_steps,
                    "num_episodes": dataset_b.n_episodes,
                    "trained_critic": info_b,
                    "labeled_by_critic": info_a,
                },
            ],
        }
        merged_path = merge_labeled_datasets_to_memmap(
            dataset_a=dataset_a,
            advantages_a=advantages_a,
            dataset_b=dataset_b,
            advantages_b=advantages_b,
            output_path=output_path,
            advantage_key=advantage_key,
            output_filename=output_filename,
            top_percentages=top_percentages,
            label_info=label_info,
            copy_batch_size=copy_batch_size,
        )
    finally:
        dataset_a.close()
        dataset_b.close()

    cprint(f"[Done] merged cross-fit dataset: {merged_path}", "green")
    cprint(f"[Done] labels: data/{advantage_key}", "green")
    cprint(f"[Done] global quantiles: {merged_path / output_filename}", "green")


if __name__ == "__main__":
    main()
