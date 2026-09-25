import os
from pathlib import Path

import numpy as np


MEMMAP_ZARR_DIR = "data.zarr"
SUPPORTED_STORAGE_MODES = ("auto", "memory", "memmap")


class LazyMemmapArray:
    """Process-local, read-only NPY memmap opened lazily by each worker."""

    def __init__(self, path):
        self.path = str(Path(path))
        self._array = None
        self._pid = None

        probe = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.shape = probe.shape
        self.dtype = probe.dtype
        self._close_array(probe)

    @staticmethod
    def _close_array(array):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _get_array(self):
        pid = os.getpid()
        if self._array is None or self._pid != pid:
            if self._array is not None:
                self._close_array(self._array)
            self._array = np.load(
                self.path, mmap_mode="r", allow_pickle=False
            )
            self._pid = pid
        return self._array

    def read_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(
                f"memmap indices must be one-dimensional, got {indices.shape}"
            )
        if indices.size > 0:
            if indices.min() < 0 or indices.max() >= self.shape[0]:
                raise IndexError(
                    f"indices [{indices.min()}, {indices.max()}] are outside "
                    f"the first dimension of {self.path}: {self.shape[0]}"
                )
        # Return an owned, writable C-contiguous array. This keeps the mapped
        # dataset read-only and makes torch.from_numpy safe in DataLoader workers.
        return np.array(
            self._get_array()[indices], copy=True, order="C"
        )

    def read_index(self, index):
        index = int(index)
        if index < 0 or index >= self.shape[0]:
            raise IndexError(
                f"index {index} is outside the first dimension of "
                f"{self.path}: {self.shape[0]}"
            )
        return np.array(
            self._get_array()[index], copy=True, order="C"
        )

    def close(self):
        if self._array is not None:
            self._close_array(self._array)
            self._array = None
            self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def resolve_dataset_storage(zarr_path, storage_mode, visual_keys):
    """
    Resolve legacy Zarr-in-RAM storage or the large-dataset memmap layout.

    Returns:
        (resolved_mode, nonvisual_zarr_path, visual_readers)
    """
    if storage_mode not in SUPPORTED_STORAGE_MODES:
        raise ValueError(
            f"storage_mode must be one of {SUPPORTED_STORAGE_MODES}, "
            f"got {storage_mode!r}"
        )

    dataset_root = Path(os.path.expanduser(str(zarr_path)))
    converted_zarr_path = dataset_root / MEMMAP_ZARR_DIR
    has_converted_layout = converted_zarr_path.is_dir()

    if storage_mode == "auto":
        resolved_mode = "memmap" if has_converted_layout else "memory"
    else:
        resolved_mode = storage_mode

    if resolved_mode == "memory":
        if has_converted_layout:
            raise ValueError(
                f"{dataset_root} uses the memmap dataset layout. "
                "Use storage_mode='auto' or storage_mode='memmap'."
            )
        return resolved_mode, str(dataset_root), {}

    if not has_converted_layout:
        raise FileNotFoundError(
            f"memmap dataset is missing {converted_zarr_path}"
        )

    readers = {}
    for key in visual_keys:
        npy_path = dataset_root / f"{key}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(
                f"memmap visual array is missing: {npy_path}"
            )
        readers[key] = LazyMemmapArray(npy_path)

    return resolved_mode, str(converted_zarr_path), readers


def validate_visual_lengths(visual_readers, n_steps):
    for key, reader in visual_readers.items():
        if len(reader.shape) < 1 or reader.shape[0] != n_steps:
            raise ValueError(
                f"{key}.npy has shape {reader.shape}, but data.zarr has "
                f"{n_steps} time steps"
            )
