import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from compute_crossfit_GAE_merge import (
    MEMMAP_FORMAT_ATTR,
    merge_labeled_datasets_to_memmap,
    open_dataset_readonly,
)


def _make_legacy_dataset(path, episode_lengths, value_offset):
    n_steps = int(sum(episode_lengths))
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    frame_values = np.arange(n_steps, dtype=np.float32) + value_offset
    data.array(
        "state",
        data=np.stack((frame_values, frame_values + 1), axis=-1),
        chunks=(2, 2),
    )
    data.array(
        "action",
        data=frame_values[:, None],
        chunks=(2, 1),
    )
    data.array(
        "intervention",
        data=(np.arange(n_steps) % 2 == 0),
        chunks=(2,),
    )
    images = np.arange(n_steps * 2 * 2 * 3, dtype=np.uint8).reshape(
        n_steps, 2, 2, 3
    )
    data.array("img", data=images + int(value_offset), chunks=(2, 2, 2, 3))
    data.array(
        "wrist_img",
        data=images + int(value_offset) + 1,
        chunks=(2, 2, 2, 3),
    )
    ends = np.cumsum(episode_lengths, dtype=np.int64)
    meta.array("episode_ends", data=ends, chunks=(len(ends),), compressor=None)
    meta.array(
        "success",
        data=np.arange(len(episode_lengths)) % 2 == 0,
        chunks=(len(episode_lengths),),
        compressor=None,
    )
    root.attrs["action_index_offset_frames"] = 1
    return root


def _make_memmap_copy(source_path, destination_path):
    source = zarr.open(str(source_path), mode="r")
    destination = zarr.open(str(destination_path / "data.zarr"), mode="w")
    output_data = destination.require_group("data")
    output_meta = destination.require_group("meta")
    destination_path.mkdir(parents=True, exist_ok=True)
    for key in source["data"].keys():
        values = np.asarray(source["data"][key][:])
        if key in {"img", "wrist_img"}:
            np.save(destination_path / f"{key}.npy", values)
        else:
            output_data.array(
                key,
                data=values,
                chunks=source["data"][key].chunks,
                compressor=source["data"][key].compressor,
            )
    for key in source["meta"].keys():
        output_meta.array(
            key,
            data=source["meta"][key][:],
            chunks=source["meta"][key].chunks,
            compressor=source["meta"][key].compressor,
        )
    destination.attrs.update(dict(source.attrs))
    destination.attrs[MEMMAP_FORMAT_ATTR] = {
        "version": 1,
        "visual_keys": ["img", "wrist_img"],
    }


class CrossfitGAEMergeTest(unittest.TestCase):
    def test_merges_crossfit_labels_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            path_a = base / "a.zarr"
            path_b_legacy = base / "b.zarr"
            path_b = base / "b_memmap"
            output = base / "merged"
            root_a = _make_legacy_dataset(path_a, [2, 3], 0)
            root_b = _make_legacy_dataset(path_b_legacy, [1, 2], 20)
            state_a = np.asarray(root_a["data"]["state"][:])
            state_b = np.asarray(root_b["data"]["state"][:])
            image_a = np.asarray(root_a["data"]["img"][:])
            image_b = np.asarray(root_b["data"]["img"][:])
            _make_memmap_copy(path_b_legacy, path_b)
            advantages_a = np.linspace(-0.4, 0.0, 5, dtype=np.float32)
            advantages_b = np.linspace(0.1, 0.3, 3, dtype=np.float32)

            dataset_a = open_dataset_readonly(path_a)
            dataset_b = open_dataset_readonly(path_b)
            try:
                merge_labeled_datasets_to_memmap(
                    dataset_a=dataset_a,
                    advantages_a=advantages_a,
                    dataset_b=dataset_b,
                    advantages_b=advantages_b,
                    output_path=output,
                    advantage_key="advantage_gae",
                    output_filename="advantage_quantiles_gae.json",
                    top_percentages=[50],
                    label_info={
                        "generated_at": "test",
                        "method": "two_way_crossfit_finite_window_gae",
                        "advantage_key": "advantage_gae",
                    },
                    copy_batch_size=2,
                )
            finally:
                dataset_a.close()
                dataset_b.close()

            merged = zarr.open(str(output / "data.zarr"), mode="r")
            np.testing.assert_array_equal(
                merged["data"]["state"][:],
                np.concatenate((state_a, state_b)),
            )
            np.testing.assert_allclose(
                merged["data"]["advantage_gae"][:],
                np.concatenate((advantages_a, advantages_b)),
            )
            np.testing.assert_array_equal(
                merged["meta"]["episode_ends"][:],
                np.asarray([2, 5, 6, 8]),
            )
            np.testing.assert_array_equal(
                merged["meta"]["advantage_crossfit_fold"][:],
                np.asarray([0, 0, 1, 1], dtype=np.int8),
            )
            self.assertEqual(
                merged.attrs[MEMMAP_FORMAT_ATTR]["visual_keys"],
                ["img", "wrist_img"],
            )
            np.testing.assert_array_equal(
                np.load(output / "img.npy"),
                np.concatenate((image_a, image_b)),
            )
            with (output / "advantage_quantiles_gae.json").open(
                "r", encoding="utf-8"
            ) as file:
                statistics = json.load(file)
            self.assertEqual(statistics["num_samples"], 8)
            self.assertEqual(statistics["num_episodes"], 4)
            self.assertAlmostEqual(
                statistics["advantage_quantiles"][0]["advantage_threshold"],
                float(np.quantile(np.concatenate((advantages_a, advantages_b)), 0.5)),
            )

            self.assertNotIn("advantage_gae", zarr.open(str(path_a), mode="r")["data"])
            self.assertNotIn(
                "advantage_gae",
                zarr.open(str(path_b / "data.zarr"), mode="r")["data"],
            )

    def test_rejects_mismatched_data_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            path_a = base / "a.zarr"
            path_b = base / "b.zarr"
            _make_legacy_dataset(path_a, [2], 0)
            root_b = _make_legacy_dataset(path_b, [2], 10)
            del root_b["data"]["intervention"]
            dataset_a = open_dataset_readonly(path_a)
            dataset_b = open_dataset_readonly(path_b)
            try:
                with self.assertRaisesRegex(ValueError, "data keys differ"):
                    merge_labeled_datasets_to_memmap(
                        dataset_a,
                        np.zeros(2, dtype=np.float32),
                        dataset_b,
                        np.zeros(2, dtype=np.float32),
                        base / "merged",
                        "advantage_gae",
                        "stats.json",
                        [50],
                        {},
                    )
            finally:
                dataset_a.close()
                dataset_b.close()


if __name__ == "__main__":
    unittest.main()
