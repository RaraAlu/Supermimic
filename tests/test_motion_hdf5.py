"""CPU checks of whitelist selection, full-source boundaries and reference provenance."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

from supermimic.tasks.tracking.constants import G1_JOINT_NAMES
from supermimic.tasks.tracking.motion import BIN_FRAMES, MIN_TAIL_STARTS, MotionLib, bin_counts, check_reference, resolve_motion_args
from supermimic.tasks.tracking.motion_hdf5 import INDEX_FORMAT, IndexedG1Dataset, file_sha256, partition_clips
from supermimic.utils.kinematics import G1Kinematics


class IndexedMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = G1Kinematics("cpu")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "manifest.json").write_text("{}", encoding="utf-8")
        self.shard = self.source / "shard_000.h5"
        self.lengths = np.array([1030, 51, 4, 90], dtype="int64")
        self.starts = np.cumsum(self.lengths) - self.lengths
        n = int(self.lengths.sum())
        self.p = np.zeros((n, 3), dtype="float32")
        self.p[:, 2] = 0.8
        for i, (start, length) in enumerate(zip(self.starts, self.lengths)):
            self.p[start:start + length, 0] = 100 * i + np.arange(length) * 0.02
        self.quat = np.zeros((n, 4), dtype="float32")
        self.joints = np.zeros((n, 29), dtype="float32")
        for i, (start, length) in enumerate(zip(self.starts, self.lengths)):
            self.quat[start:start + length, 0] = np.cos(i * 0.05)
            self.quat[start:start + length, 3] = np.sin(i * 0.05)
            self.joints[start:start + length] = i * 0.05
        with h5py.File(self.shard, "w") as f:
            f.attrs.update(format="teleopit_motion_hdf5", version=1)
            for key, data in (("root_pos", self.p), ("root_quat_w", self.quat), ("joint_pos", self.joints),
                              ("source_clip_starts", self.starts), ("source_clip_lengths", self.lengths),
                              ("source_clip_fps", np.full(4, 50, dtype="int64"))):
                f[key] = data
            # Deliberately unrelated storage windows must never determine source lengths.
            f["clip_starts"] = [0, 1024]
            f["clip_lengths"] = [1024, 6]
        self.index = self.root / "index.h5"
        self.ids = np.array([2, 0, 1], dtype="int64")
        with h5py.File(self.index, "w") as f:
            f.attrs.update(format=INDEX_FORMAT, complete=True, fps=50, training_ready=False)
            c = f.create_group("clips")
            c["source_relative_path"] = np.array(["short.bvh", "long.bvh", "edge.bvh"], dtype=h5py.string_dtype())
            c["g1_h5"] = np.array([self.shard.name] * 3, dtype=h5py.string_dtype())
            c["g1_source_clip_index"] = self.ids
            c["g1_source_start"] = self.starts[self.ids]
            c["g1_source_frames"] = self.lengths[self.ids]
            c["start"] = [-999, -999, -999]
            c["length"] = [0, 1, 999999]  # Paired lengths are intentionally unusable.
        self.bind()

    def bind(self):
        stat = self.shard.stat()
        run = {"format": INDEX_FORMAT.removesuffix("_index"), "g1_root": str(self.source),
               "g1_joint_names": G1_JOINT_NAMES,
               "g1_manifest_sha256": file_sha256(self.source / "manifest.json"),
               "g1_snapshot": {"shards": {self.shard.name: {
                   "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(self.shard)}}}}
        (self.root / "run.json").write_text(json.dumps(run), encoding="utf-8")

    def test_full_whitelist_order_and_readonly_sources(self):
        before = file_sha256(self.shard), file_sha256(self.index)
        ds = IndexedG1Dataset(self.index, verify_hashes=True)
        self.assertEqual([c.source_clip_index for c in ds.clips], [2, 0, 1])
        self.assertEqual([c.length for c in ds.clips], [4, 1030, 51])
        self.assertEqual(ds.summary()["selected_frames"], 1085)
        self.assertEqual(ds.summary()["shorter_than_33_frames"], 1)
        for i, arrays in zip(self.ids, list(ds.iter_clips())):
            for actual, expected in zip(arrays, (self.p, self.quat, self.joints)):
                np.testing.assert_array_equal(actual, expected[self.starts[i]:self.starts[i] + self.lengths[i]])
        self.assertEqual(before, (file_sha256(self.shard), file_sha256(self.index)))

    def test_paired_windows_are_optional(self):
        with h5py.File(self.index, "r+") as f:
            del f["clips/start"]
            del f["clips/length"]
        self.assertEqual(IndexedG1Dataset(self.index).summary()["selected_frames"], 1085)

    def test_reference_derivatives_bins_and_future_frames(self):
        lib = MotionLib.from_index(self.index, self.kin, "cpu")
        self.assertEqual(lib.clip_len.tolist(), [4, 1030, 51])
        torch.testing.assert_close(lib.root_v[:, 0], torch.ones(1085), atol=5e-4, rtol=0)
        torch.testing.assert_close(lib.root_w, torch.zeros_like(lib.root_w))
        torch.testing.assert_close(lib.qd, torch.zeros_like(lib.qd))
        for key in ("body_p", "body_v", "body_w"):
            self.assertTrue(torch.isfinite(getattr(lib, key)).all())
        body_R = lib.body_R[:]
        self.assertTrue(torch.isfinite(body_R).all())
        torch.testing.assert_close(body_R @ body_R.transpose(-1, -2), torch.eye(3).expand_as(body_R), atol=1e-5, rtol=0)
        # Both sides of the source's 1024-frame storage split remain one time axis.
        f = lib.clip_start[1] + 1023
        self.assertEqual(lib.future_frames(f[None], 3).tolist(), [[1028, 1029, 1030]])
        for clip in range(3):
            end = lib.clip_end[clip]
            self.assertEqual(lib.future_frames(end[None], 5).tolist(), [[int(end)] * 5])
            bins = lib.bin_start[lib.bin_clip == clip]
            self.assertTrue(bool((bins < end).all()))
            # bins partition the clip's start frames [clip_start, clip_end) in order
            starts, ends = bins, lib.bin_end[lib.bin_clip == clip]
            self.assertEqual(int(starts[0]), int(lib.clip_start[clip]))
            self.assertEqual(int(ends[-1]), int(end))
            self.assertTrue(torch.equal(starts[1:], ends[:-1]))
            for b in torch.nonzero(lib.bin_clip == clip).flatten().tolist():
                self.assertTrue(bool((lib.frame_bin[lib.bin_start[b]:lib.bin_end[b]] == b).all()))
        self.assertEqual(int(lib.frame_bin.max()), lib.num_bins - 1)

    def test_explicit_subset_has_distinct_fingerprint(self):
        full = IndexedG1Dataset(self.index)
        subset = IndexedG1Dataset(self.index, limit=1)
        self.assertEqual(len(subset.clips), 1)
        self.assertNotEqual(full.fingerprint, subset.fingerprint)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            check_reference(full.summary(), subset.summary())
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                IndexedG1Dataset(self.index, limit=limit)

    def test_native_rate_rotation_and_joint_velocity(self):
        with h5py.File(self.shard, "r+") as f:
            angle = np.arange(self.lengths[0], dtype="float32") * 0.02
            f["root_quat_w"][:self.lengths[0], 0] = np.cos(angle / 2)
            f["root_quat_w"][:self.lengths[0], 3] = np.sin(angle / 2)
            f["joint_pos"][:self.lengths[0]] = angle[:, None] * 0.1
        self.bind()
        lib = MotionLib.from_index(self.index, self.kin, "cpu")
        s = slice(int(lib.clip_start[1]), int(lib.clip_end[1]) + 1)
        self.assertEqual(len(lib.root_p[s]), 1030)  # Native 50 Hz: no interpolation or frame loss.
        torch.testing.assert_close(lib.root_w[s, 2], torch.ones(1030), atol=8e-5, rtol=0)
        torch.testing.assert_close(lib.qd[s], torch.full((1030, 29), 0.1), atol=8e-6, rtol=0)

    def test_bad_fps_locator_duplicates_and_paths(self):
        for column, value, error in (("g1_source_frames", [4, 1024, 51], "locator mismatch"),
                                     ("g1_source_clip_index", [2, 0, 0], "duplicate"),
                                     ("g1_source_clip_index", [2, 0, 99], "out of range"),
                                     ("g1_h5", ["../outside.h5"] * 3, "unsafe")):
            with self.subTest(column=column, value=value):
                with h5py.File(self.index, "r+") as f:
                    original = f[f"clips/{column}"][:]
                    f[f"clips/{column}"][:] = value
                with self.assertRaisesRegex(ValueError, error):
                    IndexedG1Dataset(self.index)
                with h5py.File(self.index, "r+") as f:
                    f[f"clips/{column}"][:] = original
        with h5py.File(self.shard, "r+") as f:
            f["source_clip_fps"][0] = 30
        self.bind()
        with self.assertRaisesRegex(ValueError, "native 50 Hz"):
            IndexedG1Dataset(self.index)

    def test_nonfinite_and_nonunit_data_rejected(self):
        for key, value, error in (("root_pos", float("nan"), "NaN or Inf"),
                                  ("root_quat_w", 2, "unit length")):
            with self.subTest(key=key):
                with h5py.File(self.shard, "r+") as f:
                    original = f[key][0, 0]
                    f[key][0, 0] = value
                self.bind()
                ds = IndexedG1Dataset(self.index)
                with self.assertRaisesRegex(ValueError, error):
                    list(ds.iter_clips())
                with h5py.File(self.shard, "r+") as f:
                    f[key][0, 0] = original

    def test_binding_change_and_wrong_joint_order_rejected(self):
        ds = IndexedG1Dataset(self.index)
        with h5py.File(self.shard, "r+") as f:
            f["root_pos"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "source changed"):
            list(ds.iter_clips())
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            IndexedG1Dataset(self.index)
        self.bind()
        path = self.root / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["g1_joint_names"] = list(reversed(G1_JOINT_NAMES))
        path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "joint order"):
            IndexedG1Dataset(self.index)

    def test_source_root_relocation_and_csv_regression(self):
        moved = self.root / "moved"
        self.source.rename(moved)
        self.assertEqual(len(IndexedG1Dataset(self.index, moved).clips), 3)
        csv = self.root / "legacy.csv"
        raw = np.zeros((4, 36), dtype="float32")
        raw[:, 0] = np.arange(4) / 30
        raw[:, 6] = 1  # xyzw input
        np.savetxt(csv, raw, delimiter=",")
        lib = MotionLib.from_dir(str(self.root), self.kin, "cpu")
        torch.testing.assert_close(lib.root_p[:, 0], torch.arange(6) * 0.02)
        torch.testing.assert_close(lib.root_R[:], torch.eye(3).expand(6, 3, 3))

    def test_selection_restoration(self):
        args = SimpleNamespace(motion_dir=None, motion_index=None, motion_root=None, motion_limit=None)
        saved = {"motion_index": str(self.index), "motion_limit": 2}
        resolve_motion_args(args, saved)
        self.assertEqual(args.motion_index, str(self.index))
        self.assertEqual(args.motion_limit, 2)
        args = SimpleNamespace(motion_dir=None, motion_index=None, motion_root=None, motion_limit=None)
        resolve_motion_args(args)
        self.assertTrue(Path(args.motion_dir).as_posix().endswith("data/lafan1_g1"))

    def test_shards_cover_whitelist_without_splitting_sources(self):
        full = MotionLib.from_index(self.index, self.kin, "cpu")
        shards = [MotionLib.from_index(self.index, self.kin, "cpu", rank=r, world_size=2) for r in range(2)]
        ids = [i for shard in shards for i in shard.partition["clip_indices"]]
        self.assertEqual(sorted(ids), list(range(3)))
        self.assertEqual(sum(shard.num_frames for shard in shards), full.num_frames)
        self.assertEqual(shards[0].global_num_bins, full.num_bins)
        for shard in shards:
            self.assertEqual(shard.global_description, full.global_description)
            for local, original in enumerate(shard.partition["clip_indices"]):
                self.assertEqual(int(shard.clip_len[local]), int(full.clip_len[original]))
                a, b = int(shard.clip_start[local]), int(full.clip_start[original])
                n = int(shard.clip_len[local])
                torch.testing.assert_close(shard.root_p[a:a+n], full.root_p[b:b+n], rtol=0, atol=0)
        self.assertEqual(shards[0].partition["frames_per_rank"], [1030, 55])
        with self.assertRaisesRegex(ValueError, "per rank"):
            MotionLib.from_index(self.index, self.kin, "cpu", rank=0, world_size=4)

    def test_balancing_is_deterministic_for_different_world_sizes(self):
        lengths = [900, 300, 200, 100, 80, 60, 40, 20, 2]
        for world_size in (1, 2, 3, 8):
            groups = partition_clips(lengths, world_size)
            self.assertEqual(groups, partition_clips(lengths, world_size))
            self.assertEqual(sorted(i for group in groups for i in group), list(range(len(lengths))))
            self.assertTrue(all(groups))
        self.assertEqual(partition_clips(lengths, 1), [list(range(len(lengths)))])


class ResetBinTests(unittest.TestCase):
    def test_short_final_bin_merges_into_previous(self):
        # clip length -> bins; a clip of n frames has n - 1 reset starts
        cases = {2: 1, 26: 1, 51: 1, 52: 1, 75: 1, 76: 2, 101: 2, 304: 6, 326: 7, 1030: 21}
        counts = bin_counts(torch.tensor(list(cases), dtype=torch.long))
        self.assertEqual(counts.tolist(), list(cases.values()))
        self.assertEqual(MIN_TAIL_STARTS, BIN_FRAMES // 2)

    def test_every_bin_of_a_multi_bin_clip_has_enough_starts(self):
        lengths = torch.arange(2, 700)
        counts = bin_counts(lengths)
        starts = lengths - 1
        last = starts - BIN_FRAMES * (counts - 1)  # start frames in each clip's final bin
        multi = counts > 1
        self.assertTrue(bool((last[multi] >= MIN_TAIL_STARTS).all()))
        self.assertTrue(bool((last[multi] < BIN_FRAMES + MIN_TAIL_STARTS).all()))
        self.assertTrue(torch.equal(last[~multi], starts[~multi]))


if __name__ == "__main__":
    unittest.main()
