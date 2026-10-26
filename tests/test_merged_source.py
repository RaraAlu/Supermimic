"""Tests for reading a single merged teleopit_motion_hdf5 source without an index."""

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from supermimic.tasks.tracking.motion_hdf5 import MergedG1Dataset


def write_source(path: Path, lengths=(10, 6, 4), *, format="teleopit_motion_hdf5",
                 version=1, fps=50, names=True, contiguous=True) -> None:
    frames = sum(lengths)
    starts = np.cumsum((0,) + tuple(lengths[:-1])).astype(np.int64)
    if not contiguous:
        starts = starts + 1
    with h5py.File(path, "w") as out:
        out.attrs["format"] = format
        out.attrs["version"] = version
        out.attrs["fps"] = fps
        out["body_names"] = np.array([f"body_{i}" for i in range(30)], dtype=h5py.string_dtype("utf-8"))
        out["root_pos"] = np.zeros((frames, 3), dtype=np.float32) + np.arange(frames, dtype=np.float32)[:, None]
        out["root_quat_w"] = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1))
        out["joint_pos"] = np.zeros((frames, 29), dtype=np.float32) + np.arange(frames, dtype=np.float32)[:, None]
        out["source_clip_starts"] = starts
        out["source_clip_lengths"] = np.asarray(lengths, dtype=np.int64)
        out["source_clip_fps"] = np.full(len(lengths), 50, dtype=np.int64)
        if names:
            clips = out.create_group("clips")
            clips["source_relative_path"] = np.array([f"path/clip_{i}.bvh" for i in range(len(lengths))],
                                                     dtype=h5py.string_dtype("utf-8"))


class MergedSourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.path = self.root / "merged.h5"
        write_source(self.path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_clips_boundaries_and_names(self):
        dataset = MergedG1Dataset(self.path)
        self.assertEqual([c.length for c in dataset.clips], [10, 6, 4])
        self.assertEqual([c.start for c in dataset.clips], [0, 10, 16])
        self.assertEqual(dataset.clips[1].name, "path/clip_1.bvh")
        self.assertEqual(dataset.whitelist_clips, 3)
        summary = dataset.summary()
        self.assertEqual(summary["selected_frames"], 20)
        self.assertEqual(summary["fps"], 50)
        self.assertEqual(summary["format"], "supermimic_merged_g1_v1")

    def test_iter_clips_matches_source(self):
        dataset = MergedG1Dataset(self.path)
        with h5py.File(self.path, "r") as source:
            expected = [tuple(source[key][c.start:c.start + c.length] for key in
                              ("root_pos", "root_quat_w", "joint_pos")) for c in dataset.clips]
        for got, want in zip(dataset.iter_clips(), expected):
            for a, b in zip(got, want):
                np.testing.assert_array_equal(a, b)

    def test_limit_and_fingerprint_survive_copy(self):
        full = MergedG1Dataset(self.path)
        limited = MergedG1Dataset(self.path, limit=2)
        self.assertEqual(len(limited.clips), 2)
        self.assertNotEqual(limited.fingerprint, full.fingerprint)
        copy = self.root / "copy.h5"
        shutil.copyfile(self.path, copy)
        self.assertEqual(MergedG1Dataset(copy).fingerprint, full.fingerprint)

    def test_names_fallback_when_clips_group_missing(self):
        path = self.root / "noname.h5"
        write_source(path, names=False)
        self.assertEqual(MergedG1Dataset(path).clips[0].name, "clip_000000")

    def test_rejects_wrong_format_or_fps(self):
        bad_format = self.root / "bad_format.h5"
        write_source(bad_format, format="mjbones_human23_g1_nominal_hdf5_v1")
        with self.assertRaises(ValueError):
            MergedG1Dataset(bad_format)
        bad_fps = self.root / "bad_fps.h5"
        write_source(bad_fps, fps=30)
        with self.assertRaises(ValueError):
            MergedG1Dataset(bad_fps)

    def test_rejects_noncontiguous_partition(self):
        path = self.root / "gap.h5"
        write_source(path, contiguous=False)
        with self.assertRaises(ValueError):
            MergedG1Dataset(path)


if __name__ == "__main__":
    unittest.main()


class LoaderWiringTests(unittest.TestCase):
    """Guards the classmethod wiring used by the script entry points."""

    def test_entry_points_are_classmethods(self):
        from supermimic.tasks.tracking.motion import MotionLib

        for name in ("from_dir", "from_index", "from_source"):
            with self.subTest(name=name):
                self.assertIsInstance(MotionLib.__dict__.get(name), (classmethod, staticmethod))
        # from_args forwards to ``cls``, so it must be a real classmethod.
        self.assertIsInstance(MotionLib.__dict__.get("from_args"), classmethod)

    def test_from_args_binds_without_an_instance(self):
        from types import SimpleNamespace

        from supermimic.tasks.tracking.motion import MotionLib

        bound = MotionLib.from_args
        self.assertTrue(callable(bound))
        # A plain function would bind its first argument as ``self`` and expose a
        # three-parameter signature; a classmethod hides ``cls``.
        import inspect
        self.assertEqual(list(inspect.signature(bound).parameters)[:3], ["args", "kin", "device"])
        self.assertFalse(hasattr(SimpleNamespace(), "from_args"))
