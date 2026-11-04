"""scripts/make_motion_subset.py：抽取结果可直接作--motion_source；整组抽取时同组片段不分开。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_motion_subset as subset  # noqa: E402
from supermimic.tasks.tracking.motion_hdf5 import MergedG1Dataset  # noqa: E402


def clip_table(motions=40, actors=4):
    """每个动作名由多名动作者演出，每段都有镜像；末尾再放一段没有镜像的动作。"""
    rows = []
    for m in range(motions):
        for a in range(actors):
            base = f"bvh/d{m % 3}/motion_{m:03d}__A{a:03d}"
            rows += [(f"{base}.bvh", f"A{a:03d}", False), (f"{base}_M.bvh", f"A{a:03d}", True)]
    rows.append(("bvh/d0/solo_000__A000.bvh", "A000", False))
    return rows


def write_source(path: Path, rows, seed=0) -> None:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(4, 12, len(rows)).astype(np.int64)
    frames = int(lengths.sum())
    with h5py.File(path, "w") as out:
        out.attrs["format"] = "teleopit_motion_hdf5"
        out.attrs["version"] = 1
        out.attrs["fps"] = 50
        out["body_names"] = np.array([f"body_{i}" for i in range(30)], dtype=h5py.string_dtype("utf-8"))
        # 每帧写入全局帧号，便于核对片段内容没有错位
        index = np.arange(frames, dtype=np.float32)[:, None]
        out["root_pos"] = np.tile(index, (1, 3))
        out["root_quat_w"] = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1))
        out["joint_pos"] = np.tile(index, (1, 29))
        out["source_clip_starts"] = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        out["source_clip_lengths"] = lengths
        out["source_clip_fps"] = np.full(len(rows), 50, dtype=np.int64)
        clips = out.create_group("clips")
        string = h5py.string_dtype("utf-8")
        clips.create_dataset("source_relative_path", data=np.array([r[0] for r in rows], dtype=object), dtype=string)
        clips.create_dataset("actor_label", data=np.array([r[1] for r in rows], dtype=object), dtype=string)
        clips["mirrored"] = np.array([r[2] for r in rows], dtype=bool)


def run(*argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        subset.main([str(a) for a in argv])
    return json.loads(out.getvalue())


def read_names(path):
    with h5py.File(path, "r") as f:
        return f["clips"]["source_relative_path"].asstr()[:].tolist()


class GroupKeyTests(unittest.TestCase):
    def test_mirror_group_keeps_directory_and_actor(self):
        self.assertEqual(subset.group_key("bvh/d0/walk_001__A012_M.bvh", "mirror"), "bvh/d0/walk_001__A012")
        self.assertEqual(subset.group_key("bvh/d0/walk_001__A012.bvh", "mirror"), "bvh/d0/walk_001__A012")

    def test_motion_group_merges_mirrors_actors_and_directories(self):
        keys = {subset.group_key(n, "motion") for n in
                ("bvh/d0/walk_001__A012.bvh", "bvh/d0/walk_001__A012_M.bvh", "bvh/d1/walk_001__A013.bvh")}
        self.assertEqual(keys, {"walk_001"})

    def test_take_number_stays_part_of_the_motion(self):
        self.assertNotEqual(subset.group_key("walk_001__A012.bvh", "motion"), subset.group_key("walk_002__A012.bvh", "motion"))

    def test_only_a_trailing_mirror_suffix_counts(self):
        self.assertEqual(subset.group_key("walk_M_slow__A001.bvh", "motion"), "walk_M_slow")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            subset.group_key("walk.bvh", "actor")


class SubsetFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.h5"
        self.rows = clip_table()
        write_source(self.source, self.rows)

    def test_clip_subset_matches_the_documented_selection(self):
        # 已有用法：结果只由NUM_CLIPS和SEED决定，且可复现
        out = self.root / "sub.h5"
        summary = run(self.source, out, 5, 3)
        ids = np.sort(np.random.default_rng(3).choice(len(self.rows), 5, replace=False))
        self.assertEqual(summary["clips"], 5)
        with h5py.File(out, "r") as g, h5py.File(self.source, "r") as f:
            np.testing.assert_array_equal(g["subset_source_ids"][:], ids)
            starts, lengths = f["source_clip_starts"][:], f["source_clip_lengths"][:]
            expected = np.concatenate([f["joint_pos"][starts[i]:starts[i] + lengths[i]] for i in ids])
            np.testing.assert_array_equal(g["joint_pos"][:], expected)
        self.assertEqual(len(MergedG1Dataset(out).clips), 5)

    def test_group_subset_keeps_groups_whole(self):
        out = self.root / "sub.h5"
        run(self.source, out, 6, 1, "--by_group", "--group_by", "mirror")
        names = read_names(out)
        keys = [subset.group_key(n, "mirror") for n in names]
        self.assertEqual(len(set(keys)), 6)
        # 除无镜像的那段外，每组都带着镜像
        for key in set(keys):
            if "solo" not in key:
                self.assertEqual(keys.count(key), 2)
        self.assertEqual(len(MergedG1Dataset(out).clips), len(names))

    def test_default_grouping_keeps_mirrors_and_actors_of_a_motion_together(self):
        out = self.root / "sub.h5"
        run(self.source, out, 5, 2, "--by_group")
        keys = [subset.group_key(n, "motion") for n in read_names(out)]
        self.assertEqual(len(set(keys)), 5)
        for key in set(keys):
            if "solo" not in key:
                self.assertEqual(keys.count(key), 8)  # 四名动作者，各带镜像


if __name__ == "__main__":
    unittest.main()
