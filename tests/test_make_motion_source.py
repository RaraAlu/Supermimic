"""scripts/make_motion_source.py：CSV、索引与合并源三种输入转出的文件可直接作--motion_source，内容与原入口逐位相同。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_motion_source  # noqa: E402
from supermimic.tasks.tracking.constants import G1_JOINT_NAMES  # noqa: E402
from supermimic.tasks.tracking.motion import MotionLib, load_csv  # noqa: E402
from supermimic.tasks.tracking.motion_hdf5 import INDEX_FORMAT, MergedG1Dataset, file_sha256  # noqa: E402
from supermimic.utils.kinematics import G1Kinematics  # noqa: E402


def run(*argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        make_motion_source.main([str(a) for a in argv])
    return json.loads(out.getvalue())


class MakeMotionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = G1Kinematics("cpu")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "csv"
        self.csv.mkdir()
        rng = np.random.default_rng(0)
        for name, frames, height in (("a_walk", 61, 0.8), ("b_turn", 47, 0.75), ("c_high", 40, 2.1)):
            t = np.arange(frames) / 30.0
            data = np.zeros((frames, 36))
            data[:, 0], data[:, 2] = 0.5 * t, height
            yaw = 0.8 * t
            data[:, 5], data[:, 6] = np.sin(yaw / 2), np.cos(yaw / 2)  # xyzw
            data[:, 7:] = 0.3 * np.sin(t[:, None] * 3 + rng.uniform(0, 3, 29))
            np.savetxt(self.csv / f"{name}.csv", data, delimiter=",")

    def test_csv_matches_the_csv_entry_bit_for_bit(self):
        out = self.root / "all.h5"
        summary = run("--csv_dir", self.csv, "--out", out)
        self.assertEqual((summary["clips"], summary["dropped"]), (3, {"exclude": [], "start_height": []}))
        ds = MergedG1Dataset(out)
        self.assertEqual([c.name for c in ds.clips], ["a_walk", "b_turn", "c_high"])
        for clip, arrays, f in zip(ds.clips, ds.iter_clips(), sorted(self.csv.glob("*.csv"))):
            for got, want in zip(arrays, load_csv(str(f))):
                np.testing.assert_array_equal(got, want.numpy())
        a = MotionLib.from_dir(str(self.csv), self.kin, "cpu")
        b = MotionLib.from_source(str(out), self.kin, "cpu")
        self.assertEqual(a.names, b.names)
        for key in ("root_p", "root_quat", "root_v", "root_w", "q", "qd", "body_p", "clip_start", "clip_end", "frame_bin"):
            torch.testing.assert_close(getattr(b, key), getattr(a, key), rtol=0, atol=0)

    def test_filters_and_provenance(self):
        out = self.root / "filtered.h5"
        summary = run("--csv_dir", self.csv, "--out", out, "--exclude", "b_turn", "missing", "--max_start_height", 1.5)
        self.assertEqual(summary["dropped"], {"exclude": ["b_turn"], "start_height": ["c_high"]})
        self.assertEqual(summary["exclude_not_found"], ["missing"])
        with h5py.File(out, "r") as f:
            self.assertEqual(f["clips/source_relative_path"].asstr()[:].tolist(), ["a_walk"])
            self.assertEqual(f["clips/input_index"][:].tolist(), [0])
            made_by = json.loads(f.attrs["made_by"])
        self.assertEqual((made_by["input_clips"], made_by["max_start_height"]), (3, 1.5))
        # 再过滤一个合并源：片段名与数据照搬
        again = self.root / "again.h5"
        run("--motion_source", self.root / "filtered.h5", "--out", again)
        x, y = MergedG1Dataset(out), MergedG1Dataset(again)
        for u, v in zip(next(x.iter_clips()), next(y.iter_clips())):
            np.testing.assert_array_equal(u, v)

    def test_refuses_existing_output_and_empty_result(self):
        out = self.root / "x.h5"
        out.write_bytes(b"")
        with self.assertRaises(SystemExit):
            run("--csv_dir", self.csv, "--out", out)
        with self.assertRaises(SystemExit):
            run("--csv_dir", self.csv, "--out", self.root / "none.h5", "--max_start_height", 0.1)
        self.assertFalse((self.root / "none.h5").exists())
        self.assertFalse((self.root / "none.h5.partial").exists())

    def test_index_selection_in_index_order(self):
        source = self.root / "g1"
        source.mkdir()
        (source / "manifest.json").write_text("{}", encoding="utf-8")
        shard = source / "shard_000.h5"
        lengths = np.array([30, 12, 20], dtype=np.int64)
        n = int(lengths.sum())
        p = np.zeros((n, 3), dtype=np.float32)
        p[:, 0], p[:, 2] = np.arange(n), 0.8
        quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
        with h5py.File(shard, "w") as f:
            f.attrs.update(format="teleopit_motion_hdf5", version=1)
            f["root_pos"], f["root_quat_w"], f["joint_pos"] = p, quat, np.zeros((n, 29), dtype=np.float32)
            f["source_clip_starts"], f["source_clip_lengths"] = np.cumsum(lengths) - lengths, lengths
            f["source_clip_fps"] = np.full(3, 50, dtype=np.int64)
        index = self.root / "index.h5"
        with h5py.File(index, "w") as f:
            f.attrs.update(format=INDEX_FORMAT, complete=True, fps=50)
            c = f.create_group("clips")
            c["source_relative_path"] = np.array(["x/third.bvh", "x/first.bvh"], dtype=h5py.string_dtype())
            c["g1_h5"] = np.array([shard.name] * 2, dtype=h5py.string_dtype())
            c["g1_source_clip_index"] = np.array([2, 0], dtype=np.int64)
        stat = shard.stat()
        (self.root / "run.json").write_text(json.dumps({
            "format": INDEX_FORMAT.removesuffix("_index"), "g1_root": str(source), "g1_joint_names": G1_JOINT_NAMES,
            "g1_manifest_sha256": file_sha256(source / "manifest.json"),
            "g1_snapshot": {"shards": {shard.name: {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                                    "sha256": file_sha256(shard)}}}}), encoding="utf-8")
        out = self.root / "from_index.h5"
        summary = run("--motion_index", index, "--out", out, "--exclude", "first")
        self.assertEqual((summary["clips"], summary["frames"]), (1, 20))
        ds = MergedG1Dataset(out)
        self.assertEqual([c.name for c in ds.clips], ["x/third.bvh"])
        np.testing.assert_array_equal(next(ds.iter_clips())[0], p[42:62])


if __name__ == "__main__":
    unittest.main()
