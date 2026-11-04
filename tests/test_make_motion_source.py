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
        self.assertEqual((summary["clips"], summary["dropped"]), (3, {"exclude": [], "start_height": [], "clearance": []}))
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
        self.assertEqual(summary["dropped"], {"exclude": ["b_turn"], "start_height": ["c_high"], "clearance": []})
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

    def test_npz_kept_at_50hz_and_checked(self):
        npz = self.root / "npz"
        npz.mkdir()
        p, quat, j = (x.numpy() for x in load_csv(str(self.csv / "a_walk.csv")))
        arrays = {"root_pos": p, "root_quat_wxyz": quat, "joint_pos": j}
        np.savez(npz / "dance.npz", fps=np.float32(50), joint_names=np.array(G1_JOINT_NAMES), **arrays)
        out = self.root / "npz.h5"
        self.assertEqual(run("--npz_dir", npz, "--out", out)["clips"], 1)
        ds = MergedG1Dataset(out)
        self.assertEqual([c.name for c in ds.clips], ["dance"])
        for got, want in zip(next(ds.iter_clips()), (p, quat, j)):
            np.testing.assert_array_equal(got, want)
        for bad in ({"fps": np.float32(30)}, {"joint_names": np.array(G1_JOINT_NAMES[::-1])}):
            np.savez(npz / "dance.npz", **arrays, **bad)
            with self.assertRaises(SystemExit):
                run("--npz_dir", npz, "--out", self.root / "bad.h5")
        self.assertFalse((self.root / "bad.h5.partial").exists())

    def write_shards(self, tamper=False):
        """两个分片、三段动作的数据集目录；窗口用的clip_*数组故意与源片段不同，只有source_clip_*才对。"""
        root = self.root / "shards"
        root.mkdir()
        names = ("a_walk", "b_turn", "c_high")
        arrays = [tuple(x.numpy() for x in load_csv(str(self.csv / f"{n}.csv"))) for n in names]
        lines = ["relative_csv,h5,clip_index,frame_start,frames,padding_frames"]
        for shard, members in (("shard_000.h5", [0, 1]), ("shard_001.h5", [2])):
            lengths = np.array([len(arrays[m][0]) for m in members], dtype=np.int64)
            starts = np.cumsum(lengths) - lengths
            with h5py.File(root / shard, "w") as f:
                f["source_clip_starts"], f["source_clip_lengths"] = starts, lengths
                f["source_clip_fps"] = np.full(len(lengths), 50, dtype=np.int64)
                f["clip_starts"], f["clip_lengths"] = np.array([0, 1]), np.array([1, 1])
                for key, column in (("root_pos", 0), ("root_quat_w", 1), ("joint_pos", 2)):
                    f[key] = np.concatenate([arrays[m][column] for m in members])
            for local, m in enumerate(members):
                start = int(starts[local]) + (1 if tamper and m == 1 else 0)
                lines.append(f"sub/{names[m]}.csv,{shard},{local},{start},{lengths[local]},0")
        (root / "clips.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return root, arrays

    def test_shard_dir_merges_source_clips_with_names_from_the_table(self):
        root, arrays = self.write_shards()
        out = self.root / "merged.h5"
        self.assertEqual(run("--shard_dir", root, "--out", out)["clips"], 3)
        ds = MergedG1Dataset(out)
        self.assertEqual([c.name for c in ds.clips], ["sub/a_walk.csv", "sub/b_turn.csv", "sub/c_high.csv"])
        for got, want in zip(ds.iter_clips(), arrays):
            for g, w in zip(got, want):
                np.testing.assert_array_equal(g, w)

    def test_shard_dir_filters_like_other_inputs(self):
        root, _ = self.write_shards()
        out = self.root / "merged.h5"
        summary = run("--shard_dir", root, "--out", out, "--max_start_height", 1.5, "--exclude", "b_turn")
        self.assertEqual(summary["clips"], 1)
        self.assertEqual(summary["dropped"], {"exclude": ["sub/b_turn.csv"], "start_height": ["sub/c_high.csv"], "clearance": []})
        with h5py.File(out, "r") as f:
            self.assertEqual(json.loads(f.attrs["made_by"])["input"]["shard_dir"], str(root))

    def test_shard_dir_refuses_a_table_that_disagrees_with_the_shard(self):
        root, _ = self.write_shards(tamper=True)
        with self.assertRaises(SystemExit):
            run("--shard_dir", root, "--out", self.root / "bad.h5")
        self.assertFalse((self.root / "bad.h5.partial").exists())

    def test_max_clearance_drops_only_clips_floating_for_the_whole_clip(self):
        # 零关节角时最低身体比根低约0.75 m；grounded全段贴地，airborne中途离地但有帧贴地，floating整段离地0.3 m
        frames = 30
        lift = {"grounded": np.zeros(frames), "airborne": np.r_[np.zeros(5), np.full(frames - 5, 0.5)], "floating": np.full(frames, 0.3)}
        folder = self.root / "npz"
        folder.mkdir()
        probe = self.kin.forward(torch.zeros(1, 3), torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29), all_bodies=True)[0]
        floor = -float(probe[..., 2].min()) + 0.04  # 让最低身体离地0.04 m的根高度
        for name, up in lift.items():
            root = np.zeros((frames, 3))
            root[:, 2] = floor + up
            np.savez(folder / f"{name}.npz", root_pos=root, root_quat_wxyz=np.tile([1.0, 0, 0, 0], (frames, 1)), joint_pos=np.zeros((frames, 29)))
        out = self.root / "grounded.h5"
        summary = run("--npz_dir", folder, "--out", out, "--max_clearance", 0.1)
        self.assertEqual(summary["dropped"]["clearance"], ["floating"])
        self.assertEqual([c.name for c in MergedG1Dataset(out).clips], ["airborne", "grounded"])
        with h5py.File(out, "r") as f:
            self.assertEqual(json.loads(f.attrs["made_by"])["max_clearance"], 0.1)

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
