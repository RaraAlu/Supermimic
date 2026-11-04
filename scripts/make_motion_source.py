"""把动作数据整理成一个合并源HDF5（teleopit_motion_hdf5 v1，50 Hz），训练与评估用--motion_source直接读取。

输入五选一：
    --csv_dir DIR          每个文件一段动作的G1 CSV（与LAFAN1相同的36列：根位置xyz、根四元数xyzw、29个关节角），
                           按--csv_fps重采样到50 Hz，做法与--motion_dir相同
    --npz_dir DIR          每个文件一段50 Hz的G1动作npz（root_pos、root_quat_wxyz、joint_pos，可带fps与joint_names），不重采样
    --shard_dir DIR        分片数据集目录：clips.csv给出片段名和位置，shard_*.h5存源片段，无填充、50 Hz（如Teleopit导出的amass）
    --motion_index INDEX   配对索引选入的源片段（需要同目录的run.json，--motion_root可覆盖源根），按索引行顺序
    --motion_source SRC    另一个合并源文件，用于再过滤
过滤：--exclude按片段名或文件名（不含扩展名）去掉片段；--max_start_height去掉首帧根高度超过该值的片段
（例如从高台起步的动作，平地上无法复现）；--max_clearance去掉全段最低身体（连杆原点，脚放平时约0.04 m）都高于该值的片段
（参考整段悬空，机器人从首帧起自由落体）。不修改输入文件；逐段读写，内存只占一段动作。

在仓库根目录运行，只用CPU：
    python scripts/make_motion_source.py --motion_index paired_50hz/index.h5 --out boneseed.h5 --max_start_height 1.5
    python scripts/make_motion_source.py --csv_dir data/lafan1_g1 --out lafan1.h5
    python scripts/make_motion_source.py --shard_dir datasets/pass_hdf5_source/amass --out amass.h5 --max_clearance 0.1
输出写clips/source_relative_path（片段名）与clips/input_index（在输入中的序号），文件属性made_by记录输入与过滤条件；
最后打印片段数、帧数、时长与去掉的片段。数据指纹随片段选择改变，旧检查点不能直接在新文件上续训或评估。
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.tasks.tracking.motion_hdf5 import ARRAY_SHAPES, IndexedG1Dataset, MergedG1Dataset  # noqa: E402

FPS = 50


def clips_from_args(args):
    """返回 (片段名列表, 逐段产出 (根位置, 根四元数wxyz, 关节角) 的迭代器)。"""
    if args.csv_dir:
        from supermimic.tasks.tracking.motion import load_csv

        files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
        if not files:
            sys.exit(f"no *.csv in {args.csv_dir}")
        return [Path(f).stem for f in files], (tuple(x.numpy() for x in load_csv(f, args.csv_fps)) for f in files)
    if args.npz_dir:
        files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
        if not files:
            sys.exit(f"no *.npz in {args.npz_dir}")
        return [Path(f).stem for f in files], (load_npz(f) for f in files)
    if args.shard_dir:
        return clips_from_shards(args.shard_dir)
    dataset =IndexedG1Dataset(args.motion_index, args.motion_root) if args.motion_index else MergedG1Dataset(args.motion_source)
    return [c.name for c in dataset.clips], dataset.iter_clips()


def clips_from_shards(shard_dir):
    """读分片数据集：片段名和位置来自clips.csv，并与分片自身的source_clip_*核对，对不上就报错而不是读错帧。"""
    root = Path(shard_dir)
    with (root / "clips.csv").open(encoding="utf-8", newline="") as table:
        rows = list(csv.DictReader(table))
    if not rows:
        sys.exit(f"no clips in {root / 'clips.csv'}")
    return [r["relative_csv"] for r in rows], iter_shard_clips(root, rows)


def iter_shard_clips(root, rows):
    """按clips.csv的行序逐段产出；同一分片连续的行只打开一次。"""
    from supermimic.tasks.tracking.motion_hdf5 import validate_pose_arrays

    f, opened = None, None
    try:
        for row in rows:
            if row["h5"] != opened:
                if f is not None:
                    f.close()
                f, opened = h5py.File(root / row["h5"], "r"), row["h5"]
            i, start, n = int(row["clip_index"]), int(row["frame_start"]), int(row["frames"])
            if (int(f["source_clip_starts"][i]), int(f["source_clip_lengths"][i]), int(f["source_clip_fps"][i])) != (start, n, FPS):
                sys.exit(f"{row['relative_csv']}: clips.csv disagrees with source_clip_* in {row['h5']}")
            if int(row.get("padding_frames") or 0):
                sys.exit(f"{row['relative_csv']}: padded clips are not supported")
            arrays = tuple(np.asarray(f[k][start:start + n], dtype=np.float32) for k in ARRAY_SHAPES)
            validate_pose_arrays(*arrays, row["relative_csv"])
            yield arrays
    finally:
        if f is not None:
            f.close()


def load_npz(path):
    """读一段npz动作，返回(根位置, 根四元数wxyz, 关节角)；带fps或joint_names时须为50 Hz与G1_JOINT_NAMES的顺序。"""
    from supermimic.tasks.tracking.constants import G1_JOINT_NAMES
    from supermimic.tasks.tracking.motion_hdf5 import validate_pose_arrays

    with np.load(path, allow_pickle=False) as z:
        if "fps" in z.files and float(z["fps"]) != FPS:
            sys.exit(f"{path}: fps {float(z['fps'])}, expected {FPS} (no resampling for npz)")
        if "joint_names" in z.files and z["joint_names"].tolist() != G1_JOINT_NAMES:
            sys.exit(f"{path}: joint_names differ from G1_JOINT_NAMES")
        arrays = tuple(np.asarray(z[k], dtype=np.float32) for k in ("root_pos", "root_quat_wxyz", "joint_pos"))
    validate_pose_arrays(*arrays, path)
    return arrays


class ClearanceCheck:
    """片段全段的最低身体高度。只在给了--max_clearance时才建，其他用法不必加载运动学模型。"""

    def __init__(self):
        from supermimic.utils.kinematics import G1Kinematics

        self.kin = G1Kinematics("cpu")

    def __call__(self, arrays) -> float:
        import torch

        from supermimic.utils.rotations import quat_to_mat

        p, quat, q = (torch.from_numpy(np.asarray(a, dtype="float32")) for a in arrays)
        zero_v, zero_qd = torch.zeros_like(p), torch.zeros_like(q)
        return float(self.kin.forward(p, quat_to_mat(quat), zero_v, zero_v, q, zero_qd, all_bodies=True)[0][..., 2].min())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv_dir", help="directory of G1 motion CSVs (one clip per file)")
    source.add_argument("--npz_dir", help="directory of 50 Hz G1 motion .npz files (one clip per file)")
    source.add_argument("--shard_dir", help="sharded dataset directory with clips.csv and shard_*.h5 (no padding, 50 Hz)")
    source.add_argument("--motion_index", help="paired index whose selected source clips are merged")
    source.add_argument("--motion_source", help="another merged source to filter")
    parser.add_argument("--motion_root", default=None, help="source root overriding run.json (with --motion_index)")
    parser.add_argument("--csv_fps", type=float, default=30.0, help="frame rate of the CSV files (resampled to 50 Hz)")
    parser.add_argument("--out", required=True, help="output .h5 (must not exist)")
    parser.add_argument("--exclude", nargs="*", default=(), help="clip names or file stems to leave out")
    parser.add_argument("--max_start_height", type=float, default=None, help="leave out clips whose first root height exceeds this (m)")
    parser.add_argument("--max_clearance", type=float, default=None, help="leave out clips whose lowest body stays above this for the whole clip (m)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; choose a new path")
    exclude = set(args.exclude)
    names, arrays_iter = clips_from_args(args)
    tmp = out.with_name(out.name + ".partial")
    try:
        kept, lengths, dropped = write(args, tmp, names, arrays_iter, exclude)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    missing = exclude - set(dropped["exclude"]) - {Path(n).stem for n in dropped["exclude"]}
    frames = int(sum(lengths))
    print(json.dumps({"out": str(out), "clips": len(kept), "frames": frames, "hours": round(frames / FPS / 3600, 3),
                      "dropped": dropped, "exclude_not_found": sorted(missing)}, ensure_ascii=False, indent=1))


def write(args, tmp, names, arrays_iter, exclude):
    """逐段写入tmp，返回保留的片段名、长度与去掉的片段。被过滤的片段也照常读出，读取器在读完最后一段后才检查文件在读取期间是否变化。"""
    kept, lengths, dropped, inputs = [], [], {"exclude": [], "start_height": [], "clearance": []}, []
    clearance = ClearanceCheck() if args.max_clearance is not None else None
    with h5py.File(tmp, "w") as f:
        f.attrs.update(format="teleopit_motion_hdf5", version=1, fps=FPS)
        data = {k: f.create_dataset(k, shape=(0, w), maxshape=(None, w), dtype="float32", chunks=(4096, w))
                for k, w in ARRAY_SHAPES.items()}
        frames = 0
        for i, (arrays, name) in enumerate(zip(arrays_iter, names)):  # 读取器放在前面，zip才会让它跑完收尾检查
            if name in exclude or Path(name).stem in exclude:
                dropped["exclude"].append(name)
                continue
            if args.max_start_height is not None and float(arrays[0][0, 2]) > args.max_start_height:
                dropped["start_height"].append(name)
                continue
            if clearance and clearance(arrays) > args.max_clearance:
                dropped["clearance"].append(name)
                continue
            n = len(arrays[0])
            for key, value in zip(ARRAY_SHAPES, arrays):
                data[key].resize(frames + n, axis=0)
                data[key][frames:] = np.asarray(value, dtype="float32")
            frames += n
            kept.append(name)
            lengths.append(n)
            inputs.append(i)
        if not kept:
            sys.exit("every clip was filtered out")
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        f["source_clip_starts"] = np.cumsum(lengths_arr) - lengths_arr
        f["source_clip_lengths"] = lengths_arr
        f["source_clip_fps"] = np.full(len(lengths_arr), FPS, dtype=np.int64)
        group = f.create_group("clips")
        group["source_relative_path"] = np.array(kept, dtype=h5py.string_dtype("utf-8"))
        group["input_index"] = np.asarray(inputs, dtype=np.int64)
        made_by = {"input": {k: getattr(args, k) for k in ("csv_dir", "csv_fps", "npz_dir", "shard_dir", "motion_index", "motion_root", "motion_source")
                             if getattr(args, k) is not None},
                   "exclude": sorted(exclude), "max_start_height": args.max_start_height, "max_clearance": args.max_clearance,
                   "input_clips": len(names), "dropped": dropped}
        f.attrs["made_by"] = json.dumps(made_by, ensure_ascii=False)
    return kept, lengths, dropped


if __name__ == "__main__":
    main()
