"""从 teleopit_motion_hdf5 源文件抽取子集，在 CPU 上生成新文件，不修改源文件。

用法：
    python scripts/make_motion_subset.py SOURCE.h5 OUT.h5 NUM_CLIPS SEED               随机抽取完整片段
    python scripts/make_motion_subset.py SOURCE.h5 OUT.h5 NUM_GROUPS SEED --by_group   整组抽取，同组片段不分开

分组（--group_by）：mirror把一段动作和它的镜像（文件名以_M结尾）归为一组；motion（默认）在此基础上再合并不同动作者的同名动作。

输出保留文件属性和片段身份列，使 --motion_source 可直接加载；subset_source_ids 记录源片段编号。输出的数据指纹与源文件不同。"""
import argparse
import json
import re
import sys
from pathlib import PurePosixPath

import h5py
import numpy as np

GROUP_BY = ("mirror", "motion")
MIRROR_SUFFIX = "_M"
ACTOR_SUFFIX = re.compile(r"__A\d+$")
ARRAYS = ("root_pos", "root_quat_w", "joint_pos")
CLIP_COLUMNS = ("source_relative_path", "actor_label", "mirrored")


def group_key(name: str, group_by: str = "motion") -> str:
    """片段名对应的分组键。mirror保留目录，只去掉镜像后缀；motion只看文件名，再去掉动作者后缀。"""
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}")
    path = PurePosixPath(name)
    stem = path.stem.removesuffix(MIRROR_SUFFIX)
    if group_by == "mirror":
        return str(path.with_name(stem))
    return ACTOR_SUFFIX.sub("", stem)


def group_ids(names, group_by: str) -> np.ndarray:
    """把片段名映射为连续的组编号（按分组键排序，与片段顺序无关）。"""
    return np.unique([group_key(str(n), group_by) for n in names], return_inverse=True)[1]


def write_subset(f: h5py.File, dst, ids: np.ndarray) -> np.ndarray:
    """把源文件f中编号为ids（已排序）的完整片段写入dst，返回它们的长度。逐段读取，内存占用约等于输出大小。"""
    starts, lengths = f["source_clip_starts"][:], f["source_clip_lengths"][:]
    arrays = {k: np.concatenate([f[k][starts[i]:starts[i] + lengths[i]] for i in ids]) for k in ARRAYS}
    sub_len = lengths[ids]
    with h5py.File(dst, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        for k, v in arrays.items():
            g.create_dataset(k, data=v.astype(np.float32))
        # 子集帧已重新拼接，必须由选中长度重算连续起点，不能直接沿用源文件的偏移。
        g.create_dataset("source_clip_lengths", data=sub_len)
        g.create_dataset("source_clip_starts", data=np.concatenate(([0], np.cumsum(sub_len)[:-1])).astype(np.int64))
        g.create_dataset("source_clip_fps", data=f["source_clip_fps"][:][ids])
        g.create_dataset("body_names", data=f["body_names"][:])
        if "clips" in f:
            clips = g.create_group("clips")
            for key in CLIP_COLUMNS:
                if key not in f["clips"]:
                    continue
                column = f["clips"][key]
                if column.dtype.kind == "O":
                    clips.create_dataset(key, data=np.array(column.asstr()[:][ids], dtype=object), dtype=h5py.string_dtype())
                else:
                    clips.create_dataset(key, data=column[:][ids])
        g.create_dataset("subset_source_ids", data=ids)
    return sub_len


def read_names(f: h5py.File) -> np.ndarray:
    """读取片段名；分组依赖它，缺失时报错而不是退回按段抽取。"""
    if "clips" not in f or "source_relative_path" not in f["clips"]:
        raise ValueError("grouping needs clips/source_relative_path in the source")
    return f["clips"]["source_relative_path"].asstr()[:]


def subset_main(argv) -> dict:
    """随机抽取片段（默认）或整组抽取（--by_group）。"""
    parser = argparse.ArgumentParser(description="随机抽取子集")
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("num", type=int, help="片段数；--by_group时为组数")
    parser.add_argument("seed", type=int)
    parser.add_argument("--by_group", action="store_true", help="整组抽取，同组片段不分开")
    parser.add_argument("--group_by", choices=GROUP_BY, default="motion")
    args = parser.parse_args(argv)
    with h5py.File(args.source, "r") as f:
        lengths = f["source_clip_lengths"][:]
        rng = np.random.default_rng(args.seed)
        if args.by_group:
            groups = group_ids(read_names(f), args.group_by)
            chosen = rng.choice(groups.max() + 1, args.num, replace=False)
            ids = np.flatnonzero(np.isin(groups, chosen))
        else:
            # 无放回抽取片段并按原编号排序，既固定随机选择，又保持源片段的相对次序。
            ids = np.sort(rng.choice(len(lengths), args.num, replace=False))
        sub_len = write_subset(f, args.out, ids)
    return {"clips": int(len(ids)), "frames": int(sub_len.sum()), "mean_len": float(sub_len.mean()),
            "full_mean_len": float(lengths.mean()), "min_len": int(sub_len.min()), "max_len": int(sub_len.max())}


def main(argv=None):
    """抽取子集，结果打印为JSON。"""
    print(json.dumps(subset_main(sys.argv[1:] if argv is None else list(argv)), ensure_ascii=False))


if __name__ == "__main__":
    main()
