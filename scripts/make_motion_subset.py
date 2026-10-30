"""从 teleopit_motion_hdf5 源文件随机抽取完整片段，在 CPU 上生成子集，不修改源文件。

用法：python scripts/make_motion_subset.py SOURCE.h5 OUT.h5 NUM_CLIPS SEED

保留片段和文件属性，使 --motion_source 可直接加载；subset_source_ids 记录源片段编号。子集的数据指纹与源文件不同。"""
import json
import sys

import h5py
import numpy as np

src, dst, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with h5py.File(src, "r") as f:
    starts, lengths = f["source_clip_starts"][:], f["source_clip_lengths"][:]
    names = f["clips"]["source_relative_path"].asstr()[:] if "source_relative_path" in f["clips"] else None
    # 无放回抽取片段并按原编号排序，既固定随机选择，又保持源片段的相对次序。
    ids = np.sort(np.random.default_rng(seed).choice(len(lengths), n, replace=False))
    arrays = {k: np.concatenate([f[k][starts[i]:starts[i] + lengths[i]] for i in ids]) for k in ("root_pos", "root_quat_w", "joint_pos")}
    with h5py.File(dst, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        for k, v in arrays.items():
            g.create_dataset(k, data=v.astype(np.float32))
        # 子集帧已重新拼接，必须由选中长度重算连续起点，不能直接沿用源文件的偏移。
        sub_len = lengths[ids]
        g.create_dataset("source_clip_lengths", data=sub_len)
        g.create_dataset("source_clip_starts", data=np.concatenate(([0], np.cumsum(sub_len)[:-1])).astype(np.int64))
        g.create_dataset("source_clip_fps", data=f["source_clip_fps"][:][ids])
        g.create_dataset("body_names", data=f["body_names"][:])
        if names is not None:
            g.create_group("clips").create_dataset("source_relative_path", data=np.array(names[ids], dtype=object), dtype=h5py.string_dtype())
        g.create_dataset("subset_source_ids", data=ids)
print(json.dumps({"clips": n, "frames": int(sub_len.sum()), "mean_len": float(sub_len.mean()),
                  "full_mean_len": float(lengths.mean()), "min_len": int(sub_len.min()), "max_len": int(sub_len.max())}))
