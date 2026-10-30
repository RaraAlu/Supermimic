# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""为 mjlab 内置 G1 提供完整片段的 50 Hz 参考数据、导数和正向运动学。"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from supermimic.tasks.tracking.motion_hdf5 import NUM_J, reference_bytes_per_frame
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.rotations import mat_to_6d, mat_to_quat, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
# 每个未来帧的目标特征：当前偏航系下的根位置与6D朝向、根系速度、q、qd
GOAL_FRAME_DIM = 3 + 6 + 3 + 3 + NUM_J + NUM_J
BIN_FRAMES = 50  # 用于基于回报的重置采样，每箱 1 秒
MIN_TAIL_STARTS = BIN_FRAMES // 2  # 过短的末箱并入前一箱
FK_CHUNK = 65536


def bin_counts(lengths: torch.Tensor) -> torch.Tensor:
    """计算各片段 n - 1 个可用起始帧的重置分箱数，末帧不能作为起点。

    末箱起始帧少于 MIN_TAIL_STARTS 时并入前一箱，避免极短回合的外推回报使基于回报的采样集中到片段末尾。"""
    starts = lengths - 1
    full, tail = starts // BIN_FRAMES, starts % BIN_FRAMES
    return full + ((tail >= MIN_TAIL_STARTS) | (full == 0)).long()


class RotationView:
    """以 wxyz 四元数存储旋转，索引时转换为矩阵，保留 lib.root_R[f] 的调用方式。"""

    def __init__(self, quat: torch.Tensor):
        """保存按 wxyz 排列的四元数张量。"""
        self.quat = quat

    def __getitem__(self, idx) -> torch.Tensor:
        """将所选四元数即时转换为旋转矩阵。"""
        return quat_to_mat(self.quat[idx])

    def __len__(self) -> int:
        """返回存储的旋转数量。"""
        return len(self.quat)

    @property
    def shape(self) -> torch.Size:
        """返回按矩阵表示时的张量形状。"""
        return self.quat.shape[:-1] + (3, 3)


def resolve_motion_args(args, saved=None):
    """除非调用者明确选择其他数据源，否则恢复运行时保存的动作选择。"""
    saved = saved or {}
    source = getattr(args, "motion_source", None)
    if args.motion_dir is None and args.motion_index is None and source is None:
        args.motion_dir = saved.get("motion_dir")
        args.motion_index = saved.get("motion_index")
        args.motion_source = saved.get("motion_source")
        if args.motion_limit is None:
            args.motion_limit = saved.get("motion_limit")
        if args.motion_root is None:
            args.motion_root = saved.get("motion_root")
    if args.motion_dir is None and args.motion_index is None and getattr(args, "motion_source", None) is None:
        args.motion_dir = str(Path(__file__).resolve().parents[3] / "data" / "lafan1_g1")
    if args.motion_root is not None and args.motion_index is None:
        raise ValueError("--motion_root requires --motion_index")
    for key in ("motion_dir", "motion_index", "motion_source", "motion_root"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, str(Path(value).expanduser().resolve()))


def check_reference(current, saved):
    """检查当前参考数据指纹是否与保存的运行或检查点一致。"""
    if saved and current["fingerprint"] != saved["fingerprint"]:
        raise ValueError("reference fingerprint mismatch: index, source or selection differs from the checkpoint/run")


class MotionLib:
    """管理分片后的参考动作、派生运动学量、重置分箱和目标特征。"""
    def __init__(self, files: list[str], kin: G1Kinematics, device: str, src_fps: float = 30.0,
                 rank=0, world_size=1):
        """从动作 CSV 读取并重采样片段，构建当前进程的参考动作库。"""
        from supermimic.tasks.tracking.motion_hdf5 import file_sha256, validate_pose_arrays

        if not files or src_fps <= 0:
            raise ValueError("need at least one CSV motion and a positive source FPS")
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        clips = []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=",", ndmin=2), dtype=torch.float32)
            if raw.shape[1] < 36:
                raise ValueError(f"{f}: expected at least 36 CSV columns")
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            validate_pose_arrays(p.numpy(), quat.numpy(), j.numpy(), f)
            # 重采样至 50 Hz
            T_src = raw.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            clips.append((p, quat, j))
        digest = hashlib.sha256(json.dumps(
            {"csv": [(os.path.basename(f), file_sha256(Path(f))) for f in files], "fps": src_fps},
            sort_keys=True,
        ).encode()).hexdigest()
        self.description = {"format": "csv", "source_fps": src_fps, "fps": 50,
                            "selected_clips": len(clips), "fingerprint": digest}
        lens = [len(p) for p, _, _ in clips]
        ids = self._select(lens, rank, world_size)
        self._build([names[i] for i in ids], [lens[i] for i in ids], (clips[i] for i in ids), kin, device)

    def _select(self, lens, rank, world_size):
        """按片段长度将完整动作分配到各进程，并选择本进程的片段。"""
        from supermimic.tasks.tracking.motion_hdf5 import partition_clips

        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        groups = partition_clips(lens, world_size)
        ids = groups[rank]
        self.global_description = dict(self.description, selected_frames=sum(lens),
                                       frame_hours=sum(lens) / 50 / 3600,
                                       shorter_than_33_frames=sum(n < 33 for n in lens))
        self.partition = {"algorithm": "full_clip_lpt_v1", "rank": rank, "world_size": world_size,
                          "clip_indices": ids,
                          "frames_per_rank": [sum(lens[i] for i in group) for group in groups],
                          "clips_per_rank": [len(group) for group in groups]}
        self.global_num_bins = int(bin_counts(torch.tensor(lens, dtype=torch.long)).sum())
        frames = sum(lens[i] for i in ids)
        self.description = dict(self.global_description, selected_clips=len(ids), selected_frames=frames,
                                frame_hours=frames / 50 / 3600,
                                shorter_than_33_frames=sum(lens[i] < 33 for i in ids), partition=self.partition)
        return ids

    def _build(self, names, lens, clips, kin, device):
        """从选定片段构建设备上的参考状态、运动学结果和重置分箱。"""
        self.device, self.names = device, names
        self.num_frames = int(sum(lens))
        tracked = len(kin.tracked_idx)
        per_frame = reference_bytes_per_frame(tracked)
        estimate = self.num_frames * per_frame
        self.global_description["reference_tensor_bytes"] = self.global_description["selected_frames"] * per_frame
        self.description.update(selected_frames=self.num_frames, reference_tensor_bytes=estimate)
        if torch.device(device).type == "cuda":
            free, _ = torch.cuda.mem_get_info(torch.device(device))
            if estimate * 1.1 + 2 * 1024**3 > free:
                raise MemoryError(
                    f"Reference tensors need about {estimate / 1024**3:.2f} GiB before training; "
                    f"only {free / 1024**3:.2f} GiB is free. Use an explicit --motion_limit for a smoke "
                    "test, or a larger-memory device. No clips were silently discarded."
                )
        # 将完整片段拼接到一条存储轴；clip_end 是末帧的包含式索引，
        # 与用于数组切片的右开边界不同。
        lengths = torch.tensor(lens, dtype=torch.long, device=device)
        self.clip_len = lengths
        self.clip_start = torch.cumsum(lengths, 0) - lengths
        self.clip_end = self.clip_start + lengths - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lens), device=device), lengths)
        for key, shape in (("root_p", (3,)), ("root_quat", (4,)), ("root_v", (3,)),
                           ("root_w", (3,)), ("q", (NUM_J,)), ("qd", (NUM_J,))):
            setattr(self, key, torch.empty((self.num_frames, *shape), device=device))
        # 逐片段计算导数再写入连续存储，避免差分跨越两个无关动作的拼接边界。
        offset = 0
        for i, (p, quat, j) in enumerate(clips):
            if i >= len(lens):
                raise ValueError("source yielded extra clips")
            expected = lens[i]
            p, quat, j = (torch.as_tensor(x, dtype=torch.float32, device="cpu") for x in (p, quat, j))
            if len(p) != expected:
                raise ValueError("source clip length changed during reference loading")
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # 对旋转做中心差分，得到世界系角速度
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            part = slice(offset, offset + expected)
            for key, value in (("root_p", p), ("root_quat", mat_to_quat(R)), ("root_v", v), ("root_w", w), ("q", j), ("qd", qd)):
                getattr(self, key)[part] = value.to(device)
            offset += expected
        if offset != self.num_frames:
            raise ValueError("source did not yield every selected clip")
        for key, shape in (("body_p", (tracked, 3)), ("body_quat", (tracked, 4)),
                           ("body_v", (tracked, 3)), ("body_w", (tracked, 3)),
                           ("root_v_local", (3,)), ("root_w_local", (3,))):
            setattr(self, key, torch.empty((self.num_frames, *shape), device=device))
        # 矩阵仅在分块计算时存在，常驻动作库保存四元数。
        self.root_R = RotationView(self.root_quat)
        self.body_R = RotationView(self.body_quat)
        with torch.no_grad():
            for s in range(0, self.num_frames, FK_CHUNK):
                e = min(s + FK_CHUNK, self.num_frames)
                R = self.root_R[s:e]
                out = kin.forward(self.root_p[s:e], R, self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                for key, value in zip(("body_p", "body_quat", "body_v", "body_w"), out):
                    getattr(self, key)[s:e] = mat_to_quat(value) if key == "body_quat" else value
                # 逐帧预计算目标中不变的部分：根坐标系下的根速度
                Rt = R.transpose(-1, -2)
                self.root_v_local[s:e] = (Rt @ self.root_v[s:e].unsqueeze(-1)).squeeze(-1)
                self.root_w_local[s:e] = (Rt @ self.root_w[s:e].unsqueeze(-1)).squeeze(-1)
        # 源片段时间轴上的分箱不使用存储窗口。
        # 最后一帧只能作为目标，不能作为重置起点，因为起点必须有下一帧。
        counts = bin_counts(lengths)
        first_bin = torch.cumsum(counts, 0) - counts
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lens), device=device), counts)
        self.num_bins = int(counts.sum())
        local_bin = torch.arange(self.num_bins, device=device) - first_bin[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local_bin * BIN_FRAMES
        # 每箱重置起点为 [bin_start, bin_end)，最后一箱延伸到片段末帧。
        last = local_bin == counts[self.bin_clip] - 1
        self.bin_end = torch.where(last, self.clip_end[self.bin_clip], self.bin_start + BIN_FRAMES)
        self.frame_bin = torch.empty(self.num_frames, dtype=torch.long, device=device)
        for start in range(0, self.num_frames, 65536):
            end = min(start + 65536, self.num_frames)
            ids = self.clip_id[start:end]
            local = (torch.arange(start, end, device=device) - self.clip_start[ids]) // BIN_FRAMES
            self.frame_bin[start:end] = first_bin[ids] + torch.minimum(local, counts[ids] - 1)

    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str, limit: int | None = None,
                 rank=0, world_size=1) -> MotionLib:
        """从目录中的动作文件创建参考动作库。"""
        if limit is not None and limit < 1:
            raise ValueError("motion_limit must be positive")
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return MotionLib(files if limit is None else files[:limit], kin, device, rank=rank, world_size=world_size)

    @classmethod
    def from_index(cls, path, kin, device, source_root=None, limit=None, rank=0, world_size=1):
        """从配对数据索引及其源片段创建参考动作库。"""
        from supermimic.tasks.tracking.motion_hdf5 import IndexedG1Dataset

        dataset = IndexedG1Dataset(path, source_root, limit)
        instance = cls.__new__(cls)
        instance.description = dataset.summary()
        ids = instance._select([c.length for c in dataset.clips], rank, world_size)
        dataset.clips = [dataset.clips[i] for i in ids]
        instance._build([c.name for c in dataset.clips], [c.length for c in dataset.clips],
                        dataset.iter_clips(), kin, device)
        return instance

    @classmethod
    def from_source(cls, path, kin, device, limit=None, rank=0, world_size=1):
        """从单个合并 HDF5 源文件创建参考动作库。"""
        from supermimic.tasks.tracking.motion_hdf5 import MergedG1Dataset

        dataset = MergedG1Dataset(path, limit)
        instance = cls.__new__(cls)
        instance.description = dataset.summary()
        ids = instance._select([c.length for c in dataset.clips], rank, world_size)
        dataset.clips = [dataset.clips[i] for i in ids]
        instance._build([c.name for c in dataset.clips], [c.length for c in dataset.clips],
                        dataset.iter_clips(), kin, device)
        return instance

    @classmethod
    def from_args(cls, args, kin, device, rank=0, world_size=1):
        """根据命令行中的数据源选项创建参考动作库。"""
        if getattr(args, "motion_source", None):
            return cls.from_source(args.motion_source, kin, device, args.motion_limit, rank, world_size)
        if args.motion_index:
            return cls.from_index(args.motion_index, kin, device, args.motion_root, args.motion_limit, rank, world_size)
        return cls.from_dir(args.motion_dir, kin, device, args.motion_limit, rank, world_size)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """返回 f+1 到 f+K 的帧索引，超出片段末尾时截断，形状为 (B, K)。"""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """仅从参考动作生成目标特征：以当前参考帧为基准的 K 个未来帧。"""
        fut = self.future_frames(f, K)  # (B,K)
        # 目标以当前参考帧的偏航和位置为基准，不随机器人的跟踪误差漂移；
        # 未来索引在各自片段末尾截断，不会读到下一个动作。
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,GOAL_FRAME_DIM)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        """按帧索引返回根部位置、旋转、速度和关节状态。"""
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    """沿时间维计算有限差分：内部使用中心差分，两端使用单边差分。"""
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
