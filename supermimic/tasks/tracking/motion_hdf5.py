"""依据配对数据索引读取完整 G1 源动作片段，不修改索引或源数据。"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from supermimic.tasks.tracking.constants import G1_JOINT_NAMES

INDEX_FORMAT = "mjbones_human23_g1_nominal_hdf5_v1_index"
NUM_J = len(G1_JOINT_NAMES)
ARRAY_SHAPES = {"root_pos": 3, "root_quat_w": 4, "joint_pos": NUM_J}


def reference_bytes_per_frame(tracked: int) -> int:
    """估算每帧常驻参考数据的字节数：根部位置、四元数和速度，关节状态，根系速度，各刚体状态及两个 int64 映射。"""
    return 4 * (3 + 4 + 3 + 3 + NUM_J + NUM_J + 3 + 3 + tracked * (3 + 4 + 3 + 3)) + 16


def partition_clips(lengths, world_size):
    """按片段长度从长到短确定性地分配，保持每个源片段完整。"""
    if world_size < 1 or len(lengths) < world_size or any(n < 2 for n in lengths):
        raise ValueError("need at least one complete motion (>=2 frames) per rank")
    if world_size == 1:
        return [list(range(len(lengths)))]
    groups = [[] for _ in range(world_size)]
    heap = [(0, rank) for rank in range(world_size)]
    # 最长片段优先放到当前总帧数最少的进程；以原索引和进程号打破并列，保证可复现。
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        frames, rank = heapq.heappop(heap)
        groups[rank].append(i)
        heapq.heappush(heap, (frames + int(lengths[i]), rank))
    return [sorted(group) for group in groups]


def file_sha256(path: Path) -> str:
    """分块读取文件并计算 SHA-256 摘要，避免一次将整个文件载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pose_arrays(p, quat, joints, name: str) -> None:
    """验证位置、四元数和关节数组的形状、有限性及四元数单位长度。"""
    n = len(p)
    for key, array, width in (("root_pos", p, 3), ("root_quat_w", quat, 4), ("joint_pos", joints, NUM_J)):
        if array.shape != (n, width) or n < 2:
            raise ValueError(f"{name}: invalid {key} shape {array.shape}; need at least two frames")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: {key} contains NaN or Inf")
    if np.max(np.abs(np.linalg.norm(quat.astype(np.float64), axis=-1) - 1)) > 1e-4:
        raise ValueError(f"{name}: root quaternion must be unit length (wxyz)")


@dataclass(frozen=True)
class SourceClip:
    """记录源动作片段的名称、分片、索引、起点和长度。"""
    name: str
    shard: str
    source_clip_index: int
    start: int
    length: int


class IndexedG1Dataset:
    """按身份白名单选择源片段，不使用配对帧或训练窗口视图。

    不读取配对片段的起点、长度或源文件中的 clip_* 窗口数组；长度和偏移始终取自源文件的 source_clip_* 元数据。"""

    def __init__(self, index_path, source_root=None, limit=None, *, verify_hashes=False):
        """读取并校验索引及源数据元信息，按选项限制片段数量和验证文件摘要。"""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("motion_limit must be positive")
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        run_path = self.index_path.with_name("run.json")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("format") != INDEX_FORMAT.removesuffix("_index"):
            raise ValueError("unsupported paired run format")
        if run.get("g1_joint_names") != G1_JOINT_NAMES:
            raise ValueError("G1 joint order does not match Supermimic")
        self.source_root = Path(source_root or run["g1_root"]).expanduser().resolve(strict=True)
        self.index_sha256 = file_sha256(self.index_path)
        manifest_sha = run.get("g1_manifest_sha256")
        if not manifest_sha or file_sha256(self.source_root / "manifest.json") != manifest_sha:
            raise ValueError("G1 source manifest changed; use the source bound to this index")
        snapshot = run["g1_snapshot"]["shards"]
        # 索引只决定片段身份白名单；实际时间范围稍后从源文件读取，避免把配对窗口误当完整动作。
        with h5py.File(self.index_path, "r") as index:
            if index.attrs.get("format") != INDEX_FORMAT or not index.attrs.get("complete", False):
                raise ValueError("requires a complete Mjbones paired index")
            if index.attrs.get("fps") != 50:
                raise ValueError("the index must declare 50 Hz")
            group = index["clips"]
            names = group["source_relative_path"].asstr()[:]
            shards = group["g1_h5"].asstr()[:]
            ids = group["g1_source_clip_index"][:]
            self.whitelist_clips = len(ids)
            for key in ("source_relative_path", "g1_h5", "g1_source_clip_index"):
                if group[key].shape != (self.whitelist_clips,):
                    raise ValueError(f"invalid index column {key}")
            if not self.whitelist_clips or ids.dtype.kind not in "iu":
                raise ValueError("index must contain integer G1 source clip identities")
            if len(set(names)) != len(ids) or len(set(zip(shards, ids.tolist()))) != len(ids):
                raise ValueError("duplicate source identity in index")
            expected = {}
            for key in ("g1_source_start", "g1_source_frames"):
                if key in group:
                    if group[key].shape != ids.shape or group[key].dtype.kind not in "iu":
                        raise ValueError(f"invalid index column {key}")
                    expected[key] = group[key][:]

        # 按源分片集中验证文件，但将结果写回索引原顺序，保持片段选择及指纹稳定。
        records = [None] * self.whitelist_clips
        self.source_stats, self.source_hashes = {}, {}
        source_clips = 0
        for shard in sorted(set(shards)):
            path = self._source_path(shard)
            stat = path.stat()
            bound = snapshot.get(shard)
            if not bound or "sha256" not in bound:
                raise ValueError(f"source shard is not bound by run.json: {shard}")
            # 默认先比较文件大小和修改时间；显式要求或元信息变化时才重算完整摘要，降低大文件启动成本。
            same_stat = (stat.st_size, stat.st_mtime_ns) == (bound["bytes"], bound["mtime_ns"])
            if (verify_hashes or not same_stat) and file_sha256(path) != bound["sha256"]:
                raise ValueError(f"G1 source checksum changed: {shard}")
            self.source_stats[shard] = (stat.st_size, stat.st_mtime_ns)
            self.source_hashes[shard] = bound["sha256"]
            rows = np.flatnonzero(shards == shard)
            selected_ids = ids[rows]
            with h5py.File(path, "r") as source:
                if source.attrs.get("format") != "teleopit_motion_hdf5" or source.attrs.get("version") != 1:
                    raise ValueError(f"requires minimal G1 source HDF5, not an old precomputed cache: {shard}")
                starts = source["source_clip_starts"][:]
                lengths = source["source_clip_lengths"][:]
                fps = source["source_clip_fps"][:]
                if any(a.ndim != 1 or a.shape != starts.shape or a.dtype.kind not in "iu"
                       for a in (starts, lengths, fps)):
                    raise ValueError(f"invalid source clip metadata: {shard}")
                source_clips += len(starts)
                if np.any(selected_ids < 0) or np.any(selected_ids >= len(starts)):
                    raise ValueError(f"source clip identity out of range: {shard}")
                frames = source["joint_pos"].shape[0]
                for key, width in ARRAY_SHAPES.items():
                    ds = source[key]
                    if ds.shape != (frames, width) or ds.dtype != np.dtype("float32"):
                        raise ValueError(f"{shard}: {key} must be float32 ({frames}, {width})")
                # 将索引身份映射到源片段的真实边界，并核对可选定位元数据；
                # 同时要求原生 50 Hz，避免加载阶段隐式重采样改变索引含义。
                ss, nn = starts[selected_ids], lengths[selected_ids]
                if np.any(ss < 0) or np.any(nn < 2) or np.any(ss + nn > frames):
                    raise ValueError(f"invalid selected source bounds: {shard}")
                if np.any(fps[selected_ids] != 50):
                    raise ValueError(f"selected G1 source clips must be native 50 Hz: {shard}")
                for key, values in (("g1_source_start", ss), ("g1_source_frames", nn)):
                    if key in expected and not np.array_equal(expected[key][rows], values):
                        raise ValueError(f"index/source locator mismatch ({key}): {shard}")
                for row, start, length in zip(rows, ss, nn):
                    records[row] = SourceClip(str(names[row]), shard, int(ids[row]), int(start), int(length))
        self.source_clips = source_clips
        self.clips = records if limit is None else records[:limit]
        self.verify_hashes = verify_hashes
        # 指纹绑定索引、源摘要和实际选择的片段；修改数量限制也会改变身份，防止恢复训练时混用数据。
        payload = {"index_sha256": self.index_sha256, "source_hashes": self.source_hashes,
                   "length_policy": "full_g1_source", "clips": [asdict(c) for c in self.clips]}
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _source_path(self, name: str) -> Path:
        """解析源分片路径并检查它位于允许的源目录内。"""
        relative = PurePosixPath(name)
        if not name or str(relative) != name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"unsafe source shard path: {name}")
        path = (self.source_root / Path(*relative.parts)).resolve(strict=True)
        if not path.is_relative_to(self.source_root):
            raise ValueError(f"source shard escapes its root: {name}")
        return path

    def iter_clips(self):
        """逐个读取所选源片段，返回根部位置、四元数和关节位置数组。"""
        with ExitStack() as stack:
            # 按需打开并复用分片句柄，ExitStack 负责异常时关闭；读取前后检查元信息以发现文件变化。
            handles = {}
            for clip in self.clips:
                if clip.shard not in handles:
                    path = self._source_path(clip.shard)
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != self.source_stats[clip.shard]:
                        raise ValueError(f"source changed after catalog validation: {clip.shard}")
                    handles[clip.shard] = stack.enter_context(h5py.File(path, "r"))
                source = handles[clip.shard]
                sl = slice(clip.start, clip.start + clip.length)
                arrays = tuple(source[key][sl] for key in ARRAY_SHAPES)
                validate_pose_arrays(*arrays, clip.name)
                yield arrays
            for shard in handles:
                stat = self._source_path(shard).stat()
                if (stat.st_size, stat.st_mtime_ns) != self.source_stats[shard]:
                    raise ValueError(f"source changed during reference loading: {shard}")

    def summary(self):
        """返回动作来源、选择和指纹等可用于检查点校验的元数据。"""
        frames = sum(c.length for c in self.clips)
        return {"format": "supermimic_indexed_g1_v1", "index": str(self.index_path),
                "source_root": str(self.source_root), "index_sha256": self.index_sha256,
                "fingerprint": self.fingerprint, "length_policy": "full_g1_source", "fps": 50,
                "whitelist_clips": self.whitelist_clips, "selected_clips": len(self.clips),
                "source_clips_in_referenced_shards": self.source_clips,
                "selected_frames": frames, "frame_hours": frames / 50 / 3600,
                "shorter_than_33_frames": sum(c.length < 33 for c in self.clips),
                "source_hashes_verified": self.verify_hashes,
                "source_binding": "manifest, source metadata, stat; SHA256 if requested or stat changed"}


class MergedG1Dataset:
    """无需索引即可读取单个合并的 teleopit_motion_hdf5 数据源。

    片段边界来自文件自身的 source_clip_* 数组；身份标识由结构元数据生成摘要，同一文件复制到其他机器后保持稳定。"""

    def __init__(self, path, limit=None):
        """读取合并源文件的片段边界，验证结构并生成稳定身份信息。"""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("motion_limit must be positive")
        self.path = Path(path).expanduser().resolve(strict=True)
        self.stats = self.path.stat()
        with h5py.File(self.path, "r") as source:
            if source.attrs.get("format") != "teleopit_motion_hdf5":
                raise ValueError("requires a teleopit_motion_hdf5 v1 source")
            if int(source.attrs.get("version", 0)) != 1:
                raise ValueError("requires teleopit_motion_hdf5 version 1")
            if int(source.attrs.get("fps", 0)) != 50:
                raise ValueError("the source must declare 50 Hz")
            frames = int(source["root_pos"].shape[0])
            for key, width in ARRAY_SHAPES.items():
                block = source[key]
                if block.shape != (frames, width) or block.dtype != np.dtype("float32"):
                    raise ValueError(f"{key} must be float32 ({frames}, {width})")
            starts = source["source_clip_starts"][:]
            lengths = source["source_clip_lengths"][:]
            fps = source["source_clip_fps"][:]
            names = None
            if "clips" in source and "source_relative_path" in source["clips"]:
                names = source["clips"]["source_relative_path"].asstr()[:]
        # 先验证所有片段连续覆盖整个帧轴，再应用数量限制，避免截取前缀掩盖源文件结构错误。
        count = len(lengths)
        if count < 1 or len(starts) != count:
            raise ValueError("source_clip_starts and source_clip_lengths must agree")
        if np.any(fps != 50):
            raise ValueError("every source clip must declare 50 Hz")
        if np.any(lengths < 2):
            raise ValueError("every source clip needs at least two frames")
        if int(starts[0]) != 0 or not np.array_equal(starts[1:], starts[:-1] + lengths[:-1]):
            raise ValueError("source clips must partition the frame axis contiguously")
        if int(starts[-1] + lengths[-1]) != frames:
            raise ValueError("source clips do not cover the whole file")
        if names is not None and len(names) != count:
            raise ValueError("clips/source_relative_path must have one row per clip")
        if limit is not None:
            starts, lengths = starts[:limit], lengths[:limit]
            names = None if names is None else names[:limit]
            count = len(lengths)
        keep = np.arange(count, dtype=np.int64)
        self.clips = [SourceClip(name=str(names[i]) if names is not None else f"clip_{i:06d}",
                                 shard=self.path.name, source_clip_index=int(keep[i]),
                                 start=int(starts[i]), length=int(lengths[i]))
                      for i in range(count)]
        self.whitelist_clips = int(len(lengths))
        self.frames = int(lengths.sum())
        self.limit = limit
        # 合并源的身份来自结构元数据及大小，不是动作数组内容的完整哈希；
        # 复制路径不影响指纹，但不能据此检测所有等大小内容修改。
        digest = hashlib.sha256()
        digest.update(json.dumps({"format": "merged_g1_v1", "fps": 50, "frames": frames,
                                  "clips": int(len(lengths)),
                                  "bytes": self.stats.st_size},
                                 sort_keys=True).encode())
        digest.update(np.ascontiguousarray(starts.astype(np.int64)).tobytes())
        digest.update(np.ascontiguousarray(lengths.astype(np.int64)).tobytes())
        self.fingerprint = digest.hexdigest()

    def iter_clips(self):
        """逐个读取合并文件中选定的完整动作片段。"""
        before = self.path.stat()
        with h5py.File(self.path, "r") as source:
            for clip in self.clips:
                sl = slice(clip.start, clip.start + clip.length)
                arrays = tuple(source[key][sl] for key in ARRAY_SHAPES)
                validate_pose_arrays(*arrays, clip.name)
                yield arrays
        after = self.path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during reference loading")

    def summary(self):
        """返回合并源文件及所选片段的摘要和指纹。"""
        frames = sum(c.length for c in self.clips)
        return {"format": "supermimic_merged_g1_v1", "source": str(self.path),
                "fingerprint": self.fingerprint, "length_policy": "full_g1_source", "fps": 50,
                "whitelist_clips": self.whitelist_clips, "selected_clips": len(self.clips),
                "selected_frames": frames, "frame_hours": frames / 50 / 3600,
                "shorter_than_33_frames": sum(c.length < 33 for c in self.clips),
                "source_bytes": self.stats.st_size,
                "source_binding": "structural metadata digest; file size and mtime checked on load"}
