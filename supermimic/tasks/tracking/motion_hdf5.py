"""Read full G1 source clips selected by a paired-data index, without editing either."""

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
    """Resident reference bytes per frame: root p/quat/v/w, q, qd, root-frame v/w, per-body p/quat/v/w, two int64 maps."""
    return 4 * (3 + 4 + 3 + 3 + NUM_J + NUM_J + 3 + 3 + tracked * (3 + 4 + 3 + 3)) + 16


def partition_clips(lengths, world_size):
    """Deterministic longest-first packing, preserving whole source clips."""
    if world_size < 1 or len(lengths) < world_size or any(n < 2 for n in lengths):
        raise ValueError("need at least one complete motion (>=2 frames) per rank")
    if world_size == 1:
        return [list(range(len(lengths)))]
    groups = [[] for _ in range(world_size)]
    heap = [(0, rank) for rank in range(world_size)]
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        frames, rank = heapq.heappop(heap)
        groups[rank].append(i)
        heapq.heappush(heap, (frames + int(lengths[i]), rank))
    return [sorted(group) for group in groups]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pose_arrays(p, quat, joints, name: str) -> None:
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
    name: str
    shard: str
    source_clip_index: int
    start: int
    length: int


class IndexedG1Dataset:
    """An identity whitelist, not a paired-frame or training-window view.

    Neither paired clips/start,length nor source clip_* window arrays are consumed.
    Length and offsets always come from the source source_clip_* metadata.
    """

    def __init__(self, index_path, source_root=None, limit=None, *, verify_hashes=False):
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

        records = [None] * self.whitelist_clips
        self.source_stats, self.source_hashes = {}, {}
        source_clips = 0
        for shard in sorted(set(shards)):
            path = self._source_path(shard)
            stat = path.stat()
            bound = snapshot.get(shard)
            if not bound or "sha256" not in bound:
                raise ValueError(f"source shard is not bound by run.json: {shard}")
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
        payload = {"index_sha256": self.index_sha256, "source_hashes": self.source_hashes,
                   "length_policy": "full_g1_source", "clips": [asdict(c) for c in self.clips]}
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _source_path(self, name: str) -> Path:
        relative = PurePosixPath(name)
        if not name or str(relative) != name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"unsafe source shard path: {name}")
        path = (self.source_root / Path(*relative.parts)).resolve(strict=True)
        if not path.is_relative_to(self.source_root):
            raise ValueError(f"source shard escapes its root: {name}")
        return path

    def iter_clips(self):
        with ExitStack() as stack:
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
    """A single merged ``teleopit_motion_hdf5`` source, read without an index.

    Clip boundaries come from the file's own ``source_clip_*`` arrays, so no
    pairing index is needed. Identity is a digest over the file's structural
    metadata, which stays stable when the same file is copied between machines.
    """

    def __init__(self, path, limit=None):
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
        digest = hashlib.sha256()
        digest.update(json.dumps({"format": "merged_g1_v1", "fps": 50, "frames": frames,
                                  "clips": int(len(lengths)),
                                  "bytes": self.stats.st_size},
                                 sort_keys=True).encode())
        digest.update(np.ascontiguousarray(starts.astype(np.int64)).tobytes())
        digest.update(np.ascontiguousarray(lengths.astype(np.int64)).tobytes())
        self.fingerprint = digest.hexdigest()

    def iter_clips(self):
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
        frames = sum(c.length for c in self.clips)
        return {"format": "supermimic_merged_g1_v1", "source": str(self.path),
                "fingerprint": self.fingerprint, "length_policy": "full_g1_source", "fps": 50,
                "whitelist_clips": self.whitelist_clips, "selected_clips": len(self.clips),
                "selected_frames": frames, "frame_hours": frames / 50 / 3600,
                "shorter_than_33_frames": sum(c.length < 33 for c in self.clips),
                "source_bytes": self.stats.st_size,
                "source_binding": "structural metadata digest; file size and mtime checked on load"}
