"""验证索引化完整动作源的 G1 数据视图，不启动 MuJoCo 或训练。"""

import argparse
import json
from pathlib import Path
import sys

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--motion_index", required=True)
parser.add_argument("--motion_root", default=None)
parser.add_argument("--motion_limit", type=int, default=None)
parser.add_argument("--verify_hashes", action="store_true")
parser.add_argument("--verify_frames", action="store_true")
parser.add_argument("--report", default=None)
parser.add_argument("--num_shards", type=int, default=1, help="metadata-only planning for the intended world size")
args = parser.parse_args()

from supermimic.tasks.tracking.motion_hdf5 import IndexedG1Dataset, partition_clips, reference_bytes_per_frame
from supermimic.tasks.tracking.constants import TRACKED_BODIES

dataset = IndexedG1Dataset(args.motion_index, args.motion_root, args.motion_limit,
                           verify_hashes=args.verify_hashes)
result = dataset.summary()
# 仅用长度元信息规划分片及常驻参考张量大小，不需要真正构造仿真环境。
groups = partition_clips([c.length for c in dataset.clips], args.num_shards)
bytes_per_frame = reference_bytes_per_frame(len(TRACKED_BODIES))
result["partition_plan"] = {
    "algorithm": "full_clip_lpt_v1", "world_size": args.num_shards,
    "clips_per_rank": [len(group) for group in groups],
    "frames_per_rank": [sum(dataset.clips[i].length for i in group) for group in groups],
    "reference_gib_per_rank": [sum(dataset.clips[i].length for i in group) * bytes_per_frame / 1024**3 for group in groups],
}
# 可选完整读取会逐片验证形状、有限性和四元数，并核对总帧数，开销高于仅检查目录元数据。
if args.verify_frames:
    result["verified_frames"] = sum(len(p) for p, _, _ in dataset.iter_clips())
    if result["verified_frames"] != result["selected_frames"]:
        raise ValueError("not every selected source frame was verified")
print(json.dumps(result, indent=2, ensure_ascii=False))
if args.report:
    Path(args.report).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
