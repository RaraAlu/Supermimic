"""将训练目录中的 train_log.csv、sampling_log.csv 和 diag_log.csv 转成 TensorBoard 标量，不修改 CSV。

标签为 train/<列名>、sampling/<列名> 和 diag/<列名>，以迭代数为步数；缺失文件跳过。
适用于未启用 TensorBoard 的训练，可与仅追加 CSV 的训练进程同时运行。恢复训练的重复迭代保留最后完整行，
忽略未写完的末行；重复转换用 purge_step 隐藏先前事件。

用法：python scripts/csv_to_tensorboard.py RUN_DIR [--out RUN_DIR/tensorboard]"""

import argparse
import csv
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("run", help="training directory with train_log.csv (and sampling_log.csv, diag_log.csv)")
parser.add_argument("--out", default=None, help="event directory (default: RUN/tensorboard, where training writes)")
args = parser.parse_args()


def read_rows(path):
    """按迭代顺序返回各迭代最后一条完整记录。"""
    if not path.exists():
        return (), []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        by_it = {}
        for r in reader:
            try:
                values = {k: float(v) for k, v in r.items()}
            except (TypeError, ValueError):  # 末行可能尚未写完：缺失字段为 None，截断的数字无法转换
                continue
            # 同一迭代可能因恢复训练出现多次，用最后一条完整记录覆盖前值，再按迭代排序。
            by_it[int(values["iteration"])] = values
    return reader.fieldnames, [by_it[k] for k in sorted(by_it)]


run = Path(args.run)
groups = [(name, *read_rows(run / f"{name}_log.csv")) for name in ("train", "sampling", "diag")]
first = min((rows[0]["iteration"] for _, _, rows in groups if rows), default=None)
if first is None:
    raise SystemExit(f"no complete rows in {run}/train_log.csv or sampling_log.csv")
# 以所有日志最早迭代为清理边界，隐藏此前转换的后续事件，避免重复导入出现重叠曲线。
writer = SummaryWriter(args.out or str(run / "tensorboard"), purge_step=int(first))
for group, names, rows in groups:
    for r in rows:
        for name in names[1:]:
            if math.isfinite(r[name]):
                writer.add_scalar(f"{group}/{name}", r[name], int(r["iteration"]))
    print(f"{group}: {len(rows)} rows, iterations {int(rows[0]['iteration']) if rows else '-'}..{int(rows[-1]['iteration']) if rows else '-'}")
writer.close()
