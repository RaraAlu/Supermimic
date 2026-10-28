"""Write a run's train_log.csv, sampling_log.csv and diag_log.csv as TensorBoard scalars (read-only on the CSV files).

Tags match training's --tensorboard output: train/<column>, sampling/<column> and diag/<column>, stepped by
iteration; a file the run does not have (diag_log.csv exists from the run that added it on) is skipped.
For runs that started without TensorBoard logging; safe beside a running training, which only appends to the
CSV files. A resumed row replaces the earlier row of the same iteration, a partly written last line is
skipped, and rerunning hides the events of the previous conversion (purge_step), so it can be repeated.

    python scripts/csv_to_tensorboard.py RUN_DIR [--out RUN_DIR/tensorboard]
"""

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
    """Last complete row per iteration, in iteration order."""
    if not path.exists():
        return (), []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        by_it = {}
        for r in reader:
            try:
                values = {k: float(v) for k, v in r.items()}
            except (TypeError, ValueError):  # partly written last line: a missing field is None, a cut number fails
                continue
            by_it[int(values["iteration"])] = values
    return reader.fieldnames, [by_it[k] for k in sorted(by_it)]


run = Path(args.run)
groups = [(name, *read_rows(run / f"{name}_log.csv")) for name in ("train", "sampling", "diag")]
first = min((rows[0]["iteration"] for _, _, rows in groups if rows), default=None)
if first is None:
    raise SystemExit(f"no complete rows in {run}/train_log.csv or sampling_log.csv")
writer = SummaryWriter(args.out or str(run / "tensorboard"), purge_step=int(first))
for group, names, rows in groups:
    for r in rows:
        for name in names[1:]:
            if math.isfinite(r[name]):
                writer.add_scalar(f"{group}/{name}", r[name], int(r["iteration"]))
    print(f"{group}: {len(rows)} rows, iterations {int(rows[0]['iteration']) if rows else '-'}..{int(rows[-1]['iteration']) if rows else '-'}")
writer.close()
