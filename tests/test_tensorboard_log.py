"""scripts/csv_to_tensorboard.py: tags, resumed rows and a partly written last line."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[1]


def scalars(path):
    acc = EventAccumulator(str(path))
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


class CsvToTensorboardTests(unittest.TestCase):
    def test_rows_become_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            # 100 is logged twice (resume from 50); the last line is cut while training writes it
            (run / "train_log.csv").write_text(
                "iteration,wall_h,v_loss\n50,0.1,0.5\n100,0.2,9.0\n100,0.2,0.25\n150,0.3,0.125\n200,0.", encoding="utf-8")
            (run / "sampling_log.csv").write_text("iteration,samp_top1\n50,0.1\n100,0.2\n", encoding="utf-8")
            (run / "diag_log.csv").write_text("iteration,v_mean\n100,9.5\n", encoding="utf-8")  # a run from before it existed has none
            for _ in range(2):  # a rerun replaces the earlier conversion instead of adding to it
                subprocess.run([sys.executable, str(REPO / "scripts" / "csv_to_tensorboard.py"), str(run)],
                               check=True, capture_output=True)
            got = scalars(run / "tensorboard")
            self.assertEqual(set(got), {"train/wall_h", "train/v_loss", "sampling/samp_top1", "diag/v_mean"})
            self.assertEqual(got["diag/v_mean"], [(100, 9.5)])
            self.assertEqual(got["train/v_loss"], [(50, 0.5), (100, 0.25), (150, 0.125)])
            self.assertEqual([s for s, _ in got["sampling/samp_top1"]], [50, 100])


if __name__ == "__main__":
    unittest.main()
