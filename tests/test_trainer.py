"""训练器中可单独检查的部分：学习率调度、存活权重、GAE目标、批量划分与训练日志；不启动仿真或GPU。"""

import csv
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch

from supermimic.tasks.tracking.rl.trainer import LOG_TABLES, Trainer, gae_targets, lr_at, make_layout, split_sizes, survival
from supermimic.utils.distributed import DistributedContext
from supermimic.utils.run_log import RunLog


class ScheduleTests(unittest.TestCase):
    def test_cosine_decay_to_three_percent(self):
        self.assertEqual(lr_at(0, 2e-4, 100), 2e-4)
        self.assertAlmostEqual(lr_at(50, 2e-4, 100), 2e-4 * (0.03 + 0.97 * 0.5))
        self.assertAlmostEqual(lr_at(100, 2e-4, 100), 2e-4 * 0.03)
        self.assertAlmostEqual(lr_at(10**6, 2e-4, 100), 2e-4 * 0.03)


class SurvivalTests(unittest.TestCase):
    def test_unrolled_product_matches_cumprod_with_gradients(self):
        # CUDA graph用逐步连乘代替cumprod，前向与梯度都必须一致
        keep = torch.rand(6, 4, dtype=torch.float64).requires_grad_(True)
        keep2 = keep.detach().clone().requires_grad_(True)
        a, b = survival(keep, unrolled=False), survival(keep2, unrolled=True)
        torch.testing.assert_close(a, b)
        weights = torch.randn(6, 4, dtype=torch.float64)
        (a * weights).sum().backward(), (b * weights).sum().backward()
        torch.testing.assert_close(keep.grad, keep2.grad)


class GaeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.H, self.B, self.gamma = 5, 3, 0.9
        self.V = torch.randn(self.H + 1, self.B, dtype=torch.float64)
        self.r = torch.randn(self.H, self.B, dtype=torch.float64)
        self.keep = torch.rand(self.H, self.B, dtype=torch.float64)

    def test_lambda_zero_is_one_step_td(self):
        targets = gae_targets(self.V, self.r, self.keep, self.gamma, 0.0)
        torch.testing.assert_close(targets, self.r + self.gamma * self.keep * self.V[1:])

    def test_lambda_one_is_survival_weighted_return_with_bootstrap(self):
        targets = gae_targets(self.V, self.r, self.keep, self.gamma, 1.0)
        for t in range(self.H):
            G, c = torch.zeros(self.B, dtype=torch.float64), torch.ones(self.B, dtype=torch.float64)
            for j in range(t, self.H):
                G = G + self.gamma ** (j - t) * c * self.r[j]
                c = c * self.keep[j]
            G = G + self.gamma ** (self.H - t) * c * self.V[-1]
            torch.testing.assert_close(targets[t], G)


class LayoutTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(batch_scope="global", num_envs=10, wm_batch=7, pi_batch=5, wm_update="steps", wm_batches=10, lr=2e-4,
                    lr_scaling="none", buffer_steps=100, wm_window=8, warmup_steps=64, no_critic=False, horizon=32,
                    critic_minibatches=4)
        return SimpleNamespace(**{**base, **kw})

    def parallel(self, rank, world):
        p = DistributedContext.__new__(DistributedContext)
        p.rank, p.world_size = rank, world
        return p

    def test_global_sizes_split_across_ranks(self):
        layouts = [make_layout(self.args(), self.parallel(r, 3)) for r in range(3)]
        self.assertEqual([l.local_envs for l in layouts], [4, 3, 3])
        self.assertEqual(sum(l.local_wm_batch for l in layouts), 7)
        self.assertEqual({l.global_envs for l in layouts}, {10})
        self.assertEqual({(l.wm_steps, l.wm_step_batch == l.local_wm_batch) for l in layouts}, {(10, True)})

    def test_per_gpu_scope_single_update_and_sqrt_lr(self):
        layout = make_layout(self.args(batch_scope="per_gpu", wm_update="single", lr_scaling="sqrt"), self.parallel(1, 2))
        self.assertEqual((layout.global_envs, layout.local_envs, layout.local_wm_batch), (20, 10, 7))
        self.assertEqual((layout.wm_steps, layout.wm_step_batch), (1, 70))
        self.assertAlmostEqual(layout.base_lr, 2e-4 * math.sqrt(2))

    def test_rejects_short_buffer(self):
        with self.assertRaises(ValueError):
            make_layout(self.args(buffer_steps=4), self.parallel(0, 1))

    def test_split_sizes_match_tensor_split(self):
        for n, parts in ((10, 4), (3, 4), (128, 4)):
            self.assertEqual(split_sizes(n, parts), [len(c) for c in torch.tensor_split(torch.arange(n), parts)])


class ReplayStartTests(unittest.TestCase):
    def test_short_replay_sampling_is_bounded(self):
        # 回放中没有完整想象长度的起点时，有限次重采样后报错，不无限循环
        fake = SimpleNamespace(buffer=SimpleNamespace(sample_starts=lambda b, h: ({"f": torch.tensor([1])}, {})),
                               HZ=32, Hh=1, motion=SimpleNamespace(clip_end=torch.tensor([1]), clip_id=torch.tensor([0, 0])))
        with self.assertRaisesRegex(RuntimeError, "too few starts"):
            Trainer.sample_valid_starts(fake, 1)


class TermRowsTests(unittest.TestCase):
    def test_summary_shares_do_not_depend_on_detail_limit(self):
        motion = SimpleNamespace(names=[f"clip{i}" for i in range(12)],
                                 bin_start=torch.arange(12) * 100, clip_start=torch.arange(12) * 100)

        def report(top):
            return {"terms": 12, "ends": 24, "starts": 36, "clips": 12,
                    "top": [(i, 1, 2, 3, i, 1) for i in range(min(top, 12))]}

        fake = SimpleNamespace(motion=motion, parallel=DistributedContext("cpu"),
                               env=SimpleNamespace(sampler=SimpleNamespace(window_report=report)))
        for limit in (0, 1, 5, 20):
            with self.subTest(limit=limit):
                fake.args = SimpleNamespace(term_top=limit)
                summary, lines = Trainer.term_rows(fake, 49)
                self.assertEqual(summary["termwin"], [50, 12, 12, 1 / 12, 10 / 12, 36, 24])
                self.assertEqual(len(lines), min(limit, 12))
                if lines:
                    self.assertEqual(lines[0], [50, "clip0", 1, 2, 0.5, 3, 0.0, 1, 1 / 12])


class RunLogTests(unittest.TestCase):
    LINES = "term_bins"  # 多行、含文字列、不写TensorBoard的表

    def rows(self, it):
        return {group: [it] + [0.5 * it + i for i in range(len(columns) - 1)] for group, (_, columns) in LOG_TABLES.items()
                if group != self.LINES}

    def test_csv_and_tensorboard_rows(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as tmp:
            log = RunLog(tmp, LOG_TABLES, tensorboard=True)
            log.write(self.rows(50))
            log.write_lines(self.LINES, [[50, "a/b.csv", 3, 8, 0.375, 40, 2.0, 2, 0.5], [50, "c.csv", 1, 2, 0.5, 10, 0.0, 1, 0.1666667]])
            log.close()
            log = RunLog(tmp, LOG_TABLES)  # 恢复训练：追加写入，不重复表头
            log.write(self.rows(100))
            log.write_lines(self.LINES, [[100, "a/b.csv", 2, 6, 0.3333333, 30, 2.0, 2, 1.0]])
            log.close()
            for group, (name, columns) in LOG_TABLES.items():
                with open(Path(tmp) / name, encoding="utf-8") as f:
                    lines = list(csv.reader(f))
                self.assertEqual(lines[0], list(columns))
                if group == self.LINES:
                    self.assertEqual([line[0] for line in lines[1:]], ["50", "50", "100"])
                    self.assertEqual(lines[1][1:4], ["a/b.csv", "3", "8"])
                    continue
                self.assertEqual([line[0] for line in lines[1:]], ["50", "100"])
                self.assertEqual(lines[1][1], "25")  # 浮点数按6位有效数字写出
            acc = EventAccumulator(str(Path(tmp) / "tensorboard"))
            acc.Reload()
            tags = set(acc.Tags()["scalars"])
            self.assertEqual(tags, {f"{g}/{c}" for g, (_, columns) in LOG_TABLES.items() if g != self.LINES for c in columns[1:]})
            self.assertEqual([(e.step, e.value) for e in acc.Scalars("train/wall_h")], [(50, 25.0)])


if __name__ == "__main__":
    unittest.main()
