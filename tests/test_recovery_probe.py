"""scripts/diag_recovery_probe.py中不依赖仿真的部分：状态分类、候选恢复奖励、回报定义和起点选择。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diag_recovery_probe as probe  # noqa: E402

TERMS = dict(gamma=0.99, term_height=0.25, term_tilt=0.8, tz_d=0.05, tt_d=0.1, dt=0.02)


class ClassifyTests(unittest.TestCase):
    def label(self, p_z, e_z, e_tilt, v_z=0.0, perturbed=False):
        out = probe.classify(torch.tensor([p_z]), torch.tensor([v_z]), torch.tensor([e_z]), torch.tensor([e_tilt]), torch.tensor([perturbed]))
        return {k for k, v in out.items() if v.item()}

    def test_partition_is_exclusive_and_complete(self):
        n = 2000
        p, v, z, tilt = torch.rand(n), 2 * torch.rand(n) - 1, torch.rand(n), 2 * torch.rand(n)
        out = probe.classify(p, v, z, tilt, torch.rand(n) < 0.5)
        total = sum(out[c].int() for c in probe.PARTITION)
        self.assertTrue((total == 1).all())

    def test_reference_that_is_low_is_not_a_fall(self):
        self.assertEqual(self.label(0.3, 0.02, 0.05), {"normal_low"})
        self.assertEqual(self.label(0.3, 0.5, 0.05), {"fallen"})

    def test_deviating_while_upright_is_imbalance(self):
        self.assertEqual(self.label(0.75, 0.02, 0.5), {"imbalance"})
        self.assertEqual(self.label(0.75, 0.02, 0.05), {"normal_high"})

    def test_tracking_after_a_push_is_not_the_clean_baseline(self):
        self.assertEqual(self.label(0.75, 0.02, 0.05, perturbed=True), {"disturbed_tracked"})
        self.assertEqual(self.label(0.3, 0.02, 0.05, perturbed=True), {"disturbed_tracked"})
        # 没有受扰、却自己偏离的状态仍按偏离算
        self.assertEqual(self.label(0.3, 0.5, 0.05, perturbed=True), {"fallen"})

    def test_rising_overlaps_the_partition(self):
        self.assertEqual(self.label(0.3, 0.5, 0.05, v_z=0.5), {"fallen", "rising"})


class CrossesPushTests(unittest.TestCase):
    def test_only_transitions_after_the_push_inside_the_window_are_invalid(self):
        # 冲量写在转移10之前；从t0=8开始时，第0、1个转移（t=8、9）仍然连续
        flags = probe.crosses_push(torch.tensor(10), torch.tensor(8), torch.arange(5))
        self.assertEqual(flags.tolist(), [False, False, True, True, True])

    def test_window_starting_at_the_push_is_valid(self):
        # s[10]已含冲量，从这里开始的推演不跨越它
        self.assertFalse(probe.crosses_push(torch.tensor(10), torch.tensor(10), torch.arange(5)).any())

    def test_environments_without_a_push_are_never_invalid(self):
        self.assertFalse(probe.crosses_push(torch.tensor(-1), torch.arange(5), torch.tensor(3)).any())


class RecoveryRewardTests(unittest.TestCase):
    def test_deviation_direction_matches_the_design(self):
        d = lambda z, tilt: probe.deviation(torch.tensor(z), torch.tensor(tilt), 0.25, 0.8, 0.05, 0.1).item()  # noqa: E731
        self.assertLess(d(0.02, 0.05), 0.02)
        self.assertGreater(d(0.6, 1.5), 0.99)

    def test_recover_reward_keeps_gradient_where_tracking_kernels_die(self):
        tilt = torch.tensor(1.5, requires_grad=True)
        probe.recover_reward(torch.tensor(0.6), tilt).backward()
        self.assertGreater(abs(tilt.grad.item()), 0.1)
        # 对照第4.3节：跟踪核在同样误差下几乎没有梯度
        tilt = torch.tensor(1.5, requires_grad=True)
        torch.exp(-tilt**2 / 0.16).backward()
        self.assertLess(abs(tilt.grad.item()), 1e-4)


class ObjectiveTests(unittest.TestCase):
    def terms(self, keep=1.0, e_z=0.0, e_tilt=0.0, H=4, B=2):
        shape = (H, B)
        r = torch.full(shape, 0.1)
        return r, r.clone(), torch.full(shape, keep), torch.full(shape, e_z), torch.full(shape, e_tilt)

    def test_track_keep_is_the_discounted_sum(self):
        out = probe.objectives(*self.terms(), **TERMS)
        expected = 0.1 * sum(0.99**t for t in range(4))
        self.assertAlmostEqual(out["track_keep"][0].item(), expected, places=5)

    def test_first_reward_is_not_gated_by_survival(self):
        # 与Trainer._policy_step相同：第t步奖励乘此前各步的存活概率，第0步权重为1
        out = probe.objectives(*self.terms(keep=0.0), **TERMS)
        self.assertAlmostEqual(out["track_surv"][0].item(), 0.1, places=6)
        self.assertGreater(out["track_keep"][0].item(), 0.1)

    def test_recover_equals_tracking_when_not_deviating(self):
        out = probe.objectives(*self.terms(), **TERMS)
        self.assertAlmostEqual(out["recover"][0].item(), out["track_keep"][0].item(), places=2)

    def test_recover_task_excludes_regularisation(self):
        r, track, keep, e_z, e_tilt = self.terms()
        out = probe.objectives(r - 0.05, track, keep, e_z, e_tilt, **TERMS)  # 正则项为-0.05
        self.assertAlmostEqual(out["recover"][0].item() - out["recover_task"][0].item(), -0.05 * sum(0.99**t for t in range(4)), places=5)

    def test_recover_pays_the_recovery_term_when_deviating(self):
        r, track, keep, e_z, e_tilt = self.terms(e_z=0.6, e_tilt=1.5)
        out = probe.objectives(r, track * 0, keep, e_z, e_tilt, **TERMS)
        expected_step = (probe.recover_reward(torch.tensor(0.6), torch.tensor(1.5)) * 0.02 + 0.1).item()
        self.assertAlmostEqual(out["recover"][0].item(), expected_step * sum(0.99**t for t in range(4)), places=4)


class SamplingTests(unittest.TestCase):
    def test_frames_leave_room_inside_the_clip(self):
        start = torch.tensor([0, 10, 100, 1100])
        length = torch.tensor([10, 90, 1000, 200])
        frames = probe.pick_frames(start, length, 150, 5000)
        self.assertTrue((frames >= 100).all())
        clip_end = (start + length)[torch.bucketize(frames, start, right=True) - 1]
        self.assertTrue((frames + 150 < clip_end).all())
        self.assertGreaterEqual(int((frames >= 1100).sum()), 1)  # 第二段够长的片段也会被选到

    def test_no_long_clip_is_an_error(self):
        with self.assertRaises(ValueError):
            probe.pick_frames(torch.tensor([0]), torch.tensor([10]), 100, 4)

    def test_pushes_are_horizontal_and_in_range(self):
        out = probe.sample_pushes(4000, 0.75, (1.0, 5.0), 3.0, (50, 150), "cpu")
        self.assertTrue((out["dv"][:, 2] == 0).all())
        speed = out["dv"].norm(dim=-1)
        self.assertGreaterEqual(speed.min().item(), 1.0 - 1e-5)
        self.assertLessEqual(speed.max().item(), 5.0 + 1e-5)
        pushed = out["step"] >= 0
        self.assertAlmostEqual(pushed.float().mean().item(), 0.75, delta=0.05)
        self.assertTrue(((out["step"][pushed] >= 50) & (out["step"][pushed] <= 150)).all())
        self.assertTrue((out["step"][~pushed] == -1).all())
        self.assertLessEqual(out["dw"].abs().max().item(), 3.0)

    def test_starts_cover_categories_and_fill_quota(self):
        a = torch.zeros(10, 4, dtype=torch.bool)
        b = torch.zeros(10, 4, dtype=torch.bool)
        a[:, :2] = True
        b[3, 3] = True  # 只有一个样本的类别
        t, e, label = probe.pick_starts({"a": a, "b": b, "none": torch.zeros_like(a)}, 12)
        self.assertEqual(len(t), 12)
        self.assertEqual(int((label == 2).sum()), 0)
        self.assertGreaterEqual(int((label == 1).sum()), 1)
        masks = torch.stack((a, b))
        self.assertTrue(masks[label, t, e].all())

    def test_describe_counts_nonfinite_separately(self):
        out = probe.describe(torch.tensor([1.0, 3.0, float("nan"), float("inf")]))
        self.assertEqual((out["n"], out["nonfinite"]), (4, 2))
        self.assertAlmostEqual(out["mean"], 2.0)

    def test_refit_norm_matches_the_batch(self):
        class Norm:
            mean, var, count = torch.zeros(2), torch.ones(2), torch.zeros(())

        x = torch.randn(500, 2) * torch.tensor([3.0, 0.5]) + 1
        probe.refit_norm(Norm, x)
        self.assertTrue(torch.allclose(Norm.mean, x.mean(0)))
        self.assertTrue(torch.allclose(Norm.var, x.var(0, unbiased=False)))
        self.assertEqual(Norm.count.item(), 500)


if __name__ == "__main__":
    unittest.main()
