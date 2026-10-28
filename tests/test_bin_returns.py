"""Simultaneous episode returns must contribute once per bin, without colliding CUDA writes."""

import unittest

import torch

from types import SimpleNamespace

from supermimic.tasks.tracking.reset_sampling import BinSampler, episode_bin_scores, global_bin_probabilities, update_bin_returns
from supermimic.utils.distributed import DistributedContext, LocalContext

GAMMA, STEP_MAX = 0.99, 0.1


def episodes(rewards, skip, terminated=False, at_clip_end=False, left=1000):
    """Reward sums of one episode per reward list, in the layout TrackingSim passes to episode_bin_scores."""
    disc = torch.tensor([sum(GAMMA**t * r for t, r in enumerate(rs)) for rs in rewards])
    ret = torch.tensor([float(sum(rs)) for rs in rewards])
    head_disc = torch.tensor([sum(GAMMA**t * r for t, r in enumerate(rs[:skip])) for rs in rewards])
    head_ret = torch.tensor([float(sum(rs[:skip])) for rs in rewards])
    steps = torch.tensor([float(len(rs)) for rs in rewards])
    n = len(rewards)
    return (disc, ret, head_disc, head_ret, steps, torch.full((n,), terminated), torch.full((n,), at_clip_end),
            torch.full((n,), float(left)))


class BinScoreTests(unittest.TestCase):
    def test_no_skip_reproduces_the_previous_score(self):
        disc, ret, hd, hr, steps, term, end, left = episodes([[0.09] * 7, [-0.2, 0.08, 0.08]], 0, at_clip_end=True, left=7)
        boot = GAMMA**steps * (ret / steps) / (1 - GAMMA)
        g_max = STEP_MAX / (1 - GAMMA)
        score, valid = episode_bin_scores(disc, ret, hd, hr, steps, term, end, left, GAMMA, STEP_MAX)
        torch.testing.assert_close(score, (disc + boot) / g_max)
        self.assertTrue(bool(valid.all()))
        score, _ = episode_bin_scores(disc, ret, hd, hr, steps, term, end, left, GAMMA, STEP_MAX, "terminal")
        torch.testing.assert_close(score, disc / (g_max * (1 - GAMMA**left)))

    def test_skip_removes_the_reset_transient(self):
        # same steady tracking reward; the second episode pays a three-step transient after its reset
        rewards = [[0.09] * 6, [-0.2, -0.1, 0.0, 0.09, 0.09, 0.09]]
        kept, _ = episode_bin_scores(*episodes(rewards, 0, at_clip_end=True, left=6), GAMMA, STEP_MAX)
        skipped, valid = episode_bin_scores(*episodes(rewards, 3, at_clip_end=True, left=6), GAMMA, STEP_MAX, skip_steps=3)
        self.assertLess(float(kept[1]), float(kept[0]) - 0.5)  # extrapolated transient makes the bin look hard
        torch.testing.assert_close(skipped[1], skipped[0])
        torch.testing.assert_close(skipped[0], torch.tensor(0.9))  # 0.09 per step, extended, over G_max
        self.assertTrue(bool(valid.all()))

    def test_episodes_within_the_transient(self):
        truncated = episodes([[-0.3, 0.05]], 3, at_clip_end=True, left=2)
        _, valid = episode_bin_scores(*truncated, GAMMA, STEP_MAX, skip_steps=3)
        self.assertFalse(bool(valid[0]))  # nothing after the transient: no evidence about the bin
        fell = episodes([[-0.3, 0.05]], 3, terminated=True, left=2)
        for mode in ("bootstrap", "terminal"):
            score, valid = episode_bin_scores(*fell, GAMMA, STEP_MAX, mode, skip_steps=3)
            self.assertTrue(bool(valid[0]))  # falling during the transient is a real failure
            self.assertTrue(bool(torch.isfinite(score).all()))
            self.assertEqual(float(score[0]), 0.0)

    def test_unallowed_and_unscored_bins(self):
        scores = torch.tensor([0.9, 0.5, 0.0, 0.0])
        allowed = torch.tensor([True, True, True, False])
        scored = torch.tensor([True, True, False, False])
        probs = global_bin_probabilities(scores, torch.full((4,), 0.25), 1.0, 4, LocalContext, None, allowed, scored)
        self.assertEqual(float(probs[3]), 0.0)  # never a reset start
        torch.testing.assert_close(probs[2], torch.tensor(0.5 / 3))  # uniform share only, despite its score 0
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        self.assertGreater(float(probs[1]), float(probs[0]))  # the lower-scored bin gets the softmax share
        everywhere = torch.ones(4, dtype=torch.bool)
        masked = global_bin_probabilities(scores, torch.full((4,), 0.25), 1.0, 4, LocalContext, None, everywhere, everywhere)
        plain = global_bin_probabilities(scores, torch.full((4,), 0.25), 1.0, 4, LocalContext)
        torch.testing.assert_close(masked, plain)

    def test_no_scored_bin_samples_uniformly(self):
        allowed = torch.tensor([True, True, False, True])
        probs = global_bin_probabilities(torch.zeros(4), torch.full((4,), 0.25), 1.0, 4, LocalContext, None, allowed,
                                         torch.zeros(4, dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(probs).all()))
        torch.testing.assert_close(probs, torch.tensor([1 / 3, 1 / 3, 0.0, 1 / 3]))

    def test_collapsed_scores_do_not_concentrate_the_softmax(self):
        n = 100
        healthy = torch.linspace(0.6, 0.9, n)
        collapsed = healthy.clone()
        collapsed[:10] = torch.linspace(-3.0, -0.5, 10)  # failing bins whose penalties outweigh the tracking reward
        ema = torch.full((n,), 1.0 / n)
        probs = global_bin_probabilities(collapsed, ema.clone(), 1.0, n, LocalContext)
        self.assertLess(float(probs.max()), 0.4 / 10 + 0.1)  # the failing bins tie instead of one taking the whole share
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        torch.testing.assert_close(probs[:10], probs[0].expand(10))
        # scores inside [0, 1] are untouched
        clamped_free = global_bin_probabilities(healthy, ema.clone(), 1.0, n, LocalContext)
        manual = torch.softmax(-25.0 * healthy.double(), 0).float()
        kept = 0.999 * ema + 0.001 * manual.float()  # the EMA share is updated in place before it is read
        torch.testing.assert_close(clamped_free, 0.5 / n + 0.4 * manual.float() + 0.1 * kept / kept.sum(), atol=1e-7, rtol=1e-5)

    def test_non_finite_score_neither_updates_a_bin_nor_poisons_the_weights(self):
        disc = torch.tensor([1.0, float("nan"), 2.0])
        ret, hd, hr = disc.clone(), torch.zeros(3), torch.zeros(3)
        steps = torch.full((3,), 200.0)
        score, valid = episode_bin_scores(disc, ret, hd, hr, steps, torch.zeros(3, dtype=torch.bool),
                                          torch.zeros(3, dtype=torch.bool), torch.full((3,), 1000.0), GAMMA, STEP_MAX)
        self.assertEqual(valid.tolist(), [True, False, True])
        probs = global_bin_probabilities(torch.tensor([0.8, float("nan"), 0.7, float("inf")]), torch.full((4,), 0.25), 1.0, 4,
                                         LocalContext)
        self.assertTrue(bool(torch.isfinite(probs).all()))
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


class BinReturnTests(unittest.TestCase):
    def test_visits_and_first_update_replacement(self):
        values = torch.zeros(3)
        visits = torch.zeros(3, dtype=torch.long)
        update_bin_returns(values, torch.tensor([0, 0, 1]), torch.tensor([0.6, 0.8, 0.5]), visits=visits)
        torch.testing.assert_close(values, torch.tensor([0.14, 0.1, 0.0]))  # blended into the initial 0
        self.assertEqual(visits.tolist(), [2, 1, 0])
        values = torch.zeros(3)
        visits = torch.zeros(3, dtype=torch.long)
        update_bin_returns(values, torch.tensor([0, 0, 1]), torch.tensor([0.6, 0.8, 0.5]), visits=visits, replace_first=True)
        torch.testing.assert_close(values, torch.tensor([0.7, 0.5, 0.0]))  # first update takes the mean
        update_bin_returns(values, torch.tensor([1]), torch.tensor([1.0]), visits=visits, replace_first=True)
        torch.testing.assert_close(values, torch.tensor([0.7, 0.6, 0.0]))  # later updates blend again
        self.assertEqual(visits.tolist(), [2, 2, 0])

    def test_unvisited_bins_take_the_visited_mean(self):
        parallel = DistributedContext("cpu")
        scores = torch.tensor([0.9, 0.5, 0.0, 0.0])
        visits = torch.tensor([3, 1, 0, 0])
        with_prior = global_bin_probabilities(scores, torch.full((4,), 0.25), 1.0, 4, parallel, visits)
        expected = global_bin_probabilities(torch.tensor([0.9, 0.5, 0.7, 0.7]), torch.full((4,), 0.25), 1.0, 4, parallel)
        torch.testing.assert_close(with_prior, expected)
        zero_prior = global_bin_probabilities(scores, torch.full((4,), 0.25), 1.0, 4, parallel)
        self.assertGreater(float(zero_prior[2]), float(with_prior[2]))  # 0 makes an unvisited bin look hardest
        torch.testing.assert_close(with_prior.sum(), torch.tensor(1.0))

    def check_device(self, device):
        values = torch.tensor([5., 10., 30., 40.], device=device)
        bins = torch.tensor([2, 0, 2, 2, 0], device=device)
        returns = torch.tensor([10., 2., 20., 30., 8.], device=device)
        update_bin_returns(values, bins, returns)
        torch.testing.assert_close(values, torch.tensor([5., 10., 28., 40.], device=device))
        before = values.clone()
        update_bin_returns(values, bins[:0], returns[:0])
        self.assertTrue(torch.equal(values, before))
        bins = torch.tensor([2, 0], device=device)
        returns = torch.tensor([1., 9.], device=device)
        expected = values.clone()
        expected[bins] = 0.8 * expected[bins] + 0.2 * returns
        update_bin_returns(values, bins, returns)
        self.assertTrue(torch.equal(values, expected))

    def test_cpu_group_mean_and_unique_compatibility(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_group_mean_and_repeatability(self):
        self.check_device("cuda")
        generator = torch.Generator(device="cuda").manual_seed(7)
        bins = torch.randint(0, 128, (4096,), device="cuda", generator=generator)
        returns = torch.randn(4096, device="cuda", generator=generator)
        reference = None
        for _ in range(40):
            values = torch.zeros(128, device="cuda")
            update_bin_returns(values, bins, returns)
            if reference is None:
                reference = values.clone()
            else:
                self.assertTrue(torch.equal(values, reference))


class BinSamplerTests(unittest.TestCase):
    def setUp(self):
        # 两段：长段300帧分成6箱，短段60帧只有1箱（不足reset_margin，只能从首帧起步）
        lengths = torch.tensor([300, 60])
        clip_start = torch.tensor([0, 300])
        bin_start = torch.tensor([0, 50, 100, 150, 200, 250, 300])
        self.motion = SimpleNamespace(
            num_bins=7, global_num_bins=7, bin_clip=torch.tensor([0, 0, 0, 0, 0, 0, 1]), clip_start=clip_start,
            clip_end=clip_start + lengths - 1, clip_id=torch.repeat_interleave(torch.arange(2), lengths),
            bin_start=bin_start, bin_end=torch.tensor([50, 100, 150, 200, 250, 299, 359]))
        self.sampler = BinSampler(self.motion, "cpu", GAMMA, reset_margin=100, skip_steps=50)

    def test_reset_range_respects_clip_end_margin(self):
        s = self.sampler
        self.assertEqual(s.allowed.tolist(), [True, True, True, True, False, False, True])
        self.assertEqual(s.scored.tolist(), [True, True, True, True, False, False, False])
        torch.manual_seed(0)
        frames, bins = s.sample(5000)
        self.assertTrue(bool((frames >= self.motion.bin_start[bins]).all() and (frames < s.reset_end[bins]).all()))
        self.assertTrue(bool((frames[bins == 3] < 199).all()))  # 不在段尾100帧内起步
        self.assertTrue(bool((frames[bins == 6] == 300).all()))  # 短段只从首帧起步
        stats = s.stats()
        self.assertEqual(stats[3].item(), 5000)  # 重置次数，读取后清零
        self.assertEqual(s.stats()[3].item(), 0)

    def test_record_updates_scores_of_valid_episodes(self):
        s = self.sampler
        # 第一个回合在过渡期内摔倒（计0分），第二个回合截断在过渡期后（按平均奖励外推）
        steps = torch.tensor([10.0, 80.0])
        ret = torch.tensor([0.5, 8.0])
        s.record(torch.tensor([0, 50]), torch.tensor([0, 1]), ret * 0.9, ret, ret * 0.9, torch.tensor([0.5, 4.0]),
                 steps, torch.tensor([True, False]), torch.tensor([False, False]), STEP_MAX)
        self.assertEqual(s.visits.tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(s.scores[0].item(), 0.0)
        self.assertGreater(s.scores[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
