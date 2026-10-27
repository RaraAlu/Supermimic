"""Simultaneous episode returns must contribute once per bin, without colliding CUDA writes."""

import unittest

import torch

from supermimic.utils.distributed import DistributedContext, episode_bin_scores, global_bin_probabilities, update_bin_returns

GAMMA, STEP_MAX = 0.99, 0.1


def episodes(rewards, terminated=False, at_clip_end=False, left=1000):
    """Reward sums of one episode per reward list, in the layout TrackingSim passes to episode_bin_scores."""
    disc = torch.tensor([sum(GAMMA**t * r for t, r in enumerate(rs)) for rs in rewards])
    ret = torch.tensor([float(sum(rs)) for rs in rewards])
    first = torch.tensor([float(rs[0]) for rs in rewards])
    steps = torch.tensor([float(len(rs)) for rs in rewards])
    n = len(rewards)
    return (disc, ret, first, steps, torch.full((n,), terminated), torch.full((n,), at_clip_end),
            torch.full((n,), float(left)))


class BinScoreTests(unittest.TestCase):
    def test_keep_reproduces_the_previous_score(self):
        disc, ret, first, steps, term, end, left = episodes([[0.09] * 7, [-0.2, 0.08, 0.08]], at_clip_end=True, left=7)
        boot = GAMMA**steps * (ret / steps) / (1 - GAMMA)
        g_max = STEP_MAX / (1 - GAMMA)
        score, valid = episode_bin_scores(disc, ret, first, steps, term, end, left, GAMMA, STEP_MAX, skip_first=False)
        torch.testing.assert_close(score, (disc + boot) / g_max)
        self.assertTrue(bool(valid.all()))
        score, _ = episode_bin_scores(disc, ret, first, steps, term, end, left, GAMMA, STEP_MAX, "terminal", skip_first=False)
        torch.testing.assert_close(score, disc / (g_max * (1 - GAMMA**left)))

    def test_skip_removes_the_reset_penalty_from_short_episodes(self):
        # same tracking reward; the second episode pays a large first-step smoothness penalty
        args = episodes([[0.09, 0.09, 0.09], [-0.2, 0.09, 0.09]], at_clip_end=True, left=3)
        kept, _ = episode_bin_scores(*args, GAMMA, STEP_MAX, skip_first=False)
        skipped, valid = episode_bin_scores(*args, GAMMA, STEP_MAX, skip_first=True)
        self.assertLess(float(kept[1]), float(kept[0]) - 0.5)  # extrapolated penalty makes the bin look hard
        torch.testing.assert_close(skipped[1], skipped[0])
        torch.testing.assert_close(skipped[0], torch.tensor(0.9))  # 0.09 per step, extended, over G_max
        self.assertTrue(bool(valid.all()))

    def test_one_step_episodes(self):
        truncated = episodes([[-0.3]], at_clip_end=True, left=1)
        _, valid = episode_bin_scores(*truncated, GAMMA, STEP_MAX)
        self.assertFalse(bool(valid[0]))  # nothing after the reset step: no evidence about the bin
        fell = episodes([[-0.3]], terminated=True, left=1)
        for mode in ("bootstrap", "terminal"):
            score, valid = episode_bin_scores(*fell, GAMMA, STEP_MAX, mode)
            self.assertTrue(bool(valid[0]))  # falling at once is a real failure
            self.assertTrue(bool(torch.isfinite(score).all()))
            self.assertEqual(float(score[0]), 0.0)


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


if __name__ == "__main__":
    unittest.main()
