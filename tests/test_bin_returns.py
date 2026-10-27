"""Simultaneous episode returns must contribute once per bin, without colliding CUDA writes."""

import unittest

import torch

from supermimic.utils.distributed import DistributedContext, global_bin_probabilities, update_bin_returns


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
