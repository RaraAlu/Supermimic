"""Simultaneous episode returns must contribute once per bin, without colliding CUDA writes."""

import unittest

import torch

from supermimic.utils.distributed import update_bin_returns


class BinReturnTests(unittest.TestCase):
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
