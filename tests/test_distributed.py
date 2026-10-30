"""CPU/Gloo tests of real collective operations; no CUDA or simulator required."""

import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from supermimic.tasks.tracking.rl.networks import RunningNorm
from supermimic.tasks.tracking.reset_sampling import BinSampler, cap_clip_share, global_bin_probabilities
from supermimic.utils.distributed import DistributedContext, LocalContext, average_gradients


def collective_worker(rank, root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    parallel = DistributedContext("cpu", init_method=(Path(root) / "store").as_uri())
    try:
        assert parallel.local_size(8, "batch") == 4
        assert parallel.local_size(7, "batch") == (4 if rank == 0 else 3)
        try:
            parallel.local_size(1, "batch")
        except ValueError:
            pass
        else:
            raise AssertionError("an empty local batch was accepted")

        torch.manual_seed(10 + rank)
        model = torch.nn.Linear(2, 1)
        parallel.broadcast_module(model)
        baseline = copy.deepcopy(model)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        base_opt = torch.optim.Adam(baseline.parameters(), lr=0.01)
        data = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        target = torch.arange(5, dtype=torch.float32).reshape(5, 1) / 10
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            base_opt.zero_grad(set_to_none=True)
            local = slice(0, 3) if rank == 0 else slice(3, 5)
            ((model(data[local]) - target[local]) ** 2).mean().backward()
            ((baseline(data) - target) ** 2).mean().backward()
            average_gradients(model, len(data[local]))
            for p, expected in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(p.grad, expected.grad, rtol=2e-6, atol=1e-7)
            # Clip only after the global mean gradient has been formed.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)
            opt.step()
            base_opt.step()
            for p, expected in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(p, expected, rtol=2e-6, atol=1e-7)
        parallel.require_equal([p.detach().tolist() for p in model.parameters()], "updated parameters")

        # The fixed-count path used inside CUDA graphs matches the general path bitwise.
        general, fixed = copy.deepcopy(model), copy.deepcopy(model)
        for m in (general, fixed):
            ((m(data[local]) - target[local]) ** 2).mean().backward()
        average_gradients(general, len(data[local]))
        average_gradients(fixed, len(data[local]), total_count=len(data))
        for p, q in zip(general.parameters(), fixed.parameters()):
            assert torch.equal(p.grad, q.grad)
        fixed.bias.grad = None
        try:
            average_gradients(fixed, len(data[local]), total_count=len(data))
        except ValueError:
            pass
        else:
            raise AssertionError("the fixed-count path accepted a missing gradient")
        try:
            parallel.broadcast_module(torch.nn.Linear(1 + rank, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched model layouts were not rejected before broadcast")

        unused = torch.nn.Linear(1, 1)
        unused.register_parameter("spare", torch.nn.Parameter(torch.ones(1)))
        unused.weight.grad = torch.full_like(unused.weight, 4) if rank == 0 else None
        average_gradients(unused)
        torch.testing.assert_close(unused.weight.grad, torch.full_like(unused.weight, 2))
        assert unused.bias.grad is None and unused.spare.grad is None

        # Unequal sample counts and different means expose incorrect averaging of local variances.
        all_x = torch.tensor([[10000., -3.], [10001., 2.], [10004., 7.], [10009., 1.]])
        x = all_x[:1] if rank == 0 else all_x[1:]
        norm = RunningNorm(2, max_count=8)
        norm.update(x)
        assert float(norm.count) == 4
        torch.testing.assert_close(norm.mean, all_x.mean(0))
        torch.testing.assert_close(norm.var, all_x.var(0, unbiased=False))
        norm.update(x + 3)
        expected = torch.cat((all_x, all_x + 3))
        assert float(norm.count) == 8
        torch.testing.assert_close(norm.mean, expected.mean(0))
        torch.testing.assert_close(norm.var, expected.var(0, unbiased=False))
        norm.update(x * 10)
        assert float(norm.count) == 8
        parallel.require_equal([norm.mean.tolist(), norm.var.tolist(), float(norm.count)], "normalizer state")

        all_returns = torch.tensor([0., 0.02, 0.1, 0.3, 0.8])
        local = slice(0, 2) if rank == 0 else slice(2, 5)
        returns = all_returns[local]
        ema = torch.full_like(returns, 1 / 5)
        expected_ema = torch.full_like(all_returns, 1 / 5)
        for _ in range(3):
            probs = global_bin_probabilities(returns, ema, 1., 5, parallel)
            soft = torch.softmax(-25 * all_returns, 0)
            expected_ema.mul_(0.999).add_(0.001 * soft)
            expected = 0.5 / 5 + 0.4 * soft + 0.1 * expected_ema / expected_ema.sum()
            torch.testing.assert_close(probs, expected[local])
            assert abs(float(parallel.reduce(probs.sum())) - 1) < 1e-6

        # Unvisited bins score the mean over every rank's visited bins.
        all_visits = torch.tensor([1, 0, 1, 1, 0])
        prior = global_bin_probabilities(returns, torch.full_like(returns, 1 / 5), 1., 5, parallel, all_visits[local])
        filled = torch.where(all_visits > 0, all_returns, all_returns[all_visits > 0].mean())
        reference = global_bin_probabilities(filled, torch.full_like(all_returns, 1 / 5), 1., 5, SimpleNamespace(reduce=lambda x, op=None: torch.as_tensor(x, dtype=torch.float64)))
        torch.testing.assert_close(prior, reference[local])

        # No scored bin on one rank still sums to 1 globally; on no rank, sampling is uniform.
        everything = torch.ones_like(returns, dtype=torch.bool)
        one_rank = global_bin_probabilities(returns, torch.full_like(returns, 1 / 5), 1., 5, parallel, None, everything,
                                            everything if rank == 1 else ~everything)
        assert abs(float(parallel.reduce(one_rank.sum())) - 1) < 1e-6
        none = global_bin_probabilities(returns, torch.full_like(returns, 1 / 5), 1., 5, parallel, None, everything, ~everything)
        torch.testing.assert_close(none, torch.full_like(returns, 1 / 5))

        # 片段份额上限：片段分在两个rank上（rank0两段，rank1一段），与单进程持有全部片段时逐片段一致
        torch.testing.assert_close(parallel.all_gather(torch.tensor([rank, 10 + rank])), torch.tensor([[0, 10], [1, 11]]))
        all_soft = torch.softmax(-25 * all_returns.double(), 0)
        all_clip = torch.tensor([0, 1, 2, 2, 2])  # 第3段跨分箱2–4，整段在rank1
        local_clip = all_clip[local] - (0 if rank == 0 else 2)
        capped, stats = cap_clip_share(all_soft[local], local_clip, 2 if rank == 0 else 1, 0.4, parallel)
        reference, ref_stats = cap_clip_share(all_soft, all_clip, 3, 0.4, LocalContext)
        torch.testing.assert_close(capped, reference[local])
        torch.testing.assert_close(stats, ref_stats)
        assert int(stats[1]) == 2 and abs(float(parallel.reduce(capped.sum())) - 1) < 1e-12

        # BinSampler.prepare：每个分箱自成一段长片段，都可重置、可评分
        n = len(returns)
        bin_start = torch.arange(n) * 2000
        motion = SimpleNamespace(num_bins=n, global_num_bins=5, bin_clip=torch.arange(n), clip_start=bin_start,
                                 clip_end=bin_start + 1000, bin_start=bin_start, bin_end=bin_start + 50)
        sampler = BinSampler(motion, "cpu", 0.99, parallel)
        assert bool(sampler.allowed.all() and sampler.scored.all())
        sampler.scores, sampler.ema = returns.clone(), ema.clone()
        sampler.prepare(rank == 0)  # Only one rank has completed episodes.
        soft = torch.softmax(-25 * all_returns, 0)
        expected_ema.mul_(0.999).add_(0.001 * soft)
        expected = 0.5 / 5 + 0.4 * soft + 0.1 * expected_ema / expected_ema.sum()
        torch.testing.assert_close(sampler.probs, expected[local])
        before = sampler.ema.clone()
        sampler.prepare(False)  # No rank resets: do not advance the EMA.
        torch.testing.assert_close(sampler.ema, before, rtol=0, atol=0)

        # A failure on only one rank must reach its peers before the next collective.
        try:
            with parallel.coordinated("injected failure"):
                if rank == 1:
                    raise ValueError("expected test error")
        except RuntimeError as exc:
            assert "rank 1" in str(exc) and "expected test error" in str(exc)
        else:
            raise AssertionError("rank-local failure was not coordinated")
        try:
            parallel.require_equal(rank, "injected mismatch")
        except ValueError:
            pass
        else:
            raise AssertionError("rank mismatch was not rejected")
        (Path(root) / f"rank_{rank}.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    finally:
        parallel.close()


class DistributedTests(unittest.TestCase):
    def test_cuda_binding_uses_local_rank_not_global_rank(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "5", "LOCAL_RANK": "1"}), \
             patch("torch.cuda.device_count", return_value=4), \
             patch("torch.cuda.set_device") as set_device, \
             patch("torch.distributed.init_process_group") as init:
            parallel = DistributedContext("cuda:0")
            self.assertEqual(parallel.device, "cuda:1")
            set_device.assert_called_once_with(1)
            self.assertEqual(init.call_args.kwargs["backend"], "nccl")
            self.assertEqual(init.call_args.kwargs["rank"], 5)

    def test_single_process_is_a_noop(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            parallel = DistributedContext("cpu")
            self.assertFalse(parallel.enabled)
            self.assertEqual(parallel.local_size(7, "batch"), 7)
            model = torch.nn.Linear(1, 1)
            (model(torch.ones(2, 1)) ** 2).sum().backward()
            before = [p.grad.clone() for p in model.parameters()]
            average_gradients(model)
            for p, expected in zip(model.parameters(), before):
                torch.testing.assert_close(p.grad, expected, atol=0, rtol=0)
            parallel.close()

    def test_real_two_process_collectives(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.spawn(collective_worker, args=(root,), nprocs=2, join=False)
            deadline = time.monotonic() + 75
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    for process in context.processes:
                        process.terminate()
                    for process in context.processes:
                        process.join(timeout=5)
                    self.fail("Gloo collective test timed out")
            for rank in range(2):
                self.assertTrue(json.loads((Path(root) / f"rank_{rank}.json").read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
