"""Small torch.distributed primitives for the custom world-model training loop."""

from contextlib import contextmanager
from datetime import timedelta
import atexit
import os

import torch
import torch.distributed as dist


def distributed_active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


@torch.no_grad()
def global_moments(x):
    """Merge batch moments, including between-rank variance, without gathering samples."""
    if not distributed_active():
        return x.shape[0], x.mean(0), x.var(0, unbiased=False)
    values = x.double()
    # filled on the device, not copied from a host list, so this also runs inside a captured CUDA graph
    totals = torch.cat((torch.full((1,), len(x), dtype=values.dtype, device=values.device), values.sum(0)))
    dist.all_reduce(totals)
    n, mean = totals[0], totals[1:] / totals[0].clamp(min=1)
    # the check needs the host, which a graph capture cannot wait on; the eager calls before capture run it
    if not (values.is_cuda and torch.cuda.is_current_stream_capturing()) and n <= 0:
        raise ValueError("normalizer received no samples on any rank")
    m2 = ((values - mean) ** 2).sum(0)
    dist.all_reduce(m2)
    return n.float(), mean.float(), (m2 / n).float()


@torch.no_grad()
def average_gradients(module, sample_count=1, total_count=None):
    """Reduce before clipping; preserve globally-unused parameters as grad=None.

    With total_count, the global sample count fixed for the run, every parameter must have a gradient on
    every rank. The reduction then needs no host round trip, so it can run inside a CUDA graph.
    """
    if not distributed_active():
        return
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return
    if total_count is not None:
        if any(p.grad is None for p in params):
            raise ValueError("average_gradients(total_count=...) requires a gradient for every parameter")
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        flat.mul_(sample_count)
        dist.all_reduce(flat)
        flat.div_(total_count)
        offset = 0
        for p in params:
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
        return
    info = torch.tensor([sample_count, *[p.grad is not None for p in params]],
                        device=params[0].device, dtype=torch.float64)
    dist.all_reduce(info)
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    flat.mul_(sample_count)
    dist.all_reduce(flat)
    flat.div_(info[0].to(flat.dtype))
    offset = 0
    used = info[1:].tolist()
    for p, count in zip(params, used):
        value = flat[offset:offset + p.numel()].view_as(p)
        if count:
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
        offset += p.numel()


@torch.no_grad()
def update_bin_returns(estimates, bins, returns, alpha=0.2, visits=None, replace_first=False):
    """Apply one EMA update per bin using its mean episode return in this simulator step.

    Several environments can finish in the same bin. Sort them in a stable order and reduce each
    segment before writing: index_put_ with duplicate destinations silently loses observations and
    has undefined results on CUDA. Unique bins retain the original single-episode update exactly.
    ``visits`` counts finished episodes per bin; with ``replace_first`` a bin's first update takes
    the observed mean instead of blending it into the initial value.
    """
    if bins.numel() == 0:
        return
    order = torch.argsort(bins, stable=True)
    unique, counts = torch.unique_consecutive(bins[order], return_counts=True)
    means = torch.segment_reduce(returns[order], "mean", lengths=counts)
    rate = alpha
    if visits is not None:
        if replace_first:
            rate = torch.where(visits[unique] == 0, 1.0, alpha).to(estimates.dtype)
        visits[unique] += counts.to(visits.dtype)
    estimates[unique] = (1 - rate) * estimates[unique] + rate * means


@torch.no_grad()
def episode_bin_scores(disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, left, gamma, step_max,
                       clip_end="bootstrap", max_steps=500, skip_steps=0):
    """Normalized reset-bin scores of finished episodes, and which episodes may update their bin.

    disc_ret and ret are the discounted and plain reward sums from the start frame, head_disc and
    head_ret the same sums over the first skip_steps steps, steps the episode length, left the steps
    from the start frame to the clip end. Truncated episodes extend their return with their mean
    reward ("bootstrap"); with "terminal", an episode that reaches the clip end stops there and is
    scored against the maximum attainable from its start, and only the max_steps cap is extended.

    With skip_steps > 0 the score is the return after the reset transient: right after a reset the
    policy recovers from reset noise, a zero previous action and a repeated observation history,
    which costs reward for about a second regardless of the bin, and in a short episode that cost
    dominates the mean reward the bootstrap extends. An episode that falls within the transient
    still scores 0; one truncated within it carries no other evidence and does not update its bin.
    """
    if skip_steps:
        disc_ret, ret = (disc_ret - head_disc) / gamma**skip_steps, ret - head_ret
        steps, left, max_steps = steps - skip_steps, left - skip_steps, max_steps - skip_steps
    valid = terminated | (steps > 0)
    ended = terminated | at_clip_end if clip_end == "terminal" else terminated
    G = disc_ret + torch.where(ended, 0.0, gamma**steps * (ret / steps.clamp(min=1)) / (1 - gamma))
    G_max = torch.full_like(G, step_max / (1 - gamma))
    if clip_end == "terminal":
        G_max = G_max * torch.where(left < max_steps, 1 - gamma ** left.clamp(min=1), 1.0)
    return G / G_max, valid


@torch.no_grad()
def global_bin_probabilities(returns, ema, maximum_return, num_bins, parallel, visits=None, allowed=None, scored=None):
    """Global 50/40/10 reset weights; sampling remains conditional on the local shard.

    With ``visits``, bins without a finished episode score the global mean of visited bins.
    ``allowed`` marks bins that may be reset to (the others get probability 0, and the uniform share
    is spread over allowed bins only); ``scored`` marks allowed bins whose score enters the softmax
    and the EMA. Bins that can never produce a score take only the uniform share, so an unscorable
    bin stuck at its initial score cannot attract the adaptive share. With no scored bin on any rank
    (every clip too short to score), the adaptive half joins the uniform share.
    """
    if allowed is None:
        allowed = torch.ones_like(returns, dtype=torch.bool)
    else:
        num_bins = parallel.reduce(allowed.sum().double())
    scored = allowed if scored is None else scored & allowed
    if visits is not None:
        seen = (visits > 0) & scored
        total, count = parallel.reduce(torch.stack(((returns * seen).double().sum(), seen.sum().double())))
        returns = torch.where(seen, returns, (total / count.clamp(min=1)).to(returns.dtype))
    logits = torch.where(scored, -25.0 * returns / maximum_return, -torch.inf)
    maximum = parallel.reduce(logits.max(), dist.ReduceOp.MAX)
    exp = torch.where(scored, torch.exp(logits.double() - maximum), 0.0)
    norm = parallel.reduce(exp.sum())  # >= 1 (the global maximum contributes exp(0)) unless nothing is scored
    soft = (exp / norm.clamp(min=1e-300)).to(returns.dtype)
    ema.mul_(0.999).add_(0.001 * soft)
    kept = torch.where(scored, ema, 0.0)
    total_ema = parallel.reduce(kept.double().sum()).clamp(min=1e-300)
    probs = 0.5 / num_bins + 0.4 * soft + 0.1 * kept / total_ema.to(ema.dtype)
    probs = torch.where(norm > 0, probs, 1.0 / num_bins)
    return torch.where(allowed, probs, 0.0).to(returns.dtype)


class LocalContext:
    """Single-process stand-in for DistributedContext.reduce."""

    enabled = False

    @staticmethod
    def reduce(values, op=None):
        return values.double() if torch.is_tensor(values) else torch.as_tensor(values, dtype=torch.float64)


class DistributedContext:
    def __init__(self, device, *, init_method="env://"):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size or self.local_rank < 0:
            raise ValueError("invalid torchrun rank/world size")
        self.enabled = self.world_size > 1
        self.device = str(device)
        if self.enabled:
            if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
                raise ValueError("multi-process training requires torchrun RANK and LOCAL_RANK")
            if torch.device(device).type == "cuda":
                if self.local_rank >= torch.cuda.device_count():
                    raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
                backend = "nccl"
            elif torch.device(device).type == "cpu":
                backend = "gloo"
            else:
                raise ValueError("distributed training supports CUDA/NCCL or CPU/Gloo")
            dist.init_process_group(backend=backend, init_method=init_method,
                                    rank=self.rank, world_size=self.world_size, timeout=timedelta(minutes=10))
            atexit.register(self.close)

    @property
    def primary(self):
        return self.rank == 0

    def local_size(self, total, name):
        if total < self.world_size:
            raise ValueError(f"global {name}={total} must provide at least one sample per rank (WORLD_SIZE={self.world_size})")
        quotient, remainder = divmod(total, self.world_size)
        return quotient + int(self.rank < remainder)

    def reduce(self, values, op=dist.ReduceOp.SUM):
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.device).clone()
        if self.enabled:
            dist.all_reduce(tensor, op=op)
        return tensor

    def gather(self, value):
        if not self.enabled:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def require_equal(self, value, name):
        values = self.gather(value)
        if any(v != values[0] for v in values):
            raise ValueError(f"{name} differs across ranks")

    @contextmanager
    def coordinated(self, stage):
        error = None
        try:
            yield
        except Exception as exc:
            if not self.enabled:
                raise
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [e for e in self.gather(error) if e is not None]
        if errors:
            raise RuntimeError(f"{stage} failed: " + "; ".join(errors))

    @torch.no_grad()
    def broadcast_module(self, module):
        if self.enabled:
            self.require_equal([(key, tuple(value.shape), str(value.dtype)) for key, value in module.state_dict().items()],
                               "model state layout")
            for value in module.state_dict().values():
                dist.broadcast(value, src=0)

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
