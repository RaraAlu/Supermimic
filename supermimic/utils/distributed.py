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
