"""Replay a fixed-shape step as a CUDA graph after a few eager calls."""

import torch
from torch.utils._pytree import tree_map


class GraphedCall:
    """Run fn eagerly for its first calls, then capture it once as a CUDA graph and replay it.

    Tensor inputs are copied into static buffers before each replay and outputs are cloned, so a call
    returns what the eager call would. fn must keep static shapes and must not sync with the host;
    random ops draw from the default CUDA generator, which each replay advances like an eager call.
    The eager calls are real updates that also warm up cuBLAS and the allocator before capture.
    With thread_local, only the capturing thread's CUDA calls are checked: under NCCL the communicator's
    watchdog thread keeps querying CUDA events during capture.
    """

    def __init__(self, fn, eager_calls: int = 3, enabled: bool = True, thread_local: bool = False):
        self.fn, self.eager_calls, self.calls, self.graph = fn, eager_calls, 0, None
        self.enabled, self.thread_local = enabled, thread_local

    def __call__(self, *inputs):
        if not self.enabled or self.calls < self.eager_calls:
            self.calls += 1
            return self.fn(*inputs)
        if self.graph is None:
            self.static_in = tree_map(torch.clone, inputs)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager calls' cached blocks before the graph takes its own pool
            self.graph = torch.cuda.CUDAGraph()
            # records only; the replay below performs this call
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local" if self.thread_local else "global"):
                self.static_out = self.fn(*self.static_in)
        tree_map(lambda dst, src: dst.copy_(src), self.static_in, inputs)
        self.graph.replay()
        return tree_map(torch.clone, self.static_out)

    def close(self):
        """Release captured NCCL operations before destroying their process group."""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
            del self.static_in, self.static_out
        self.calls = 0
