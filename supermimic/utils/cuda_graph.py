"""先执行若干次普通调用，再以 CUDA 图重放固定形状的计算步骤。"""

import torch
from torch.utils._pytree import tree_map


class GraphedCall:
    """先普通执行 fn，再捕获一次 CUDA 图并在后续调用中重放。

    每次重放前将输入张量复制到固定缓冲区，返回时克隆输出，避免后续重放覆盖结果。
    fn 的张量形状必须固定且不能与主机同步；随机操作使用默认 CUDA 生成器，每次重放都会推进其状态。
    捕获前的普通调用会实际更新状态，同时预热 cuBLAS 和内存分配器。
    thread_local 模式只检查捕获线程的 CUDA 调用，允许 NCCL 看门狗线程在捕获期间查询事件。
    """

    def __init__(self, fn, eager_calls: int = 3, enabled: bool = True, thread_local: bool = False):
        """配置被包装函数、普通调用次数、图开关及捕获线程范围。"""
        self.fn, self.eager_calls, self.calls, self.graph = fn, eager_calls, 0, None
        self.enabled, self.thread_local = enabled, thread_local

    def __call__(self, *inputs):
        """按预热、首次捕获或重放阶段执行函数，并返回独立的输出张量。"""
        if not self.enabled or self.calls < self.eager_calls:
            self.calls += 1
            return self.fn(*inputs)
        if self.graph is None:
            # 捕获图绑定张量地址，因此建立长期缓冲区；后续调用只复制内容，不替换输入对象。
            self.static_in = tree_map(torch.clone, inputs)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # 在图建立自己的内存池之前释放普通调用缓存的内存块
            self.graph = torch.cuda.CUDAGraph()
            # 捕获只记录操作；下面的 replay 才执行本次调用
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local" if self.thread_local else "global"):
                self.static_out = self.fn(*self.static_in)
        tree_map(lambda dst, src: dst.copy_(src), self.static_in, inputs)
        self.graph.replay()
        # 图输出缓冲区会被下一次重放覆盖，返回副本使调用者可安全保存本次结果。
        return tree_map(torch.clone, self.static_out)

    def close(self):
        """销毁进程组之前释放捕获的 NCCL 操作，并重置预热计数。"""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
            del self.static_in, self.static_out
        self.calls = 0
