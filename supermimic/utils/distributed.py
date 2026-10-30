"""自定义世界模型训练循环使用的分布式基础操作。"""

from contextlib import contextmanager
from datetime import timedelta
import atexit
import os

import torch
import torch.distributed as dist


def distributed_active():
    """判断分布式进程组是否已初始化且包含多个进程。"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


@torch.no_grad()
def global_moments(x):
    """合并各进程的批次统计量，包含进程间的方差，无需收集原始样本。"""
    if not distributed_active():
        return x.shape[0], x.mean(0), x.var(0, unbiased=False)
    values = x.double()
    # 在设备上填充而非从主机列表复制，因此也能在捕获的 CUDA 图中运行
    totals = torch.cat((torch.full((1,), len(x), dtype=values.dtype, device=values.device), values.sum(0)))
    dist.all_reduce(totals)
    n, mean = totals[0], totals[1:] / totals[0].clamp(min=1)
    # 此检查需要主机参与，图捕获无法等待；捕获前的普通调用会执行检查
    if not (values.is_cuda and torch.cuda.is_current_stream_capturing()) and n <= 0:
        raise ValueError("normalizer received no samples on any rank")
    # 第二次归约围绕全局均值计算平方偏差，包含进程间均值差，避免漏算跨卡方差。
    m2 = ((values - mean) ** 2).sum(0)
    dist.all_reduce(m2)
    return n.float(), mean.float(), (m2 / n).float()


@torch.no_grad()
def average_gradients(module, sample_count=1, total_count=None):
    """在梯度裁剪之前按样本数归约梯度；全局未使用的参数保留 grad=None。

    指定固定的全局样本数 total_count 时，每个进程的每个参数都必须有梯度。此时归约无需与主机往返，可在 CUDA 图中运行。"""
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
    # 一般路径先统计全局样本数和各参数是否被使用；本卡缺失梯度以零参与归约，
    # 所有卡都未使用的参数则保持 None，避免优化器误更新其状态。
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
    """单进程下替代 DistributedContext 的归约接口。"""

    enabled = False

    @staticmethod
    def reduce(values, op=None):
        """单进程归约：将输入转换为双精度张量。"""
        return values.double() if torch.is_tensor(values) else torch.as_tensor(values, dtype=torch.float64)


class DistributedContext:
    """管理分布式进程组、集合通信及跨进程错误协调。"""
    def __init__(self, device, *, init_method="env://"):
        """从环境变量读取进程编号，并按设备初始化 NCCL 或 Gloo 进程组。"""
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
        """判断当前进程是否为主进程。"""
        return self.rank == 0

    def local_size(self, total, name):
        """将全局数量尽量均匀地分配给各进程，并确保每个进程至少获得一个。"""
        if total < self.world_size:
            raise ValueError(f"global {name}={total} must provide at least one sample per rank (WORLD_SIZE={self.world_size})")
        quotient, remainder = divmod(total, self.world_size)
        return quotient + int(self.rank < remainder)

    def reduce(self, values, op=dist.ReduceOp.SUM):
        """将输入转为设备上的双精度张量，并在启用分布式时执行全归约。"""
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.device).clone()
        if self.enabled:
            dist.all_reduce(tensor, op=op)
        return tensor

    def gather(self, value):
        """收集所有进程的 Python 对象；单进程时返回单元素列表。"""
        if not self.enabled:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def require_equal(self, value, name):
        """检查所有进程提供的值是否一致，不一致时抛出异常。"""
        values = self.gather(value)
        if any(v != values[0] for v in values):
            raise ValueError(f"{name} differs across ranks")

    @contextmanager
    def coordinated(self, stage):
        """协调各进程的阶段性错误，使所有进程在集合通信后报告失败。"""
        error = None
        try:
            yield
        except Exception as exc:
            if not self.enabled:
                raise
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        # 等各进程离开阶段后统一收集异常；阶段内部若已有不匹配的集合通信，
        # 此包装本身无法解除阻塞，因此各卡仍须保持通信顺序一致。
        errors = [e for e in self.gather(error) if e is not None]
        if errors:
            raise RuntimeError(f"{stage} failed: " + "; ".join(errors))

    @torch.no_grad()
    def broadcast_module(self, module):
        """验证模型状态布局一致，再从主进程广播所有状态张量。"""
        if self.enabled:
            self.require_equal([(key, tuple(value.shape), str(value.dtype)) for key, value in module.state_dict().items()],
                               "model state layout")
            for value in module.state_dict().values():
                dist.broadcast(value, src=0)

    def close(self):
        """在已初始化时销毁分布式进程组。"""
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
