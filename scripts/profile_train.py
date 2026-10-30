"""分析训练迭代耗时，通过优化器钩子驱动 torch.profiler，无需修改训练脚本。

策略优化器每迭代更新一次；跳过 --wait 次更新，再预热并记录 --active 次迭代。
命令行中 -- 后的参数全部转交训练脚本。"""

import argparse
import json
from pathlib import Path
import re
import runpy
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--report", required=True, help="JSON output: per-op rows, iteration times, peak memory")
parser.add_argument("--wait", type=int, default=30, help="policy updates to skip before profiling")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--active", type=int, default=5, help="recorded iterations")
parser.add_argument("train_args", nargs=argparse.REMAINDER)
args = parser.parse_args()
train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

KERNEL_CATEGORIES = {
    "gemm": r"gemm|gemv|cutlass|xmma|cublas",
    "copy/cast": r"copy|Memcpy|Memset|cast",
    "layernorm": r"layer_norm",
    "reduce": r"reduce|Reduce|cumprod|scan",
    "index/gather": r"index|gather|scatter",
}
HOST_EVENTS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch", "cudaStreamSynchronize", "cudaMemcpyAsync")

prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1),
)
step_times = []


def on_step(opt, *_):
    """仅在策略优化器更新时同步设备、记录时间并推进分析器。"""
    if opt.defaults.get("betas") == (0.9, 0.95):  # 策略优化器每次迭代更新一次
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        prof.step()


def is_kernel(name):
    """过滤框架事件名称，识别需要统计的设备计算内核。"""
    return not (name.startswith(("aten::", "autograd", "cuda", "cu")) or "ProfilerStep" in name or "#" in name)


def summarize():
    """汇总采样区间的耗时、内核或训练指标，并生成可供比较的报告。"""
    rows = [{"name": e.key, "count": e.count, "cpu_self_us": e.self_cpu_time_total,
             "device_self_us": getattr(e, "self_device_time_total", 0)} for e in prof.key_averages()]
    iteration_s = [b - a for a, b in zip(step_times[:-1], step_times[1:])][-args.active:]
    kernels = [r for r in rows if r["device_self_us"] > 0 and is_kernel(r["name"])]
    # 按内核名称归类并折算为每迭代耗时与调用数；未匹配项归入其他类别，不丢弃已识别内核。
    categories = {name: {"ms": 0.0, "kernels": 0.0} for name in [*KERNEL_CATEGORIES, "elementwise/other"]}
    for r in kernels:
        name = next((k for k, pattern in KERNEL_CATEGORIES.items() if re.search(pattern, r["name"])), "elementwise/other")
        categories[name]["ms"] += r["device_self_us"] / args.active / 1e3
        categories[name]["kernels"] += r["count"] / args.active
    # 主机发射/同步开销单独报告，帮助区分 CPU 调度瓶颈与设备计算耗时。
    host = {r["name"]: {"calls": r["count"] / args.active, "cpu_ms": r["cpu_self_us"] / args.active / 1e3}
            for r in rows if r["name"] in HOST_EVENTS}
    tiny = [r for r in kernels if r["device_self_us"] < 10 * r["count"]]
    summary = {
        "iteration_s_median": statistics.median(iteration_s) if iteration_s else None,
        "kernel_busy_ms_per_iteration": sum(c["ms"] for c in categories.values()),
        "kernels_per_iteration": sum(c["kernels"] for c in categories.values()),
        "kernels_under_10us_per_iteration": sum(r["count"] for r in tiny) / args.active,
        "categories": categories,
        "host": host,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    Path(args.report).write_text(json.dumps({"summary": summary, "train_args": train_args, "rows": rows}, indent=1),
                                 encoding="utf-8")
    print(json.dumps(summary, indent=1))


register_optimizer_step_post_hook(on_step)
sys.argv = [str(REPO / "supermimic" / "scripts" / "train.py"), *train_args]
prof.start()
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
# 即使训练异常退出，也停止分析器并尝试输出已有样本，便于定位失败前的瓶颈。
finally:
    prof.stop()
    summarize()
