"""Profile supermimic.scripts.train unchanged and summarize where an iteration's time goes.

The policy optimizer steps once per iteration, so a global optimizer-step hook drives a
torch.profiler schedule: skip --wait iterations of policy updates, warm up, then record --active.
All arguments after ``--`` go to the training script, for example:

    python scripts/profile_train.py --report logs/profile.json -- --num_envs 2048 --pi_batch 2048 --iterations 60 --wm_only_iters 10
"""

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
    if opt.defaults.get("betas") == (0.9, 0.95):  # the policy optimizer: once per iteration
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        prof.step()


def is_kernel(name):
    return not (name.startswith(("aten::", "autograd", "cuda", "cu")) or "ProfilerStep" in name or "#" in name)


def summarize():
    rows = [{"name": e.key, "count": e.count, "cpu_self_us": e.self_cpu_time_total,
             "device_self_us": getattr(e, "self_device_time_total", 0)} for e in prof.key_averages()]
    iteration_s = [b - a for a, b in zip(step_times[:-1], step_times[1:])][-args.active:]
    kernels = [r for r in rows if r["device_self_us"] > 0 and is_kernel(r["name"])]
    categories = {name: {"ms": 0.0, "kernels": 0.0} for name in [*KERNEL_CATEGORIES, "elementwise/other"]}
    for r in kernels:
        name = next((k for k, pattern in KERNEL_CATEGORIES.items() if re.search(pattern, r["name"])), "elementwise/other")
        categories[name]["ms"] += r["device_self_us"] / args.active / 1e3
        categories[name]["kernels"] += r["count"] / args.active
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
finally:
    prof.stop()
    summarize()
