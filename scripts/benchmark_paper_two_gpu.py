"""使用全部动作数据，按论文表 A.5 的两种批量解释执行有上限的双卡计时。

论文及其 TeX 源码仅明确环境数为每卡数量，因此分别测量全局批量与每卡批量两种解释，
不视为已获作者确认。每次运行最多 1000 次迭代，不提供长时间训练模式。"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def training_arguments(interpretation):
    # 两种解释均显式传入全局数量，避免环境数与优化器批量的作用范围
    # 意外一起变化；不设置 motion_limit，以使用全部可用片段。
    """按全局或每卡批量解释生成显式训练参数，保持环境总数一致。"""
    scale = 1 if interpretation == "global" else 2
    return ["--batch_scope", "global", "--num_envs", "8192",
            "--wm_batch", str(2048 * scale), "--pi_batch", str(4096 * scale),
            "--wm_batches", "10", "--wm_window", "8", "--horizon", "32",
            "--goal_frames", "10", "--hist", "10", "--buffer_steps", "1000",
            "--utd", "0.4", "--lr", "0.0002", "--lr_scaling", "sqrt",
            "--lr_decay_iters", "250000", "--gamma", "0.99", "--lam", "0.95",
            "--action_noise", "0.1", "--critic_epochs", "5", "--critic_minibatches", "4",
            "--amp", "none", "--warmup_steps", "64", "--wm_only_iters", "0",
            "--compile", "on", "--cuda_graph", "on", "--seed", "0"]


def summarize(path, skip, iterations):
    """汇总采样区间的耗时、内核或训练指标，并生成可供比较的报告。"""
    with (path / "train_log.csv").open() as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if int(row["iteration"]) > skip]
    if not rows or int(rows[-1]["iteration"]) != iterations:
        raise RuntimeError(f"Incomplete short run: {path}")
    for row in all_rows:
        if any(not math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError(f"Nonfinite training log: {path}")
    seconds = statistics.mean(1 / float(row["it_per_s"]) for row in rows)
    shards = [json.loads(p.read_text()) for p in sorted((path / "shards/world_2").glob("rank_*.json"))]
    if len(shards) != 2 or any(s["runtime"]["local_envs"] != 4096 for s in shards):
        raise RuntimeError("Expected exactly two ranks with 4096 environments each")
    return {"seconds_per_iteration": seconds, "iterations_per_second": 1 / seconds,
            "projected_320k_hours": seconds * 320000 / 3600,
            "training_loop_seconds": float(all_rows[-1]["wall_h"]) * 3600,
            "timed_iterations": iterations - skip,
            "last_wm_loss": float(rows[-1]["wm_loss"]),
            "last_sim_reward": float(rows[-1]["sim_rew"]),
            "shards": shards}


def main():
    """验证短程计时参数，分配双卡运行组合并保存两种批量解释的比较结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_source", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1],
                        help="two GPUs: sequential runs; four GPUs: two concurrent two-GPU runs")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--skip", type=int, default=50)
    parser.add_argument("--rounds", type=int, choices=[1, 2], default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.iterations <= 1000 or args.iterations % 10:
        parser.error("short runs require 10..1000 iterations, in multiples of 10")
    if not 0 <= args.skip < args.iterations or args.skip % 10:
        parser.error("skip must be a multiple of 10 below iterations")
    if len(args.gpus) not in (2, 4) or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("provide two or four distinct nonnegative GPU IDs")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run this launcher directly, not inside torchrun")
    output = args.output_dir.resolve()
    # 先生成完整执行计划；两卡依次运行两种解释，四卡可在独立设备组并行运行。
    plans = []
    for round_index in range(args.rounds):
        pairs = [args.gpus[:2], args.gpus[2:]] if len(args.gpus) == 4 else [args.gpus[:], args.gpus[:]]
        if round_index:
            pairs = pairs[::-1] if len(args.gpus) == 4 else [pair[::-1] for pair in pairs]
        for interpretation, pair in zip(("global", "per_gpu"), pairs):
            path = output / f"round{round_index}_{interpretation}"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                       "-m", "supermimic.scripts.train", *training_arguments(interpretation),
                       "--motion_source", str(Path(args.motion_source).resolve()),
                       "--iterations", str(args.iterations), "--log_every", "10",
                       "--ckpt_every", str(args.iterations), "--log_dir", str(path)]
            plans.append({"round": round_index, "interpretation": interpretation, "gpus": pair,
                          "path": str(path), "command": command})
    batches = ([[p] for p in plans] if len(args.gpus) == 2 else
               [[p for p in plans if p["round"] == r] for r in range(args.rounds)])
    groups = [[p["path"] for p in batch] for batch in batches]
    # 预览模式只输出命令和并发分组，不创建结果目录或启动训练。
    if args.dry_run:
        print(json.dumps({"plans": plans, "concurrent_groups": groups}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    (output / "source.patch").write_bytes(diff)
    (output / "launcher.py").write_bytes(Path(__file__).read_bytes())
    metadata = {"paper": "https://arxiv.org/pdf/2610.28480v1", "batch_scope_confirmed_by_author": False,
                "interpretation": "two interpretations: global batches or per-GPU batches; see docs/analysis/paper_two_gpu.md",
                "concurrent_groups": groups,
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "diff_sha256": hashlib.sha256(diff).hexdigest(), "plans": plans,
                "initial_replay_collection_steps": 64, "wm_only_iterations": 0,
                "skip": args.skip, "started_unix": time.time()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records = []
    for batch in batches:
        running = []
        try:
            for plan in batch:
                path = Path(plan["path"])
                path.mkdir()
                (path / "command.json").write_text(json.dumps(plan["command"], indent=2))
                log = (path / "console.log").open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, plan["gpus"])),
                           OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONPATH=str(ROOT))
                start = time.time()
                process = subprocess.Popen(plan["command"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                running.append((process, log, plan, start))
                print("START " + json.dumps({**plan, "command": "command.json", "pid": process.pid}), flush=True)
            # 轮询所有并行任务，任一失败即进入清理；每个成功结果立即落盘，保留已完成的测量。
            pending = list(running)
            while pending:
                for item in pending[:]:
                    process, log, plan, start = item
                    status = process.poll()
                    if status is None:
                        continue
                    log.close()
                    if status:
                        raise RuntimeError(f"{plan['path']}: exit {status}; see console.log")
                    record = {key: value for key, value in plan.items() if key != "command"}
                    record.update(summarize(Path(plan["path"]), args.skip, args.iterations))
                    record["process_wall_seconds"] = time.time() - start
                    records.append(record)
                    (output / "results.json").write_text(json.dumps(records, indent=2))
                    print("RESULT " + json.dumps({k: v for k, v in record.items() if k != "shards"}), flush=True)
                    pending.remove(item)
                if pending:
                    time.sleep(1)
        # torchrun 会启动多个工作进程，因此按进程组终止整个任务；超时后强制清理，避免留下 GPU 占用。
        finally:
            for process, log, *_ in running:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process, log, *_ in running:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                log.close()
    # 跨轮次取耗时中位数，再外推长程训练时间；外推仅基于短程稳态吞吐，不包含启动成本。
    summary = {}
    for name in ("global", "per_gpu"):
        seconds = [r["seconds_per_iteration"] for r in records if r["interpretation"] == name]
        middle = statistics.median(seconds)
        summary[name] = {"seconds_per_iteration": middle, "iterations_per_second": 1 / middle,
                         "projected_320k_hours": middle * 320000 / 3600,
                         "round_seconds_per_iteration": seconds}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
