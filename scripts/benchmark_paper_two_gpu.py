"""Bounded, full-data two-GPU timing under both readings of paper Table A.5.

Only simulator environments are explicitly labelled per GPU in the paper and its TeX source.
Both global and per-GPU training batch interpretations are measured, without assuming author confirmation.
Every run stops after at most 1000 iterations. There is no long-training mode.
"""

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
    # Use explicit global counts for both cases, so environment and optimizer batch scopes
    # cannot accidentally change together. No motion_limit: use every available clip.
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
