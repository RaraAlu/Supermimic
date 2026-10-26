"""Compare collection optimizations in paired training runs, swapping GPUs on the second round.

    python scripts/benchmark_collect.py --gpus 0 1 2 3 --output_dir logs/collect_ab \
        --motion_source SOURCE --motion_limit 5000

Runs the four combinations of --collect_overlap/--cuda_graph_collect with otherwise identical
paper-sized batches. Extra arguments override training defaults. Uses the fixed duplicate-bin
update for every variant, excludes startup iterations, and records the source diff fingerprint.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("baseline", "off", "off"), ("post", "on", "off"),
            ("graph", "off", "on"), ("both", "on", "on")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip", type=int, default=30)
    args, extra = parser.parse_known_args()
    if args.rounds < 1 or args.iterations <= args.skip or len(set(args.gpus)) != len(args.gpus):
        parser.error("need distinct GPUs, positive rounds, and iterations > skip")
    if args.iterations % 10 or args.skip < 0:
        parser.error("iterations must be a multiple of 10 and skip must be nonnegative")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    (output / "source.patch").write_bytes(diff)
    metadata = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "diff_sha256": hashlib.sha256(diff).hexdigest(), "extra_args": extra,
                "gpus": args.gpus, "rounds": args.rounds, "iterations": args.iterations, "skip": args.skip}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results = []
    for round_index in range(args.rounds):
        devices = args.gpus if round_index % 2 == 0 else args.gpus[::-1]
        for offset in range(0, len(VARIANTS), len(devices)):
            running = []
            try:
                for gpu, (name, overlap, graph) in zip(devices, VARIANTS[offset:offset + len(devices)]):
                    path = output / f"round{round_index}_{name}_gpu{gpu}"
                    path.mkdir()
                    command = [sys.executable, "-m", "supermimic.scripts.train",
                               "--num_envs", "4096", "--wm_batch", "2048", "--pi_batch", "4096",
                               "--buffer_steps", "1000", "--warmup_steps", "64", "--wm_only_iters", "5",
                               "--compile", "on", "--cuda_graph", "on", "--amp", "bf16", *extra,
                               "--iterations", str(args.iterations), "--log_every", "10", "--ckpt_every", "100000",
                               "--collect_overlap", overlap, "--cuda_graph_collect", graph, "--log_dir", str(path)]
                    (path / "command.json").write_text(json.dumps(command, indent=2))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                               TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONPATH=str(ROOT))
                    log = (path / "console.log").open("w")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running.append((process, log, path, name, gpu))
                    print(f"START round={round_index} variant={name} gpu={gpu} pid={process.pid}", flush=True)
                for process, log, path, name, gpu in running:
                    status = process.wait()
                    log.close()
                    if status:
                        raise RuntimeError(f"{path}: exit {status}; see console.log")
                    with (path / "train_log.csv").open() as stream:
                        rows = [r for r in csv.DictReader(stream) if int(r["iteration"]) > args.skip]
                    if not rows or int(rows[-1]["iteration"]) != args.iterations:
                        raise RuntimeError(f"{path}: incomplete timed interval (iterations must be a multiple of 10)")
                    seconds = statistics.mean(1 / float(row["it_per_s"]) for row in rows)
                    record = {"round": round_index, "variant": name, "gpu": gpu, "seconds_per_iteration": seconds,
                              "last_wm_loss": float(rows[-1]["wm_loss"]), "last_sim_reward": float(rows[-1]["sim_rew"])}
                    results.append(record)
                    print("RESULT " + json.dumps(record), flush=True)
            finally:
                for process, log, *_ in running:
                    if process.poll() is None:
                        process.terminate()
                for process, log, *_ in running:
                    if process.poll() is None:
                        process.wait()
                    log.close()
            (output / "results.json").write_text(json.dumps(results, indent=2))
    medians = {name: statistics.median(r["seconds_per_iteration"] for r in results if r["variant"] == name)
               for name, *_ in VARIANTS}
    print("SUMMARY " + json.dumps({name: {"seconds_per_iteration": value,
          "time_reduction_pct": 100 * (1 - value / medians["baseline"])} for name, value in medians.items()}), flush=True)


if __name__ == "__main__":
    main()
