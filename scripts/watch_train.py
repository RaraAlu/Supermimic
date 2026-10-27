"""Watch a SuperMimic run's logs; exit when something needs attention or a milestone is reached.

usage: python3 scripts/watch_train.py RUN_DIR [--milestone 10000] [--ref REF_RUN_DIR] [--poll 120]
exit 10: anomaly, 11: milestone, 0: run finished. Prints a status block either way.
Thresholds come from the healthy part of the previous 2-GPU run (2.6 it/s, survival ~0.999,
termination < 1%), except ep_ret_norm, which that run's tail-bin bug drove negative.
"""

import argparse
import csv
import math
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--milestone", type=int, default=10000)
p.add_argument("--ref", default=None)
p.add_argument("--poll", type=int, default=120)
p.add_argument("--start_after", type=int, default=0, help="only report milestones beyond this iteration")
a = p.parse_args()


def rows(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    except (FileNotFoundError, ValueError):
        return []


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def nearest(ref, it):
    return min(ref, key=lambda r: abs(r["iteration"] - it)) if ref else None


def status(tr, sp, ref):
    last = tr[-1]
    w = tr[-10:]
    lines = [f"iteration {int(last['iteration'])}  wall {last['wall_h']:.2f} h  {mean([r['it_per_s'] for r in w]):.2f} it/s",
             f"  wm_err8 {mean([r['wm_err8'] for r in w]):.4f}  imag_rew {mean([r['imag_rew'] for r in w]):.3f}  "
             f"surv {mean([r['imag_survival'] for r in w]):.4f}  sim_rew {mean([r['sim_rew'] for r in w]):.3f}  "
             f"ep_len {mean([r['ep_len'] for r in w]):.1f}  ep_ret {mean([r['ep_ret_norm'] for r in w]):.3f}  "
             f"term {mean([r['term_frac'] for r in w]):.4f}  lr {last['lr']:.3g}"]
    if sp:
        s = sp[-10:]
        lines.append(f"  sampling: entropy {mean([r['samp_entropy'] for r in s]):.3f}  top1 {mean([r['samp_top1'] for r in s]):.4f}  "
                     f"visited {sp[-1]['bins_visited']:.3f}  tail2s {mean([r['reset_tail_frac'] for r in s]):.3f}")
    r = nearest(ref, last["iteration"])
    if r:
        lines.append(f"  previous run at it {int(r['iteration'])}: wm_err8 {r['wm_err8']:.4f}  imag_rew {r['imag_rew']:.3f}  "
                     f"sim_rew {r['sim_rew']:.3f}  ep_ret {r['ep_ret_norm']:.3f}  {r['it_per_s']:.2f} it/s")
    return "\n".join(lines)


def anomalies(tr, sp):
    out = []
    last = tr[-1]
    it = last["iteration"]
    for r in tr[-3:]:
        bad = [k for k, v in r.items() if not math.isfinite(v)]
        if bad:
            out.append(f"non-finite {bad} at it {int(r['iteration'])}")
    if len(tr) >= 3 and all(r["it_per_s"] < 2.0 for r in tr[-3:]):
        out.append(f"throughput below 2.0 it/s for 3 logs ({tr[-1]['it_per_s']:.2f})")
    if it >= 2000 and len(tr) >= 30:
        roll = [mean([r["wm_err8"] for r in tr[i - 10:i]]) for i in range(10, len(tr) + 1)]
        if roll[-1] > 1.5 * min(roll):
            out.append(f"wm_err8 rolling mean {roll[-1]:.4f} > 1.5x its best {min(roll):.4f}")
        sim = [mean([r["sim_rew"] for r in tr[i - 10:i]]) for i in range(10, len(tr) + 1)]
        if sim[-1] < max(sim) - 0.05:
            out.append(f"sim_rew rolling mean {sim[-1]:.3f} fell > 0.05 below its best {max(sim):.3f}")
        w = tr[-10:]
        if mean([r["imag_survival"] for r in w]) < 0.95:
            out.append("imagined survival below 0.95")
        if mean([r["term_frac"] for r in w]) > 0.05:
            out.append("termination fraction above 5%")
        if mean([r["ep_ret_norm"] for r in w]) < 0.3:
            out.append("ep_ret_norm rolling mean below 0.3 (tail-bin symptom)")
    if sp and it >= 2000:
        s = sp[-10:]
        if mean([r["reset_tail_frac"] for r in s]) > 0.6:
            out.append("more than 60% of resets within 2 s of the clip end")
        if mean([r["samp_top1"] for r in s]) > 0.2:
            out.append("one reset bin draws more than 20% of resets")
    return out


ref = rows(os.path.join(a.ref, "train_log.csv")) if a.ref else []
seen_milestone = a.start_after // a.milestone
while True:
    exit_json = os.path.join(a.run, "exit.json")
    tr = rows(os.path.join(a.run, "train_log.csv"))
    sp = rows(os.path.join(a.run, "sampling_log.csv"))
    if os.path.exists(exit_json):
        print("RUN ENDED:", open(exit_json).read().strip())
        if tr:
            print(status(tr, sp, ref))
        sys.exit(0)
    if tr:
        bad = anomalies(tr, sp)
        if bad:
            print("ANOMALY:\n  " + "\n  ".join(bad))
            print(status(tr, sp, ref))
            sys.exit(10)
        m = int(tr[-1]["iteration"]) // a.milestone
        if m > seen_milestone:
            print(f"MILESTONE {m * a.milestone}")
            print(status(tr, sp, ref))
            sys.exit(11)
    time.sleep(a.poll)
