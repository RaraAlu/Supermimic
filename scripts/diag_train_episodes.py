"""Training-condition episode diagnostic for a SuperMimic checkpoint (read-only).

Replays the policy under the training collection setup of supermimic/scripts/train.py:
domain randomization, reset noise, return-based bin starts, observation noise, action noise
N(0, 0.1^2) clamped to +-20, and the 500-step episode cap. Each finished episode is logged
with its ep_ret_norm (exactly as the training log computes it) and the split of its return
into tracking reward, action-smoothness penalty and joint-limit penalty.

Run from the repository root; --src selects the source tree (default: this repository):
    python scripts/diag_train_episodes.py --src <src> --ckpt <ckpt.pt> --motion_source <subset.h5> --out <dir>
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=str(Path(__file__).resolve().parents[1]),
                    help="source tree to import supermimic from (default: this repository; pass a checkpoint's commit to replay it)")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_source", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bin_clip_end", default=None, help="passed to TrackingSim when the source supports it")
parser.add_argument("--bin_prior", default=None, help="passed to TrackingSim when the source supports it")
parser.add_argument("--bin_skip_steps", type=int, default=None, help="passed to TrackingSim when the source supports it")
parser.add_argument("--reset_margin", type=int, default=None, help="passed to TrackingSim when the source supports it")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))

import torch

import supermimic.tasks.tracking.rl.networks as networks
from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, RewardCfg
from supermimic.tasks.tracking.motion import CONTROL_DT, MotionLib
from supermimic.tasks.tracking.rl.networks import Policy
from supermimic.utils.kinematics import G1Kinematics

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
kin = G1Kinematics(dev)
motion = MotionLib.from_source(args.motion_source, kin, dev)
mdp = MDP(kin, motion, kin.default_q)
pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()  # normalizer statistics stay as saved; training would keep updating them
w_rate, w_lim = RewardCfg.w_action_rate, RewardCfg.w_joint_limit

N = args.num_envs
bin_kw = {k: v for k, v in (("bin_clip_end", args.bin_clip_end), ("bin_prior", args.bin_prior), ("bin_skip_steps", args.bin_skip_steps),
                                       ("reset_margin", args.reset_margin)) if v is not None}
env = TrackingSim(N, motion, mdp, dev, gamma=a_.get("gamma", 0.99), **bin_kw)
hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
acc_names = ("track", "rate_pen", "lim_pen", "r", "rate0", "lim0", "r0", "rate_max", "a_absmax", "sat_steps", "rate_max_step")
acc = {k: torch.zeros(N, device=dev) for k in acc_names}
f_start = env.f.clone()
episodes = []  # list of dicts of CPU tensors per step with finished episodes

t0 = time.time()
with torch.no_grad():
    for step in range(args.steps):
        S = env.state()
        ep_step = env.ep_step.clone()
        fresh = ep_step == 0
        f_start = torch.where(fresh, env.f, f_start)
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(hist), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
        a = torch.clamp(pi(hist, motion.goal(env.f, K)) + args.action_noise * torch.randn(N, 29, device=dev), -20, 20)
        out = env.step(a)
        r, terms, done = out["r"], out["terms"], out["done"]
        rate_pen = w_rate * terms["action_rate"] * CONTROL_DT
        lim_pen = w_lim * terms["joint_limit"] * CONTROL_DT
        acc["track"] += terms["track"]
        acc["rate_pen"] += rate_pen
        acc["lim_pen"] += lim_pen
        acc["r"] += r
        for k, v in (("rate0", rate_pen), ("lim0", lim_pen), ("r0", r)):
            acc[k] = torch.where(fresh, v, acc[k])
        worse = rate_pen < acc["rate_max"]
        acc["rate_max"] = torch.where(worse, rate_pen, acc["rate_max"])
        acc["rate_max_step"] = torch.where(worse, ep_step.float(), acc["rate_max_step"])
        acc["a_absmax"] = torch.maximum(acc["a_absmax"], a.abs().amax(-1))
        acc["sat_steps"] += (a.abs() >= 19.99).any(-1).float()
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            T = (ep_step[ids] + 1).float()
            fs = f_start[ids]
            clip = motion.clip_id[fs]
            rec = {k: acc[k][ids].clone() for k in acc_names}
            rec |= {"T": T, "term": out["term"][ids].float(), "ep_ret_norm": out["ep_ret_norm"],
                    "start_offset": (fs - motion.clip_start[clip]).float(), "clip_len": motion.clip_len[clip].float(),
                    "clip": clip.float(), "friction": out["xi"][ids, 0]}
            episodes.append({k: v.cpu() for k, v in rec.items()})
            for k in acc_names:
                acc[k][ids] = 0.0
elapsed = time.time() - t0

E = {k: torch.cat([e[k] for e in episodes]) for k in episodes[0]}
n = len(E["T"])
den = E["T"] * MAX_STEP_REWARD
# my own split must reproduce the logged metric
check = (E["r"] / den - E["ep_ret_norm"]).abs().max().item()
parts = {"track": E["track"] / den, "rate_pen": E["rate_pen"] / den, "lim_pen": E["lim_pen"] / den}
first_rate = E["rate0"] / den
first_lim = E["lim0"] / den


def q(x, ps=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    x = x.float()
    return {f"p{p * 100:g}": round(float(torch.quantile(x, p)), 4) for p in ps} | {"mean": round(float(x.mean()), 4)}


def bucket_rows(mask_fn, labels):
    rows = []
    for label, m in zip(labels, mask_fn):
        if m.sum() == 0:
            continue
        rows.append({"bucket": label, "episodes": int(m.sum()), "share": round(float(m.float().mean()), 4),
                     "mean_ep_ret_norm": round(float(E["ep_ret_norm"][m].mean()), 4),
                     "contrib_to_overall_mean": round(float(E["ep_ret_norm"][m].sum() / n), 4),
                     "mean_track": round(float(parts["track"][m].mean()), 4),
                     "mean_rate_pen": round(float(parts["rate_pen"][m].mean()), 4),
                     "mean_lim_pen": round(float(parts["lim_pen"][m].mean()), 4),
                     "mean_first_step_rate_pen": round(float(first_rate[m].mean()), 4),
                     "term_frac": round(float(E["term"][m].mean()), 4)})
    return rows


T = E["T"]
by_T = bucket_rows([T == 1, (T >= 2) & (T <= 5), (T >= 6) & (T <= 20), (T >= 21) & (T <= 100), (T >= 101) & (T <= 499), T >= 500],
                   ["T=1", "T=2-5", "T=6-20", "T=21-100", "T=101-499", "T=500"])
order = torch.argsort(E["ep_ret_norm"])
worst_k = max(1, n // 100)
worst = order[:worst_k]
summary = {
    "ckpt": args.ckpt, "iteration": ck.get("iteration"), "num_envs": N, "steps": args.steps, "elapsed_s": round(elapsed, 1),
    "episodes": n, "split_check_max_abs_err": check,
    "ep_ret_norm": q(E["ep_ret_norm"]),
    "step_weighted_return": round(float(E["r"].sum() / den.sum()), 4),
    "mean_parts": {k: round(float(v.mean()), 4) for k, v in parts.items()},
    "mean_first_step_parts": {"rate_pen": round(float(first_rate.mean()), 4), "lim_pen": round(float(first_lim.mean()), 4)},
    "T": q(T), "term_frac": round(float(E["term"].mean()), 4),
    "by_episode_length": by_T,
    "worst_1pct": {"episodes": worst_k, "share_of_mean": round(float(E["ep_ret_norm"][worst].sum() / n), 4),
                   "mean_without_them": round(float(E["ep_ret_norm"][order[worst_k:]].mean()), 4),
                   "mean_parts": {k: round(float(v[worst].mean()), 4) for k, v in parts.items()},
                   "mean_first_step_rate_pen": round(float(first_rate[worst].mean()), 4),
                   "T": q(T[worst]), "a_absmax": q(E["a_absmax"][worst]), "saturated_any_step": round(float((E["sat_steps"][worst] > 0).float().mean()), 4),
                   "rate_max_at_step0": round(float((E["rate_max_step"][worst] == 0).float().mean()), 4),
                   "term_frac": round(float(E["term"][worst].mean()), 4)},
    "all_a_absmax": q(E["a_absmax"]),
    "saturated_episode_frac": round(float((E["sat_steps"] > 0).float().mean()), 4),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
cols = ["ep_ret_norm", "T", "term", "track", "rate_pen", "lim_pen", "rate0", "lim0", "r0", "rate_max", "rate_max_step", "a_absmax",
        "sat_steps", "start_offset", "clip_len", "clip", "friction"]
with open(out_dir / "worst_episodes.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["clip_name"] + cols + ["track_norm", "rate_pen_norm", "lim_pen_norm"])
    for i in order[:200].tolist():
        w.writerow([motion.names[int(E["clip"][i])]] + [f"{float(E[c][i]):.4f}" for c in cols]
                   + [f"{float(parts[k][i]):.4f}" for k in ("track", "rate_pen", "lim_pen")])
torch.save(E, out_dir / "episodes.pt")
probs = getattr(env, "bin_probs", None)
torch.save({"bin_start": motion.bin_start.cpu(), "bin_clip": motion.bin_clip.cpu(), "bin_G": env.bin_G.cpu(),
            "bin_probs": None if probs is None else probs.cpu(),
            "bin_visits": getattr(env, "bin_visits", torch.zeros(0)).cpu()}, out_dir / "bins.pt")
print(json.dumps(summary, indent=2, ensure_ascii=False))
