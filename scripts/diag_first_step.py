"""First-step penalty diagnostic for a SuperMimic checkpoint (Appendix F.1 rollout, read-only).

Plays every clip of a motion source once from its first frame, exactly like
supermimic/scripts/evaluate.py, and additionally records per clip the reward terms of the
first control step (where a_prev = 0) and the normalized return without that step.

Run from the repository root; --src selects the source tree (default: this repository):
    python scripts/diag_first_step.py --src <src> --ckpt <ckpt.pt> --motion_source <subset.h5> --out <dir>
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
parser.add_argument("--transient_steps", type=int, default=20)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))

import torch

import supermimic.tasks.tracking.rl.networks as networks
from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, RewardCfg
from supermimic.tasks.tracking.motion import CONTROL_DT, MotionLib
from supermimic.tasks.tracking.rl.networks import Policy
from supermimic.utils.kinematics import G1Kinematics

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
C = len(motion.names)
pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
w_rate, w_lim = RewardCfg.w_action_rate, RewardCfg.w_joint_limit

env = TrackingSim(C, motion, mdp, dev, eval_mode=True)
ids = torch.arange(C, device=dev)
clip_end = motion.clip_end[ids]
T_tr = args.transient_steps
rec = {k: torch.zeros(C, device=dev) for k in
       ("ret", "r0", "track0", "rate0", "lim0", "a0_norm", "a0_absmax", "rate_rest_sum", "lim_rest_sum", "a_norm_rest_sum", "steps")}
trans_rate = torch.zeros(T_tr, device=dev)
trans_lim = torch.zeros(T_tr, device=dev)
trans_r = torch.zeros(T_tr, device=dev)
trans_n = torch.zeros(T_tr, device=dev)

t0 = time.time()
with torch.no_grad():
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[ids].clone())
    hist = None
    step = 0
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        out = env.step(a)
        r, terms = out["r"], out["terms"]
        af = alive.float()
        rec["ret"] += r * af
        rec["steps"] += af
        if step == 0:
            rec["r0"] = r * af
            rec["track0"] = terms["track"] * af
            rec["rate0"] = terms["action_rate"] * af
            rec["lim0"] = terms["joint_limit"] * af
            rec["a0_norm"] = a.norm(dim=-1) * af
            rec["a0_absmax"] = a.abs().amax(-1) * af
        else:
            rec["rate_rest_sum"] += terms["action_rate"] * af
            rec["lim_rest_sum"] += terms["joint_limit"] * af
            rec["a_norm_rest_sum"] += a.norm(dim=-1) * af
        if step < T_tr:
            trans_rate[step] = (terms["action_rate"] * af).sum()
            trans_lim[step] = (terms["joint_limit"] * af).sum()
            trans_r[step] = (r * af).sum()
            trans_n[step] = af.sum()
        env.f = torch.minimum(env.f, clip_end - 1)
        step += 1
elapsed = time.time() - t0

denom = (motion.clip_len - 1).float() * MAX_STEP_REWARD
failed = env.failed_eval.clone()
norm = rec["ret"] / denom
norm_wo_first = (rec["ret"] - rec["r0"]) / ((motion.clip_len - 2).float() * MAX_STEP_REWARD)
# counterfactual: same rollout, first step's action-smoothness penalty removed
norm_no_rate0 = (rec["ret"] - w_rate * rec["rate0"] * CONTROL_DT) / denom
rest = (rec["steps"] - 1).clamp(min=1)
pen_rate0 = w_rate * rec["rate0"] * CONTROL_DT / MAX_STEP_REWARD  # in normalized-step units
pen_lim0 = w_lim * rec["lim0"] * CONTROL_DT / MAX_STEP_REWARD


def q(x, ps=(0.0, 0.1, 0.5, 0.9, 1.0)):
    x = x.float().cpu()
    return {f"p{int(p * 100)}": round(float(torch.quantile(x, p)), 4) for p in ps} | {"mean": round(float(x.mean()), 4)}


summary = {
    "ckpt": args.ckpt, "iteration": ck.get("iteration"), "clips": C, "frames": int(motion.clip_len.sum()),
    "elapsed_s": round(elapsed, 1), "steps": step, "n_failed": int(failed.sum()),
    "norm_return": round(float(norm.mean()), 4),
    "norm_return_wo_first_step": round(float(norm_wo_first.mean()), 4),
    "norm_return_without_first_step_action_rate": round(float(norm_no_rate0.mean()), 4),
    "norm_return_framew": round(float(rec["ret"].sum() / denom.sum()), 4),
    "first_step": {
        "reward_normalized": q(rec["r0"] / MAX_STEP_REWARD),
        "track_normalized": q(rec["track0"] / MAX_STEP_REWARD),
        "action_rate_raw_sq_norm": q(rec["rate0"]),
        "action_rate_penalty_normalized": q(pen_rate0),
        "joint_limit_raw": q(rec["lim0"]),
        "joint_limit_penalty_normalized": q(pen_lim0),
        "action_l2": q(rec["a0_norm"]),
        "action_absmax": q(rec["a0_absmax"]),
    },
    "later_steps_mean_per_clip": {
        "action_rate_raw_sq_norm": q(rec["rate_rest_sum"] / rest),
        "joint_limit_raw": q(rec["lim_rest_sum"] / rest),
        "action_l2": q(rec["a_norm_rest_sum"] / rest),
    },
    "transient_mean_over_clips": {
        "action_rate_raw": [round(float(v), 3) for v in (trans_rate / trans_n.clamp(min=1)).cpu()],
        "joint_limit_raw": [round(float(v), 4) for v in (trans_lim / trans_n.clamp(min=1)).cpu()],
        "reward_normalized": [round(float(v), 4) for v in (trans_r / trans_n.clamp(min=1) / MAX_STEP_REWARD).cpu()],
    },
    # fraction of the mean shortfall (1 - norm) caused by the first step's two penalties
    "first_step_share_of_deficit": round(float((-(pen_rate0 + pen_lim0) / (motion.clip_len - 1)).mean() / (1 - norm).mean()), 4),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with open(out_dir / "per_clip.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "clip_len", "failed", "norm", "norm_wo_first", "norm_no_rate0", "r0_norm", "rate0_raw", "lim0_raw",
                "a0_l2", "rate_rest_mean", "a_l2_rest_mean"])
    for i in range(C):
        w.writerow([motion.names[i], int(motion.clip_len[i]), int(failed[i]), f"{norm[i]:.4f}", f"{norm_wo_first[i]:.4f}",
                    f"{norm_no_rate0[i]:.4f}", f"{rec['r0'][i] / MAX_STEP_REWARD:.4f}", f"{rec['rate0'][i]:.3f}",
                    f"{rec['lim0'][i]:.4f}", f"{rec['a0_norm'][i]:.3f}", f"{rec['rate_rest_sum'][i] / rest[i]:.4f}",
                    f"{rec['a_norm_rest_sum'][i] / rest[i]:.3f}"])
print(json.dumps(summary, indent=2, ensure_ascii=False))
