# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.
Failures are also counted without a truncated goal: if the action that triggered termination was computed
from future reference frames clamped at the clip end (f + K beyond the clip end), it counts only toward n_failed.

Run as a watcher: ``python -m supermimic.scripts.evaluate --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
source = parser.add_mutually_exclusive_group()
source.add_argument("--motion_dir", type=str, default=None)
source.add_argument("--motion_index", type=str, default=None)
source.add_argument("--motion_source", type=str, default=None)
parser.add_argument("--motion_root", type=str, default=None)
parser.add_argument("--motion_limit", type=int, default=None)
parser.add_argument("--eval_batch_size", type=int, default=4096, help="parallel environments per batch (paper: 4096 per GPU)")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    parser.error("evaluation is single-process and batched; run without torchrun")

import numpy as np
import torch

import supermimic.tasks.tracking.rl.networks as networks
from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, OBS_DIM, MDP
from supermimic.tasks.tracking.motion import GOAL_FRAME_DIM, MotionLib, check_reference, resolve_motion_args
from supermimic.tasks.tracking.rl.networks import Policy
from supermimic.utils.kinematics import G1Kinematics

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
if args.eval_batch_size < 1:
    raise ValueError("eval_batch_size must be positive")
initial_ck = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
run_args = Path(args.run_dir) / "args.json"
saved_args = initial_ck["args"] if initial_ck else json.loads(run_args.read_text(encoding="utf-8"))
resolve_motion_args(args, saved_args)
kin = G1Kinematics(dev)
motion = MotionLib.from_args(args, kin, dev)
mdp = MDP(kin, motion, kin.default_q)
C = len(motion.names)
if args.record and C > args.eval_batch_size:
    raise ValueError("trajectory recording requires a single evaluation batch; explicitly increase --eval_batch_size")


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    check_reference(motion.description, ck.get("reference"))
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(env.N, device=dev)
    ret = torch.zeros(C, device=dev)
    failed = torch.zeros(C, dtype=torch.bool, device=dev)
    truncated = torch.zeros(C, dtype=torch.bool, device=dev)
    fail_step = torch.zeros(C, dtype=torch.long, device=dev)
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    # A batch runs until its longest clip ends, so batches take clips longest-first to keep lengths
    # similar. A single batch keeps library order (trajectory recording relies on it).
    if C <= env.N:
        order = torch.arange(C, device=dev)
    else:
        order = torch.argsort(motion.clip_len, descending=True, stable=True)
    for start in range(0, C, env.N):
        count = min(env.N, C - start)
        selected = order[start:start + count]
        # Pad the last batch with an inactive duplicate; it never enters the scores.
        clips = torch.cat((selected, selected[-1:].expand(env.N - count)))
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start[clips].clone())
        env.done_eval[count:] = True
        clip_end = motion.clip_end[clips]
        batch_ret = torch.zeros(env.N, device=dev)
        fail_f = torch.zeros(env.N, dtype=torch.long, device=dev)  # reference frame before the step that triggered termination
        hist = None
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(env.f, K))
            alive = ~env.done_eval.clone()
            if record:
                from supermimic.utils.rotations import mat_to_quat

                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            f_prev, failed_prev = env.f.clone(), env.failed_eval.clone()
            out = env.step(a)
            batch_ret += torch.where(alive, out["r"], 0.0)
            fail_f = torch.where(env.failed_eval & ~failed_prev, f_prev, fail_f)
            env.f = torch.minimum(env.f, clip_end - 1)
        ret[selected] = batch_ret[:count]
        failed[selected] = env.failed_eval[:count]
        # The goal uses frames f+1..f+K clamped at the clip end; when f + K exceeds it, the future reference is frozen.
        truncated[selected] = (env.failed_eval & (fail_f + K > clip_end))[:count]
        fail_step[selected] = (fail_f - motion.clip_start[clips] + 1)[:count]
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(failed.sum().item()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "n_failed_guided": int((failed & ~truncated).sum().item()),
        "failed_goal_truncated": [motion.names[i] for i in truncated.nonzero().flatten().tolist()],
        # transition (from 1) that triggered termination, and the clip's total transitions
        "fail_step": {motion.names[i]: [int(fail_step[i]), int(motion.clip_len[i]) - 1] for i in failed.nonzero().flatten().tolist()},
        "reference": motion.description,
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R[:].cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


def make_env() -> TrackingSim:
    return TrackingSim(min(C, args.eval_batch_size), motion, mdp, dev, eval_mode=True)


env = None
if args.ckpt:
    ck = initial_ck
    check_reference(motion.description, ck.get("reference"))
    env = make_env()
    res = evaluate(env, ck, args.record)
    result_path = str(Path(args.ckpt).with_suffix("")) + "_eval.json"
    Path(result_path).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
eval_reference = Path(args.run_dir) / "eval_reference.json"
if eval_reference.exists():
    check_reference(motion.description, json.loads(eval_reference.read_text(encoding="utf-8")))
elif Path(out_csv).exists():
    raise ValueError("existing eval.csv has no reference provenance; use a separate evaluation run_dir")
else:
    eval_reference.write_text(json.dumps(motion.description, indent=2, ensure_ascii=False), encoding="utf-8")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv, encoding="utf-8").read().strip().splitlines()[1:]}
else:
    open(out_csv, "w", encoding="utf-8").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
    for c in todo:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        check_reference(motion.description, ck.get("reference"))
        env = env or make_env()
        t0 = time.time()
        res = evaluate(env, ck)
        line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
        open(out_csv, "a", encoding="utf-8").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w", encoding="utf-8"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}, "
              f"excluding truncated goal {res['n_failed_guided']} ({time.time() - t0:.0f}s)", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
