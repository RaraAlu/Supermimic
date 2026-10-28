"""Difficulty label per 1 s start position (read-only): does the checkpoint fall within 2 s from there?

For every clip and every start offset 0, 50, 100, ... (< n - 1), the policy plays from the reference
state at that frame under the evaluation setup of Appendix F.1 (friction 1.0, no randomization or
noise) for up to --steps control steps or until the clip ends. A position is labeled hard if the
rollout triggers the training termination. The labels are independent of any reset-bin layout.
"""

import argparse
import json
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_source", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--batch", type=int, default=2048)
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

dev = args.device
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
subject.set_amp(a_)
motion, mdp = subject.reference(args.motion_source)
pi = subject.policy(ck)

starts = []
for c in range(len(motion.names)):
    s0, n = int(motion.clip_start[c]), int(motion.clip_len[c])
    starts += [s0 + o for o in range(0, n - 1, 50)]
starts = torch.tensor(starts, device=dev)
P = len(starts)
failed = torch.zeros(P, dtype=torch.bool, device=dev)
env = subject.make_env(min(args.batch, P), motion, mdp, eval_mode=True)
with torch.no_grad():
    for b0 in range(0, P, env.N):
        cnt = min(env.N, P - b0)
        frames = torch.cat((starts[b0:b0 + cnt], starts[b0 + cnt - 1:b0 + cnt].expand(env.N - cnt)))
        ids = torch.arange(env.N, device=dev)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, frames.clone())
        env.done_eval[cnt:] = True
        clip_end = motion.clip_end[motion.clip_id[frames]]
        hist = None
        for _ in range(args.steps):
            if bool(env.done_eval.all()):
                break
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            env.step(pi(hist, motion.goal(env.f, K)))
            env.f = torch.minimum(env.f, clip_end - 1)
        failed[b0:b0 + cnt] = env.failed_eval[:cnt]

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
clip = motion.clip_id[starts]
torch.save({"start": starts.cpu(), "clip": clip.cpu(), "offset": (starts - motion.clip_start[clip]).cpu(),
            "clip_len": motion.clip_len[clip].cpu(), "failed": failed.cpu()}, out / "labels.pt")
summary = {"positions": P, "hard": int(failed.sum()), "hard_frac": round(float(failed.float().mean()), 4),
           "clips_with_hard": int(len(torch.unique(clip[failed]))), "subject": subject.provenance()}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
