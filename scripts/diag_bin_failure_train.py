"""训练条件下每个1 s起点的失败概率（只读）：检查点从该起点出发2 s内会不会触发终止。

对每个片段每隔50帧的起点（与重置分箱同尺度），在训练采集条件下（域随机化、重置噪声、观测噪声、
动作噪声N(0, 0.1^2)截断到±20）各重复--repeats次，到--steps步或第一次回合结束为止；
回合因触发训练终止而结束记一次失败，片段结束或超时不算。输出每个起点的失败次数，
用来回答“训练中的终止集中在哪些片段、哪类动作”：课程把重置集中到失败概率高的分箱，
所以高失败概率的起点就是训练终止的来源。与diag_bin_difficulty.py（评估条件、无噪声）不同，
这里的随机化与噪声会暴露评估条件下看不到的脆弱位置。

在仓库根目录运行；--src选择被测源码树（默认本仓库）：
    python scripts/diag_bin_failure_train.py --ckpt CKPT.pt --motion_source SOURCE.h5 --out DIR
"""

import argparse
import json
import time
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_source", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数，每个起点占--repeats个")
parser.add_argument("--repeats", type=int, default=4, help="每个起点的重复次数（噪声与随机化各不相同）")
parser.add_argument("--steps", type=int, default=100, help="每次最多播放的控制步数（100步为2 s）")
parser.add_argument("--stride", type=int, default=50, help="起点间隔帧数")
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

from supermimic.tasks.tracking.mdp import NUM_J, OBS_DIM

torch.manual_seed(args.seed)
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
    starts += [s0 + o for o in range(0, n - 1, args.stride)]
starts = torch.tensor(starts, device=dev)
P, R = len(starts), args.repeats
# 每个起点连续占用 R 个环境，方便结束后还原为（起点数, 重复次数）并统计失败频次。
per_batch = max(args.num_envs // R, 1)
N = per_batch * R
fails = torch.zeros(P, dtype=torch.long, device=dev)
# 训练条件的环境：未显式传入的采样选项取源码树默认值，终止后环境自动重置，之后的步数忽略
env = subject.make_env(N, motion, mdp, run_args=a_, inherit=("gamma",))
hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
t0 = time.time()
with torch.no_grad():
    for b0 in range(0, P, per_batch):
        cnt = min(per_batch, P - b0)
        block = torch.cat((starts[b0:b0 + cnt], starts[b0 + cnt - 1:b0 + cnt].expand(per_batch - cnt)))
        env.reset(torch.arange(N, device=dev), block.repeat_interleave(R))
        ended = torch.zeros(N, dtype=torch.bool, device=dev)
        failed = torch.zeros(N, dtype=torch.bool, device=dev)
        for _ in range(args.steps):
            S = env.state()
            fresh = env.ep_step == 0
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(hist),
                               torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
            a = torch.clamp(pi(hist, motion.goal(env.f, K)) + args.action_noise * torch.randn(N, NUM_J, device=dev), -20, 20)
            out = env.step(a)
            # 训练环境会自动重置，但诊断只关心指定起点的第一回合；用 ended 屏蔽后续回合的失败。
            failed |= out["term"] & ~ended
            ended |= out["done"]
            if bool(ended.all()):
                break
        # 沿重复次数求和，并丢弃为填满最后批次而复制的起点，保证每个真实起点权重相同。
        fails[b0:b0 + cnt] = failed.view(per_batch, R)[:cnt].sum(-1)
elapsed = time.time() - t0

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
clip = motion.clip_id[starts]
torch.save({"start": starts.cpu(), "clip": clip.cpu(), "offset": (starts - motion.clip_start[clip]).cpu(),
            "clip_len": motion.clip_len[clip].cpu(), "fails": fails.cpu(), "repeats": R}, out / "labels.pt")
summary = {"positions": P, "repeats": R, "steps": args.steps, "fail_prob": round(float(fails.sum()) / (P * R), 5),
           "positions_with_failure": int((fails > 0).sum()), "clips_with_failure": int(len(torch.unique(clip[fails > 0]))),
           "elapsed_s": round(elapsed, 1), "subject": subject.provenance()}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
