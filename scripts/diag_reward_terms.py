"""按附录F.1条件（整个参考文件一批、首帧起步、TF32）评估检查点，把归一化回报拆成各奖励项，可选叠加训练时的动作噪声。

各项按“每个活跃步、除以单步最大奖励”平均，跟踪六项之和加两项惩罚即归一化回报的逐步均值；
每步用拆出的各项重算奖励并与环境的奖励比较（reward_check_maxdiff），口径不一致时一眼可见。
另记各关节相对参考的有符号与绝对误差、软限位越界量与动作变化率（关节顺序同constants.G1_JOINT_NAMES），
用来找退步集中在哪些关节。
--noise与--obs_noise分开时，仿真执行加噪声的动作，而观测里的上一动作换成干净动作加--obs_noise的噪声，
用来区分退步来自执行噪声还是观测中的噪声。在仓库根目录运行，--src 指定被测源码树，默认使用本仓库：
    python scripts/diag_reward_terms.py --ckpt CKPT --motion_source shard_000.h5 --out res.json [--noise 0.1 --seed 0]
"""
import argparse
import json
import time

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_source", required=True, help="reference file evaluated as one batch (e.g. one source shard)")
parser.add_argument("--out", required=True, help="output JSON")
parser.add_argument("--noise", type=float, default=0.0, help="std of the action noise executed in simulation (training: 0.1)")
parser.add_argument("--obs_noise", type=float, default=None,
                    help="std of the noise in the previous action the policy observes; default: the same as --noise (as in training)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, RewardCfg
from supermimic.tasks.tracking.motion import CONTROL_DT
from supermimic.utils.rotations import rot_angle_sq

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
torch.manual_seed(args.seed)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
subject.set_amp(a_)
pi = subject.policy(ck)
motion, mdp = subject.reference(args.motion_source, a_)
C = len(motion.names)
env = subject.make_env(C, motion, mdp, eval_mode=True)
ids = torch.arange(C, device=dev)
clip_end = motion.clip_end[ids]
c = RewardCfg
NAMES = ("body_pos", "body_ori", "body_lv", "body_av", "root_pos", "root_ori", "action_rate", "joint_limit")
W = torch.tensor([1, 1, 1, 1, 0.5, 0.5, c.w_action_rate, c.w_joint_limit], device=dev)
acc = torch.zeros(C, len(NAMES), device=dev)
raw = torch.zeros(C, 8, device=dev)  # 各项原始误差（未经指数核）与惩罚原值
steps = torch.zeros(C, device=dev)
ret = torch.zeros(C, device=dev)
amax_hit = torch.zeros(C, device=dev)
NJ = motion.q.shape[-1]
jerr = torch.zeros(C, NJ, device=dev)  # 关节角相对参考的有符号误差
jabs = torch.zeros(C, NJ, device=dev)
jlim = torch.zeros(C, NJ, device=dev)
arate_j = torch.zeros(C, NJ, device=dev)


def components(S, f, a, a_prev):
    p, R, v, w, q, qd = S
    bp, bR, bv, bw = mdp.kin.forward(p, R, v, w, q, qd)
    rp, rR = motion.root_p[f], motion.root_R[f]
    ref_bp, dR = mdp._anchored_ref_bodies(p, R, rp, rR, f)
    ref_bR = dR.unsqueeze(-3) @ motion.body_R[f]
    e = torch.stack((((ref_bp - bp) ** 2).sum(-1).mean(-1), rot_angle_sq(ref_bR, bR).mean(-1),
                     ((motion.body_v[f] - bv) ** 2).sum(-1).mean(-1), ((motion.body_w[f] - bw) ** 2).sum(-1).mean(-1),
                     ((rp - p) ** 2).sum(-1), rot_angle_sq(rR, R),
                     ((a - a_prev) ** 2).sum(-1),
                     (torch.relu(q - mdp.soft_hi) + torch.relu(mdp.soft_lo - q)).sum(-1)), -1)
    std = torch.tensor([c.body_pos_std, c.body_ori_std, c.body_linvel_std, c.body_angvel_std, c.root_pos_std, c.root_ori_std], device=dev)
    k = torch.cat((torch.exp(-e[:, :6] / std ** 2), e[:, 6:]), -1)
    return k, e


t0 = time.time()
with torch.no_grad():
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[ids].clone())
    hist, maxdiff = None, 0.0
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        a_clean = a
        if args.noise > 0:
            a = torch.clamp(a + args.noise * torch.randn_like(a), -20, 20)  # 与训练采集相同
        alive = (~env.done_eval).float()
        f_prev, a_prev = env.f.clone(), env.a_prev.clone()
        out = env.step(a)
        on = args.noise if args.obs_noise is None else args.obs_noise
        if on != args.noise:  # 观测里的上一动作换成干净动作加指定噪声，仿真执行的动作不变
            env.a_prev = a_clean + on * torch.randn_like(a_clean) if on > 0 else a_clean.clone()
        k, e = components(env.state(), f_prev + 1, a, a_prev)
        r_chk = (k * W).sum(-1) * CONTROL_DT
        maxdiff = max(maxdiff, float(((r_chk - out["r"]).abs() * alive).max()))
        acc += k * alive[:, None]
        raw += e * alive[:, None]
        steps += alive
        ret += out["r"] * alive
        amax_hit += (a.abs() > 3.0).float().mean(-1) * alive
        q1, rq = env.state()[4], motion.q[f_prev + 1]
        jerr += (q1 - rq) * alive[:, None]
        jabs += (q1 - rq).abs() * alive[:, None]
        jlim += (torch.relu(q1 - mdp.soft_hi) + torch.relu(mdp.soft_lo - q1)) * alive[:, None]
        arate_j += (a - a_prev) ** 2 * alive[:, None]
        env.f = torch.minimum(env.f, clip_end - 1)
n = steps.clamp(min=1)[:, None]
per_step = (acc / n) * W * CONTROL_DT / MAX_STEP_REWARD  # 每项对逐步归一化回报的贡献
norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
res = {"ckpt": args.ckpt, "iteration": ck.get("iteration"), "subject": subject.provenance(), "motion_source": args.motion_source,
       "noise": args.noise, "obs_noise": args.obs_noise, "clips": C,
       "n_failed": int(env.failed_eval.sum()), "norm_return": round(float(norm.mean()), 4),
       "reward_check_maxdiff": maxdiff, "elapsed_s": round(time.time() - t0, 1),
       "contrib": {nm: round(float(per_step[:, i].mean()), 5) for i, nm in enumerate(NAMES)},
       "kernel_mean": {nm: round(float((acc / n)[:, i].mean()), 5) for i, nm in enumerate(NAMES)},
       "raw_error_mean": {nm: round(float((raw / n)[:, i].mean()), 5) for i, nm in enumerate(NAMES)},
       "frac_action_abs_gt3": round(float((amax_hit / n[:, 0]).mean()), 5),
       "joint_signed_err": [round(float(x), 4) for x in (jerr / n).mean(0)],
       "joint_abs_err": [round(float(x), 4) for x in (jabs / n).mean(0)],
       "joint_limit": [round(float(x), 5) for x in (jlim / n).mean(0)],
       "joint_action_rate": [round(float(x), 4) for x in (arate_j / n).mean(0)]}
open(args.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("ckpt", "subject")}, ensure_ascii=False))
