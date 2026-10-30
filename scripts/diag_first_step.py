"""诊断检查点的首步惩罚，按论文附录 F.1 从每个片段首帧评估，不修改检查点。

流程与 evaluate.py 一致，另记录首个控制步（a_prev = 0）的各奖励项及去掉首步后的归一化回报；
失败的统计口径也与 evaluate.py 相同，并记录触发终止的步数与当时目标是否已截断在段末。
在仓库根目录运行，--src 指定被测源码树，默认使用本仓库。"""

import argparse
import csv
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
parser.add_argument("--transient_steps", type=int, default=20)
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, RewardCfg
from supermimic.tasks.tracking.motion import CONTROL_DT

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
subject.set_amp(a_)
motion, mdp = subject.reference(args.motion_source)
C = len(motion.names)
pi = subject.policy(ck)
w_rate, w_lim = RewardCfg.w_action_rate, RewardCfg.w_joint_limit

env = subject.make_env(C, motion, mdp, eval_mode=True)
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
    fail_f = torch.zeros(C, dtype=torch.long, device=dev)  # 触发终止那一步推进前的参考帧
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        f_prev, failed_prev = env.f.clone(), env.failed_eval.clone()
        out = env.step(a)
        fail_f = torch.where(env.failed_eval & ~failed_prev, f_prev, fail_f)
        r, terms = out["r"], out["terms"]
        af = alive.float()
        rec["ret"] += r * af
        rec["steps"] += af
        # 首步的前一动作为零，单独保存其惩罚及动作大小，避免与稳定跟踪阶段混在一起。
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
        # 瞬态曲线每个时间位置只统计仍活跃的片段，分母随片段结束而变化。
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
# 目标取f+1..f+K帧并在段末截断；f + K超过段末时策略看到的未来参考已停在末帧。
truncated = failed & (fail_f + K > clip_end)
fail_step = torch.where(failed, fail_f - motion.clip_start[ids] + 1, 0)
norm = rec["ret"] / denom
# 去掉首步时分子与理论可得回报都减去一步；下面仅移除惩罚的反事实指标仍保留原分母。
norm_wo_first = (rec["ret"] - rec["r0"]) / ((motion.clip_len - 2).float() * MAX_STEP_REWARD)
# 反事实对照：保持同一轨迹，仅去掉首步的动作平滑惩罚
norm_no_rate0 = (rec["ret"] - w_rate * rec["rate0"] * CONTROL_DT) / denom
rest = (rec["steps"] - 1).clamp(min=1)
pen_rate0 = w_rate * rec["rate0"] * CONTROL_DT / MAX_STEP_REWARD  # 以归一化控制步为单位
pen_lim0 = w_lim * rec["lim0"] * CONTROL_DT / MAX_STEP_REWARD


def q(x, ps=(0.0, 0.1, 0.5, 0.9, 1.0)):
    """计算指定分位数及均值，并将结果舍入为便于 JSON 输出的数值。"""
    x = x.float().cpu()
    return {f"p{int(p * 100)}": round(float(torch.quantile(x, p)), 4) for p in ps} | {"mean": round(float(x.mean()), 4)}


summary = {
    "ckpt": args.ckpt, "iteration": ck.get("iteration"), "subject": subject.provenance(), "clips": C, "frames": int(motion.clip_len.sum()),
    "elapsed_s": round(elapsed, 1), "steps": step, "n_failed": int(failed.sum()),
    "n_failed_guided": int((failed & ~truncated).sum()), "goal_frames": K,
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
    # 首步两项惩罚占平均回报缺口（1 - norm）的比例
    "first_step_share_of_deficit": round(float((-(pen_rate0 + pen_lim0) / (motion.clip_len - 1)).mean() / (1 - norm).mean()), 4),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with open(out_dir / "per_clip.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "clip_len", "failed", "norm", "norm_wo_first", "norm_no_rate0", "r0_norm", "rate0_raw", "lim0_raw",
                "a0_l2", "rate_rest_mean", "a_l2_rest_mean", "fail_step", "goal_truncated"])
    for i in range(C):
        w.writerow([motion.names[i], int(motion.clip_len[i]), int(failed[i]), f"{norm[i]:.4f}", f"{norm_wo_first[i]:.4f}",
                    f"{norm_no_rate0[i]:.4f}", f"{rec['r0'][i] / MAX_STEP_REWARD:.4f}", f"{rec['rate0'][i]:.3f}",
                    f"{rec['lim0'][i]:.4f}", f"{rec['a0_norm'][i]:.3f}", f"{rec['rate_rest_sum'][i] / rest[i]:.4f}",
                    f"{rec['a_norm_rest_sum'][i] / rest[i]:.3f}", int(fail_step[i]), int(truncated[i])])
print(json.dumps(summary, indent=2, ensure_ascii=False))
