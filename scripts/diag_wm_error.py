"""在新采集的策略轨迹上测量世界模型 k 步预测误差，按物理单位分量报告，不修改检查点。

论文图 3(c) 的 8 步误差为 0.007–0.011，仅统计状态、按预设标称误差归一化并在线测量，
但未提供标称误差及聚合方式。训练指标 wm_err8 使用单步增量的运行标准差和回放窗口，定义不同。
本脚本在论文对应的策略轨迹上报告物理误差，并比较多种候选标称尺度。
策略使用训练时的域随机化、重置噪声、分箱起点、观测和动作噪声采集转移；
世界模型从记录的起点按记录的动作推演 k 步，与训练拟合方式一致。"""

import argparse
import json
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True, help="checkpoint with wm, pi and args")
parser.add_argument("--motion_source", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--warmup", type=int, default=100, help="steps before recording (decorrelates the initial resets)")
parser.add_argument("--steps", type=int, default=400, help="recorded steps per environment")
parser.add_argument("--window", type=int, default=8)
parser.add_argument("--stride", type=int, default=4, help="spacing of window starts along each trajectory")
parser.add_argument("--chunk", type=int, default=16384, help="windows per world-model batch")
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

from supermimic.tasks.tracking.mdp import NUM_J, OBS_DIM, RewardCfg, normalize_xi, unflatten_state

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh, k = a_["goal_frames"], a_["hist"], args.window
subject.set_amp(a_)
motion, mdp = subject.reference(args.motion_source, a_)
wm = subject.world_model(ck)
pi = subject.policy(ck)

# ---- 在训练配置下用检查点策略采集新转移 ----
N, T = args.num_envs, args.steps
env = subject.make_env(N, motion, mdp, run_args=a_, inherit=("gamma", "bin_clip_end", "bin_prior", "bin_clip_cap"))
hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
rec = {"s": torch.empty(T, N, subject.STATE_DIM, device=dev), "s1": torch.empty(T, N, subject.STATE_DIM, device=dev),
       "a": torch.empty(T, N, NUM_J, device=dev), "xi": torch.empty(T, N, subject.XI_DIM, device=dev),
       "done": torch.empty(T, N, dtype=torch.bool, device=dev)}
with torch.no_grad():
    for step in range(args.warmup + T):
        S = env.state()
        fresh = env.ep_step == 0
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(hist), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
        a = torch.clamp(pi(hist, motion.goal(env.f, K)) + args.action_noise * torch.randn(N, NUM_J, device=dev), -20, 20)
        out = env.step(a)
        t = step - args.warmup
        if t >= 0:
            for key in rec:
                rec[key][t] = out[key]

# ---- 局部坐标系 70 维误差的候选标称尺度：位置、旋转、线速度、角速度、q、qd ----
def per_dim(pos, rot, lin, ang, q, qd):
    """按位置、旋转、线速度、角速度、关节位置和速度展开各分量的归一化尺度。"""
    return torch.cat([torch.full((n,), float(x)) for n, x in ((3, pos), (3, rot), (3, lin), (3, ang), (NUM_J, q), (NUM_J, qd))]).to(dev)


nan = float("nan")  # 候选方案未定义的维度不参与均值计算
err_floor = per_dim(3e-3, 5e-3, 2e-2, 5e-2, 3e-3, 5e-2)  # supermimic/tasks/tracking/rl/trainer.py的ERR_FLOOR
c = RewardCfg
# 方案 D：用 0.05 的投影重力噪声代替姿态尺度；根部位置和线速度不属于观测
candidates = {
    "A_training_wm_err8 (max(sigma_delta, floor))": torch.maximum(wm.out_norm.std(), err_floor),
    "B_fixed_floor_constants (train.py ERR_FLOOR)": err_floor,
    "C_reward_scales_root_only (Table A.2)": per_dim(c.body_pos_std, c.body_ori_std, c.body_linvel_std, c.body_angvel_std, nan, nan),
    "D_sensor_noise (Table A.1 half-widths)": per_dim(nan, 0.05, nan, 0.2, 0.01, 0.5),
}

# ---- 从每个记录窗口的起点执行 k 步世界模型推演 ----
starts = torch.arange(0, T - k + 1, args.stride, device=dev)
comp = ("pos_m", "rot_rad", "linvel_mps", "angvel_radps", "q_rms_rad", "qd_rms_radps")
step_sum = {name: torch.zeros(k, device=dev, dtype=torch.float64) for name in comp}
cand_sum = {name: torch.zeros(k, device=dev, dtype=torch.float64) for name in candidates}
count = torch.zeros(k, device=dev, dtype=torch.float64)
last = {name: [] for name in comp}  # 每个窗口第 k 步的分量误差，仅统计截至该步仍有效的窗口
sq_sum = torch.zeros(k, 70, device=dev, dtype=torch.float64)


def components(e):
    """将逐维误差聚合为根部向量范数及关节位置、速度的均方根误差。"""
    return torch.stack((e[:, 0:3].norm(dim=-1), e[:, 3:6].norm(dim=-1), e[:, 6:9].norm(dim=-1), e[:, 9:12].norm(dim=-1),
                        e[:, 12:12 + NUM_J].pow(2).mean(-1).sqrt(), e[:, 12 + NUM_J:].pow(2).mean(-1).sqrt()), -1)


with torch.no_grad():
    # 将窗口起点与环境编号组成笛卡尔积，再分块推演，避免一次展开所有序列耗尽显存。
    idx_t = starts.repeat_interleave(N)
    idx_e = torch.arange(N, device=dev).repeat(len(starts))
    for c0 in range(0, len(idx_t), args.chunk):
        t0, e0 = idx_t[c0:c0 + args.chunk], idx_e[c0:c0 + args.chunk]
        rows = t0.unsqueeze(-1) + torch.arange(k, device=dev)
        envs = e0.unsqueeze(-1).expand_as(rows)
        s, s1, act, xi, done = (rec[key][rows, envs] for key in ("s", "s1", "a", "xi", "done"))
        # 仅当转移 0 至 j-1 均未结束回合时，第 j 个转移才有效，与 ReplayBuffer.sample_windows 一致
        mask = torch.cumprod(torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), -1).float(), -1)
        # 窗口内只在首步使用真实状态，后续递归预测但沿用记录的动作，以隔离模型误差。
        S = unflatten_state(s[:, 0])
        for j in range(k):
            S = wm.step(S, act[:, j], normalize_xi(xi[:, j]))
            e = wm.state_error(S, unflatten_state(s1[:, j]))  # 形状为（B, 70），使用局部坐标系下的物理单位
            m = mask[:, j]
            cm = components(e)
            for i, name in enumerate(comp):
                step_sum[name][j] += (cm[:, i] * m).sum()
            # 各候选尺度只在其定义的维度上平均平方误差；物理分量范数与归一化均方误差分开汇报。
            for name, nom in candidates.items():
                use = ~torch.isnan(nom)
                cand_sum[name][j] += (((e[:, use] / nom[use]) ** 2).mean(-1) * m).sum()
            sq_sum[j] += (e.double() ** 2 * m.double().unsqueeze(-1)).sum(0)
            count[j] += m.sum()
            if j == k - 1:
                keep = m > 0
                for i, name in enumerate(comp):
                    last[name].append(cm[keep, i].cpu())


def q(x, ps=(0.5, 0.9, 0.99)):
    """计算指定分位数及均值，并将结果舍入为便于 JSON 输出的数值。"""
    return {f"p{int(p * 100)}": round(float(torch.quantile(x, p)), 5) for p in ps} | {"mean": round(float(x.mean()), 5)}


last = {name: torch.cat(v) for name, v in last.items()}
sigma_delta = wm.out_norm.std()
blocks = {"pos": slice(0, 3), "rot": slice(3, 6), "linvel": slice(6, 9), "angvel": slice(9, 12),
          "q": slice(12, 12 + NUM_J), "qd": slice(12 + NUM_J, 70)}
# 先累计逐轴平方误差再开方，得到总体 RMS；这与先对每个窗口取范数再平均的指标不同。
rms_k = (sq_sum[k - 1] / count[k - 1]).sqrt()
summary = {
    "ckpt": args.ckpt, "iteration": ck.get("iteration"), "subject": subject.provenance(), "num_envs": N, "recorded_steps": T, "window": k,
    "windows_valid_at_step_k": int(count[k - 1]), "windows_total": int(len(idx_t)),
    f"step_{k}_physical": {name: q(v) for name, v in last.items()},
    "mean_per_step_physical": {name: [round(float(x), 5) for x in (step_sum[name] / count)] for name in comp},
    f"step_{k}_normalized_mean_sq": {name: round(float(cand_sum[name][k - 1] / count[k - 1]), 5) for name in candidates},
    "mean_over_steps_normalized_mean_sq": {name: round(float((cand_sum[name] / count).mean()), 5) for name in candidates},
    "per_axis_rms_step_k_by_block": {b: [round(float(x), 5) for x in rms_k[s]] if b in ("pos", "rot", "linvel", "angvel")
                                     else round(float(rms_k[s].pow(2).mean().sqrt()), 5) for b, s in blocks.items()},
    "sigma_delta_by_block (one-step increment std)": {b: round(float(sigma_delta[s].median()), 5) for b, s in blocks.items()},
    "training_log_note": "compare candidate A at step k with train_log.csv wm_err8 at the checkpoint's iteration",
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
