"""交叉评估训练状态中策略与价值网络的偏差，不修改检查点。

输入包含 wm、pi、critic 和 args 的完整训练状态（如 last.pt）。每个策略采集新轨迹，
每个价值网络分别评估这些轨迹。真实偏差为 V(x_t) 与 n 步真实回报之差：终止截断回报，
尾部使用同一价值网络的 gamma^n V(x_{t+n})，超时截断的窗口丢弃，并按窗口内是否失败分组。
想象偏差包括 32 步世界模型推演的 V 与 lambda 目标残差、末端价值、策略目标的自举项，
以及起点想象目标与真实回报之差。另报告真实起点与想象末端的归一化输入绝对值及末端价值梯度范数；
运行归一化器在 2e8 个样本后冻结。新环境分箱得分为零、起点近似均匀，使用训练时的动作与观测噪声。

用法：python scripts/diag_critic_bias.py --states A.pt B.pt --motion_source SUBSET.h5 --out DIR"""

import argparse
import json
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--states", nargs="+", required=True, help="checkpoints with wm, pi, critic and args")
parser.add_argument("--motion_source", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--warmup", type=int, default=100, help="steps before recording (decorrelates the initial resets)")
parser.add_argument("--starts", type=int, default=10, help="start steps per environment")
parser.add_argument("--start_stride", type=int, default=10)
parser.add_argument("--mc_steps", type=int, default=200, help="real-return window (tail weight gamma^n)")
parser.add_argument("--chunk", type=int, default=4096, help="imagined sequences per batch")
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import torch

from supermimic.tasks.tracking.mdp import NUM_J, OBS_DIM, normalize_xi, unflatten_state

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

cks = {Path(p).stem: torch.load(p, map_location="cpu", weights_only=False) for p in args.states}
a0 = next(iter(cks.values()))["args"]
K, Hh, HZ, gamma, lam = a0["goal_frames"], a0["hist"], a0["horizon"], a0["gamma"], a0["lam"]
subject.set_amp(a0)
motion, mdp = subject.reference(args.motion_source, a0)
models = {name: (subject.world_model(ck), subject.policy(ck), subject.critic(ck)) for name, ck in cks.items()}
survival_prob = subject.survival_prob(mdp)


def critic_input(S, f, xi_n):
    """拼接状态的价值网络特征与未来动作目标，供价值估计使用。"""
    return torch.cat((mdp.critic_features(S, f, xi_n), motion.goal(f, K)), -1)


def stats(x):
    """统计样本数量、均值和第 10、50、90 百分位数。"""
    x = x.float()
    return {"mean": round(float(x.mean()), 4), "p10": round(float(torch.quantile(x, 0.1)), 4),
            "p50": round(float(torch.quantile(x, 0.5)), 4), "p90": round(float(torch.quantile(x, 0.9)), 4),
            "n": int(x.numel())}


def collect(pi):
    """重新采集轨迹，保留逐步状态用于真实回报计算，并记录起点观测历史。"""
    torch.manual_seed(args.seed)
    N = args.num_envs
    # 记录长度覆盖最后一个采样起点的完整真实回报窗口，并额外保留尾部自举状态。
    T = (args.starts - 1) * args.start_stride + args.mc_steps + 1
    env = subject.make_env(N, motion, mdp, run_args=a0, inherit=("gamma", "bin_clip_end", "bin_prior"))
    hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
    rec = {"s": torch.empty(T, N, subject.STATE_DIM, device=dev), "f": torch.empty(T, N, dtype=torch.long, device=dev),
           "xi": torch.empty(T, N, subject.XI_DIM, device=dev), "r": torch.empty(T, N, device=dev),
           "term": torch.empty(T, N, dtype=torch.bool, device=dev), "done": torch.empty(T, N, dtype=torch.bool, device=dev)}
    starts = {"hist": [], "a_prev": []}
    with torch.no_grad():
        for step in range(args.warmup + T):
            S = env.state()
            fresh = env.ep_step == 0
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(hist), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
            t = step - args.warmup
            # 预热阶段只让环境分布展开；正式记录后按间隔保存起点历史，供同起点想象推演使用。
            if t >= 0 and t % args.start_stride == 0 and t // args.start_stride < args.starts:
                starts["hist"].append(hist.clone()), starts["a_prev"].append(env.a_prev.clone())
            a = torch.clamp(pi(hist, motion.goal(env.f, K)) + args.action_noise * torch.randn(N, NUM_J, device=dev), -20, 20)
            out = env.step(a)
            if t >= 0:
                for key in rec:
                    rec[key][t] = out[key]
    return rec, {k: torch.stack(v) for k, v in starts.items()}


def real_returns(rec, critic):
    """计算每个起点的 n 步回报；超时截断窗口时标记为无效。"""
    n, S0 = args.mc_steps, args.starts
    xs = [critic_input(unflatten_state(rec["s"][t]), rec["f"][t], normalize_xi(rec["xi"][t]))
          for t in range(0, S0 * args.start_stride, args.start_stride)]
    G, valid, failed = [], [], []
    for i in range(S0):
        t0 = i * args.start_stride
        g = torch.zeros_like(rec["r"][0])
        live = torch.ones_like(g, dtype=torch.bool)
        ok = torch.ones_like(live)
        # 只累加首次回合结束之前的奖励；失败是合法终点，超时则使该窗口失效，
        # 避免将重置后的状态误作原回合的延续。
        for j in range(n):
            g += torch.where(live, gamma**j * rec["r"][t0 + j], 0.0)
            ok &= ~(live & rec["done"][t0 + j] & ~rec["term"][t0 + j])
            live &= ~rec["done"][t0 + j]
        # 仅对仍存活的窗口补上 n 步后的价值，且尾部使用当前被测价值网络，保持比较口径一致。
        tail = critic(critic_input(unflatten_state(rec["s"][t0 + n]), rec["f"][t0 + n], normalize_xi(rec["xi"][t0 + n])))
        G.append(g + torch.where(live, gamma**n * tail, 0.0)), valid.append(ok), failed.append(~live)
    return torch.stack(xs), torch.stack(G), torch.stack(valid), torch.stack(failed)


def imagine(wm, pi, rec, st):
    """对拥有完整时域的起点执行训练式想象推演，不计算梯度。"""
    idx = torch.arange(0, args.starts * args.start_stride, args.start_stride, device=dev)
    s, f, xi = rec["s"][idx].flatten(0, 1), rec["f"][idx].flatten(), rec["xi"][idx].flatten(0, 1)
    o_hist, a_prev = st["hist"].flatten(0, 1), st["a_prev"].flatten(0, 1)
    # 先标记片段内能容纳完整想象时域的起点；后续批量计算可包含占位项，最终统计必须应用此掩码。
    ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
    out = {"X": [], "r": [], "keep": [], "ok": ok}
    with torch.no_grad():
        for c0 in range(0, len(f), args.chunk):
            sl = slice(c0, c0 + args.chunk)
            S, ff, x, oh, ap = unflatten_state(s[sl]), f[sl].clamp(max=motion.num_frames - HZ - 1), xi[sl], o_hist[sl], a_prev[sl]
            xn = normalize_xi(x)
            X, R, Kp = [critic_input(S, ff, xn)], [], []
            for _ in range(HZ):
                a = pi(oh, motion.goal(ff, K))
                S1 = wm.step(S, a, xn)
                f1 = ff + 1
                R.append(mdp.reward(S1, f1, a, ap)), Kp.append(survival_prob(S1, f1))
                oh = torch.cat((oh[:, 1:], mdp.add_noise(mdp.obs(S1, a, x, f1)).unsqueeze(1)), 1)
                S, ff, ap = S1, f1, a
                X.append(critic_input(S, ff, xn))
            out["X"].append(torch.stack(X)), out["r"].append(torch.stack(R)), out["keep"].append(torch.stack(Kp))
    return {k: torch.cat(v, 1) if k != "ok" else v for k, v in out.items()}


summary = {"states": {n: int(ck["iteration"]) for n, ck in cks.items()}, "subject": subject.provenance(), "num_envs": args.num_envs, "mc_steps": args.mc_steps,
           "horizon": HZ, "tail_weight": round(gamma**args.mc_steps, 4), "pairs": {}}
# 每个策略只采集一组轨迹，再交叉使用各检查点的价值网络评分，分离轨迹分布与价值估计的影响。
for pn, (wm, pi, _) in models.items():
    rec, st = collect(pi)
    im = imagine(wm, pi, rec, st)
    ok = im["ok"]
    c_t = torch.cat((torch.ones_like(im["keep"][:1]), torch.cumprod(im["keep"], 0)[:-1]), 0)
    disc = gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    alive_end = torch.cumprod(im["keep"], 0)[-1]
    roll = {"real_reward_per_step": round(float(rec["r"].mean() / 0.1), 4), "real_term_frac_per_step": round(float(rec["term"].float().mean()), 5),
            "imag_reward_per_step": round(float((c_t * im["r"]).sum(0)[ok].mean() / HZ / 0.1), 4),
            "imag_survival_end": round(float(alive_end[ok].mean()), 4)}
    for cn, (_, _, critic) in models.items():
        with torch.no_grad():
            xs, G, valid, failed = real_returns(rec, critic)
            V0 = critic(xs)
            V = torch.stack([critic(im["X"][t]) for t in range(HZ + 1)])
            # 从末端价值逆序递推 lambda 回报，再恢复时间顺序与逐步价值对齐。
            Rl, tg = V[-1], []
            for t in range(HZ - 1, -1, -1):
                Rl = im["r"][t] + gamma * im["keep"][t] * ((1 - lam) * V[t + 1] + lam * Rl)
                tg.append(Rl)
            tg = torch.stack(tg[::-1])
            resid = (V[:-1] - tg)[:, ok]
            boot = gamma**HZ * alive_end * V[-1]
            z0, zH = critic.norm(xs).abs().amax(-1).flatten(), critic.norm(im["X"][-1]).abs().amax(-1)[ok]
            frac0, fracH = (critic.norm(xs).abs() > 5).float().mean(), (critic.norm(im["X"][-1][ok]).abs() > 5).float().mean()
        # 其余统计在无梯度模式计算；仅这一段对末端输入求导，以测量价值外推的敏感程度。
        xe = im["X"][-1][ok][:4096].clone().requires_grad_(True)
        gn = torch.autograd.grad(critic(xe).sum(), xe)[0].norm(dim=-1)
        b = (V0 - G)[valid]
        fl = failed[valid]
        okf = ok.view(args.starts, -1)
        optimism = (tg[0].view(args.starts, -1) - G)[valid & okf]
        summary["pairs"][f"policy {pn} / critic {cn}"] = {
            **({"rollout": roll} if cn == pn else {}),
            "real_V": stats(V0[valid]), "real_return": stats(G[valid]),
            "real_bias_all": stats(b), "real_bias_failing_starts": stats(b[fl]) if fl.any() else None,
            "real_bias_surviving_starts": stats(b[~fl]), "failing_start_frac": round(float(fl.float().mean()), 4),
            "imag_residual_V_minus_target": stats(resid), "imag_residual_t0": stats(resid[0]), "imag_residual_t31": stats(resid[-1]),
            "imag_V_start": stats(V[0][ok]), "imag_V_end": stats(V[-1][ok]), "imag_bootstrap_term": stats(boot[ok]),
            "imag_target_t0_minus_real_return": stats(optimism),
            "z_max_real_starts": stats(z0), "z_max_imag_end": stats(zH),
            "frac_abs_z_gt5_real": round(float(frac0), 5), "frac_abs_z_gt5_imag_end": round(float(fracH), 5),
            "grad_norm_V_imag_end": stats(gn),
        }
        print(f"[{pn} policy | {cn} critic] real bias {stats(b)['mean']:+.3f}  imag resid {stats(resid)['mean']:+.3f}  "
              f"V_end {stats(V[-1][ok])['mean']:.2f}  optimism {stats(optimism)['mean']:+.3f}  zmax_end p90 {stats(zH)['p90']:.1f}", flush=True)
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
