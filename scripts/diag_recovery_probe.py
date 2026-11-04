"""M1b恢复可行性探针：把跟踪中的策略推倒、继续推进物理，判断世界模型解析梯度能否用于恢复任务。

不修改检查点和训练代码。设计见docs/teleop/causal_intent_encoder.md第4.1、4.3节与M1b，回答三个问题：
1. 世界模型在各类状态上的多步误差有多大；用推倒数据微调后能否拟合，正常状态的精度会不会变差。
2. 回报回传到动作的梯度：现有目标（跟踪奖励＋软存活）、恢复窗口内不截断（keep=1）和候选恢复奖励分别如何。
3. 从相同物理起点执行模型优化出的动作，物理回报是否真的改善，是否好过同样幅度的随机扰动。

采集用评估模式的环境：不做域随机化，失败后不重置，物理连续推进。现有检查点的策略不会起身，因此这里覆盖不到
真正的起身，也只有受轻推后自己接回的那部分重接（归入disturbed_tracked）；rising只统计低姿态且向上运动的状态
（主要是跟踪地面动作和弹跳），摘要里按实际样本数报告。
全库检查点的结果只作可行性判断，不下泛化结论。"""

import argparse
import copy
import json
from pathlib import Path

import torch

import _subject

PARTITION = ("normal_high", "normal_low", "disturbed_tracked", "imbalance", "fallen")  # 互斥且完整
CATEGORIES = PARTITION + ("rising",)  # rising与上面重叠，单独统计
COMPONENTS = ("pos_m", "rot_rad", "linvel_mps", "angvel_radps", "q_rms_rad", "qd_rms_radps", "norm_mse")
BLOCKS = {"pos": slice(0, 3), "rot": slice(3, 6), "linvel": slice(6, 9), "angvel": slice(9, 12), "q": slice(12, 41), "qd": slice(41, 70)}
OBJECTIVES = ("track_surv", "track_keep", "recover", "recover_task")


# ---------------- 不依赖仿真的计算 ----------------
def classify(p_z, v_z, e_z, e_tilt, perturbed, low=0.45, tracked_z=0.1, tracked_tilt=0.2) -> dict:
    """按跟踪误差、骨盆高度和是否已受扰分类。误差相对参考，所以参考本来就低时算normal_low，不算倒地。

    normal_*只含受扰之前的状态，是干净的基线；受扰后高度和倾斜仍在门槛内（含自己接回的）算disturbed_tracked。"""
    tracked = (e_z < tracked_z) & (e_tilt < tracked_tilt)
    is_low = p_z < low
    return {"normal_high": tracked & ~is_low & ~perturbed, "normal_low": tracked & is_low & ~perturbed,
            "disturbed_tracked": tracked & perturbed, "imbalance": ~tracked & ~is_low, "fallen": ~tracked & is_low,
            "rising": (p_z < 0.6) & (v_z > 0.2)}


def crosses_push(push_step, t0, j):
    """窗口从t0的真实状态开始，第j个转移（从0计）之前是否施加过冲量。

    冲量在转移p之前写入，s[p]不等于s1[p-1]，模型无从得知，所以t0<p<=t0+j的转移不能算模型误差，也不能进入拟合。"""
    return (push_step > t0) & (push_step <= t0 + j)


def deviation(e_z, e_tilt, term_height, term_tilt, tz, tt):
    """第4.3节的偏离程度d_t：1减去固定的偏差相似度核，正常时接近0，偏离时接近1。"""
    return 1 - torch.sigmoid((term_height - e_z) / tz) * torch.sigmoid((term_tilt - e_tilt) / tt)


def recover_reward(e_z, e_tilt):
    """第4.3节的候选r_recover：尺度与倒地误差同量级的宽核。"""
    return torch.exp(-e_z**2 / 0.5**2) + torch.exp(-e_tilt**2 / 1.0**2)


def objectives(r, track, keep, e_z, e_tilt, gamma, term_height, term_tilt, tz_d, tt_d, dt, lam=1.0) -> dict:
    """想象回报（不含critic），各输入形状为(H, B)。r和track已乘dt，r减track即正则项。

    track_surv是现有训练目标；track_keep相当于恢复窗口内不截断；recover再按d_t在跟踪与恢复之间混合；
    recover_task是recover去掉正则项，用来区分姿态真的变好和只是动作变平滑。"""
    disc = gamma ** torch.arange(r.shape[0], dtype=r.dtype, device=r.device).unsqueeze(-1)
    alive = torch.cumprod(keep, 0)
    c_t = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
    d = deviation(e_z, e_tilt, term_height, term_tilt, tz_d, tt_d)
    task = (1 - d) * track + lam * d * recover_reward(e_z, e_tilt) * dt
    return {"track_surv": (disc * c_t * r).sum(0), "track_keep": (disc * r).sum(0), "recover": (disc * (task + r - track)).sum(0),
            "recover_task": (disc * task).sum(0)}


def pick_frames(clip_start, clip_len, need, n, generator=None):
    """随机取n个起始帧，保证其后还有need帧留在同一片段内。没有够长的片段时报错，不悄悄截断。"""
    eligible = (clip_len > need).nonzero().flatten()
    if len(eligible) == 0:
        raise ValueError(f"no clip is longer than {need} frames")
    clips = eligible[torch.randint(len(eligible), (n,), device=clip_len.device, generator=generator)]
    room = clip_len[clips] - need
    offset = (torch.rand(n, device=clip_len.device, generator=generator) * room).long()
    return clip_start[clips] + offset


def sample_pushes(n, push_frac, speed, spin, after, device, generator=None) -> dict:
    """每个环境的瞬时冲量：水平方向随机的线速度增量和随机角速度增量；未被选中的环境step为-1，作无扰动对照。"""
    pushed = torch.rand(n, device=device, generator=generator) < push_frac
    step = torch.randint(after[0], after[1] + 1, (n,), device=device, generator=generator)
    angle = torch.rand(n, device=device, generator=generator) * 2 * torch.pi
    size = speed[0] + (speed[1] - speed[0]) * torch.rand(n, device=device, generator=generator)
    dv = torch.stack((size * torch.cos(angle), size * torch.sin(angle), torch.zeros(n, device=device)), -1)
    dw = spin * (2 * torch.rand(n, 3, device=device, generator=generator) - 1)
    return {"step": torch.where(pushed, step, -torch.ones_like(step)), "dv": dv, "dw": dw, "speed": size}


def pick_starts(masks: dict, n, generator=None):
    """从(T, E)形状的类别掩码中各取约n/类别数个起点，凑足n个；样本不足的类别由其余类别补足。返回t、e和类别编号。"""
    ts, es, labels = [], [], []
    quota = n // len(masks)
    for label, mask in enumerate(masks.values()):
        where = mask.nonzero()
        take = where[torch.randperm(len(where), device=mask.device, generator=generator)[:quota]]
        ts.append(take[:, 0]), es.append(take[:, 1]), labels.append(torch.full((len(take),), label, device=mask.device))
    t, e, label = torch.cat(ts), torch.cat(es), torch.cat(labels)
    if len(t) == 0:
        raise ValueError("no start states in any category")
    extra = torch.randint(len(t), (n - len(t),), device=t.device, generator=generator)
    return torch.cat((t, t[extra])), torch.cat((e, e[extra])), torch.cat((label, label[extra]))


def describe(x, ps=(0.5, 0.9)) -> dict:
    """有限样本的均值和分位数；非有限样本单独计数，推演发散时不影响其余统计。"""
    x = x.detach().double().flatten()
    finite = x[torch.isfinite(x)]
    out = {"n": int(x.numel()), "nonfinite": int(x.numel() - finite.numel())}
    if finite.numel():
        out["mean"] = round(float(finite.mean()), 6)
        out |= {f"p{int(p * 100)}": round(float(torch.quantile(finite, p)), 6) for p in ps}
    return out


def components(e):
    """把70维局部误差聚合为根部范数和关节均方根，与diag_wm_error.py相同。"""
    return torch.stack((e[..., 0:3].norm(dim=-1), e[..., 3:6].norm(dim=-1), e[..., 6:9].norm(dim=-1), e[..., 9:12].norm(dim=-1),
                        e[..., 12:41].pow(2).mean(-1).sqrt(), e[..., 41:70].pow(2).mean(-1).sqrt()), -1)


def refit_norm(norm, x):
    """用这批样本重估归一化统计量，模拟训练从一开始就包含这些数据。"""
    norm.mean.copy_(x.mean(0))
    norm.var.copy_(x.var(0, unbiased=False))
    norm.count.fill_(len(x))


# ---------------- 探针 ----------------
class Probe:
    """持有检查点、环境和采集到的连续轨迹，对给定世界模型做误差、梯度和物理三项检查。"""

    def __init__(self, args):
        self.args = args
        self.subject = _subject.Subject(args.src, args.device)
        from supermimic.tasks.tracking.mdp import CONTROL_DT, NUM_J, OBS_DIM, normalize_xi, unflatten_state
        from supermimic.tasks.tracking.rl.trainer import ERR_FLOOR
        from supermimic.utils.rotations import mat_to_quat

        self.CONTROL_DT, self.NUM_J, self.OBS_DIM = CONTROL_DT, NUM_J, OBS_DIM
        self.normalize_xi, self.unflatten, self.mat_to_quat = normalize_xi, unflatten_state, mat_to_quat
        self.dev = args.device
        self.floor = ERR_FLOOR.to(self.dev)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        self.ck, a_ = ck, ck["args"]
        self.K, self.Hh = a_["goal_frames"], a_["hist"]
        self.subject.set_amp(a_)
        self.motion, self.mdp = self.subject.reference(args.motion_source, a_)
        self.wm = self.subject.world_model(ck).requires_grad_(False)
        self.pi = self.subject.policy(ck).requires_grad_(False)
        self.gamma = a_.get("gamma", 0.99)
        self.env = self.subject.make_env(args.num_envs, self.motion, self.mdp, run_args=a_, eval_mode=True)
        self.N, self.T, self.H = args.num_envs, args.steps, a_["horizon"]
        self.ids = torch.arange(self.N, device=self.dev)
        self.std_ref = torch.maximum(self.wm.out_norm.std(), self.floor)  # 不同世界模型的归一化误差都用检查点的尺度，才可比

    # ---- 采集 ----
    def collect(self):
        """从够长的片段中途开始跟踪，到点施加冲量，失败后继续推进，记录连续转移。"""
        a, env, mdp = self.args, self.env, self.mdp
        need = self.T + self.H + 2
        env.reset(self.ids, pick_frames(self.motion.clip_start, self.motion.clip_len, need, self.N))
        self.pushes = sample_pushes(self.N, a.push_frac, (a.push_speed_min, a.push_speed_max), a.push_spin,
                                    (a.push_after_min, a.push_after_max), self.dev)
        rec = {"s": torch.empty(self.T, self.N, self.subject.STATE_DIM, device=self.dev),
               "s1": torch.empty(self.T, self.N, self.subject.STATE_DIM, device=self.dev),
               "a": torch.empty(self.T, self.N, self.NUM_J, device=self.dev),
               "a_prev": torch.empty(self.T, self.N, self.NUM_J, device=self.dev),
               "xi": torch.empty(self.T, self.N, self.subject.XI_DIM, device=self.dev),
               "f": torch.empty(self.T, self.N, dtype=torch.long, device=self.dev)}
        hist = torch.zeros(self.N, self.Hh, self.OBS_DIM, device=self.dev)
        with torch.no_grad():
            for t in range(self.T):
                due = (self.pushes["step"] == t).nonzero().flatten()
                if len(due):
                    self.apply_push(due, self.pushes["dv"][due], self.pushes["dw"][due])
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
                hist = torch.where((env.ep_step == 0)[:, None, None], o.unsqueeze(1).expand_as(hist), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))
                action = self.pi(hist, self.motion.goal(env.f, self.K)) + a.action_noise * torch.randn(self.N, self.NUM_J, device=self.dev)
                out = env.step(torch.clamp(action, -20, 20))
                for key in rec:
                    rec[key][t] = out[key]
        # 仿真偶尔发散时整条轨迹弃用；其余环境的转移连续，窗口内不会跨重置。
        good = torch.isfinite(rec["s"]).all(-1).all(0) & torch.isfinite(rec["s1"]).all(-1).all(0)
        self.dropped = int((~good).sum())
        perm = good.nonzero().flatten()
        perm = perm[torch.randperm(len(perm), device=self.dev)]
        cut = max(1, len(perm) // 5)
        self.hold, self.train = perm[:cut], perm[cut:]
        self.rec = rec
        self.masks = self.label_all()

    def apply_push(self, ids, dv, dw):
        """把冲量直接加到根部速度上，并确认写入的坐标系与env.state()读出的一致。"""
        _, _, v, w, _, _ = self.env.state()
        v1, w1 = v[ids] + dv, w[ids] + dw
        self.env.robot.write_root_link_velocity_to_sim(torch.cat((v1, w1), -1), env_ids=ids)
        _, _, v2, w2, _, _ = self.env.state()
        if (v2[ids] - v1).abs().max() > 1e-3 or (w2[ids] - w1).abs().max() > 1e-3:
            raise RuntimeError("root velocity write does not round-trip through env.state()")

    def label_all(self) -> dict:
        """每个记录状态的类别掩码，形状(T, N)。"""
        S = self.unflatten(self.rec["s"])
        f = self.rec["f"]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        step = self.pushes["step"]
        perturbed = (step >= 0) & (torch.arange(self.T, device=self.dev).unsqueeze(-1) >= step)
        return classify(S[0][..., 2], S[2][..., 2], (rp[..., 2] - S[0][..., 2]).abs(), (rR[..., 2, 2] - S[1][..., 2, 2]).abs(), perturbed)

    def coverage(self) -> dict:
        """各类状态占比和摔倒比例，用来判断采集是否覆盖到要测的状态。"""
        out = {"dropped_envs": self.dropped, "kept_envs": int(len(self.hold) + len(self.train))}
        kept = torch.cat((self.hold, self.train))
        out["state_fraction"] = {c: round(float(self.masks[c][:, kept].float().mean()), 4) for c in CATEGORIES}
        ever_fallen = self.masks["fallen"].any(0)
        pushed = self.pushes["step"] >= 0
        for name, group in (("pushed", pushed), ("not_pushed", ~pushed)):
            members = kept[group[kept]]
            out[f"ever_fallen_{name}"] = {"envs": int(len(members)), "fraction": round(float(ever_fallen[members].float().mean()), 4) if len(members) else None}
        speed, members = self.pushes["speed"], kept[pushed[kept]]
        out["ever_fallen_by_push_speed"] = {
            f"{lo:.1f}-{hi:.1f}": round(float(ever_fallen[m].float().mean()), 4)
            for lo, hi in zip(torch.linspace(self.args.push_speed_min, self.args.push_speed_max, 5)[:-1].tolist(),
                              torch.linspace(self.args.push_speed_min, self.args.push_speed_max, 5)[1:].tolist())
            for m in [members[(speed[members] >= lo) & (speed[members] < hi)]] if len(m)}
        return out

    def increments(self, envs) -> dict:
        """各类状态一步真实增量的均方根，与世界模型输出尺度对比：重尾增量会抬高尺度、压低正常状态的权重。"""
        S, S1 = self.unflatten(self.rec["s"][:, envs]), self.unflatten(self.rec["s1"][:, envs])
        d = self.wm.targets(S, S1)
        out = {"out_norm_std": {b: round(float(self.wm.out_norm.std()[s].median()), 5) for b, s in BLOCKS.items()}}
        for c in CATEGORIES:
            m = self.masks[c][:, envs]
            if m.any():
                out[c] = {b: round(float(d[m][:, s].pow(2).mean().sqrt()), 5) for b, s in BLOCKS.items()}
        return out

    # ---- 误差 ----
    @torch.no_grad()
    def model_errors(self, wm, envs) -> dict:
        """从各类状态出发按记录的动作推演，报告指定步的物理误差；类别按窗口起点的状态算。"""
        a, rec = self.args, self.rec
        k = max(a.report_steps)
        starts = torch.arange(0, self.T - k + 1, a.stride, device=self.dev)
        idx_t, idx_e = starts.repeat_interleave(len(envs)), envs.repeat(len(starts))
        stats, used = {j: [] for j in a.report_steps}, {j: [] for j in a.report_steps}
        for c0 in range(0, len(idx_t), a.chunk):
            t0, e0 = idx_t[c0:c0 + a.chunk], idx_e[c0:c0 + a.chunk]
            rows = t0.unsqueeze(-1) + torch.arange(k, device=self.dev)
            cols = e0.unsqueeze(-1).expand_as(rows)
            S = self.unflatten(rec["s"][t0, e0])
            for j in range(k):
                S = wm.step(S, rec["a"][rows[:, j], cols[:, j]], self.normalize_xi(rec["xi"][rows[:, j], cols[:, j]]))
                if j + 1 in stats:
                    e = wm.state_error(S, self.unflatten(rec["s1"][rows[:, j], cols[:, j]]))
                    stats[j + 1].append(torch.cat((components(e), ((e / self.std_ref) ** 2).mean(-1, keepdim=True)), -1).cpu())
                    used[j + 1].append((~crosses_push(self.pushes["step"][e0], t0, j)).cpu())
        out = {}
        for j, chunks in stats.items():
            all_stats, keep = torch.cat(chunks), torch.cat(used[j])
            out[f"step_{j}"] = {}
            for c in CATEGORIES:
                m = self.masks[c][idx_t, idx_e].cpu() & keep
                if m.any():
                    out[f"step_{j}"][c] = {name: describe(all_stats[m, i]) for i, name in enumerate(COMPONENTS)}
        return out

    # ---- 回报 ----
    def terms(self, Ss, As, Aps, Fs):
        """轨迹上的奖励、跟踪项、软存活和根部误差，形状(H, B)。"""
        H, B = len(Ss), len(Fs[0])
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        S = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f = flat(Fs)
        r, parts = self.mdp.reward(S, f, flat(As), flat(Aps), terms=True)
        keep = self.mdp.soft_survival(S, f)
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z, e_tilt = (rp[..., 2] - S[0][..., 2]).abs(), (rR[..., 2, 2] - S[1][..., 2, 2]).abs()
        return tuple(x.view(H, B) for x in (r, parts["track"], keep, e_z, e_tilt))

    def returns(self, Ss, As, Aps, Fs) -> dict:
        """按第4.3节的三个目标计算轨迹回报。"""
        c, a = self.mdp.cfg, self.args
        return objectives(*self.terms(Ss, As, Aps, Fs), self.gamma, c.term_height, c.term_tilt, a.tz_d, a.tt_d, self.CONTROL_DT)

    def replay_history(self, t, e):
        """由记录的状态重建观测历史（不加噪声），与Trainer.policy_update的做法相同。"""
        rows = t.unsqueeze(-1) - torch.arange(self.Hh - 1, -1, -1, device=self.dev)
        cols = e.unsqueeze(-1).expand_as(rows)
        r = self.rec
        return self.mdp.obs(self.unflatten(r["s"][rows, cols]), r["a_prev"][rows, cols], r["xi"][rows, cols], r["f"][rows, cols])

    def imagine_policy(self, wm, t, e, deltas=None):
        """从记录的状态闭环想象H步。deltas加在每步动作上，用来取回报对动作的梯度。"""
        r = self.rec
        S, xi, f, ap = self.unflatten(r["s"][t, e]), r["xi"][t, e], r["f"][t, e], r["a_prev"][t, e]
        xi_n, hist = self.normalize_xi(xi), self.replay_history(t, e)
        Ss, As, Aps, Fs = [], [], [], []
        for h in range(self.H):
            a = self.pi(hist, self.motion.goal(f, self.K))
            if deltas is not None:
                a = a + deltas[h]
            Aps.append(ap)
            S = wm.step(S, a, xi_n)
            f = f + 1
            hist = torch.cat((hist[:, 1:], self.mdp.obs(S, a, xi, f).unsqueeze(1)), 1)
            Ss.append(S), As.append(a), Fs.append(f)
            ap = a
        return Ss, As, Aps, Fs

    def imagine_actions(self, wm, S, ap, xi_n, f, actions):
        """按给定动作序列开环想象。"""
        Ss, As, Aps, Fs = [], [], [], []
        for a in actions:
            Aps.append(ap)
            S = wm.step(S, a, xi_n)
            f = f + 1
            Ss.append(S), As.append(a), Fs.append(f)
            ap = a
        return Ss, As, Aps, Fs

    # ---- 梯度 ----
    def gradients(self, wm, envs) -> dict:
        """回报对每步动作的梯度范数。倒地时跟踪核和软存活接近0，梯度是否还在正是这里要看的。"""
        a = self.args
        masks = {c: self.masks[c][self.Hh:self.T - self.H, envs] for c in PARTITION}
        t, e, label = pick_starts(masks, a.grad_starts)
        t, e = t + self.Hh, envs[e]
        deltas = [torch.zeros(len(t), self.NUM_J, device=self.dev, requires_grad=True) for _ in range(self.H)]
        J = self.returns(*self.imagine_policy(wm, t, e, deltas))
        out = {}
        for name in OBJECTIVES:
            g = torch.stack(torch.autograd.grad(J[name].sum(), deltas, retain_graph=True)).norm(dim=-1)  # (H, B)
            total = g.pow(2).sum(0).sqrt()
            out[name] = {}
            for i, c in enumerate(PARTITION):
                m = label == i
                if m.any():
                    out[name][c] = {"J": describe(J[name][m]), "grad_first": describe(g[0, m]), "grad_mid": describe(g[self.H // 2, m]),
                                    "grad_last": describe(g[-1, m]), "grad_total": describe(total[m]),
                                    "vanished_frac": round(float((total[m] < a.vanish).float().mean()), 4)}
        return out

    # ---- 物理检验 ----
    def restore(self, s, f, a_prev):
        """把物理状态写回记录值。接触和warm-start被清除，各对照组条件相同。"""
        env = self.env
        p, R, v, w, q, qd = self.unflatten(s)
        env.sim.reset(self.ids)
        env.scene.reset(self.ids)
        env.robot.write_root_link_pose_to_sim(torch.cat((p, self.mat_to_quat(R)), -1), env_ids=self.ids)
        env.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=self.ids)
        env.robot.write_joint_state_to_sim(q, qd, env_ids=self.ids)
        env.robot.set_joint_position_target(q, env_ids=self.ids)
        env.f[:], env.a_prev[:], env.ep_step[:] = f, a_prev, 1
        env.refresh()
        back = env.state()
        worst = max(float((x - y).abs().max()) for x, y in zip(back, (p, R, v, w, q, qd)))
        if worst > 1e-3:
            raise RuntimeError(f"state restore does not round-trip (max error {worst:.2e})")

    @torch.no_grad()
    def run_physical(self, s, f, a_prev, actions) -> dict:
        """从记录状态开环执行动作序列，返回物理回报和末端的高度、倾斜误差。"""
        self.restore(s, f, a_prev)
        Ss, Aps, Fs = [], [], []
        ap = a_prev
        for a in actions:
            out = self.env.step(a)
            Ss.append(self.unflatten(out["s1"])), Aps.append(ap), Fs.append(out["f"] + 1)
            ap = a
        terms = self.terms(Ss, list(actions), Aps, Fs)
        c, o = self.mdp.cfg, self.args
        return objectives(*terms, self.gamma, c.term_height, c.term_tilt, o.tz_d, o.tt_d, self.CONTROL_DT) | {"e_z_end": terms[3][-1], "e_tilt_end": terms[4][-1]}

    def physical_check(self, wm, envs) -> dict:
        """第4.1节的检验：沿模型梯度优化动作，到物理里执行，看回报是否真的改善、是否好过同幅度随机扰动。"""
        a, r = self.args, self.rec
        masks = {c: self.masks[c][self.Hh:self.T - self.H, envs] for c in PARTITION}
        t, e, label = pick_starts(masks, self.N)
        t, e = t + self.Hh, envs[e]
        s, f, ap, xi = r["s"][t, e], r["f"][t, e], r["a_prev"][t, e], r["xi"][t, e]
        S0, xi_n = self.unflatten(s), self.normalize_xi(xi)
        with torch.no_grad():
            base = torch.stack(self.imagine_policy(wm, t, e)[1])  # (H, N, 29)
        out = {}
        physical_base = self.run_physical(s, f, ap, list(base))
        for name in a.physical_objectives:
            actions = base.clone().requires_grad_(True)
            opt = torch.optim.Adam([actions], lr=a.opt_lr)
            for _ in range(a.opt_steps):
                loss = -self.returns(*self.imagine_actions(wm, S0, ap, xi_n, f, list(actions)))[name].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    actions.clamp_(-20, 20)
            best = actions.detach()
            with torch.no_grad():
                model_base = self.returns(*self.imagine_actions(wm, S0, ap, xi_n, f, list(base)))[name]
                model_best = self.returns(*self.imagine_actions(wm, S0, ap, xi_n, f, list(best)))[name]
            change = (best - base).transpose(0, 1).flatten(1).norm(dim=-1)  # 每个环境整段动作的改变量
            noise = torch.randn_like(base)
            noise_size = noise.transpose(0, 1).flatten(1).norm(dim=-1)
            randomized = base + noise * (change / noise_size.clamp_min(1e-9)).view(1, -1, 1)
            run_best, run_rand = self.run_physical(s, f, ap, list(best)), self.run_physical(s, f, ap, list(randomized))
            phys_best, phys_rand = run_best[name], run_rand[name]
            out[name] = {}
            for i, c in enumerate(PARTITION):
                m = label == i
                if not m.any():
                    continue
                d_model, d_best, d_rand = (model_best - model_base)[m], (phys_best - physical_base[name])[m], (phys_rand - physical_base[name])[m]
                both = torch.stack((d_model, d_best))
                out[name][c] = {"n": int(m.sum()), "J_phys_base": describe(physical_base[name][m]), "J_model_base": describe(model_base[m]),
                                "dJ_model": describe(d_model), "dJ_phys_opt": describe(d_best), "dJ_phys_random": describe(d_rand),
                                "dTask_phys_opt": describe((run_best["recover_task"] - physical_base["recover_task"])[m]),
                                "dTask_phys_random": describe((run_rand["recover_task"] - physical_base["recover_task"])[m]),
                                "e_z_end_base": describe(physical_base["e_z_end"][m]), "d_e_z_end_opt": describe((run_best["e_z_end"] - physical_base["e_z_end"])[m]),
                                "e_tilt_end_base": describe(physical_base["e_tilt_end"][m]),
                                "d_e_tilt_end_opt": describe((run_best["e_tilt_end"] - physical_base["e_tilt_end"])[m]),
                                "phys_improved_frac": round(float((d_best > 0).float().mean()), 4),
                                "beats_random_frac": round(float((d_best > d_rand).float().mean()), 4),
                                "corr_model_phys": round(float(torch.corrcoef(both)[0, 1]), 4) if int(m.sum()) > 2 else None,
                                "action_change_rms": round(float((change[m] ** 2 / (self.H * self.NUM_J)).mean().sqrt()), 5)}
        return out

    # ---- 微调 ----
    def finetune(self, refit: bool):
        """用推倒数据微调世界模型，损失与Trainer._wm_err相同。held-out环境不参与，训练只看这批数据。"""
        a, r = self.args, self.rec
        wm = copy.deepcopy(self.wm)
        if refit:
            with torch.no_grad():
                s, s1 = r["s"][:, self.train].flatten(0, 1), r["s1"][:, self.train].flatten(0, 1)
                S, S1 = self.unflatten(s), self.unflatten(s1)
                act, xi = r["a"][:, self.train].flatten(0, 1), r["xi"][:, self.train].flatten(0, 1)
                refit_norm(wm.in_norm, wm.inputs(S, act, self.normalize_xi(xi)))
                refit_norm(wm.out_norm, wm.targets(S, S1))
        wm.requires_grad_(True)
        opt = torch.optim.Adam(wm.parameters(), lr=a.wm_lr)
        k, log = a.wm_window, []
        std = torch.maximum(wm.out_norm.std(), self.floor).detach()
        for step in range(a.finetune_steps):
            t0 = torch.randint(0, self.T - k + 1, (a.wm_batch,), device=self.dev)
            e0 = self.train[torch.randint(len(self.train), (a.wm_batch,), device=self.dev)]
            rows = t0.unsqueeze(-1) + torch.arange(k, device=self.dev)
            cols = e0.unsqueeze(-1).expand_as(rows)
            S = self.unflatten(r["s"][t0, e0])
            preds = []
            for j in range(k):
                S = wm.step(S, r["a"][rows[:, j], cols[:, j]], self.normalize_xi(r["xi"][rows[:, j], cols[:, j]]))
                preds.append(S)
            S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))
            # 与Trainer._wm_err相同：无效转移不计损失，分母固定为窗口数
            valid = ~crosses_push(self.pushes["step"][e0].unsqueeze(-1), t0.unsqueeze(-1), torch.arange(k, device=self.dev))
            err = ((wm.state_error(S_hat, self.unflatten(r["s1"][rows, cols])) / std) ** 2).mean(-1)
            loss = (err * valid).sum() / a.wm_batch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt.step()
            if step % max(1, a.finetune_steps // 5) == 0 or step == a.finetune_steps - 1:
                log.append({"step": step, "loss_per_step": round(float(loss.detach()) / k, 5)})
                print(f"  finetune step {step}: {log[-1]['loss_per_step']}", flush=True)
        return wm.requires_grad_(False).eval(), log


def compact(summary: dict) -> str:
    """把摘要压成几张便于阅读的表。"""
    lines = [f"coverage: {json.dumps(summary['coverage'], ensure_ascii=False)}"]
    for arm, res in summary["arms"].items():
        lines.append(f"\n=== {arm}")
        for step, by_cat in res["errors"].items():
            for c, stat in by_cat.items():
                lines.append(f"  err {step:8s} {c:17s} n={stat['pos_m']['n']:6d} " + " ".join(
                    f"{name}={stat[name].get('p50', float('nan')):.3f}" for name in ("pos_m", "rot_rad", "q_rms_rad", "qd_rms_radps", "norm_mse")))
        for name, by_cat in res["gradients"].items():
            for c, stat in by_cat.items():
                lines.append(f"  grad {name:12s} {c:17s} J={stat['J'].get('mean', float('nan')):+.3f} first_p50={stat['grad_first'].get('p50', float('nan')):.2e} "
                             f"total_p50={stat['grad_total'].get('p50', float('nan')):.2e} vanished={stat['vanished_frac']:.3f}")
        for name, by_cat in res["physical"].items():
            for c, stat in by_cat.items():
                lines.append(f"  phys {name:12s} {c:17s} n={stat['n']:4d} dJ_model={stat['dJ_model'].get('mean', float('nan')):+.4f} "
                             f"dJ_opt={stat['dJ_phys_opt'].get('mean', float('nan')):+.4f} dJ_rand={stat['dJ_phys_random'].get('mean', float('nan')):+.4f} "
                             f"dTask_opt={stat['dTask_phys_opt'].get('mean', float('nan')):+.4f} dTask_rand={stat['dTask_phys_random'].get('mean', float('nan')):+.4f} "
                             f"d_e_z={stat['d_e_z_end_opt'].get('mean', float('nan')):+.4f} d_e_tilt={stat['d_e_tilt_end_opt'].get('mean', float('nan')):+.4f} "
                             f"improved={stat['phys_improved_frac']:.2f} beats_rand={stat['beats_random_frac']:.2f} corr={stat['corr_model_phys']}")
    return "\n".join(lines)


def parse_args(argv=None):
    """命令行参数；推力范围和d_t温度都是待检验的候选，写进摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
    parser.add_argument("--ckpt", required=True, help="checkpoint with wm, pi and args")
    parser.add_argument("--motion_source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=450, help="recorded control steps per environment")
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--push_frac", type=float, default=0.75, help="fraction of environments that get pushed; the rest are the control")
    parser.add_argument("--push_speed_min", type=float, default=1.0, help="horizontal root speed change (m/s)")
    parser.add_argument("--push_speed_max", type=float, default=5.0)
    parser.add_argument("--push_spin", type=float, default=3.0, help="half-width of the root angular velocity change per axis (rad/s)")
    parser.add_argument("--push_after_min", type=int, default=50)
    parser.add_argument("--push_after_max", type=int, default=150)
    parser.add_argument("--report_steps", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--tz_d", type=float, default=0.05, help="height temperature of the candidate d_t")
    parser.add_argument("--tt_d", type=float, default=0.1, help="tilt temperature of the candidate d_t")
    parser.add_argument("--grad_starts", type=int, default=1024)
    parser.add_argument("--vanish", type=float, default=1e-6, help="total action gradient below this counts as vanished")
    parser.add_argument("--physical_objectives", nargs="+", default=["recover", "recover_task", "track_keep"], choices=OBJECTIVES)
    parser.add_argument("--opt_steps", type=int, default=20)
    parser.add_argument("--opt_lr", type=float, default=0.05)
    parser.add_argument("--finetune_steps", type=int, default=3000, help="0 skips the fine-tuned world models")
    parser.add_argument("--finetune_arms", nargs="+", default=["keep_norm", "refit_norm"], choices=["keep_norm", "refit_norm"])
    parser.add_argument("--wm_window", type=int, default=8)
    parser.add_argument("--wm_batch", type=int, default=2048)
    parser.add_argument("--wm_lr", type=float, default=2e-4)
    return parser.parse_args(argv)


def main():
    """采集一次，对检查点世界模型和微调后的世界模型分别做三项检查，写summary.json。"""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = Probe(args)
    probe.collect()
    summary = {"ckpt": args.ckpt, "iteration": probe.ck.get("iteration"), "subject": probe.subject.provenance(), "args": vars(args),
               "note": "existing policy cannot stand up: real stand-up and reconnection states are not covered; full-library checkpoint, feasibility only",
               "coverage": probe.coverage(), "increments_holdout": probe.increments(probe.hold), "arms": {}}
    print(json.dumps(summary["coverage"], ensure_ascii=False), flush=True)
    arms = {"checkpoint": (probe.wm, [])}
    if args.finetune_steps:
        for arm in args.finetune_arms:
            print(f"finetuning {arm}", flush=True)
            arms[f"finetuned_{arm}"] = probe.finetune(arm == "refit_norm")
    for name, (wm, log) in arms.items():
        print(f"checking {name}", flush=True)
        # 同样多的训练环境：与留出环境对比，分清剩余误差是欠拟合还是数据不够
        summary["arms"][name] = {"finetune_log": log, "errors": probe.model_errors(wm, probe.hold),
                                 "errors_train_envs": probe.model_errors(wm, probe.train[:len(probe.hold)]),
                                 "gradients": probe.gradients(wm, probe.hold), "physical": probe.physical_check(wm, probe.hold)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    text = compact(summary)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
