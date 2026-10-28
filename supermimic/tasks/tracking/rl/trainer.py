# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""SuperMimic训练器（算法A.1）：采样 → 拟合世界模型 → 经冻结的世界模型想象推演、BPTT更新策略 → critic更新。

supermimic/scripts/train.py只负责命令行、进程与运行目录；本模块持有模型、优化器、回放缓冲与仿真环境，
各更新步骤可被检查脚本与测试直接调用。随机数的消耗顺序与拆分前的train.py相同：
模型初始化 →（多卡时按rank重新设种子）→ 仿真环境 → 预热采样。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
import time

import torch
import torch.nn as nn

from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, NUM_J, OBS_DIM, normalize_xi, unflatten_state
from supermimic.tasks.tracking.motion import GOAL_FRAME_DIM
from supermimic.tasks.tracking.rl.buffer import ReplayBuffer
from supermimic.tasks.tracking.rl.networks import Critic, Policy, WorldModel
from supermimic.utils.checkpoints import rotate_states
from supermimic.utils.cuda_graph import GraphedCall
from supermimic.utils.distributed import average_gradients

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2)))  # fmt: skip

# critic diagnostics of one update (diag_log.csv): means, maxima, and bin-score statistics added when logging
DIAG_MEAN = ("v_mean", "v_end_mean", "v_capped_frac", "critic_bias", "z_gt5_frac", "v_input_gradnorm", "critic_gradnorm")
DIAG_MAX = ("v_max", "z_max")
DIAG_BINS = ("bin_score_mean", "bin_score_min", "nonfinite_scores")  # visited, scorable bins; scores skipped since the last row
TRAIN_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm",
                 "imag_survival", "sim_rew", "ep_len", "ep_ret_norm", "term_frac", "it_per_s", "lr")
SAMPLING_COLUMNS = ("iteration", "samp_entropy", "samp_top1", "bins_visited", "reset_tail_frac")
DIAG_COLUMNS = ("iteration", *DIAG_MEAN, *DIAG_MAX, *DIAG_BINS)
# 日志表：组名（TensorBoard标签前缀）→（CSV文件名, 列名）
LOG_TABLES = {"train": ("train_log.csv", TRAIN_COLUMNS), "sampling": ("sampling_log.csv", SAMPLING_COLUMNS),
              "diag": ("diag_log.csv", DIAG_COLUMNS)}


@dataclass(frozen=True)
class Layout:
    """本rank的环境数与批量；--batch_scope决定命令行数值是全局总数（global）还是每卡数值（per_gpu）。"""

    scope: int
    global_envs: int
    local_envs: int
    local_wm_batch: int
    local_pi_batch: int
    wm_step_batch: int  # 每次世界模型优化器步的窗口数
    wm_steps: int  # 每次迭代的世界模型优化器步数
    base_lr: float


def make_layout(args, parallel) -> Layout:
    scope = parallel.world_size if args.batch_scope == "per_gpu" else 1
    local_wm_batch = parallel.local_size(args.wm_batch * scope, "wm_batch")
    local_pi_batch = parallel.local_size(args.pi_batch * scope, "pi_batch")
    layout = Layout(
        scope=scope,
        global_envs=args.num_envs * scope,
        local_envs=parallel.local_size(args.num_envs * scope, "num_envs"),
        local_wm_batch=local_wm_batch,
        local_pi_batch=local_pi_batch,
        wm_step_batch=local_wm_batch * (args.wm_batches if args.wm_update == "single" else 1),
        wm_steps=1 if args.wm_update == "single" else args.wm_batches,
        base_lr=args.lr * (math.sqrt(parallel.world_size) if args.lr_scaling == "sqrt" else 1.0),
    )
    if args.buffer_steps < args.wm_window or args.warmup_steps < args.wm_window:
        raise ValueError("buffer_steps and warmup_steps must be at least wm_window")
    if not args.no_critic and local_pi_batch * args.horizon < args.critic_minibatches:
        raise ValueError("local imagined batch is smaller than critic_minibatches")
    return layout


def split_sizes(n: int, parts: int) -> list[int]:
    """Chunk sizes of torch.tensor_split(range(n), parts)."""
    return [n // parts + int(i < n % parts) for i in range(parts)]


def lr_at(it: int, base_lr: float, decay_iters: int) -> float:
    """余弦衰减到base_lr的3%，之后保持。"""
    if it >= decay_iters:
        return base_lr * 0.03
    return base_lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / decay_iters)))


def survival(keep: torch.Tensor, unrolled: bool) -> torch.Tensor:
    """Soft survival weights c_{t+1} = prod_{i<=t} (1 - d_i) along the imagined horizon (dim 0).

    Takes the per-step survival probabilities 1 - d_i themselves (MDP.soft_survival), not d_i, so a
    near-certain termination is not rounded to exactly 0 by 1 - (1 - p).
    cumprod's backward branches on zeros in its input on the host, which a captured graph cannot do,
    so graphs (unrolled) use the same running product written out step by step.
    """
    if not unrolled:
        return torch.cumprod(keep, 0)
    out, c = [], torch.ones_like(keep[0])
    for t in range(keep.shape[0]):
        c = c * keep[t]
        out.append(c)
    return torch.stack(out)


def gae_targets(V: torch.Tensor, rews: torch.Tensor, keeps: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """GAE(lambda)目标（式3）：V为(H+1, B)的估值，rews与keeps为(H, B)的奖励与存活概率，返回(H, B)。"""
    R = V[-1]
    targets = []
    for t in range(rews.shape[0] - 1, -1, -1):
        R = rews[t] + gamma * keeps[t] * ((1 - lam) * V[t + 1] + lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


class Trainer:
    """一个rank上的训练状态与各更新步骤。构造后调用collect(warmup_steps)预热，再逐次调用iteration(it)。"""

    def __init__(self, args, parallel, layout: Layout, kin, motion, resume: dict | None = None):
        self.args, self.parallel, self.layout, self.motion = args, parallel, layout, motion
        dev = self.device = parallel.device
        self.mdp = MDP(kin, motion, kin.default_q)
        self.K, self.Hh, self.HZ = args.goal_frames, args.hist, args.horizon
        goal_dim = GOAL_FRAME_DIM * self.K

        self.wm = WorldModel().to(dev)
        self.pi = Policy(OBS_DIM, goal_dim, hist=self.Hh).to(dev)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + goal_dim).to(dev)
        cuda = torch.device(dev).type == "cuda"
        self.fused_adam = args.fused_adam == "on" and cuda
        self.profile = bool(int(os.environ.get("SM_PROFILE", "0")))
        self.use_graph = args.cuda_graph == "on" or (args.cuda_graph == "auto" and cuda and not parallel.enabled and not self.profile)
        if self.use_graph and not cuda:
            raise ValueError("--cuda_graph on needs CUDA")
        # Inside graphs the gradient all-reduce uses the fixed global batch sizes (no host round trip).
        scope = layout.scope
        self.graph_wm_total = args.wm_batch * scope * (args.wm_batches if args.wm_update == "single" else 1) if self.use_graph else None
        self.graph_pi_total = args.pi_batch * scope if self.use_graph else None
        self.graph_critic = self.use_graph and args.cuda_graph_critic == "on" and not args.no_critic
        # global sample count of each critic minibatch: every rank splits its HZ * local_pi_batch rows the same way
        world = parallel.world_size
        self.graph_critic_totals = [sum(col) for col in zip(*(
            split_sizes(args.horizon * rank_batch, args.critic_minibatches)
            for rank_batch in (args.pi_batch * scope // world + int(r < args.pi_batch * scope % world) for r in range(world))))
        ] if self.graph_critic else None

        self.opt_wm = self._adam(self.wm.parameters(), self.use_graph)
        self.opt_pi = self._adam(self.pi.parameters(), self.use_graph, betas=(0.9, 0.95))
        self.opt_v = self._adam(self.critic.parameters(), self.graph_critic)
        self.start_it, self.wall_offset, self.sim_steps = 0, 0.0, 0
        if resume:
            self._resume(resume)
        for model in (self.wm, self.pi, self.critic):
            parallel.broadcast_module(model)
        if parallel.enabled:
            torch.manual_seed(args.seed + parallel.rank)

        self.err_floor = ERR_FLOOR.to(dev)
        # every per-step reward is at most MAX_STEP_REWARD (the tracking kernels are <= 1, the penalties <= 0), so no
        # discounted return, and no correct critic value, exceeds MAX_STEP_REWARD / (1 - gamma) (10 at gamma 0.99)
        self.value_cap = MAX_STEP_REWARD / (1 - args.gamma) if args.critic_cap == "on" else math.inf
        # 可编译的内核是实例属性：检查脚本可在同一Trainer上切换eager与编译版本
        self.use_compile = args.compile == "on" or (args.compile == "auto" and importlib.util.find_spec("triton") is not None)
        maybe_compile = torch.compile if self.use_compile else (lambda fn: fn)
        self.wm_rollout = maybe_compile(self._wm_rollout)
        self.wm_err = maybe_compile(self._wm_err)
        self.imag_step = maybe_compile(self._imag_step)
        self.reward_term = maybe_compile(self._reward_term)
        self.critic_input_c = maybe_compile(self._critic_input)
        print(f"torch.compile: {'on' if self.use_compile else 'off'}", flush=True)

        # Non-finite world-model/critic/policy gradients are flagged on the GPU and checked once per iteration:
        # a host sync after each of the 10 world-model and 20 critic updates stalls the GPU queue.
        # The check runs before logging and checkpointing, so a corrupted step aborts without being saved.
        self.nonfinite = torch.zeros(3, dtype=torch.bool, device=dev)
        self.wm_step = self.graphed(self._wm_step)
        self.policy_step = self.graphed(self._policy_step)
        self.critic_step = self.graphed(self._critic_step, enabled=self.graph_critic)

        self.env = TrackingSim(layout.local_envs, motion, self.mdp, dev, gamma=args.gamma, parallel=parallel,
                               graph_reward=self.use_graph and args.cuda_graph_env == "on",
                               bin_clip_end=args.bin_clip_end, bin_prior=args.bin_prior, bin_skip_steps=args.bin_skip_steps,
                               reset_margin=args.reset_margin)
        self.buffer = ReplayBuffer(args.buffer_steps, layout.local_envs, dev)
        self.obs_hist = torch.zeros(layout.local_envs, self.Hh, OBS_DIM, device=dev)
        self.collect_action = self.graphed(self._collect_action, enabled=self.use_graph and args.cuda_graph_collect == "on")

        self.step_acc = 0.0
        self.stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        self.agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        self.diag_agg = {"mean": [], "max": []}
        self.t_start = self.t_log = None

    # ---------------- 构造与恢复 ----------------
    def _adam(self, params, captured: bool, **kw):
        """Graph-captured optimizers keep their step count and learning rate on the device."""
        lr = self.layout.base_lr
        if captured:
            return torch.optim.Adam(params, lr=torch.tensor(lr, device=self.device), fused=self.fused_adam, capturable=True, **kw)
        return torch.optim.Adam(params, lr=lr, fused=self.fused_adam, **kw)

    def _resume(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])
        # load_state_dict restores the checkpoint's own flags, lr type and step placement; match this run's mode.
        # Fused and capturable kernels read the step count from device memory, the others from the host.
        for opt, captured in ((self.opt_wm, self.use_graph), (self.opt_pi, self.use_graph), (self.opt_v, self.graph_critic)):
            on_device = self.fused_adam or captured
            for group in opt.param_groups:
                group["fused"], group["foreach"], group["capturable"] = self.fused_adam, None, captured
                group["lr"] = torch.tensor(float(group["lr"]), device=self.device) if captured else float(group["lr"])
                for p in group["params"]:
                    state = opt.state.get(p)
                    if state and "step" in state:
                        state["step"] = torch.as_tensor(state["step"], dtype=torch.float32,
                                                        device=p.device if on_device else "cpu").clone()
        self.start_it, self.wall_offset, self.sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]

    def graphed(self, fn, eager_calls: int = 3, enabled: bool | None = None):
        """CUDA-graph replay of a training step (supermimic.utils.cuda_graph); enabled defaults to --cuda_graph."""
        return GraphedCall(fn, eager_calls, self.use_graph if enabled is None else enabled, thread_local=self.parallel.enabled)

    # ---------------- 可编译内核 ----------------
    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g):
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.K)

    def _reward_term(self, S, f, a, a_prev):
        return self.mdp.reward(S, f, a, a_prev), self.mdp.soft_survival(S, f)

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def cap_value(self, v):
        return v.clamp(max=self.value_cap)

    # ---------------- 世界模型 ----------------
    def _wm_step(self, s, a, xi, s1, mask):
        S_hat = self.wm_rollout(s, a, xi)
        loss, e_last = self.wm_err(S_hat, s1, mask)
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self.wm, self.layout.wm_step_batch, total_count=self.graph_wm_total)
        grad_norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.nonfinite[0].logical_or_(~torch.isfinite(grad_norm))
        self.opt_wm.step()
        m = mask[:, -1]
        error = self.parallel.reduce(torch.stack(((e_last.detach() * m).sum(), m.sum())))
        return loss.detach() / self.args.wm_window, error[0] / error[1].clamp(min=1)

    def wm_update(self):
        w = self.buffer.sample_windows(self.layout.wm_step_batch, self.args.wm_window)
        return self.wm_step(w["s"], w["a"], w["xi"], w["s1"], w["mask"])

    # ---------------- 策略与critic ----------------
    def sample_valid_starts(self, B: int):
        starts, hists, n = [], [], 0
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 256:
                raise RuntimeError("replay has too few starts with a complete imagination horizon; collect more transitions or inspect short clips")
            s, h = self.buffer.sample_starts(2 * B, self.Hh)
            f = s["f"]
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def _policy_step(self, s_flat, xi, f, a_prev, o_hist):
        """Imagine HZ steps through the frozen world model, backpropagate the return and step the policy."""
        args, HZ, dev = self.args, self.HZ, self.device
        if self.profile:
            torch.cuda.synchronize(); tq0 = time.time()
        S = unflatten_state(s_flat)
        xi_n = normalize_xi(xi)
        B = self.layout.local_pi_batch
        g0 = self.motion.goal(f, self.K)
        g = g0
        S0, f0 = S, f
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, keep = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        r, keep = r.view(HZ, B), keep.view(HZ, B)
        alive = survival(keep, self.use_graph)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, keeps = r.detach(), keep.detach()
        if not args.no_critic:
            J = J + (args.gamma**HZ) * c * self.cap_value(self.critic(self._critic_input(S, f, xi_n, g)))
        loss = -J.mean()
        if self.profile:
            torch.cuda.synchronize(); tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self.pi, self.layout.local_pi_batch, total_count=self.graph_pi_total)
        if self.profile:
            torch.cuda.synchronize(); tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_pi_grad = torch.isfinite(gn)
        self.nonfinite[2].logical_or_(~finite_pi_grad)
        # A captured graph cannot branch on the host, so it always steps; a non-finite gradient still aborts
        # the run at the end of this iteration, before anything is logged or saved.
        if self.use_graph or bool(finite_pi_grad):
            self.opt_pi.step()
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
        tq = (tq0, tq1, tq2) if self.profile else ()
        return (loss.detach(), gn.detach(), (r_sum / HZ / 0.1).mean(), c.mean().detach(), rews, keeps, xi_n,
                S_c, f_c, g_c) + tq

    def _critic_step(self, rews, keeps, xi_n, S_c, f_c, g_c):
        """Fit the critic to GAE(lambda) targets over the imagined transitions (Eq. 3).

        Besides the loss it returns the diagnostics DIAG_MEAN and DIAG_MAX measured on this update's imagined states:
        the raw critic values (mean, at the horizon end, maximum, share above the cap), their mean offset from the
        GAE targets, how far the critic's normalized inputs stray (|z| > 5 share, maximum), the critic's input
        gradient at the end states (what the policy objective differentiates through) and its parameter gradient norm.
        """
        args, HZ, critic = self.args, self.HZ, self.critic
        B = self.layout.local_pi_batch
        with torch.no_grad():
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            raw = critic(X)
            V = self.cap_value(raw)
            targets = gae_targets(V, rews, keeps, args.gamma, args.lam)  # (H, B)
            z = critic.norm(X).abs()
            diag_mean = torch.stack((raw.mean(), raw[-1].mean(), (raw > self.value_cap).float().mean(), (raw[:-1] - targets).mean(),
                                     (z > 5).float().mean()))
            diag_max = torch.stack((raw.max(), z.max()))
        x_end = X[-1].detach().clone().requires_grad_(True)
        value_grad = torch.autograd.grad(critic(x_end).sum(), x_end)[0].norm(dim=-1).mean()  # before this update's steps
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        v_loss = torch.zeros(())
        grad_sum = torch.zeros((), device=self.device)
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i, idx in enumerate(torch.tensor_split(perm, args.critic_minibatches)):
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                average_gradients(critic, len(idx), total_count=self.graph_critic_totals[i] if self.graph_critic else None)
                grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                self.nonfinite[1].logical_or_(~torch.isfinite(grad_norm))
                grad_sum = grad_sum + grad_norm.detach()
                self.opt_v.step()
        diag_mean = torch.cat((diag_mean, torch.stack((value_grad, grad_sum / (args.critic_epochs * args.critic_minibatches)))))
        return v_loss.detach(), diag_mean, diag_max

    def policy_update(self):
        with self.parallel.coordinated("policy replay sampling"):
            s, h = self.sample_valid_starts(self.layout.local_pi_batch)
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = self.mdp.add_noise(o_hist)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        loss, gn, imag_r, surv, rews, keeps, xi_n, S_c, f_c, g_c, *tq = self.policy_step(s["s"], s["xi"], s["f"], s["a_prev"], o_hist)
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        if self.args.no_critic:
            v_loss, diag_mean, diag_max = torch.zeros(()), torch.zeros(len(DIAG_MEAN)), torch.zeros(len(DIAG_MAX))
        else:
            v_loss, diag_mean, diag_max = self.critic_step(rews, keeps, xi_n, S_c, f_c, g_c)
        if self.profile:
            torch.cuda.synchronize()
            tq0, tq1, tq2 = tq
            print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
        return loss, v_loss, imag_r, gn, surv, diag_mean, diag_max

    # ---------------- 采样 ----------------
    @torch.no_grad()
    def _collect_action(self, S, a_prev, xi, f, ep_step, history):
        """Fixed-shape collection work; all changing environment tensors are explicit graph inputs."""
        o = self.mdp.add_noise(self.mdp.obs(S, a_prev, xi, f))
        fresh = ep_step == 0
        history = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(history),
                              torch.cat((history[:, 1:], o.unsqueeze(1)), 1))
        g = self.motion.goal(f, self.K)
        self.pi.obs_norm.update(o)
        self.pi.goal_norm.update(g)
        a = self.pi(history, g)
        a = torch.clamp(a + self.args.action_noise * torch.randn_like(a), -20, 20)
        return a, history

    @torch.no_grad()
    def store_transition(self, out):
        self.buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        self.wm.in_norm.update(self.wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        self.wm.out_norm.update(self.wm.targets(S0, S1))

    @torch.no_grad()
    def collect(self, n_steps: int):
        env, stats = self.env, self.stats
        for _ in range(n_steps):
            S = env.state()
            a, self.obs_hist = self.collect_action(S, env.a_prev, env.xi, env.f, env.ep_step, self.obs_hist)
            overlap = self.args.collect_overlap == "on"
            out = env.step(a, after_step=self.store_transition if overlap else None)
            if not overlap:
                self.store_transition(out)
            self.sim_steps += self.layout.global_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].sum())
                stats["ep_ret"].append(out["ep_ret_norm"].sum())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    # ---------------- 一次迭代 ----------------
    def set_lr(self, lr: float):
        for o in (self.opt_wm, self.opt_pi, self.opt_v):
            for gparam in o.param_groups:
                if torch.is_tensor(gparam["lr"]):
                    gparam["lr"].fill_(lr)
                else:
                    gparam["lr"] = lr

    def iteration(self, it: int) -> float:
        """采样、世界模型更新，过了--wm_only_iters后再更新策略与critic；返回本次学习率。梯度非有限时中止。"""
        args = self.args
        lr = lr_at(it, self.layout.base_lr, args.lr_decay_iters)
        self.set_lr(lr)
        self.step_acc += 1.0 / args.utd
        n = int(self.step_acc)
        self.step_acc -= n
        prof = self.profile and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        self.collect(n)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(self.layout.wm_steps):
            l, e8 = self.wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
        self.agg["wm"].append(l), self.agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv, dmean, dmax = self.policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            self.agg["pi"].append(pl), self.agg["v"].append(vl), self.agg["ir"].append(ir), self.agg["gn"].append(gn)
            self.agg["surv"].append(surv)
            self.diag_agg["mean"].append(dmean), self.diag_agg["max"].append(dmax)
        if bool(self.nonfinite.any()):
            which = [name for name, bad in zip(("world-model", "critic", "policy"), self.nonfinite.tolist()) if bad]
            raise FloatingPointError(f"non-finite globally averaged {' and '.join(which)} gradient at iteration {it + 1}")
        return lr

    # ---------------- 日志与检查点 ----------------
    def start_clock(self):
        """训练墙钟从这里起算（恢复时接上检查点的墙钟）。"""
        self.t_start = time.time() - self.wall_offset
        self.t_log = time.time()

    def log_rows(self, it: int, lr: float) -> dict:
        """自上次记日志以来的全局统计，按LOG_TABLES分组返回各行；所有rank都必须调用（含集合通信），并清空累计。"""
        args, parallel, env, motion, layout = self.args, self.parallel, self.env, self.motion, self.layout
        stats, agg, diag_agg = self.stats, self.agg, self.diag_agg
        keys = list(agg) + ["r"]
        values = [agg[k] if k in agg else stats[k] for k in keys]
        weights = [layout.local_wm_batch if k in ("wm", "wm8") else layout.local_envs if k == "r" else layout.local_pi_batch
                   for k in keys]
        totals = parallel.reduce([[sum(float(v) for v in vs) * weight, len(vs) * weight]
                                  for vs, weight in zip(values, weights)]).cpu()
        means = {key: float(pair[0] / pair[1].clamp(min=1)) for key, pair in zip(keys, totals)}
        episodes = parallel.reduce([sum(float(v) for v in stats[key]) for key in ("ep_len", "ep_ret", "term", "done")]).cpu()
        timing = parallel.reduce([time.time() - self.t_log, time.time() - self.t_start], torch.distributed.ReduceOp.MAX).cpu()
        ips = args.log_every / float(timing[0])
        self.t_log = time.time()
        wall_h = float(timing[1]) / 3600
        term_frac = float(episodes[2] / episodes[3].clamp(min=1))
        row = [it + 1, wall_h, self.sim_steps, *[means[k] for k in keys],
               float(episodes[0] / episodes[3].clamp(min=1)), float(episodes[1] / episodes[3].clamp(min=1)), term_frac, ips, lr]
        local_samp = env.sampling_stats()
        samp = parallel.reduce(local_samp).cpu()
        top1 = float(parallel.reduce(local_samp[1], torch.distributed.ReduceOp.MAX))
        samp_row = [it + 1, float(samp[0]) / math.log(max(motion.global_num_bins, 2)), top1,
                    float(samp[2]) / motion.global_num_bins, float(samp[4] / samp[3].clamp(min=1))]
        dev = self.device
        weight = torch.tensor([float(layout.local_pi_batch)], device=dev)
        d_mean = torch.stack(diag_agg["mean"]).mean(0).to(dev) if diag_agg["mean"] else torch.zeros(len(DIAG_MEAN), device=dev)
        d_max = torch.stack(diag_agg["max"]).max(0).values.to(dev) if diag_agg["max"] else torch.zeros(len(DIAG_MAX), device=dev)
        d_mean = parallel.reduce(torch.cat((d_mean * weight, weight))).cpu()
        d_max = parallel.reduce(d_max, torch.distributed.ReduceOp.MAX).cpu()
        scored = env.bin_scored & (env.bin_visits > 0)
        bin_sum = parallel.reduce(torch.stack((torch.where(scored, env.bin_G, 0.0).sum().double(), scored.sum().double()))).cpu()
        bin_min = float(parallel.reduce(torch.where(scored, env.bin_G, torch.inf).min(), torch.distributed.ReduceOp.MIN))
        diag_row = [it + 1, *(float(x) for x in d_mean[:-1] / d_mean[-1]), *(float(x) for x in d_max),
                    float(bin_sum[0] / bin_sum[1].clamp(min=1)), bin_min if math.isfinite(bin_min) else 0.0, float(samp[5])]
        self.stats = {k: [] for k in stats}
        self.agg = {k: [] for k in agg}
        self.diag_agg = {k: [] for k in diag_agg}
        return {"train": row, "sampling": samp_row, "diag": diag_row}

    def checkpoint(self, it: int) -> dict:
        """完整训练状态（--resume用）；其中pi、wm与元数据另存为评估用的ckpt_*.pt。"""
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
                "iteration": it + 1, "wall_time": time.time() - self.t_start, "sim_steps": self.sim_steps,
                "args": vars(self.args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reference": self.motion.global_description,
                "distributed": {"world_size": self.parallel.world_size, "partition_algorithm": self.motion.partition["algorithm"]}}

    def save(self, it: int, log_dir: str, keep_states: int):
        """写last.pt（先轮换旧的）与ckpt_{it+1}.pt；都先写临时文件再原子替换。只在主进程调用。"""
        ck = self.checkpoint(it)
        last = os.path.join(log_dir, "last.pt")
        rotate_states(last, keep_states)
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "reference", "distributed")}
        path = os.path.join(log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)

    def close(self):
        """释放捕获的CUDA graph（其中的NCCL操作须在销毁进程组之前释放）。"""
        for captured in (self.wm_step, self.policy_step, self.critic_step, self.collect_action, self.env._reward_termination):
            captured.close()
