# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
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

# 误差归一化各分量的下限：位置、旋转、线速度、角速度、关节位置和速度增量
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2)))  # fmt: skip

# 单次价值网络更新的诊断指标（diag_log.csv）；分箱得分统计在记录时补充
DIAG_MEAN = ("v_mean", "v_end_mean", "v_capped_frac", "critic_bias", "z_gt5_frac", "v_input_gradnorm", "critic_gradnorm")
DIAG_MAX = ("v_max", "z_max")
DIAG_BINS = ("bin_score_mean", "bin_score_min", "nonfinite_scores")  # 已访问且可评分的分箱；上次记录以来跳过的得分数
TRAIN_COLUMNS = ("iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm",
                 "imag_survival", "sim_rew", "ep_len", "ep_ret_norm", "term_frac", "it_per_s", "lr")
SAMPLING_COLUMNS = ("iteration", "samp_entropy", "samp_top1", "bins_visited", "reset_tail_frac")
DIAG_COLUMNS = ("iteration", *DIAG_MEAN, *DIAG_MAX, *DIAG_BINS)
# 终止集中在哪：每--term_log_every次迭代一行汇总（TensorBoard可见），以及这个窗口内终止最多的片段（term_bins.csv，含文字列）
TERMWIN_COLUMNS = ("iteration", "term_total", "term_clips", "top1_share", "top10_share", "resets_total", "episodes_total")
# terminations：窗口内从该片段起步而因终止结束的回合数；episodes：从该片段起步而结束的回合数；term_rate＝前者/后者；
# resets：窗口内被课程抽中的次数（集中程度）；top_bin_s：该片段内终止最多的1 s分箱的起点秒数；share：占窗口全部终止的比例
TERM_BINS_COLUMNS = ("iteration", "clip", "terminations", "episodes", "term_rate", "resets", "top_bin_s", "top_bin_terms", "share")
BIN_SECONDS = 50  # 一个重置分箱的帧数（1 s，50 Hz）
# 日志表：组名（TensorBoard标签前缀）→（CSV文件名, 列名）；term_bins用RunLog.write_lines写多行，不写TensorBoard
LOG_TABLES = {"train": ("train_log.csv", TRAIN_COLUMNS), "sampling": ("sampling_log.csv", SAMPLING_COLUMNS),
              "diag": ("diag_log.csv", DIAG_COLUMNS), "termwin": ("termwin_log.csv", TERMWIN_COLUMNS),
              "term_bins": ("term_bins.csv", TERM_BINS_COLUMNS)}
# --clean_action_frac > 0 时另写：不加动作噪声的环境组（clean）与加噪声组（noisy）的仿真指标，含义同train_log.csv的同名列。
# clean组的sim_rew低于noisy组，说明确定性策略在真实仿真中已不如加噪声时（世界模型只在加噪声动作上拟合的后果）。
CLEAN_COLUMNS = ("iteration", "sim_rew_clean", "sim_rew_noisy", "ep_len_clean", "ep_ret_clean", "term_frac_clean",
                 "ep_len_noisy", "ep_ret_noisy", "term_frac_noisy")
CLEAN_TABLE = {"clean": ("clean_log.csv", CLEAN_COLUMNS)}
# --bin_clip_cap > 0 时另写：clip_top1为单个片段（各分箱之和）最大的重置概率，hard_clip_top1_raw为削平前困难度部分（占重置的40%）
# 中最大的片段份额，capped_clips为被削到上限的片段数。都是记日志那一刻的快照。
CLIPCAP_COLUMNS = ("iteration", "clip_top1", "hard_clip_top1_raw", "capped_clips")
CLIPCAP_TABLE = {"clipcap": ("clipcap_log.csv", CLIPCAP_COLUMNS)}


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
    """根据全局或每卡批量配置和进程数量，计算当前进程的训练布局。"""
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
    """返回 torch.tensor_split(range(n), parts) 各分块的大小。"""
    return [n // parts + int(i < n % parts) for i in range(parts)]


def lr_at(it: int, base_lr: float, decay_iters: int) -> float:
    """余弦衰减到base_lr的3%，之后保持。"""
    if it >= decay_iters:
        return base_lr * 0.03
    return base_lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / decay_iters)))


def clean_action_split(frac: float, local_envs: int, action_noise: float, device) -> tuple:
    """前round(frac * local_envs)个环境不加动作噪声：返回(clean掩码, 逐环境噪声尺度(N, 1))；frac为0时两者都是None（论文路径）。
    frac>0时即使本rank取整后没有clean环境也返回掩码，保证各rank都参与clean_log的集合通信。"""
    if not 0 <= frac < 1:
        raise ValueError("clean_action_frac must be in [0, 1)")
    if frac == 0:
        return None, None
    clean = torch.arange(local_envs, device=device) < round(frac * local_envs)
    return clean, (action_noise * (~clean).float()).unsqueeze(-1)


def survival(keep: torch.Tensor, unrolled: bool) -> torch.Tensor:
    """计算想象时间轴（第 0 维）上的软存活权重 c_{t+1} = prod_{i<=t}(1 - d_i)。

    直接接收每步存活概率 MDP.soft_survival，避免近乎必然终止时因相减舍入为零。cumprod 的反向传播会在主机上根据输入中的零分支，捕获图无法执行这一操作，因此图模式逐步展开相同的累乘。"""
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
    """统一启用或禁用模块参数的梯度计算。"""
    for p in m.parameters():
        p.requires_grad_(flag)


class Trainer:
    """一个rank上的训练状态与各更新步骤。构造后调用collect(warmup_steps)预热，再逐次调用iteration(it)。"""

    def __init__(self, args, parallel, layout: Layout, kin, motion, resume: dict | None = None):
        """建立环境、回放缓冲区、模型、优化器与训练状态，并按需恢复检查点。"""
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
        # 图内梯度全归约使用固定的全局批量大小，无需与主机往返。
        scope = layout.scope
        self.graph_wm_total = args.wm_batch * scope * (args.wm_batches if args.wm_update == "single" else 1) if self.use_graph else None
        self.graph_pi_total = args.pi_batch * scope if self.use_graph else None
        self.graph_critic = self.use_graph and args.cuda_graph_critic == "on" and not args.no_critic
        # 每个价值网络小批次的全局样本数：各进程以相同方式切分 HZ * local_pi_batch 行
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
        # 先同步模型与归一化缓冲区，再按进程切换随机种子。
        # 这样各卡从同一组参数出发，但后续环境随机化和采样不会重复。
        for model in (self.wm, self.pi, self.critic):
            parallel.broadcast_module(model)
        if parallel.enabled:
            torch.manual_seed(args.seed + parallel.rank)

        self.err_floor = ERR_FLOOR.to(dev)
        # 每步奖励至多为 MAX_STEP_REWARD（跟踪核不超过 1，惩罚不大于 0），因此折扣回报和正确的
        # 价值估计均不超过 MAX_STEP_REWARD / (1 - gamma)（gamma 为 0.99 时为 10）。
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

        # 在 GPU 上标记世界模型、价值网络和策略的非有限梯度，每次迭代检查一次：
        # 若在 10 次世界模型更新和 20 次价值网络更新后分别同步主机，会阻塞 GPU 队列。
        # 检查先于日志与检查点保存，异常更新会中止且不会被保存。
        self.nonfinite = torch.zeros(3, dtype=torch.bool, device=dev)
        self.wm_step = self.graphed(self._wm_step)
        self.policy_step = self.graphed(self._policy_step)
        self.critic_step = self.graphed(self._critic_step, enabled=self.graph_critic)

        self.env = TrackingSim(layout.local_envs, motion, self.mdp, dev, gamma=args.gamma, parallel=parallel,
                               graph_reward=self.use_graph and args.cuda_graph_env == "on",
                               bin_clip_end=args.bin_clip_end, bin_prior=args.bin_prior, bin_skip_steps=args.bin_skip_steps,
                               reset_margin=args.reset_margin, bin_clip_cap=args.bin_clip_cap)
        if resume:
            self._restore_bins(resume)
        # 回放数据与观测历史属于当前仿真会话，不从检查点恢复；外层 setup 会重新预热。
        self.buffer = ReplayBuffer(args.buffer_steps, layout.local_envs, dev)
        self.obs_hist = torch.zeros(layout.local_envs, self.Hh, OBS_DIM, device=dev)
        # 可选（偏离论文）：本rank前clean_action_frac比例的环境采样时不加动作噪声。回放中只有加噪声的动作时，
        # 世界模型在确定性策略的平滑动作上缺少数据，想象推演用的正是确定性策略。为0时与论文完全相同。
        self.clean_envs, self.noise_scale = clean_action_split(args.clean_action_frac, layout.local_envs, args.action_noise, dev)
        self.collect_action = self.graphed(self._collect_action, enabled=self.use_graph and args.cuda_graph_collect == "on")

        self.step_acc = 0.0
        self.stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        if self.clean_envs is not None:
            self.stats |= {"r_clean": [], "r_noisy": [], "ep_clean": []}
        self.agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        self.diag_agg = {"mean": [], "max": []}
        self.t_start = self.t_log = None

    # ---------------- 构造与恢复 ----------------
    def _adam(self, params, captured: bool, **kw):
        """创建适合当前模式的 Adam 优化器；捕获图使用的步数和学习率保存在设备上。"""
        lr = self.layout.base_lr
        if captured:
            return torch.optim.Adam(params, lr=torch.tensor(lr, device=self.device), fused=self.fused_adam, capturable=True, **kw)
        return torch.optim.Adam(params, lr=lr, fused=self.fused_adam, **kw)

    def _resume(self, ck: dict):
        """恢复模型和优化器状态，并将优化器配置调整为当前运行模式。"""
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])
        # load_state_dict 会恢复检查点中的标志、学习率类型和步数位置；这里调整为当前运行模式。
        # fused 与 capturable 内核从设备内存读取步数，其他内核从主机读取。
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

    def _restore_bins(self, ck: dict):
        """恢复重置课程（分箱得分、访问数、概率EMA）。分片取决于数据划分，世界大小或分片形状不同就不恢复，课程从零开始。
        所有rank都要调用，且结论一致（集合通信）。"""
        sampler, parallel = self.env.sampler, self.parallel
        state = ck.get("bins")
        if state is None:
            print("checkpoint has no reset-curriculum state: bin scores start from zero", flush=True)
            return
        shards, rank = state["shards"], parallel.rank
        ok = (len(shards) == parallel.world_size and state["partition_algorithm"] == self.motion.partition["algorithm"]
              and all(shards[rank][k].shape == getattr(sampler, k).shape for k in ("scores", "visits", "ema")))
        # 用全局最小值统一恢复决定：任一卡分片不匹配时，所有卡都放弃恢复课程，
        # 避免只有部分进程继续进入概率更新的集合通信。
        if not bool(parallel.reduce(int(ok), torch.distributed.ReduceOp.MIN)):
            print(f"checkpoint's reset-curriculum state was saved for {len(shards)} rank(s) with "
                  f"'{state['partition_algorithm']}' and does not fit this shard: bin scores start from zero", flush=True)
            return
        sampler.load_state_dict(shards[rank])
        sampler.prepare(True)  # 多卡：由恢复的得分重算全局概率
        visited = float((sampler.visits > 0).float().mean())
        print(f"rank {rank}: restored reset-curriculum state ({100 * visited:.1f}% of this shard's bins visited)", flush=True)

    def graphed(self, fn, eager_calls: int = 3, enabled: bool | None = None):
        """包装训练步骤以支持 CUDA 图重放；enabled 默认取自 --cuda_graph。"""
        return GraphedCall(fn, eager_calls, self.use_graph if enabled is None else enabled, thread_local=self.parallel.enabled)

    # ---------------- 可编译内核 ----------------
    def _wm_rollout(self, s, a, xi):
        """从窗口首状态开始，根据动作序列和模拟上下文逐步预测世界模型状态。"""
        # 只使用窗口首个真实状态，随后递归使用模型自身的预测。
        # 这让损失包含多步误差积累，而非若干互不相关的单步拟合。
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_err(self, S_hat, s1, mask):
        """按输出尺度归一化状态误差，使用有效掩码计算窗口损失及末步误差。"""
        # 按状态增量标准差平衡各物理量；下限避免静止维度的微小方差放大误差。
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        # 终止后的转移不计损失；分母固定为窗口数，保留有效时域长度对损失的贡献。
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g):
        """执行一步策略动作、世界模型预测和带噪观测更新，并生成下一帧目标。"""
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.K)

    def _reward_term(self, S, f, a, a_prev):
        """计算想象状态的解析奖励与软存活概率。"""
        return self.mdp.reward(S, f, a, a_prev), self.mdp.soft_survival(S, f)

    def _critic_input(self, S, f, xi_n, g):
        """拼接价值网络状态特征与目标特征。"""
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def cap_value(self, v):
        """将价值估计限制在理论回报上限以内。"""
        return v.clamp(max=self.value_cap)

    # ---------------- 世界模型 ----------------
    def _wm_step(self, s, a, xi, s1, mask):
        """拟合一个世界模型窗口，归约并裁剪梯度，返回损失和末步误差。"""
        S_hat = self.wm_rollout(s, a, xi)
        loss, e_last = self.wm_err(S_hat, s1, mask)
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self.wm, self.layout.wm_step_batch, total_count=self.graph_wm_total)
        grad_norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.nonfinite[0].logical_or_(~torch.isfinite(grad_norm))
        self.opt_wm.step()
        # 末步诊断只统计仍有效的窗口，并跨卡汇总分子与样本数，避免等权平均不同样本量的均值。
        m = mask[:, -1]
        error = self.parallel.reduce(torch.stack(((e_last.detach() * m).sum(), m.sum())))
        return loss.detach() / self.args.wm_window, error[0] / error[1].clamp(min=1)

    def wm_update(self):
        """从回放缓冲区采样窗口并执行一次世界模型更新。"""
        w = self.buffer.sample_windows(self.layout.wm_step_batch, self.args.wm_window)
        return self.wm_step(w["s"], w["a"], w["xi"], w["s1"], w["mask"])

    # ---------------- 策略与critic ----------------
    def sample_valid_starts(self, B: int):
        """采样足以容纳完整想象时域的起点及历史；多次失败时报告数据不足。"""
        starts, hists, n = [], [], 0
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 256:
                raise RuntimeError("replay has too few starts with a complete imagination horizon; collect more transitions or inspect short clips")
            s, h = self.buffer.sample_starts(2 * B, self.Hh)
            f = s["f"]
            # 保留能走完整个想象时域的起点，防止参考索引跨入相邻片段；
            # 每轮多抽候选，只截取尚缺的数量。
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def _policy_step(self, s_flat, xi, f, a_prev, o_hist):
        """通过冻结的世界模型想象 HZ 步，反向传播回报并更新策略。"""
        args, HZ, dev = self.args, self.HZ, self.device
        if self.profile:
            torch.cuda.synchronize(); tq0 = time.time()
        S = unflatten_state(s_flat)
        xi_n = normalize_xi(xi)
        B = self.layout.local_pi_batch
        g0 = self.motion.goal(f, self.K)
        g = g0
        S0, f0 = S, f
        # 保存每步预测状态、动作与参考目标，随后批量计算奖励并复用于价值网络更新。
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        # 将时间与批次合并为 HZ * B，一次执行运动学和奖励计算；随后恢复时间轴计算存活权重。
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, keep = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        r, keep = r.view(HZ, B), keep.view(HZ, B)
        alive = survival(keep, self.use_graph)  # c_{t+1}
        # 第 t 步奖励使用此前各步的存活概率；本步的终止影响后续奖励和末端自举。
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, keeps = r.detach(), keep.detach()
        # 时域之外的回报由末端价值补充，并同时乘折扣与整个时域的存活概率。
        # 价值上限约束避免策略追逐不可能达到的高估回报。
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
        # 捕获的图无法在主机上分支，因此总会执行优化步骤；非有限梯度仍会在本次迭代末尾、
        # 写日志或保存之前中止运行。
        if self.use_graph or bool(finite_pi_grad):
            self.opt_pi.step()
        with torch.no_grad():
            # 价值拟合需要起点加 HZ 个后继状态，共 HZ + 1 帧；切断策略计算图，
            # 使后续价值更新不会再次反向传播到本次策略推演。
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
        tq = (tq0, tq1, tq2) if self.profile else ()
        return (loss.detach(), gn.detach(), (r_sum / HZ / 0.1).mean(), c.mean().detach(), rews, keeps, xi_n,
                S_c, f_c, g_c) + tq

    def _critic_step(self, rews, keeps, xi_n, S_c, f_c, g_c):
        """以想象转移的 GAE(lambda) 目标拟合价值网络（式 3）。

        除损失外，还返回本次想象状态上的 DIAG_MEAN 和 DIAG_MAX：原始价值的均值、末步值、最大值及超过上限的比例，相对 GAE 目标的平均偏差，归一化输入的偏离程度，以及末状态输入梯度和参数梯度范数。"""
        args, HZ, critic = self.args, self.HZ, self.critic
        B = self.layout.local_pi_batch
        with torch.no_grad():
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        # 先更新输入统计再固定本轮目标；所有小批次与训练轮次使用同一份 GAE 目标。
        critic.norm.update(X)
        with torch.no_grad():
            raw = critic(X)
            V = self.cap_value(raw)
            targets = gae_targets(V, rews, keeps, args.gamma, args.lam)  # (H, B)
            z = critic.norm(X).abs()
            diag_mean = torch.stack((raw.mean(), raw[-1].mean(), (raw > self.value_cap).float().mean(), (raw[:-1] - targets).mean(),
                                     (z > 5).float().mean()))
            diag_max = torch.stack((raw.max(), z.max()))
        # 单独测量末端价值对输入的敏感度，供判断自举梯度是否异常；该指标不参与拟合损失。
        x_end = X[-1].detach().clone().requires_grad_(True)
        value_grad = torch.autograd.grad(critic(x_end).sum(), x_end)[0].norm(dim=-1).mean()  # 本次更新的优化步骤之前
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        v_loss = torch.zeros(())
        grad_sum = torch.zeros((), device=self.device)
        # 每轮重新打乱 HZ * B 个训练状态；末端状态只用于自举，不作为本轮回归样本。
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
        """从回放中采样起点，更新策略，再使用想象转移更新价值网络。"""
        with self.parallel.coordinated("policy replay sampling"):
            s, h = self.sample_valid_starts(self.layout.local_pi_batch)
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = self.mdp.add_noise(o_hist)
        # 冻结世界模型与价值网络的参数，但保留它们对输入的导数，
        # 让回报梯度仍能穿过动力学和末端价值传到策略。
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        loss, gn, imag_r, surv, rews, keeps, xi_n, S_c, f_c, g_c, *tq = self.policy_step(s["s"], s["xi"], s["f"], s["a_prev"], o_hist)
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        # ---------------- 价值网络：想象转移的 GAE(lambda) 目标 ----------------
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
        """执行固定形状的采样工作，所有变化的环境张量均显式作为图输入。"""
        o = self.mdp.add_noise(self.mdp.obs(S, a_prev, xi, f))
        # 新回合用首帧观测填满历史，避免混入上一回合；其余环境移除最旧观测并追加当前帧。
        fresh = ep_step == 0
        history = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(history),
                              torch.cat((history[:, 1:], o.unsqueeze(1)), 1))
        g = self.motion.goal(f, self.K)
        self.pi.obs_norm.update(o)
        self.pi.goal_norm.update(g)
        a = self.pi(history, g)
        noise = torch.randn_like(a)
        a = torch.clamp(a + (self.args.action_noise * noise if self.noise_scale is None else self.noise_scale * noise), -20, 20)
        return a, history

    @torch.no_grad()
    def store_transition(self, out):
        """保存模拟转移，并更新世界模型输入和目标的归一化统计。"""
        self.buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        self.wm.in_norm.update(self.wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        self.wm.out_norm.update(self.wm.targets(S0, S1))

    @torch.no_grad()
    def collect(self, n_steps: int):
        """执行指定步数的模拟采样，写入转移并累计训练统计。"""
        env, stats = self.env, self.stats
        for _ in range(n_steps):
            S = env.state()
            a, self.obs_hist = self.collect_action(S, env.a_prev, env.xi, env.f, env.ep_step, self.obs_hist)
            # 可将转移入库安排在环境等待重置索引之前，以重叠 GPU 工作与主机等待；
            # 两条路径都只存储一次转移，使用重置前的后继状态。
            overlap = self.args.collect_overlap == "on"
            out = env.step(a, after_step=self.store_transition if overlap else None)
            if not overlap:
                self.store_transition(out)
            self.sim_steps += self.layout.global_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if self.clean_envs is not None:
                c = self.clean_envs.float()
                stats["r_clean"].append((out["r"] * c).sum() / 0.1)
                stats["r_noisy"].append((out["r"] * (1 - c)).sum() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].sum())
                stats["ep_ret"].append(out["ep_ret_norm"].sum())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())
                if self.clean_envs is not None:
                    # ep_len与ep_ret_norm按done.nonzero()的顺序排列；环境已在主机上取得重置索引，这里的布尔索引不增加同步
                    ended = self.clean_envs[out["done"]].float()
                    stats["ep_clean"].append(torch.stack(((out["ep_len"] * ended).sum(), (out["ep_ret_norm"] * ended).sum(),
                                                          (out["term"] & self.clean_envs).float().sum(), ended.sum())))

    # ---------------- 一次迭代 ----------------
    def set_lr(self, lr: float):
        """更新各优化器的学习率，兼容设备张量与普通数值。"""
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
        # 用小数累加器落实更新/数据比，保留不足一个采集步的余数，避免每轮取整产生长期偏差。
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
        # 不同指标按实际环境数或批量大小加权，再归约总和与权重；
        # 各卡分配不均时也能得到正确的全局平均值。
        keys = list(agg) + ["r"]
        values = [agg[k] if k in agg else stats[k] for k in keys]
        weights = [layout.local_wm_batch if k in ("wm", "wm8") else layout.local_envs if k == "r" else layout.local_pi_batch
                   for k in keys]
        totals = parallel.reduce([[sum(float(v) for v in vs) * weight, len(vs) * weight]
                                  for vs, weight in zip(values, weights)]).cpu()
        means = {key: float(pair[0] / pair[1].clamp(min=1)) for key, pair in zip(keys, totals)}
        # 回合指标按已结束回合数归一化；计时取最慢进程，反映同步训练实际吞吐。
        episodes = parallel.reduce([sum(float(v) for v in stats[key]) for key in ("ep_len", "ep_ret", "term", "done")]).cpu()
        timing = parallel.reduce([time.time() - self.t_log, time.time() - self.t_start], torch.distributed.ReduceOp.MAX).cpu()
        ips = args.log_every / float(timing[0])
        self.t_log = time.time()
        wall_h = float(timing[1]) / 3600
        term_frac = float(episodes[2] / episodes[3].clamp(min=1))
        row = [it + 1, wall_h, self.sim_steps, *[means[k] for k in keys],
               float(episodes[0] / episodes[3].clamp(min=1)), float(episodes[1] / episodes[3].clamp(min=1)), term_frac, ips, lr]
        sampler = env.sampler
        local_samp = sampler.stats()
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
        scored = sampler.scored & (sampler.visits > 0)
        bin_sum = parallel.reduce(torch.stack((torch.where(scored, sampler.scores, 0.0).sum().double(), scored.sum().double()))).cpu()
        bin_min = float(parallel.reduce(torch.where(scored, sampler.scores, torch.inf).min(), torch.distributed.ReduceOp.MIN))
        diag_row = [it + 1, *(float(x) for x in d_mean[:-1] / d_mean[-1]), *(float(x) for x in d_max),
                    float(bin_sum[0] / bin_sum[1].clamp(min=1)), bin_min if math.isfinite(bin_min) else 0.0, float(samp[5])]
        rows = {"train": row, "sampling": samp_row, "diag": diag_row}
        if self.clean_envs is not None:
            rows["clean"] = self._clean_row(it, stats, episodes)
        if sampler.clip_cap > 0:
            cap = parallel.reduce(sampler.clip_stats(), torch.distributed.ReduceOp.MAX).cpu()
            rows["clipcap"] = [it + 1, float(cap[0]), float(cap[1]), int(cap[2])]
        # 本次日志已消费累计量，清空窗口，避免下次重复统计；模型与课程长期状态不在此处清除。
        self.stats = {k: [] for k in stats}
        self.agg = {k: [] for k in agg}
        self.diag_agg = {k: [] for k in diag_agg}
        return rows

    def _clean_row(self, it: int, stats: dict, episodes: torch.Tensor) -> list:
        """clean_log.csv一行：两组环境分别的每步奖励与回合指标；noisy组的回合量由全部回合减去clean组得到。"""
        n_clean = float(self.clean_envs.sum())
        steps = len(stats["r_clean"])
        ep = torch.stack(stats["ep_clean"]).sum(0).tolist() if stats["ep_clean"] else [0.0] * 4
        tot = self.parallel.reduce([sum(float(v) for v in stats["r_clean"]), steps * n_clean,
                                    sum(float(v) for v in stats["r_noisy"]), steps * (self.layout.local_envs - n_clean), *ep]).cpu()
        c = tot[4:8]
        n = episodes - c  # 顺序相同：回合长度和、归一化回报和、终止数、结束回合数
        per = lambda x: [float(x[0] / x[3].clamp(min=1)), float(x[1] / x[3].clamp(min=1)), float(x[2] / x[3].clamp(min=1))]  # noqa: E731
        return [it + 1, float(tot[0] / tot[1].clamp(min=1)), float(tot[2] / tot[3].clamp(min=1)), *per(c), *per(n)]

    def gather_bins(self) -> dict:
        """各rank的重置课程状态汇总（集合通信，所有rank都要调用；只有主进程拿去存检查点）。"""
        shards = self.parallel.gather(self.env.sampler.state_dict())
        return {"shards": shards, "partition_algorithm": self.motion.partition["algorithm"]}

    def term_rows(self, it: int):
        """自上次调用以来终止集中在哪些片段：返回({"termwin": 汇总行}, [term_bins行...])；所有rank都要调用（集合通信）并清零窗口计数。"""
        motion, parallel = self.motion, self.parallel
        report = self.env.sampler.window_report(max(self.args.term_top, 10))
        top = [(motion.names[clip].replace(",", ";"), terms, ends, starts,
                (int(motion.bin_start[b]) - int(motion.clip_start[clip])) / BIN_SECONDS, bin_terms)
               for clip, terms, ends, starts, b, bin_terms in report["top"]]
        parts = parallel.gather({k: report[k] for k in ("terms", "ends", "starts", "clips")} | {"top": top})
        total, ends, starts = (sum(p[k] for p in parts) for k in ("terms", "ends", "starts"))
        clips = sorted((row for p in parts for row in p["top"]), key=lambda row: -row[1])
        share = lambda n: sum(row[1] for row in clips[:n]) / max(total, 1)  # noqa: E731
        summary = [it + 1, total, sum(p["clips"] for p in parts), share(1), share(10), starts, ends]
        lines = [[it + 1, name, n, e, n / max(e, 1), s, offset, bin_terms, n / max(total, 1)]
                 for name, n, e, s, offset, bin_terms in clips[:self.args.term_top]]
        return {"termwin": summary}, lines

    def checkpoint(self, it: int, bins: dict | None = None) -> dict:
        """完整训练状态（--resume用）；其中pi、wm与元数据另存为评估用的ckpt_*.pt。bins是gather_bins()的结果。"""
        return {"bins": bins, "wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
                "iteration": it + 1, "wall_time": time.time() - self.t_start, "sim_steps": self.sim_steps,
                "args": vars(self.args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reference": self.motion.global_description,
                "distributed": {"world_size": self.parallel.world_size, "partition_algorithm": self.motion.partition["algorithm"]}}

    def save(self, it: int, log_dir: str, keep_states: int, bins: dict | None = None):
        """写last.pt（先轮换旧的）与ckpt_{it+1}.pt；都先写临时文件再原子替换。只在主进程调用。"""
        ck = self.checkpoint(it, bins)
        last = os.path.join(log_dir, "last.pt")
        rotate_states(last, keep_states)
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        # 评估检查点只保留推理模型与溯源信息；last.pt 另包含优化器和课程状态，用于继续训练。
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "reference", "distributed")}
        path = os.path.join(log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)

    def close(self):
        """释放捕获的CUDA graph（其中的NCCL操作须在销毁进程组之前释放）。"""
        for captured in (self.wm_step, self.policy_step, self.critic_step, self.collect_action, self.env._reward_termination):
            captured.close()
