"""基于回报的重置分箱采样（论文附录D.1）：回合评分、分箱EMA、全局重置概率，以及维护这些状态的BinSampler。

求和与最大值都经过``parallel.reduce``（``DistributedContext``或``LocalContext``），
因此每个rank只持有本分片的分箱，却从同一个全局分布中采样。
"""

import math

import torch
import torch.distributed as dist

from supermimic.utils.distributed import LocalContext

TAIL_STEPS = 100  # 距片段末尾不超过此步数（2 秒）的重置计为末尾重置
CLIP_END_MODES = ("bootstrap", "terminal")
PRIORS = ("zero", "mean")


@torch.no_grad()
def update_bin_returns(estimates, bins, returns, alpha=0.2, visits=None, replace_first=False):
    """对本模拟步中结束的回合按分箱求平均回报，每箱执行一次 EMA 更新。

    多个环境可能在同一箱结束：先稳定排序并分段归约，再写入，避免 index_put_ 在重复目标下丢失观测或产生未定义 CUDA 结果。唯一分箱与原来的单回合更新一致。visits 记录每箱结束的回合数；replace_first 为真时首次更新直接采用观测均值。"""
    if bins.numel() == 0:
        return
    # 先把重复分箱的回合聚合为一条观测，再更新 EMA；同一步结束回合的排列不会改变更新次数。
    order = torch.argsort(bins, stable=True)
    unique, counts = torch.unique_consecutive(bins[order], return_counts=True)
    means = torch.segment_reduce(returns[order], "mean", lengths=counts)
    rate = alpha
    if visits is not None:
        if replace_first:
            rate = torch.where(visits[unique] == 0, 1.0, alpha).to(estimates.dtype)
        visits[unique] += counts.to(visits.dtype)
    estimates[unique] = (1 - rate) * estimates[unique] + rate * means


@torch.no_grad()
def episode_bin_scores(disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, left, gamma, step_max,
                       clip_end="bootstrap", max_steps=500, skip_steps=0):
    """计算已结束回合的归一化分箱得分及其是否可以更新分箱。

    disc_ret、ret 为起始帧起的折扣及未折扣回报，head_disc、head_ret 为前 skip_steps 步的对应回报；steps 是回合长度，left 是起点到片段末尾的步数。bootstrap 模式以平均奖励外推截断回合；terminal 模式在片段末尾结束回报，按起点可达的最大值评分，仅对 max_steps 截断进行外推。

    skip_steps 大于零时跳过重置瞬态，以免重置噪声、零前一动作和重复观测历史的恢复成本主导短回合得分。瞬态内跌倒仍记零分，瞬态内截断则不更新分箱。非有限得分也不更新，避免 NaN 永久污染采样权重。"""
    # 移除重置瞬态后，将折扣起点重新移到有效区间首步；步数及可达时域同步平移。
    if skip_steps:
        disc_ret, ret = (disc_ret - head_disc) / gamma**skip_steps, ret - head_ret
        steps, left, max_steps = steps - skip_steps, left - skip_steps, max_steps - skip_steps
    valid = terminated | (steps > 0)
    ended = terminated | at_clip_end if clip_end == "terminal" else terminated
    # 真正终止不补未来奖励；允许自举的截断回合按已观测平均奖励估算剩余无限时域。
    G = disc_ret + torch.where(ended, 0.0, gamma**steps * (ret / steps.clamp(min=1)) / (1 - gamma))
    G_max = torch.full_like(G, step_max / (1 - gamma))
    if clip_end == "terminal":
        G_max = G_max * torch.where(left < max_steps, 1 - gamma ** left.clamp(min=1), 1.0)
    score = G / G_max
    return score, valid & torch.isfinite(score)


@torch.no_grad()
def cap_clip_share(soft, bin_clip, num_clips, cap, parallel):
    """把困难度分布soft中单个片段（全部分箱之和）的份额限制在cap以内；不是论文的设置（LaoYang1994复现中的--env.hard-clip-cap）。

    超过上限的片段整段等比缩到上限，其余片段乘同一个系数分走削掉的质量，片段内各分箱的相对比例不变，总和不变。
    有质量的片段少于1/cap个时上限取1/片段数。片段不跨rank：每个rank只交换本地最大的ceil(1/cap)+1个片段质量
    （被削的片段不会更多），系数用闭式解求出，全程在设备上。返回(削平后的soft, [削平前最大片段份额, 被削平的片段数])。"""
    dev = soft.device
    mass = torch.zeros(num_clips, dtype=torch.float64, device=dev).index_add_(0, bin_clip, soft.double())
    k_max = math.ceil(1 / cap) + 1
    n_top = min(k_max, num_clips)
    ordered = mass.sort(descending=True).values
    local = torch.zeros(k_max, dtype=torch.float64, device=dev)
    local[:n_top] = ordered[:n_top]
    total, n_pos, tail = parallel.reduce(torch.stack((mass.sum(), (mass > 0).sum().double(), ordered[n_top:].sum())))
    limit = torch.maximum(torch.as_tensor(cap, dtype=torch.float64, device=dev), 1 / n_pos.clamp(min=1)) * total
    pool = parallel.all_gather(local).flatten().sort(descending=True).values
    top = pool[:k_max]  # 全局最大的k_max个片段质量，降序
    # 最大的k段削到上限时，其余片段的系数为 s_k = (总和 − k·上限) / 其余片段的质量；取最小的k，使第k+1大的片段乘s_k后不超过上限。
    # 其余质量直接求和（不用总和减前k段：其余很小时相减会丢掉有效位）
    k = torch.arange(k_max + 1, device=dev, dtype=torch.float64)
    rest = torch.cat((top.flip(0).cumsum(0).flip(0), top.new_zeros(1))) + pool[k_max:].sum() + tail
    scale = (total - k * limit) / rest.clamp(min=1e-300)
    fits = torch.cat((top, top.new_zeros(1))) * scale <= limit * (1 + 1e-9)
    capped = fits.int().argmax()
    factor = torch.where(mass > 0, torch.minimum(scale[capped], limit / mass.clamp(min=1e-300)), 0.0)
    return (soft.double() * factor[bin_clip]).to(soft.dtype), torch.stack((top[0] / total.clamp(min=1e-300), capped.double()))


@torch.no_grad()
def global_bin_probabilities(returns, ema, maximum_return, num_bins, parallel, visits=None, allowed=None, scored=None,
                             clip_cap=0.0, bin_clip=None, num_clips=0, cap_stats=None):
    """计算全局 50/40/10 重置权重，实际采样限定在本地分片。

    提供 visits 时，未完成回合的分箱使用已访问分箱的全局平均分。allowed 指定允许重置的分箱；scored 指定参与 softmax 和 EMA 的分箱。无法评分的分箱只获得均匀权重；若所有进程都没有可评分的分箱，则自适应权重也并入均匀分布。

    得分截断到 [0, maximum_return]，避免策略崩溃后大量负分使 softmax 几乎只选择一个失败分箱；正常范围内的得分不变。

    clip_cap>0 时困难度部分（及由它累积的EMA）中每个片段的份额不超过clip_cap（见cap_clip_share；bin_clip、num_clips为本分片
    分箱所属的局部片段与片段数），cap_stats（形状(2,)）原地写入cap_clip_share返回的统计；为0时与论文相同。"""
    returns = torch.nan_to_num(returns, nan=0.0).clamp(0.0, maximum_return)
    if allowed is None:
        allowed = torch.ones_like(returns, dtype=torch.bool)
    else:
        num_bins = parallel.reduce(allowed.sum().double())
    scored = allowed if scored is None else scored & allowed
    if visits is not None:
        seen = (visits > 0) & scored
        total, count = parallel.reduce(torch.stack(((returns * seen).double().sum(), seen.sum().double())))
        returns = torch.where(seen, returns, (total / count.clamp(min=1)).to(returns.dtype))
    # 低分箱获得更高权重；先求跨卡最大 logit 再计算指数，保证所有分片共享同一数值基准。
    logits = torch.where(scored, -25.0 * returns / maximum_return, -torch.inf)
    maximum = parallel.reduce(logits.max(), dist.ReduceOp.MAX)
    exp = torch.where(scored, torch.exp(logits.double() - maximum), 0.0)
    norm = parallel.reduce(exp.sum())  # 除无可评分分箱外，结果至少为 1（全局最大值贡献 exp(0)）
    soft = (exp / norm.clamp(min=1e-300)).to(returns.dtype)
    if clip_cap > 0:
        soft, stats = cap_clip_share(soft, bin_clip, num_clips, clip_cap, parallel)
        if cap_stats is not None:
            cap_stats.copy_(stats)
    # 混合均匀探索、当前困难度和缓慢变化的历史概率，避免采样只追随近期噪声。
    ema.mul_(0.999).add_(0.001 * soft)
    kept = torch.where(scored, ema, 0.0)
    total_ema = parallel.reduce(kept.double().sum()).clamp(min=1e-300)
    probs = 0.5 / num_bins + 0.4 * soft + 0.1 * kept / total_ema.to(ema.dtype)
    probs = torch.where(norm > 0, probs, 1.0 / num_bins)
    return torch.where(allowed, probs, 0.0).to(returns.dtype)


class BinSampler:
    """一个rank上分片的重置分箱状态：每个分箱的得分与访问数、概率的EMA，以及据此抽取的重置帧。

    TrackingSim在回合结束时调用record()记录得分，在重置时调用sample()抽取起始帧；
    多卡时每步先由prepare()在所有rank上同步更新全局概率。
    """

    def __init__(self, motion, device, gamma: float, parallel=None, clip_end: str = "bootstrap", prior: str = "zero",
                 skip_steps: int = 50, reset_margin: int = 100, max_steps: int = 500, clip_cap: float = 0.0):
        """配置分箱采样：clip_end 为 bootstrap 时以平均奖励外推片段末尾并按全局最大回报评分；terminal 时在末尾结束并按起点可达的最大回报评分，仅外推步数上限截断。

        prior 控制未访问分箱采用零分或已访问分箱的均值；mean 模式首次观测直接替换先验。skip_steps 跳过重置瞬态；reset_margin 排除片段末尾附近的起点，短片段仅允许首帧。无法越过瞬态的分箱及这些首帧只获得均匀权重。两者均为零时恢复旧评分方式。parallel 为 None 时使用单进程 LocalContext。

        clip_cap>0 时限制困难度部分中单个片段的份额（不是论文的设置，见cap_clip_share），0为论文原版。"""
        if clip_end not in CLIP_END_MODES or prior not in PRIORS or skip_steps < 0 or reset_margin < 0 or not 0 <= clip_cap < 1:
            raise ValueError(f"bin_clip_end must be one of {CLIP_END_MODES}, bin_prior one of {PRIORS}, "
                             "bin_skip_steps and reset_margin non-negative, and bin_clip_cap in [0, 1)")
        self.motion, self.device, self.gamma, self.parallel = motion, device, gamma, parallel
        self.clip_end, self.prior, self.skip_steps, self.max_steps = clip_end, prior, skip_steps, max_steps
        self.clip_cap = clip_cap
        self.cap_stats = torch.zeros(2, dtype=torch.float64, device=device)  # 最近一次：削平前最大片段份额、被削平的片段数
        # 分箱得分为折扣回报除以最大回报（全局最大值或起始帧可达到的最大值）
        self.scores = torch.full((motion.num_bins,), 0.0, device=device)
        self.visits = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        # 分箱起点为 [bin_start, reset_end)；除片段首帧外，排除距末尾不足 reset_margin 的起点。
        # 这些首帧不参与评分：瞬态结束到片段末尾的步数太少，一次低分就可能占据全部 softmax 权重。
        # 起点无法越过瞬态的分箱也不参与评分。
        clip_start, clip_end_frame = motion.clip_start[motion.bin_clip], motion.clip_end[motion.bin_clip]
        limit = clip_end_frame - reset_margin
        self.allowed = (motion.bin_start < limit) | (motion.bin_start == clip_start)
        self.reset_end = torch.maximum(torch.minimum(motion.bin_end, limit), motion.bin_start + 1)
        self.scored = (motion.bin_start < limit) & (clip_end_frame - motion.bin_start > skip_steps)
        num_bins = motion.global_num_bins if parallel else motion.num_bins
        self.ema = torch.full((motion.num_bins,), 1.0 / num_bins, device=device)
        self.probs = self.allowed.float() / self.allowed.sum().clamp(min=1)
        self.reset_counts = torch.zeros(2, device=device)  # 采样重置数及其中的末尾重置数
        self.nonfinite = torch.zeros((), device=device)  # 因非有限值而跳过的分箱得分数
        # 自上次window_report以来每个分箱被抽中的次数、从它起步的回合结束数与其中因终止而结束的数（日志用，不进入检查点）
        self.window_starts = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        self.window_ends = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        self.window_terms = torch.zeros(motion.num_bins, dtype=torch.long, device=device)

    def _probabilities(self, context):
        """根据当前分箱得分、访问数和概率 EMA 计算全局重置概率。"""
        visits = self.visits if self.prior == "mean" else None
        return global_bin_probabilities(self.scores, self.ema, 1.0, self.motion.global_num_bins, context,
                                        visits, self.allowed, self.scored, self.clip_cap, self.motion.bin_clip,
                                        self.motion.clip_start.numel(), self.cap_stats)

    def prepare(self, active):
        """多卡时每个仿真步都调用：有任一rank结束了回合就在所有rank上更新全局概率；单进程时不做事。"""
        # 即使只有其他进程完成回合，所有进程也必须参与。
        if self.parallel and bool(self.parallel.reduce(int(active))):
            self.probs = self._probabilities(self.parallel)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按当前概率抽n个分箱，再在各分箱的起点范围内均匀抽起始帧；返回(帧, 分箱)。"""
        m = self.motion
        if not self.parallel:
            self.probs = self._probabilities(LocalContext)
        bins = torch.multinomial(self.probs, n, replacement=True)
        start = m.bin_start[bins]
        end = self.reset_end[bins]  # 至少比片段末尾早一步
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        self.reset_counts[0] += n
        self.reset_counts[1] += (m.clip_end[m.bin_clip[bins]] - f < TAIL_STEPS).sum()
        self.window_starts.index_add_(0, bins, torch.ones_like(bins))
        return f, bins

    def record(self, start_f, start_bin, disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, step_max):
        """记录结束回合的得分（参数含义见episode_bin_scores），截断的回合以平均奖励外推。"""
        left = (self.motion.clip_end[self.motion.clip_id[start_f]] - start_f).float()  # 从起始帧到片段末尾
        score, valid = episode_bin_scores(disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, left,
                                          self.gamma, step_max, self.clip_end, self.max_steps, self.skip_steps)
        # 窗口结束/失败计数记录所有回合；只有有效且有限的得分进入课程更新，两类统计用途不同。
        self.nonfinite += (~torch.isfinite(score)).sum()
        self.window_ends.index_add_(0, start_bin, torch.ones_like(start_bin))
        died = start_bin[terminated]
        self.window_terms.index_add_(0, died, torch.ones_like(died))
        update_bin_returns(self.scores, start_bin[valid], score[valid], visits=self.visits, replace_first=self.prior == "mean")

    def state_dict(self) -> dict:
        """课程状态（本rank分片上每个分箱的得分、访问数与概率EMA），放到CPU；概率由它们重新算出。"""
        return {"scores": self.scores.cpu().clone(), "visits": self.visits.cpu().clone(), "ema": self.ema.cpu().clone()}

    def load_state_dict(self, state: dict) -> None:
        """原地写回state_dict()的内容（形状须与本分片一致，否则报错）；之后多卡需调用prepare(True)刷新全局概率。"""
        for name in ("scores", "visits", "ema"):
            src, dst = state[name], getattr(self, name)
            if src.shape != dst.shape:
                raise ValueError(f"bin state {name} has shape {tuple(src.shape)}, this shard has {tuple(dst.shape)}")
            dst.copy_(src.to(dst.device))

    @torch.no_grad()
    def window_report(self, top: int) -> dict:
        """自上次调用以来终止最多的前top个片段（按片段汇总本rank分片内的分箱），并清零计数。

        返回{"terms"、"ends"、"starts"：窗口内全部的终止数、结束的回合数与抽中次数，"clips"：有终止的片段数，
        "top"：[(局部片段序号, 终止数, 结束的回合数, 抽中数, 其中终止最多的分箱的局部序号, 该分箱的终止数), ...]}。
        终止率应除以结束的回合数（它们可能在上个窗口被抽中），不是抽中数。
        片段不跨rank，所以各rank的前top个合并后就是全局的前top个。"""
        m = self.motion
        n_clips = m.clip_start.numel()
        clip_terms = torch.zeros(n_clips, dtype=torch.long, device=self.device).index_add_(0, m.bin_clip, self.window_terms)
        clip_ends = torch.zeros(n_clips, dtype=torch.long, device=self.device).index_add_(0, m.bin_clip, self.window_ends)
        clip_starts = torch.zeros(n_clips, dtype=torch.long, device=self.device).index_add_(0, m.bin_clip, self.window_starts)
        # 先按片段汇总终止数再排名，每个入选片段再定位最常失败的分箱，便于追踪具体动作时段。
        counts, order = clip_terms.topk(min(top, n_clips))
        rows = []
        for count, clip in zip(counts.tolist(), order.tolist()):
            if count == 0:
                break
            in_clip = torch.where(m.bin_clip == clip, self.window_terms, -1)
            bin_ = int(in_clip.argmax())
            rows.append((clip, count, int(clip_ends[clip]), int(clip_starts[clip]), bin_, int(self.window_terms[bin_])))
        report = {"terms": int(self.window_terms.sum()), "ends": int(self.window_ends.sum()),
                  "starts": int(self.window_starts.sum()), "clips": int((clip_terms > 0).sum()), "top": rows}
        self.window_terms.zero_()
        self.window_ends.zero_()
        self.window_starts.zero_()
        return report

    @torch.no_grad()
    def clip_stats(self) -> torch.Tensor:
        """[本分片中最大的片段重置概率（各分箱之和）, 削平前困难度部分的最大片段份额, 被削平的片段数]。
        片段不跨rank且概率是全局分布，各项跨进程取最大值即为全局值（后两项各rank相同）。"""
        clip_p = torch.zeros(self.motion.clip_start.numel(), dtype=torch.float64, device=self.device)
        clip_p.index_add_(0, self.motion.bin_clip, self.probs.double())
        return torch.cat((clip_p.max().reshape(1), self.cap_stats))

    @torch.no_grad()
    def stats(self) -> torch.Tensor:
        """返回上次调用以来的本地统计：[熵项、最大概率、已访问分箱数、重置数、末尾重置数、跳过的得分数]。

        概率是全局分布，跨进程求和熵项与访问数、取最大概率的最大值即可得到全局统计。"""
        p = self.probs.double()
        stats = torch.stack((-(p * (p + 1e-30).log()).sum(), p.max(), (self.visits > 0).sum().double(),
                             *self.reset_counts.double(), self.nonfinite.double()))
        self.reset_counts.zero_()
        self.nonfinite.zero_()
        return stats
