"""基于回报的重置分箱采样（论文附录D.1）：回合评分、分箱EMA、全局重置概率，以及维护这些状态的BinSampler。

求和与最大值都经过``parallel.reduce``（``DistributedContext``或``LocalContext``），
因此每个rank只持有本分片的分箱，却从同一个全局分布中采样。
"""

import torch
import torch.distributed as dist

from supermimic.utils.distributed import LocalContext

TAIL_STEPS = 100  # resets this close to the clip end (2 s) are reported as tail resets
CLIP_END_MODES = ("bootstrap", "terminal")
PRIORS = ("zero", "mean")


@torch.no_grad()
def update_bin_returns(estimates, bins, returns, alpha=0.2, visits=None, replace_first=False):
    """Apply one EMA update per bin using its mean episode return in this simulator step.

    Several environments can finish in the same bin. Sort them in a stable order and reduce each
    segment before writing: index_put_ with duplicate destinations silently loses observations and
    has undefined results on CUDA. Unique bins retain the original single-episode update exactly.
    ``visits`` counts finished episodes per bin; with ``replace_first`` a bin's first update takes
    the observed mean instead of blending it into the initial value.
    """
    if bins.numel() == 0:
        return
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
    """Normalized reset-bin scores of finished episodes, and which episodes may update their bin.

    disc_ret and ret are the discounted and plain reward sums from the start frame, head_disc and
    head_ret the same sums over the first skip_steps steps, steps the episode length, left the steps
    from the start frame to the clip end. Truncated episodes extend their return with their mean
    reward ("bootstrap"); with "terminal", an episode that reaches the clip end stops there and is
    scored against the maximum attainable from its start, and only the max_steps cap is extended.

    With skip_steps > 0 the score is the return after the reset transient: right after a reset the
    policy recovers from reset noise, a zero previous action and a repeated observation history,
    which costs reward for about a second regardless of the bin, and in a short episode that cost
    dominates the mean reward the bootstrap extends. An episode that falls within the transient
    still scores 0; one truncated within it carries no other evidence and does not update its bin.
    A non-finite score (a diverged simulation) never updates its bin: one NaN estimate would stay in
    the bin for good and make every later weight computation fall back to the uniform distribution.
    """
    if skip_steps:
        disc_ret, ret = (disc_ret - head_disc) / gamma**skip_steps, ret - head_ret
        steps, left, max_steps = steps - skip_steps, left - skip_steps, max_steps - skip_steps
    valid = terminated | (steps > 0)
    ended = terminated | at_clip_end if clip_end == "terminal" else terminated
    G = disc_ret + torch.where(ended, 0.0, gamma**steps * (ret / steps.clamp(min=1)) / (1 - gamma))
    G_max = torch.full_like(G, step_max / (1 - gamma))
    if clip_end == "terminal":
        G_max = G_max * torch.where(left < max_steps, 1 - gamma ** left.clamp(min=1), 1.0)
    score = G / G_max
    return score, valid & torch.isfinite(score)


@torch.no_grad()
def global_bin_probabilities(returns, ema, maximum_return, num_bins, parallel, visits=None, allowed=None, scored=None):
    """Global 50/40/10 reset weights; sampling remains conditional on the local shard.

    With ``visits``, bins without a finished episode score the global mean of visited bins.
    ``allowed`` marks bins that may be reset to (the others get probability 0, and the uniform share
    is spread over allowed bins only); ``scored`` marks allowed bins whose score enters the softmax
    and the EMA. Bins that can never produce a score take only the uniform share, so an unscorable
    bin stuck at its initial score cannot attract the adaptive share. With no scored bin on any rank
    (every clip too short to score), the adaptive half joins the uniform share.

    Scores are clamped to [0, maximum_return], the range a working policy produces. Below 0 the
    penalties outweigh the tracking reward; after a policy collapse many bins score far below 0,
    and exp(-25 * score) then turns the softmax into a single bin that takes the whole 40% share
    (0.48 of the weight in the 2026-10-28 run) and fills the replay buffer with one failure.
    Clamped, all failing bins tie and share it. Scores in range are unchanged.
    """
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
    logits = torch.where(scored, -25.0 * returns / maximum_return, -torch.inf)
    maximum = parallel.reduce(logits.max(), dist.ReduceOp.MAX)
    exp = torch.where(scored, torch.exp(logits.double() - maximum), 0.0)
    norm = parallel.reduce(exp.sum())  # >= 1 (the global maximum contributes exp(0)) unless nothing is scored
    soft = (exp / norm.clamp(min=1e-300)).to(returns.dtype)
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
                 skip_steps: int = 50, reset_margin: int = 100, max_steps: int = 500):
        """clip_end: "bootstrap" extends episodes that reach the clip end with their mean reward and
        scores bins against the global maximum return; "terminal" ends the return at the clip end and
        scores each episode against the maximum attainable from its start frame (only the 500-step
        cap is bootstrapped). prior: unvisited bins score 0 ("zero") or the mean of visited bins
        ("mean", whose first episode then replaces the prior instead of being blended into it).
        skip_steps: bins are scored from this step of each episode on, past the reset transient
        (see episode_bin_scores). reset_margin: no reset starts within this many steps of a clip's
        end (a clip shorter than that starts only at its first frame); such first frames and bins
        that cannot outlast the transient take only the uniform share. Both 0 reproduce the earlier
        scores. parallel为None时按单进程处理（LocalContext）。"""
        if clip_end not in CLIP_END_MODES or prior not in PRIORS or skip_steps < 0 or reset_margin < 0:
            raise ValueError(f"bin_clip_end must be one of {CLIP_END_MODES}, bin_prior one of {PRIORS}, "
                             "and bin_skip_steps and reset_margin non-negative")
        self.motion, self.device, self.gamma, self.parallel = motion, device, gamma, parallel
        self.clip_end, self.prior, self.skip_steps, self.max_steps = clip_end, prior, skip_steps, max_steps
        # per-bin score: discounted return divided by its maximum (the global maximum, or the start frame's attainable maximum)
        self.scores = torch.full((motion.num_bins,), 0.0, device=device)
        self.visits = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        # reset starts of a bin are [bin_start, reset_end); none within reset_margin of the clip end, except
        # a clip's first frame. Those first frames are not scored: their score rests on the few steps between
        # the transient and the clip end, and one low draw takes the whole softmax share. Neither is a bin whose
        # starts cannot outlast the transient.
        clip_start, clip_end_frame = motion.clip_start[motion.bin_clip], motion.clip_end[motion.bin_clip]
        limit = clip_end_frame - reset_margin
        self.allowed = (motion.bin_start < limit) | (motion.bin_start == clip_start)
        self.reset_end = torch.maximum(torch.minimum(motion.bin_end, limit), motion.bin_start + 1)
        self.scored = (motion.bin_start < limit) & (clip_end_frame - motion.bin_start > skip_steps)
        num_bins = motion.global_num_bins if parallel else motion.num_bins
        self.ema = torch.full((motion.num_bins,), 1.0 / num_bins, device=device)
        self.probs = self.allowed.float() / self.allowed.sum().clamp(min=1)
        self.reset_counts = torch.zeros(2, device=device)  # sampled resets, of which tail resets
        self.nonfinite = torch.zeros((), device=device)  # bin scores skipped for being non-finite
        # 自上次window_report以来每个分箱被抽中的次数、从它起步的回合结束数与其中因终止而结束的数（日志用，不进入检查点）
        self.window_starts = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        self.window_ends = torch.zeros(motion.num_bins, dtype=torch.long, device=device)
        self.window_terms = torch.zeros(motion.num_bins, dtype=torch.long, device=device)

    def _probabilities(self, context):
        visits = self.visits if self.prior == "mean" else None
        return global_bin_probabilities(self.scores, self.ema, 1.0, self.motion.global_num_bins, context,
                                        visits, self.allowed, self.scored)

    def prepare(self, active):
        """多卡时每个仿真步都调用：有任一rank结束了回合就在所有rank上更新全局概率；单进程时不做事。"""
        # Every rank participates, even when only another rank has completed episodes.
        if self.parallel and bool(self.parallel.reduce(int(active))):
            self.probs = self._probabilities(self.parallel)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按当前概率抽n个分箱，再在各分箱的起点范围内均匀抽起始帧；返回(帧, 分箱)。"""
        m = self.motion
        if not self.parallel:
            self.probs = self._probabilities(LocalContext)
        bins = torch.multinomial(self.probs, n, replacement=True)
        start = m.bin_start[bins]
        end = self.reset_end[bins]  # stays >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        self.reset_counts[0] += n
        self.reset_counts[1] += (m.clip_end[m.bin_clip[bins]] - f < TAIL_STEPS).sum()
        self.window_starts.index_add_(0, bins, torch.ones_like(bins))
        return f, bins

    def record(self, start_f, start_bin, disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, step_max):
        """记录结束回合的得分（参数含义见episode_bin_scores），截断的回合以平均奖励外推。"""
        left = (self.motion.clip_end[self.motion.clip_id[start_f]] - start_f).float()  # start frame to clip end
        score, valid = episode_bin_scores(disc_ret, ret, head_disc, head_ret, steps, terminated, at_clip_end, left,
                                          self.gamma, step_max, self.clip_end, self.max_steps, self.skip_steps)
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
    def stats(self) -> torch.Tensor:
        """Local sums since the last call: [entropy terms, max prob, visited bins, resets, tail resets, skipped scores].

        The probabilities are global (they sum to 1 across ranks), so summing the entropy terms and
        visited bins and taking the max of the max over ranks gives the global values.
        """
        p = self.probs.double()
        stats = torch.stack((-(p * (p + 1e-30).log()).sum(), p.max(), (self.visits > 0).sum().double(),
                             *self.reset_counts.double(), self.nonfinite.double()))
        self.reset_counts.zero_()
        self.nonfinite.zero_()
        return stats
