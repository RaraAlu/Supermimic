"""基于回报的重置分箱采样（论文附录D.1）：回合评分、分箱EMA与全局重置概率。

求和与最大值都经过``parallel.reduce``（``DistributedContext``或``LocalContext``），
因此每个rank只持有本分片的分箱，却从同一个全局分布中采样。
"""

import torch
import torch.distributed as dist


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
