"""根据 diag_bin_difficulty.py 的难度标签，比较 diag_train_episodes.py 产生的重置分箱方案。

用法：python scripts/diag_compare_bins.py LABELS.pt RUN_DIR [RUN_DIR ...]
仅使用各次运行后半段的回合，使分箱得分有时间适应。"""

import json
import sys

import torch

labels = torch.load(sys.argv[1], weights_only=False)
runs = sys.argv[2:]
# 按每 50 帧网格起点的全局帧号查找标签：片段起点 + offset // 50 * 50
num_clips = int(labels["clip"].max()) + 1
clip_start = torch.zeros(num_clips, dtype=torch.long).scatter_(0, labels["clip"], labels["start"] - labels["offset"])
hard_at = torch.zeros(int(labels["start"].max()) + 1, dtype=torch.bool)
hard_at[labels["start"]] = labels["failed"]
base = float(labels["failed"].float().mean())
rows = []
for run in runs:
    E = torch.load(f"{run}/episodes.pt", weights_only=False)
    B = torch.load(f"{run}/bins.pt", weights_only=False)
    n = len(E["T"])
    # 课程起初尚未适应，仅比较后半段回合；起点偏移映射回固定 50 帧标签网格。
    late = slice(n // 2, n)
    T, ret, term = E["T"][late], E["ep_ret_norm"][late], E["term"][late]
    off, clen, clip = E["start_offset"][late].long(), E["clip_len"][late].long(), E["clip"][late].long()
    left = clen - 1 - off
    is_hard = hard_at[clip_start[clip] + off // 50 * 50]
    # 同时看回合等权回报、步数加权回报与困难起点占比；短回合较多时这些指标可能走势不同。
    row = {"run": run.split("\\")[-1].split("/")[-1], "episodes_late": int(len(T)),
           "T_le_5": round(float((T <= 5).float().mean()), 4), "T_median": float(T.median()),
           "ep_ret_norm_mean": round(float(ret.mean()), 4),
           "step_weighted": round(float((E["r"][late]).sum() / (T * 0.1).sum()), 4),
           "term_frac": round(float(term.mean()), 4),
           "resets_within_2s_of_end": round(float((left < 100).float().mean()), 4),
           "resets_on_hard_positions": round(float(is_hard.float().mean()), 4),
           "hard_lift_vs_uniform": round(float(is_hard.float().mean()) / max(base, 1e-9), 2)}
    p = B.get("bin_probs")
    if p is not None:
        # 重新归一化保存的权重，再用 log(分箱数) 缩放熵，便于比较不同分箱数量的方案。
        p = p.double() / p.sum()
        row["samp_entropy"] = round(float(-(p * (p + 1e-30).log()).sum() / torch.log(torch.tensor(float(len(p))))), 4)
        row["samp_top1"] = round(float(p.max()), 5)
        v = B["bin_visits"]
        if len(v):
            row["bins_visited"] = round(float((v > 0).float().mean()), 4)
            row["prob_mass_on_bins_visited_lt3"] = round(float(p[v < 3].sum()), 4)
    rows.append(row)
print(json.dumps({"label_positions": len(labels["failed"]), "hard_position_frac": round(base, 4), "runs": rows},
                 indent=2, ensure_ascii=False))
