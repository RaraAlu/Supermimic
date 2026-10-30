"""在训练条件下重放指定片段及其同族片段：摩擦、躯干质心、关节偏移与重置扰动按训练随机化，可叠加训练的动作噪声。

用来判断训练中反复终止的片段是否本身有问题：同族片段（同一动作的其他录制、镜像）都能完成而它失败，才可能是数据问题。
每段复制--reps份，从片段首帧起步，记录终止率（按摩擦分三档）、终止落在最后K帧目标窗口内的比例与终止时离段末的步数。
片段按clips.csv找到所在的源分片，同族按去掉序号与演员后缀的名字归类。在仓库根目录运行，--src 指定被测源码树：
    python scripts/diag_clip_train_cond.py --ckpt CKPT --source_dir SHARDS --clips stem1 stem2 ... --out res.json [--noise 0.1]
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--source_dir", required=True, help="directory with the source shards and their clips.csv")
parser.add_argument("--clips", nargs="+", required=True, help="clip names (file stems, e.g. flip_360_008__A416)")
parser.add_argument("--siblings", type=int, default=3, help="clips of the same family taken from the same shard per clip")
parser.add_argument("--reps", type=int, default=64, help="randomized copies per clip")
parser.add_argument("--noise", type=float, default=0.1, help="action noise std (training: 0.1)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True, help="output JSON")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import gc

import torch

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
src = Path(args.source_dir)
rows = list(csv.DictReader(open(src / "clips.csv", encoding="utf-8")))
stem = lambda r: r["relative_csv"].rsplit("/", 1)[-1][:-4]
family = lambda s: re.sub(r"_\d{3}$", "", re.sub(r"__A\d+(_M)?$", "", s))
by_stem = {stem(r): r for r in rows}
hot = set(args.clips)
groups = collections.defaultdict(list)  # 分片 -> [(clip_index, stem, 角色)]
missing = [s for s in args.clips if s not in by_stem]
if missing:
    raise SystemExit(f"not in {src / 'clips.csv'}: {', '.join(missing)}")
for s in args.clips:
    r = by_stem[s]
    groups[r["h5"]].append((int(r["clip_index"]), s, "hot"))
    sib = [x for x in rows if x["h5"] == r["h5"] and family(stem(x)) == family(s) and stem(x) not in hot]
    for x in sib[:args.siblings]:
        if all(stem(x) != y[1] for y in groups[r["h5"]]):
            groups[r["h5"]].append((int(x["clip_index"]), stem(x), "sibling of " + s))
torch.manual_seed(args.seed)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
subject.set_amp(a_)
pi = subject.policy(ck)
res = {"ckpt": args.ckpt, "iteration": ck.get("iteration"), "noise": args.noise, "reps": args.reps, "clips": {}}
for h5, items in sorted(groups.items()):
    motion, mdp = subject.reference(str(src / h5), a_)
    sel = torch.tensor([c for c, _, _ in items], device=dev).repeat_interleave(args.reps)
    n = len(sel)
    env = subject.make_env(n, motion, mdp)  # 训练模式：构造时按训练分布随机化摩擦、质心与关节偏移
    env.eval_mode = True  # 之后不再自动重置，便于逐回合统计；重置扰动仍按训练开启
    ids = torch.arange(n, device=dev)
    clip_end = motion.clip_end[sel]
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[sel].clone())
    fail_f = torch.zeros(n, dtype=torch.long, device=dev)
    hist = None
    with torch.no_grad():
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(env.f, K))
            if args.noise > 0:
                a = torch.clamp(a + args.noise * torch.randn_like(a), -20, 20)
            f_prev, failed_prev = env.f.clone(), env.failed_eval.clone()
            env.step(a)
            fail_f = torch.where(env.failed_eval & ~failed_prev, f_prev, fail_f)
            env.f = torch.minimum(env.f, clip_end - 1)
    failed = env.failed_eval.cpu()
    to_end = (clip_end - fail_f).cpu()
    fric = env.xi[:, 0].cpu()
    for j, (c, s, role) in enumerate(items):
        sl = slice(j * args.reps, (j + 1) * args.reps)
        fl, te, fr = failed[sl], to_end[sl], fric[sl]
        lo, mid, hi = fr < 0.6, (fr >= 0.6) & (fr < 0.9), fr >= 0.9
        rate = lambda m: round(float(fl[m].float().mean()), 3) if bool(m.any()) else None
        res["clips"][s] = {"role": role, "len": int(motion.clip_len[c]), "term_rate": rate(torch.ones_like(fl)),
                           "term_rate_fric_lt06": rate(lo), "term_rate_fric_06_09": rate(mid), "term_rate_fric_ge09": rate(hi),
                           "term_in_goal_window": round(float((fl & (te < K)).float().sum() / max(int(fl.sum()), 1)), 3),
                           "median_steps_before_end": int(te[fl].median()) if bool(fl.any()) else None}
        print(f"{s[:52]:52s} {role[:14]:14s} len {int(motion.clip_len[c]):4d} term {res['clips'][s]['term_rate']} "
              f"fric<.6 {rate(lo)} .6-.9 {rate(mid)} >=.9 {rate(hi)} in_window {res['clips'][s]['term_in_goal_window']} "
              f"med_before_end {res['clips'][s]['median_steps_before_end']}", flush=True)
    res["subject"] = subject.provenance()
    del env, motion, mdp
    gc.collect()
    torch.cuda.empty_cache()
Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
