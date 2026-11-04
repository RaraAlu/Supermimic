"""按源分片对检查点做全库附录F.1评估（每段从首帧确定性播放一次，无随机化与噪声），再汇总成一份报告。

evaluate.py要求参考库与检查点的数据指纹相同，全库又放不进一次评估，所以逐个源分片调用diag_first_step.py
（播放与失败口径同evaluate.py），已有summary.json的分片跳过，中断后重跑即可续上。汇总时按clips.csv把各分片的
per_clip.csv对回片段名；给了--select_index时只统计其中选入训练的片段，--exclude的片段另报告去掉它们后的结果。

在仓库根目录运行；Rara上的全库（30个分片，约11.4万段）单卡约1.1 h：
    python scripts/eval_f1_shards.py --ckpt ckpt_218000.pt --source_dir D:/datasets/pass_hdf5_source/boneseed \
        --select_index D:/datasets/paired_50hz/index.h5 --out f1_218k \
        --exclude fall_from_2m_panic_180_R_003__A304 fall_from_2m_panic_180_R_003__A304_M jump_to_shoulder_roll_R_003__A304_M
只汇总已有结果：加--aggregate_only。输出目录下写report.txt与aggregate.json。
"""

import argparse
import collections
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="checkpoint to evaluate (not needed with --aggregate_only)")
parser.add_argument("--source_dir", required=True, help="directory with shard_*.h5 and the clips.csv that maps them to clip names")
parser.add_argument("--out", required=True, help="output directory; one shard_<n>/ per source shard")
parser.add_argument("--select_index", default=None,
                    help="index.h5 whose clips (g1_h5, g1_source_clip_index) are the training selection; default: every clip")
parser.add_argument("--exclude", nargs="*", default=(), help="clip names (file stems) also reported as excluded")
parser.add_argument("--shards", nargs="*", default=None, help="shard stems to run (default: every shard_*.h5)")
parser.add_argument("--aggregate_only", action="store_true", help="skip evaluation and only aggregate existing shards")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--src", default=None, help="source tree passed to diag_first_step.py (default: its own default)")
parser.add_argument("--goal_tail", default=None, choices=["clamp", "hold", "stand", "auto"],
                    help="passed to diag_first_step.py (default: the checkpoint's setting)")
args = parser.parse_args()

source, out = Path(args.source_dir), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
shards = args.shards or sorted(p.stem for p in source.glob("shard_*.h5"))
if not shards:
    sys.exit(f"no shard_*.h5 in {source}")

if not args.aggregate_only:
    if not args.ckpt:
        sys.exit("--ckpt is required unless --aggregate_only")
    script = Path(__file__).with_name("diag_first_step.py")
    for shard in shards:
        d = out / shard
        if (d / "summary.json").exists():
            continue
        cmd = [sys.executable, "-B", str(script), "--ckpt", args.ckpt, "--motion_source", str(source / f"{shard}.h5"),
               "--out", str(d), "--device", args.device] + (["--src", args.src] if args.src else []) \
            + (["--goal_tail", args.goal_tail] if args.goal_tail else [])
        t0 = time.time()
        with open(out / f"{shard}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{shard}: exit {code}, {time.time() - t0:.0f} s", flush=True)
        if code:
            sys.exit(f"{shard} failed; see {out / (shard + '.log')}")

# ---------------- 汇总 ----------------
keep = None
if args.select_index:
    import h5py

    with h5py.File(args.select_index, "r") as f:
        keep = set(zip(f["clips"]["g1_h5"].asstr()[:], (int(i) for i in f["clips"]["g1_source_clip_index"][:])))
by_shard = collections.defaultdict(list)
for r in csv.DictReader(open(source / "clips.csv", encoding="utf-8")):
    by_shard[r["h5"]].append(r)
exclude = set(args.exclude)

recs, done, unselected, iterations, tails = [], [], 0, set(), set()
for shard in shards:
    d = out / shard
    if not (d / "per_clip.csv").exists():
        continue
    summary_json = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    iterations.add(summary_json.get("iteration"))
    tails.add(summary_json.get("subject", {}).get("goal_tail") or "clamp")
    meta = sorted(by_shard[f"{shard}.h5"], key=lambda r: int(r["clip_index"]))
    clips = list(csv.DictReader(open(d / "per_clip.csv", encoding="utf-8")))
    if len(clips) != len(meta):
        sys.exit(f"{shard}: {len(clips)} evaluated clips but {len(meta)} rows in clips.csv")
    done.append(shard)
    for m, c in zip(meta, clips):
        # per_clip.csv按分片内的片段顺序排列，名字只是序号；逐行核对长度，防止对错行
        if int(c["clip_len"]) != int(m["frames"]):
            sys.exit(f"{shard} clip {m['clip_index']}: evaluated {c['clip_len']} frames, clips.csv says {m['frames']}")
        if keep is not None and (f"{shard}.h5", int(m["clip_index"])) not in keep:
            unselected += 1
            continue
        name = m["relative_csv"]
        recs.append({"name": name, "cat": name.split("/")[0], "len": int(c["clip_len"]), "failed": int(c["failed"]),
                     "trunc": int(c.get("goal_truncated") or 0), "step": int(c.get("fail_step") or 0), "norm": float(c["norm"]),
                     "excluded": name.rsplit("/", 1)[-1].rsplit(".", 1)[0] in exclude})
if not recs:
    sys.exit(f"no evaluated shard in {out}")

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def summary(sel, label):
    """失败数（另计不含目标截断的）与归一化回报（按段平均、按转移数加权）。"""
    n = len(sel)
    fails = sum(r["failed"] for r in sel)
    guided = sum(r["failed"] and not r["trunc"] for r in sel)
    mean = sum(r["norm"] for r in sel) / max(n, 1)
    framew = sum(r["norm"] * (r["len"] - 1) for r in sel) / max(sum(r["len"] - 1 for r in sel), 1)
    say(f"{label}：{n} 段，失败 {fails}（{100 * fails / max(n, 1):.4f}%），不计目标截断 {guided}，"
        f"归一化回报 {mean:.4f}（按帧加权 {framew:.4f}）")
    return {"clips": n, "failed": fails, "failed_guided": guided, "norm_return": round(mean, 4), "norm_return_framew": round(framew, 4)}


say(f"检查点迭代 {', '.join(str(i) for i in sorted(iterations, key=str))}；分片 {len(done)}/{len(shards)} 个"
    + (f"；未选入训练 {unselected} 段" if keep is not None else "") + f"；末尾目标 {'/'.join(sorted(tails))}")
if len(iterations) > 1 or len(tails) > 1:
    say("警告：各分片来自不同的检查点或末尾目标设置")
res = {"iteration": sorted(iterations, key=str), "goal_tail": sorted(tails), "shards": done, "all": summary(recs, "全部")}
if exclude:
    res["without_excluded"] = summary([r for r in recs if not r["excluded"]], f"去掉--exclude的{len(exclude)}段")
    say("--exclude：" + "，".join(f"{r['name'].rsplit('/', 1)[-1]} 失败={r['failed']} 回报={r['norm']:.3f}"
                                   for r in recs if r["excluded"]))
cat_n, cat_f, cat_r = collections.Counter(), collections.Counter(), collections.Counter()
for r in recs:
    cat_n[r["cat"]] += 1
    cat_f[r["cat"]] += r["failed"]
    cat_r[r["cat"]] += r["norm"]
say()
say(f"{'类别':28s} {'段数':>7s} {'失败':>5s} {'失败率%':>8s} {'回报':>7s}")
for c in sorted(cat_n, key=lambda c: (-cat_f[c], c)):
    say(f"{c:28s} {cat_n[c]:7d} {cat_f[c]:5d} {100 * cat_f[c] / cat_n[c]:8.3f} {cat_r[c] / cat_n[c]:7.4f}")
failed = sorted((r for r in recs if r["failed"]), key=lambda r: r["norm"])
say()
say(f"失败片段 {len(failed)} 段（按回报升序；终止步为终止转移的序号，距末为它到片段末帧的步数）：")
for r in failed:
    tag = "目标截断" if r["trunc"] else ("排除" if r["excluded"] else "有引导")
    say(f"  {r['norm']:.3f}  {r['len']:5d}帧  {tag:4s} 终止步{r['step']:5d} 距末{r['len'] - 1 - r['step']:5d}  {r['name']}")
res["by_category"] = {c: [cat_n[c], cat_f[c], round(cat_r[c] / cat_n[c], 4)] for c in cat_n}
res["failed"] = [r["name"] for r in failed]
res["failed_goal_truncated"] = [r["name"] for r in failed if r["trunc"]]
(out / "aggregate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
