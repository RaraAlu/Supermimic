"""按附录F.1条件（首帧起步、确定性、无随机化与噪声）重放指定片段，记录策略与参考的根位姿和关节角，供render_clips.py渲染。

片段可直接用--clips给出（clips.csv的relative_csv或文件名），也可取JSON文件里的名单，例如eval_f1_shards.py的
aggregate.json中的失败片段（--from_json aggregate.json --key failed）。片段按clips.csv找到所在的源分片，同一分片的
片段一批重放；结束（失败或播完）后继续记录--extra步，看清摔倒过程。在仓库根目录运行，--src 指定被测源码树：
    python scripts/record_clips.py --ckpt CKPT --source_dir SHARDS --from_json f1/aggregate.json --out traj.npz
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import _subject

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=_subject.DEFAULT_SRC, help=_subject.SRC_HELP)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--source_dir", required=True, help="directory with the source shards and their clips.csv")
parser.add_argument("--clips", nargs="*", default=(), help="clip names: relative_csv from clips.csv, or file stems")
parser.add_argument("--from_json", default=None, help="JSON file with a list of clip names under --key")
parser.add_argument("--key", default="failed", help="list key in --from_json (default: the failed clips of aggregate.json)")
parser.add_argument("--out", required=True, help="output .npz")
parser.add_argument("--extra", type=int, default=100, help="control steps recorded after each clip ends")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
subject = _subject.Subject(args.src, args.device)

import gc

import numpy as np
import torch

from supermimic.utils.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True  # 与diag_first_step.py一致

src = Path(args.source_dir)
wanted = list(args.clips)
if args.from_json:
    wanted += json.loads(Path(args.from_json).read_text(encoding="utf-8"))[args.key]
if not wanted:
    raise SystemExit("no clips: pass --clips and/or --from_json")
stem = lambda name: name.rsplit("/", 1)[-1].rsplit(".", 1)[0]  # noqa: E731
rows = list(csv.DictReader(open(src / "clips.csv", encoding="utf-8")))
by_name = {r["relative_csv"]: r for r in rows} | {stem(r["relative_csv"]): r for r in rows}
missing = [n for n in wanted if n not in by_name]
if missing:
    raise SystemExit(f"not in {src / 'clips.csv'}: {', '.join(missing)}")
order = {}  # relative_csv -> 输出中的序号，保持给出的顺序并去重
for n in wanted:
    order.setdefault(by_name[n]["relative_csv"], len(order))
where = collections.defaultdict(list)
for name in order:
    r = by_name[name]
    where[r["h5"]].append((int(r["clip_index"]), name))

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
subject.set_amp(a_)
pi = subject.policy(ck)
out = {}
for h5, items in sorted(where.items()):
    motion, mdp = subject.reference(str(src / h5), a_)
    sel = torch.tensor([c for c, _ in items], device=args.device)
    for c, name in items:  # 分片内片段名为clip_序号，与clips.csv的clip_index对应
        assert motion.names[c] in (f"clip_{c:06d}", Path(name).stem), (h5, c, motion.names[c], name)
    n = len(items)
    env = subject.make_env(n, motion, mdp, eval_mode=True)
    ids = torch.arange(n, device=args.device)
    clip_end = motion.clip_end[sel]
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[sel].clone())
    rec = {"root_p": [], "root_quat": [], "q": [], "f": []}
    done_at = torch.full((n,), -1, dtype=torch.long, device=args.device)
    hist, step = None, 0
    with torch.no_grad():
        while True:
            S = env.state()
            rec["root_p"].append(S[0].cpu()), rec["root_quat"].append(mat_to_quat(S[1]).cpu())
            rec["q"].append(S[4].cpu()), rec["f"].append(env.f.cpu())
            if bool(env.done_eval.all()) and bool((step - done_at >= args.extra).all()):
                break
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            env.step(pi(hist, motion.goal(env.f, K)))
            step += 1
            done_at = torch.where(env.done_eval & (done_at < 0), step, done_at)
            env.f = torch.minimum(env.f, clip_end - 1)
    T = {k: torch.stack(v).numpy() for k, v in rec.items()}
    for j, (c, name) in enumerate(items):
        s0, L = int(motion.clip_start[c]), int(motion.clip_len[c])
        fr = torch.arange(s0, s0 + L, device=args.device)
        end = int(done_at[j]) + args.extra + 1
        out[name] = {
            "rob_p": T["root_p"][:end, j], "rob_quat": T["root_quat"][:end, j], "rob_q": T["q"][:end, j],
            "rob_f": T["f"][:end, j] - s0,
            "ref_p": motion.root_p[fr].cpu().numpy(), "ref_quat": mat_to_quat(motion.root_R[fr]).cpu().numpy(),
            "ref_q": motion.q[fr].cpu().numpy(), "done_step": int(done_at[j]), "failed": bool(env.failed_eval[j]),
        }
        print(f"{h5} {name} len={L} done_step={int(done_at[j])} failed={bool(env.failed_eval[j])}", flush=True)
    del env, motion, mdp
    gc.collect()
    torch.cuda.empty_cache()

flat = {"names": np.array(sorted(out, key=order.get)), "iteration": ck.get("iteration"),
        "subject": json.dumps(subject.provenance())}
for i, name in enumerate(flat["names"]):
    for k, v in out[name].items():
        flat[f"{i}_{k}"] = v
np.savez_compressed(args.out, **flat)
print("saved", args.out, len(out))
