"""Check that CUDA-graph replay of the world-model step and the policy step matches eager execution bitwise.

Builds the training state with ``supermimic.scripts.train --iterations 0 --cuda_graph on`` (a short warmup
collection only), then on each rank runs one eager step and one graph replay from identical model, optimizer
and RNG state and compares outputs and updated parameters. Extra arguments go to the training script.

    python scripts/check_cuda_graph.py --motion_limit 8
    torchrun --standalone --nproc-per-node=2 scripts/check_cuda_graph.py --motion_limit 8 --batch_scope per_gpu

Exits non-zero if any comparison differs.
"""

import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils._pytree import tree_map

from supermimic.utils.distributed import DistributedContext

# train.py tears down the process group when it finishes; keep it alive for the comparisons below
close = DistributedContext.close
DistributedContext.close = lambda self: None

sys.argv = ["train.py", "--num_envs", "512", "--buffer_steps", "100", "--wm_batch", "512", "--pi_batch", "512",
            "--warmup_steps", "40", "--log_dir", os.path.join(REPO, "logs", "check_cuda_graph"),
            *sys.argv[1:], "--iterations", "0", "--cuda_graph", "on"]
g = runpy.run_module("supermimic.scripts.train", run_name="__main__")
wm, pi, critic, opt_wm, opt_pi = g["wm"], g["pi"], g["critic"], g["opt_wm"], g["opt_pi"]
rank = g["parallel"].rank


def snapshot(module, opt):
    params = [p.detach().clone() for p in module.parameters()]
    state = {id(p): {k: v.clone() for k, v in opt.state[p].items()} for p in module.parameters() if p in opt.state}
    return params, state


def restore(module, opt, snap):
    params, state = snap
    with torch.no_grad():
        for p, v in zip(module.parameters(), params):
            p.copy_(v)
        for p in module.parameters():  # in place: the captured graph holds these exact tensors
            for k, v in state.get(id(p), {}).items():
                opt.state[p][k].copy_(v)


def tensors(x):
    out = []
    tree_map(lambda t: out.append(t) if torch.is_tensor(t) else None, x)
    return out


def compare(name, module, opt, inputs, pre=lambda: None, post=lambda: None):
    fn = g[f"_{name}"]
    pre(); fn(*inputs); post()  # one real step first, so the optimizer state exists
    snap, rng = snapshot(module, opt), torch.cuda.get_rng_state()
    pre(); eager_out = fn(*inputs); post()
    eager_params = [p.detach().clone() for p in module.parameters()]
    graphed = g["GraphedCall"](fn, eager_calls=0)
    restore(module, opt, snap); torch.cuda.set_rng_state(rng)
    pre(); graphed(*inputs); post()  # capture, then replay
    restore(module, opt, snap); torch.cuda.set_rng_state(rng)
    pre(); graph_out = graphed(*inputs); post()  # replay from the eager call's starting state
    outs = all(torch.equal(a, b) for a, b in zip(tensors(eager_out), tensors(graph_out)))
    params = all(torch.equal(a, b.detach()) for a, b in zip(eager_params, module.parameters()))
    print(f"rank {rank} {name}: outputs bitwise equal {outs}, updated parameters bitwise equal {params}", flush=True)
    return outs and params


w = g["buffer"].sample_windows(g["local_wm_batch"], g["args"].wm_window)
ok = compare("wm_step", wm, opt_wm, (w["s"], w["a"], w["xi"], w["s1"], w["mask"]))
s, h = g["sample_valid_starts"](g["local_pi_batch"])
with torch.no_grad():
    o_hist = g["mdp"].add_noise(g["mdp"].obs(g["unflatten_state"](h["s"]), h["a_prev"], h["xi"], h["f"]))
sr = g["set_requires_grad"]
ok &= compare("policy_step", pi, opt_pi, (s["s"], s["xi"], s["f"], s["a_prev"], o_hist),
              pre=lambda: (sr(wm, False), sr(critic, False)), post=lambda: (sr(wm, True), sr(critic, True)))
close(g["parallel"])
sys.exit(0 if ok else 1)
