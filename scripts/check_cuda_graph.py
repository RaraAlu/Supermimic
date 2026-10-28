"""Check that CUDA-graph replay of the world-model, policy and critic steps and of the simulator's reward and
termination matches eager execution bitwise.

Builds the training state with ``supermimic.scripts.train.setup(... --cuda_graph on)`` (a short warmup
collection only), then on each rank runs one eager step and one graph replay from identical model, optimizer
and RNG state and compares outputs and updated parameters. Extra arguments go to the training script.

    python scripts/check_cuda_graph.py --motion_limit 8
    torchrun --standalone --nproc-per-node=2 scripts/check_cuda_graph.py --motion_limit 8 --batch_scope per_gpu

Exits non-zero if any comparison differs.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils._pytree import tree_map

from supermimic.scripts import train
from supermimic.tasks.tracking.mdp import NUM_J, unflatten_state
from supermimic.tasks.tracking.rl.trainer import set_requires_grad
from supermimic.utils.cuda_graph import GraphedCall

args, parallel, trainer = train.setup(["--num_envs", "512", "--buffer_steps", "100", "--wm_batch", "512", "--pi_batch", "512",
                                       "--warmup_steps", "40", "--log_dir", os.path.join(REPO, "logs", "check_cuda_graph"),
                                       *sys.argv[1:], "--iterations", "0", "--cuda_graph", "on"])
wm, pi, critic, opt_wm, opt_pi = trainer.wm, trainer.pi, trainer.critic, trainer.opt_wm, trainer.opt_pi
rank = parallel.rank


def snapshot(module, opt):
    tensors_ = [t.detach().clone() for t in module.state_dict(keep_vars=True).values()]  # parameters and buffers
    state = {id(p): {k: v.clone() for k, v in opt.state[p].items()} for p in module.parameters() if p in opt.state}
    return tensors_, state


def restore(module, opt, snap):
    tensors_, state = snap
    with torch.no_grad():
        for t, v in zip(module.state_dict(keep_vars=True).values(), tensors_):
            t.copy_(v)
        for p in module.parameters():  # in place: the captured graph holds these exact tensors
            for k, v in state.get(id(p), {}).items():
                opt.state[p][k].copy_(v)


def tensors(x):
    out = []
    tree_map(lambda t: out.append(t) if torch.is_tensor(t) else None, x)
    return out


def compare(name, module, opt, inputs, pre=lambda: None, post=lambda: None):
    fn = getattr(trainer, f"_{name}")
    pre(); fn(*inputs); post()  # one real step first, so the optimizer state exists
    snap, rng = snapshot(module, opt), torch.cuda.get_rng_state()
    pre(); eager_out = fn(*inputs); post()
    eager_params = [t.detach().clone() for t in module.state_dict(keep_vars=True).values()]
    graphed = trainer.graphed(fn, eager_calls=0)
    restore(module, opt, snap); torch.cuda.set_rng_state(rng)
    pre(); first_out = graphed(*inputs); post()  # capture, then replay
    first_ok = all(torch.equal(a, b) for a, b in zip(tensors(eager_out), tensors(first_out)))
    first_ok &= all(torch.equal(a, b.detach()) for a, b in zip(eager_params, module.state_dict(keep_vars=True).values()))
    restore(module, opt, snap); torch.cuda.set_rng_state(rng)
    pre(); graph_out = graphed(*inputs); post()  # replay from the eager call's starting state
    outs = all(torch.equal(a, b) for a, b in zip(tensors(eager_out), tensors(graph_out)))
    params = all(torch.equal(a, b.detach()) for a, b in zip(eager_params, module.state_dict(keep_vars=True).values()))
    print(f"rank {rank} {name}: first capture+replay bitwise equal {first_ok}, outputs bitwise equal {outs}, updated parameters and buffers bitwise equal {params}", flush=True)
    return first_ok and outs and params


w = trainer.buffer.sample_windows(trainer.layout.local_wm_batch, args.wm_window)
ok = compare("wm_step", wm, opt_wm, (w["s"], w["a"], w["xi"], w["s1"], w["mask"]))
s, h = trainer.sample_valid_starts(trainer.layout.local_pi_batch)
with torch.no_grad():
    o_hist = trainer.mdp.add_noise(trainer.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]))
sr = set_requires_grad
ok &= compare("policy_step", pi, opt_pi, (s["s"], s["xi"], s["f"], s["a_prev"], o_hist),
              pre=lambda: (sr(wm, False), sr(critic, False)), post=lambda: (sr(wm, True), sr(critic, True)))
if trainer.graph_critic:
    sr(wm, False), sr(critic, False)
    out = trainer._policy_step(s["s"], s["xi"], s["f"], s["a_prev"], o_hist)
    sr(wm, True), sr(critic, True)
    rews, keeps, xi_n, S_c, f_c, g_c = out[4:10]
    ok &= compare("critic_step", critic, trainer.opt_v, (rews, keeps, xi_n, S_c, f_c, g_c))
env = trainer.env
if trainer.collect_action.enabled:
    ok &= compare("collect_action", pi, opt_pi,
                  (env.state(), env.a_prev, env.xi, env.f, env.ep_step, trainer.obs_hist))
# Compare the actual post-processing on both sides of reset, using the same transition and the same
# starting normalizer/buffer state. Replaying physics would introduce simulator nondeterminism.
buffer, store = trainer.buffer, trainer.store_transition
norms = (wm.in_norm, wm.out_norm)
overlap_same = True
for _ in range(3):
    saved = [{k: v.clone() for k, v in norm.state_dict().items()} for norm in norms]
    ptr, size = buffer.ptr, buffer.size
    action = torch.randn(env.N, NUM_J, device=env.device)
    with torch.no_grad():
        transition = env.step(action, after_step=store)
    expected = [{k: v.clone() for k, v in norm.state_dict().items()} for norm in norms]
    row = {k: v[ptr].clone() for k, v in buffer.data.items()}
    for norm, state in zip(norms, saved):
        norm.load_state_dict(state)
    buffer.ptr, buffer.size = ptr, size
    store(transition)
    overlap_same &= all(torch.equal(v, state[k]) for norm, state in zip(norms, expected)
                        for k, v in norm.state_dict().items())
    overlap_same &= all(torch.equal(buffer.data[k][ptr], v) for k, v in row.items())
print(f"rank {rank} collection overlap: replay row and world-model normalizers bitwise equal {overlap_same}", flush=True)
ok &= overlap_same
if env._reward_termination.enabled:
    S1, f1 = env.state(), env.f + 1
    a, a_prev = torch.randn(env.N, NUM_J, device=env.device), env.a_prev.clone()
    eager_out = env._reward_termination_eager(S1, f1, a, a_prev)
    graphed = GraphedCall(env._reward_termination_eager, eager_calls=0, thread_local=parallel.enabled)
    graphed(S1, f1, a, a_prev)  # capture, then replay
    graph_out = graphed(S1, f1, a, a_prev)
    same = all(torch.equal(x, y) for x, y in zip(tensors(eager_out), tensors(graph_out)))
    print(f"rank {rank} env reward/termination: outputs bitwise equal {same} ({len(tensors(eager_out))} tensors)", flush=True)
    ok &= same
trainer.close()
parallel.close()
sys.exit(0 if ok else 1)
