# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
source = parser.add_mutually_exclusive_group()
source.add_argument("--motion_dir", type=str, default=None)
source.add_argument("--motion_index", type=str, default=None)
source.add_argument("--motion_source", type=str, default=None)
parser.add_argument("--motion_root", type=str, default=None)
parser.add_argument("--motion_limit", type=int, default=None, help="explicit development subset, never an implicit memory fallback")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--fused_adam", type=str, default="on", choices=["on", "off"],
                    help="fused CUDA Adam kernels for all three optimizers (same update up to float rounding)")
parser.add_argument("--compile", type=str, default="auto", choices=["auto", "on", "off"], help="torch.compile (auto: only if triton is installed)")
parser.add_argument("--cuda_graph", type=str, default="auto", choices=["auto", "on", "off"],
                    help="replay the world-model step and the policy imagination+BPTT step as CUDA graphs "
                         "(auto: single CUDA process without torch.compile or SM_PROFILE; multi-GPU needs on)")
parser.add_argument("--cuda_graph_critic", type=str, default="on", choices=["on", "off"],
                    help="with CUDA graphs on, also replay the critic update (GAE targets and its minibatch steps)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_scope", type=str, default="global", choices=["global", "per_gpu"],
                    help="--num_envs/--wm_batch/--pi_batch are global totals split across ranks, or per-GPU sizes (paper Table A.5)")
parser.add_argument("--lr_scaling", type=str, default="none", choices=["none", "sqrt"],
                    help="multiply --lr by sqrt(WORLD_SIZE); the paper uses sqrt(2) x for two GPUs")
parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
args = parser.parse_args()

import importlib.util

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state
from supermimic.tasks.tracking.motion import MotionLib, check_reference, resolve_motion_args
import supermimic.tasks.tracking.rl.networks as networks
from supermimic.tasks.tracking.rl.buffer import ReplayBuffer
from supermimic.tasks.tracking.rl.networks import Critic, Policy, WorldModel
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.distributed import DistributedContext, average_gradients

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
parallel = DistributedContext(args.device)
dev = parallel.device
with parallel.coordinated("argument validation"):
    resume_ck = torch.load(args.resume, map_location="cpu") if args.resume else None
    resolve_motion_args(args, resume_ck["args"] if resume_ck else None)
    scope = parallel.world_size if args.batch_scope == "per_gpu" else 1
    global_envs = args.num_envs * scope
    local_envs = parallel.local_size(global_envs, "num_envs")
    local_wm_batch = parallel.local_size(args.wm_batch * scope, "wm_batch")
    local_pi_batch = parallel.local_size(args.pi_batch * scope, "pi_batch")
    base_lr = args.lr * (math.sqrt(parallel.world_size) if args.lr_scaling == "sqrt" else 1.0)
    if args.buffer_steps < args.wm_window or args.warmup_steps < args.wm_window:
        raise ValueError("buffer_steps and warmup_steps must be at least wm_window")
    if not args.no_critic and local_pi_batch * args.horizon < args.critic_minibatches:
        raise ValueError("local imagined batch is smaller than critic_minibatches")
parallel.require_equal({k: v for k, v in vars(args).items() if k != "local_rank"}, "training arguments")

with parallel.coordinated("reference loading"):
    kin = G1Kinematics(dev)
    motion = MotionLib.from_args(args, kin, dev, parallel.rank, parallel.world_size)
    if resume_ck:
        check_reference(motion.global_description, resume_ck.get("reference"))
    if not bool((motion.clip_len > args.horizon).any()):
        raise ValueError("this reference shard has no complete imagination horizon; use fewer ranks or a larger selection")
parallel.require_equal(motion.global_description["fingerprint"], "global reference fingerprint")
with parallel.coordinated("run metadata"):
    if parallel.primary:
        os.makedirs(args.log_dir, exist_ok=True)
        reference_path = Path(args.log_dir) / "reference.json"
        if reference_path.exists():
            check_reference(motion.global_description, json.loads(reference_path.read_text(encoding="utf-8")))
        reference_path.write_text(json.dumps(motion.global_description, indent=2, ensure_ascii=False), encoding="utf-8")
        json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w", encoding="utf-8"), indent=2)
        (Path(args.log_dir) / "shards" / f"world_{parallel.world_size}").mkdir(parents=True, exist_ok=True)
with parallel.coordinated("shard manifest"):
    shard_path = Path(args.log_dir) / "shards" / f"world_{parallel.world_size}" / f"rank_{parallel.rank:04d}.json"
    local_reference = {**motion.description, "runtime": {"device": dev, "local_envs": local_envs,
                        "local_wm_batch": local_wm_batch, "local_pi_batch": local_pi_batch,
                        "batch_scope": args.batch_scope, "global_envs": global_envs, "base_lr": base_lr}}
    shard_path.write_text(json.dumps(local_reference, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"rank {parallel.rank}/{parallel.world_size} on {dev}: {len(motion.names)} full clips, "
      f"{motion.num_frames} local frames, reference {motion.description['reference_tensor_bytes'] / 1024**3:.3f} GiB", flush=True)
mdp = MDP(kin, motion, kin.default_q)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
FUSED_ADAM = args.fused_adam == "on" and torch.device(dev).type == "cuda"
PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))
USE_GRAPH = args.cuda_graph == "on" or (
    args.cuda_graph == "auto" and torch.device(dev).type == "cuda" and not parallel.enabled and not PROFILE
    and not (args.compile == "on" or (args.compile == "auto" and importlib.util.find_spec("triton") is not None)))
if USE_GRAPH and torch.device(dev).type != "cuda":
    raise ValueError("--cuda_graph on needs CUDA")
# Inside graphs the gradient all-reduce uses the fixed global batch sizes (no host round trip).
GRAPH_WM_TOTAL = args.wm_batch * scope if USE_GRAPH else None
GRAPH_PI_TOTAL = args.pi_batch * scope if USE_GRAPH else None
GRAPH_CRITIC = USE_GRAPH and args.cuda_graph_critic == "on" and not args.no_critic


def split_sizes(n: int, parts: int) -> list[int]:
    """Chunk sizes of torch.tensor_split(range(n), parts)."""
    return [n // parts + int(i < n % parts) for i in range(parts)]


# global sample count of each critic minibatch: every rank splits its HZ * local_pi_batch rows the same way
GRAPH_CRITIC_TOTALS = [sum(col) for col in zip(*(
    split_sizes(args.horizon * parallel_batch, args.critic_minibatches)
    for parallel_batch in (args.pi_batch * scope // parallel.world_size + int(r < args.pi_batch * scope % parallel.world_size)
                           for r in range(parallel.world_size))))] if GRAPH_CRITIC else None


def make_adam(params, captured: bool, **kw):
    """Graph-captured optimizers keep their step count and learning rate on the device."""
    if captured:
        return torch.optim.Adam(params, lr=torch.tensor(base_lr, device=dev), fused=FUSED_ADAM, capturable=True, **kw)
    return torch.optim.Adam(params, lr=base_lr, fused=FUSED_ADAM, **kw)


opt_wm = make_adam(wm.parameters(), USE_GRAPH)
opt_pi = make_adam(pi.parameters(), USE_GRAPH, betas=(0.9, 0.95))
opt_v = make_adam(critic.parameters(), GRAPH_CRITIC)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = resume_ck
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    for opt, captured in ((opt_wm, USE_GRAPH), (opt_pi, USE_GRAPH), (opt_v, GRAPH_CRITIC)):
        for group in opt.param_groups:  # load_state_dict restores the checkpoint's own flags and lr type
            group["fused"], group["foreach"], group["capturable"] = FUSED_ADAM, None, captured
            if captured:
                group["lr"] = torch.tensor(float(group["lr"]), device=dev)
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
for model in (wm, pi, critic):
    parallel.broadcast_module(model)
if parallel.enabled:
    torch.manual_seed(args.seed + parallel.rank)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


USE_COMPILE = args.compile == "on" or (args.compile == "auto" and importlib.util.find_spec("triton") is not None)
maybe_compile = torch.compile if USE_COMPILE else (lambda fn: fn)
wm_rollout = maybe_compile(_wm_rollout)
wm_err = maybe_compile(_wm_err)
imag_step = maybe_compile(_imag_step)
reward_term = maybe_compile(_reward_term)
critic_input_c = maybe_compile(_critic_input)
print(f"torch.compile: {'on' if USE_COMPILE else 'off'}", flush=True)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return base_lr * 0.03
    return base_lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# Non-finite world-model/critic/policy gradients are flagged on the GPU and checked once per iteration:
# a host sync after each of the 10 world-model and 20 critic updates stalls the GPU queue.
# The check runs before logging and checkpointing, so a corrupted step aborts without being saved.
nonfinite = torch.zeros(3, dtype=torch.bool, device=dev)


class GraphedCall:
    """Run fn eagerly for its first calls, then capture it once as a CUDA graph and replay it.

    Tensor inputs are copied into static buffers before each replay and outputs are cloned, so a call
    returns what the eager call would. fn must keep static shapes and must not sync with the host;
    random ops draw from the default CUDA generator, which each replay advances like an eager call.
    The eager calls are real updates that also warm up cuBLAS and the allocator before capture.
    """

    def __init__(self, fn, eager_calls: int = 3, enabled: bool | None = None):
        self.fn, self.eager_calls, self.calls, self.graph = fn, eager_calls, 0, None
        self.enabled = USE_GRAPH if enabled is None else enabled

    def __call__(self, *inputs):
        if not self.enabled or self.calls < self.eager_calls:
            self.calls += 1
            return self.fn(*inputs)
        if self.graph is None:
            self.static_in = tree_map(torch.clone, inputs)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager calls' cached blocks before the graph takes its own pool
            self.graph = torch.cuda.CUDAGraph()
            # records only; the replay below performs this call. With NCCL, the communicator's watchdog thread
            # keeps querying CUDA events during capture, so only this thread's calls are checked.
            mode = "thread_local" if parallel.enabled else "global"
            with torch.cuda.graph(self.graph, capture_error_mode=mode):
                self.static_out = self.fn(*self.static_in)
        tree_map(lambda dst, src: dst.copy_(src), self.static_in, inputs)
        self.graph.replay()
        return tree_map(torch.clone, self.static_out)


# ------------------------------------------------------------------------------------------------
def _wm_step(s, a, xi, s1, mask):
    S_hat = wm_rollout(s, a, xi)
    loss, e_last = wm_err(S_hat, s1, mask)
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    average_gradients(wm, local_wm_batch, total_count=GRAPH_WM_TOTAL)
    grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    nonfinite[0].logical_or_(~torch.isfinite(grad_norm))
    opt_wm.step()
    m = mask[:, -1]
    error = parallel.reduce(torch.stack(((e_last.detach() * m).sum(), m.sum())))
    return loss.detach() / args.wm_window, error[0] / error[1].clamp(min=1)


wm_step = GraphedCall(_wm_step)


def wm_update():
    w = buffer.sample_windows(local_wm_batch, args.wm_window)
    return wm_step(w["s"], w["a"], w["xi"], w["s1"], w["mask"])


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    attempts = 0
    while n < B:
        attempts += 1
        if attempts > 256:
            raise RuntimeError("replay has too few starts with a complete imagination horizon; collect more transitions or inspect short clips")
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


def survival(d):
    """Soft survival weights c_{t+1} = prod_{i<=t} (1 - d_i) along the imagined horizon (dim 0).

    cumprod's backward branches on zeros in its input on the host, which a captured graph cannot do,
    so graphs use the same running product written out step by step.
    """
    if not USE_GRAPH:
        return torch.cumprod(1 - d, 0)
    out, c = [], torch.ones_like(d[0])
    for t in range(d.shape[0]):
        c = c * (1 - d[t])
        out.append(c)
    return torch.stack(out)


def _policy_step(s_flat, xi, f, a_prev, o_hist):
    """Imagine HZ steps through the frozen world model, backpropagate the return and step the policy."""
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    S = unflatten_state(s_flat)
    xi_n = normalize_xi(xi)
    B = local_pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0, f0 = S, f
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = survival(d)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    average_gradients(pi, local_pi_batch, total_count=GRAPH_PI_TOTAL)
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    finite_pi_grad = torch.isfinite(gn)
    nonfinite[2].logical_or_(~finite_pi_grad)
    # A captured graph cannot branch on the host, so it always steps; a non-finite gradient still aborts
    # the run at the end of this iteration, before anything is logged or saved.
    if USE_GRAPH or bool(finite_pi_grad):
        opt_pi.step()
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
        g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
    tq = (tq0, tq1, tq2) if PROFILE else ()
    return (loss.detach(), gn.detach(), (r_sum / HZ / 0.1).mean(), c.mean().detach(), rews, dones, xi_n,
            S_c, f_c, g_c) + tq


policy_step = GraphedCall(_policy_step)


def _critic_step(rews, dones, xi_n, S_c, f_c, g_c):
    """Fit the critic to GAE(lambda) targets over the imagined transitions (Eq. 3)."""
    B = local_pi_batch
    with torch.no_grad():
        xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(X)
        R = V[-1]
        targets = []
        for t in range(HZ - 1, -1, -1):
            R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        targets = torch.stack(targets[::-1])  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    v_loss = torch.zeros(())
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i, idx in enumerate(torch.tensor_split(perm, args.critic_minibatches)):
            v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            average_gradients(critic, len(idx), total_count=GRAPH_CRITIC_TOTALS[i] if GRAPH_CRITIC else None)
            grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            nonfinite[1].logical_or_(~torch.isfinite(grad_norm))
            opt_v.step()
    return v_loss.detach()


critic_step = GraphedCall(_critic_step, enabled=GRAPH_CRITIC)


def policy_update():
    with parallel.coordinated("policy replay sampling"):
        s, h = sample_valid_starts(local_pi_batch)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    loss, gn, imag_r, surv, rews, dones, xi_n, S_c, f_c, g_c, *tq = policy_step(s["s"], s["xi"], s["f"], s["a_prev"], o_hist)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = critic_step(rews, dones, xi_n, S_c, f_c, g_c) if not args.no_critic else torch.zeros(())
    if PROFILE:
        torch.cuda.synchronize()
        tq0, tq1, tq2 = tq
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss, v_loss, imag_r, gn, surv


# ------------------------------------------------------------------------------------------------
env = TrackingSim(local_envs, motion, mdp, dev, gamma=args.gamma, parallel=parallel)
buffer = ReplayBuffer(args.buffer_steps, local_envs, dev)
obs_hist = torch.zeros(local_envs, Hh, OBS_DIM, device=dev)


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        S = env.state()
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += global_envs
        stats["r"].append(out["r"].mean() / 0.1)
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].sum())
            stats["ep_ret"].append(out["ep_ret_norm"].sum())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())

stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
# The replay buffer is not checkpointed, so resumed runs also need fresh transitions.
collect(args.warmup_steps, stats)
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a", encoding="utf-8") if parallel.primary else None
if parallel.primary and log_f.tell() == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            if torch.is_tensor(gparam["lr"]):
                gparam["lr"].fill_(lr)
            else:
                gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
    if prof:
        torch.cuda.synchronize(); tp0 = time.time()
    collect(n, stats)
    if prof:
        torch.cuda.synchronize(); tp1 = time.time()
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    if prof:
        torch.cuda.synchronize(); tp2 = time.time()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn, surv = policy_update()
        if prof:
            torch.cuda.synchronize(); tp3 = time.time()
            print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if bool(nonfinite.any()):
        which = [name for name, bad in zip(("world-model", "critic", "policy"), nonfinite.tolist()) if bad]
        raise FloatingPointError(f"non-finite globally averaged {' and '.join(which)} gradient at iteration {it + 1}")
    if (it + 1) % args.log_every == 0:
        keys = list(agg) + ["r"]
        values = [agg[k] if k in agg else stats[k] for k in keys]
        weights = [local_wm_batch if k in ("wm", "wm8") else local_envs if k == "r" else local_pi_batch for k in keys]
        totals = parallel.reduce([[sum(float(v) for v in vs) * weight, len(vs) * weight]
                                  for vs, weight in zip(values, weights)]).cpu()
        means = {key: float(pair[0] / pair[1].clamp(min=1)) for key, pair in zip(keys, totals)}
        episodes = parallel.reduce([sum(float(v) for v in stats[key]) for key in ("ep_len", "ep_ret", "term", "done")]).cpu()
        timing = parallel.reduce([time.time() - t_log, time.time() - t_start], torch.distributed.ReduceOp.MAX).cpu()
        ips = args.log_every / float(timing[0])
        t_log = time.time()
        wall_h = float(timing[1]) / 3600
        term_frac = float(episodes[2] / episodes[3].clamp(min=1))
        row = [it + 1, wall_h, sim_steps, *[means[k] for k in keys],
               float(episodes[0] / episodes[3].clamp(min=1)), float(episodes[1] / episodes[3].clamp(min=1)), term_frac, ips, lr]
        if parallel.primary:
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                  f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        with parallel.coordinated("checkpoint write"):
            if parallel.primary:
                ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                      "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                      "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": motion.global_description,
                      "distributed": {"world_size": parallel.world_size, "partition_algorithm": motion.partition["algorithm"]}}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "reference", "distributed")}
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)
if log_f:
    log_f.close()
parallel.close()
