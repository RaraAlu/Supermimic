# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Check FK vs simulator, state round-trip, and reward on reference playback."""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

import torch

from supermimic.tasks.tracking.constants import TRACKED_BODIES
from supermimic.tasks.tracking.env import TrackingSim
from supermimic.tasks.tracking.mdp import MDP
from supermimic.tasks.tracking.motion import MotionLib
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.rotations import quat_to_mat, rot_angle_sq

dev = args.device
kin = G1Kinematics(dev)
t0 = time.time()
motion = MotionLib.from_dir(os.path.join(REPO, "data", "lafan1_g1"), kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, kin.default_q)
t0 = time.time()
env = TrackingSim(args.num_envs, motion, mdp, dev, eval_mode=True)
print(f"sim init {time.time() - t0:.1f}s, cuda graphs: {env.sim.use_cuda_graph}")
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
env.reset(torch.arange(args.num_envs, device=dev), frames)
env.refresh()
S = env.state()
ref = motion.ref_state(frames)
names = ["p", "R", "v", "w", "q", "qd"]
for n, a, b in zip(names, S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
# FK vs sim body frames
d = env.robot.data
ids = env.robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0]
sp = d.body_link_pose_w[:, ids, :3]
sR = quat_to_mat(d.body_link_pose_w[:, ids, 3:7])
bp, bR, bv, bw = kin.forward(*S)
print("FK pos err", (bp - sp).norm(dim=-1).max().item())
print("FK rot err", rot_angle_sq(bR, sR).sqrt().max().item())
sv = d.body_link_vel_w[:, ids]
print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
# open-loop reference playback via joint targets
rs = []
t0 = time.time()
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    rs.append(out["r"].mean().item() / 0.1)
    if t % 25 == 0:
        print(t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())
if dev.startswith("cuda"):
    torch.cuda.synchronize()
print(f"playback: {150 * args.num_envs / (time.time() - t0):.0f} control steps/s at {args.num_envs} envs")
