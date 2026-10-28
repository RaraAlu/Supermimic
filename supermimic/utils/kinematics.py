# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Differentiable forward kinematics of the G1 in torch, built from mjlab's compiled G1 model.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.string import resolve_matching_names_values

from supermimic.tasks.tracking.constants import G1_JOINT_NAMES, TRACKED_BODIES
from supermimic.utils.rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES, model: mujoco.MjModel | None = None):
        cfg = get_g1_robot_cfg()
        m = Entity(cfg).compile() if model is None else model
        self.device = device
        self.default_q = torch.zeros(len(G1_JOINT_NAMES), dtype=torch.float32, device=device)
        ids, _, values = resolve_matching_names_values(cfg.init_state.joint_pos, G1_JOINT_NAMES)
        self.default_q[ids] = torch.tensor(values, dtype=torch.float32, device=device)
        jname_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        # bodies in tree order (mujoco orders parents before children); world (id 0) is skipped
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] > 1:
                raise RuntimeError(f"body {name} has {m.body_jntnum[b]} joints; FK supports at most one")
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            jid = m.body_jntadr[b] if m.body_jntnum[b] == 1 else -1
            if jid >= 0 and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                if pb not in mj_to_local:
                    raise RuntimeError(f"parent of {name} is not a robot body")
                parent.append(mj_to_local[pb])
                if jid >= 0:
                    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise RuntimeError(f"joint of {name} is not a hinge")
                    if np.abs(m.jnt_pos[jid]).max() > 1e-9:
                        raise RuntimeError(f"joint of {name} is not at the body origin")
                    joint_idx.append(jname_to_idx[m.joint(jid).name])
                else:
                    joint_idx.append(-1)  # fixed body: offset only
            axes.append(m.jnt_axis[jid].copy() if jid >= 0 else np.array([0.0, 0.0, 1.0]))
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        if parent.count(-1) != 1 or parent[0] != -1:
            raise RuntimeError("expected a single free-floating root body")
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in G1_JOINT_NAMES order)
        lo, hi = np.zeros(len(G1_JOINT_NAMES)), np.zeros(len(G1_JOINT_NAMES))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (pelvis) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,29) in G1_JOINT_NAMES order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            if j < 0:
                P[i], Rs[i], V[i], W[i] = pi, R_off, vi, W[par]
                continue
            Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            wi = W[par] + axis_w * qd[..., j : j + 1]
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
