# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""基于 mjlab 编译后的 G1 模型，用 PyTorch 实现可微正向运动学。

参考动作、记录的模拟器状态和世界模型预测共用这套运动学计算，以保证解析奖励与模拟器转移的奖励标签一致。"""

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
    """从 G1 模型构造可微的批量正向运动学计算。"""
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES, model: mujoco.MjModel | None = None):
        """读取模型的刚体树、关节轴、初始关节位置和关节限位，并放到指定设备。"""
        cfg = get_g1_robot_cfg()
        m = Entity(cfg).compile() if model is None else model
        self.device = device
        self.default_q = torch.zeros(len(G1_JOINT_NAMES), dtype=torch.float32, device=device)
        ids, _, values = resolve_matching_names_values(cfg.init_state.joint_pos, G1_JOINT_NAMES)
        self.default_q[ids] = torch.tensor(values, dtype=torch.float32, device=device)
        jname_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        # 按树顺序处理刚体（MuJoCo 将父节点排在子节点之前），跳过世界节点（编号 0）
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
                # 仅支持原点处的单铰链结构；提前拒绝其他关节，避免后续速度递推使用错误假设。
                if jid >= 0:
                    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise RuntimeError(f"joint of {name} is not a hinge")
                    if np.abs(m.jnt_pos[jid]).max() > 1e-9:
                        raise RuntimeError(f"joint of {name} is not at the body origin")
                    joint_idx.append(jname_to_idx[m.joint(jid).name])
                else:
                    joint_idx.append(-1)  # 固定刚体：只有偏移
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
        # 关节限位（按 G1_JOINT_NAMES 顺序）
        lo, hi = np.zeros(len(G1_JOINT_NAMES)), np.zeros(len(G1_JOINT_NAMES))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """计算刚体在世界坐标系中的位置、旋转、线速度（连杆原点）和角速度。

        p、R、v、w 分别是根部（骨盆）位置 (B,3)、旋转 (B,3,3) 及世界系线速度和角速度 (B,3)；q、qd 是按 G1_JOINT_NAMES 排列的关节位置和速度 (B,29)。all_bodies 为真时返回所有刚体，否则只返回跟踪刚体。"""
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
            # 连杆原点的线速度包含父体平移与父体旋转产生的叉积项；固定连杆也需此项。
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            if j < 0:
                P[i], Rs[i], V[i], W[i] = pi, R_off, vi, W[par]
                continue
            # 关节轴先经安装姿态转换到世界系，再叠加关节转速；位置不因该转动改变，
            # 因为构造时已要求铰链位于连杆原点。
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
