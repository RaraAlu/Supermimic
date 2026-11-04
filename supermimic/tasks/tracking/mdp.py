# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""模拟器和世界模型共用的解析 MDP 计算（论文附录 A）。

状态元组 S = (p, R, v, w, q, qd) 依次为骨盆位置、旋转矩阵、骨盆连杆在世界系的线速度与角速度，以及关节位置和速度。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.tasks.tracking.constants import TRACKED_BODIES
from supermimic.tasks.tracking.motion import CONTROL_DT, NUM_J, MotionLib
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.rotations import mat_to_6d, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat

STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J  # flatten_state的排列：p、四元数、v、w、q、qd
OBS_DIM = NUM_J + NUM_J + 3 + 3 + NUM_J + 6
# 仿真上下文 xi = [摩擦系数, 关节零位偏移, 躯干质心偏移]
XI_JOINT = slice(1, 1 + NUM_J)
XI_COM = slice(1 + NUM_J, 4 + NUM_J)
XI_DIM = 4 + NUM_J

# 表 A.1 的传感器噪声（均匀分布半宽），以及相对参考的朝向
OBS_NOISE = torch.cat(
    (
        torch.full((NUM_J,), 0.01),
        torch.full((NUM_J,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(NUM_J),
        torch.full((6,), 0.05),
    )
)
# 策略输入前的观测归一化（关节速度和角速度数值较大）
OBS_SCALE = torch.cat(
    (torch.ones(NUM_J), torch.full((NUM_J,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(NUM_J), torch.ones(6))
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    """保存跟踪奖励、惩罚和终止条件使用的参数。"""
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> torch.Tensor:
    """将状态元组拼接为张量，旋转矩阵转换为四元数。"""
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    """将扁平状态张量还原为位置、旋转、速度和关节状态元组。"""
    qd0 = 13 + NUM_J
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:qd0], x[..., qd0:STATE_DIM]


def invariant_features(S) -> torch.Tensor:
    """提取与朝向和平移无关的状态特征：高度、六维倾斜、机体系速度和关节状态。"""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J


class MDP:
    """为真实模拟与想象推演提供一致的观测、奖励、终止和价值特征。"""
    def __init__(self, kin: G1Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg = RewardCfg()):
        """保存运动学模型、参考动作、默认关节位置和奖励配置。"""
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.obs_scale = OBS_SCALE.to(dev)

    # ---------------- 观测 ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """生成无噪声本体观测，包括相对于参考的根部朝向。"""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., XI_JOINT]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        """按观测各分量的噪声尺度添加均匀随机噪声。"""
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- 奖励与终止 ----------------
    def _anchored_ref_bodies(self, p, R, rp, rR, f):
        """重新锚定到机器人根部（xy与偏航）的参考刚体位置，同BeyondMimic；另返回偏航差dR。"""
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (self.motion.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        return ref_bp, dR

    def reward(self, S, f, a, a_prev, terms: bool = False):
        """比较状态 S 与参考帧 f，计算已乘以 dt 的单步奖励。"""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp, dR = self._anchored_ref_bodies(p, R, rp, rR, f)
        # 刚体位置与姿态按当前机器人水平位置和偏航重新锚定，衡量局部动作形态；
        # 根部误差仍对照原始参考轨迹，保留对全局位置和朝向的约束。
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        # 各误差经指数核映射为有界跟踪奖励；标准差控制容忍范围，根部两项权重各为 0.5。
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        # 动作变化惩罚抑制控制抖动，软关节限位惩罚在触及硬限位之前生效；
        # 最后统一乘控制步长，使累计回报对应时间积分。
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        """按机器人与参考状态的偏差计算硬终止标志。"""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_survival(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """计算想象推演中本步未终止的可微概率。

        使用 sigmoid(-x) 而非 1 - sigmoid(x)：在 fp32 下，后者在 x 约大于 17 时因相消舍入为零并失去梯度（Barron 等，A Fix for Softmax and Cross-Entropy，arXiv:2610.31642）。训练直接累乘这些存活概率。"""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return torch.sigmoid((self.cfg.term_height - e_z) / tz) * torch.sigmoid((self.cfg.term_tilt - e_tilt) / tt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """计算可微终止概率，即 1 - soft_survival。"""
        return 1 - self.soft_survival(S, f, tz, tt)

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """构建价值网络输入：不变状态、目标、模拟上下文和机器人相对参考的误差。"""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp = self._anchored_ref_bodies(p, R, rp, rR, f)[0]
        # 把跟踪误差投到当前偏航系，使价值网络能识别偏离参考的程度而不依赖绝对朝向。
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)


CRITIC_EXTRA = NUM_J + len(TRACKED_BODIES) * 3 + 3 + 6 + XI_DIM


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """按预设尺度归一化摩擦、关节偏移和躯干质心偏移。"""
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., XI_JOINT] / 0.01, xi[..., XI_COM] / 0.05), -1)
