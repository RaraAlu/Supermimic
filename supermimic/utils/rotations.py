# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""批量旋转工具：四元数按 wxyz 排列，旋转矩阵作用于列向量。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """将批量四元数归一化后转换为旋转矩阵，保留前面的批量维度。"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """稳定地将旋转矩阵转换为四元数，结果按 wxyz 排列且 w 非负。"""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    # 分别假设四元数的某一分量为主分量构造候选；随后选绝对值最大的分量作分母，
    # 避免接近 180 度时仅用实部分量计算所造成的数值不稳定。
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """用 Rodrigues 公式将单位旋转轴（..., 3）和角度（...）转换为矩阵。"""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))  # fmt: skip


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """将旋转向量（..., 3）映射为矩阵，并在零角度附近使用平滑近似。"""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    # 小角度处用泰勒展开消除 sin(theta)/theta 和 (1-cos(theta))/theta² 的相消误差。
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """通过四元数将矩阵转换为旋转向量，在零和 π 附近保持稳定。"""
    q = mat_to_quat(R)
    return quat_to_rotvec(q)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """将单位四元数转换为旋转向量，并选择非负实部对应的表示。"""
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def skew(v: torch.Tensor) -> torch.Tensor:
    """构造向量的反对称矩阵，使矩阵乘法等价于叉积。"""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def rot_angle_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """计算两个旋转之间的测地角平方；截断余弦值以保持梯度稳定。"""
    tr = (Ra * Rb).sum((-1, -2))  # 相对旋转矩阵 Ra^T Rb 的迹
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos) ** 2


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """将 x 轴投影到水平面，从旋转矩阵提取仅含偏航角的朝向。"""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """展平旋转矩阵的前两列，得到六维连续旋转表示。"""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """沿最短弧球面插值四元数；夹角很小时退化为线性插值。"""
    dot = (q0 * q1).sum(-1, keepdim=True)
    # q 与 -q 表示同一旋转；先统一到同一半球，确保插值沿较短弧进行。
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    t = t[..., None]
    w0 = torch.where(s < 1e-6, 1 - t, torch.sin((1 - t) * theta) / (s + 1e-12))
    w1 = torch.where(s < 1e-6, t, torch.sin(t * theta) / (s + 1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)
