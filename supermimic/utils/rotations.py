# 移植自 zita-ch/supermimic_vibed（MIT 许可，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""批量旋转工具。

约定：
- 四元数按 wxyz 顺序排布，形状 (..., 4)；
- 旋转矩阵作用于列向量（p' = R p），形状 (..., 3, 3)；
- 支持前置批量维度；输入形状需满足各函数约定，退化输入与分支边界不保证梯度良好。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """四元数 (..., 4) -> 旋转矩阵 (..., 3, 3)。

    公式要求单位四元数，因此先归一化；未归一化时结果通常不正交，也不只是等比缩放。
    9 个元素按行优先堆叠后一次 reshape 成 3x3，全程向量化，无显式批量循环。
    """
    q = F.normalize(q, dim=-1)  # 入口统一单位化，防止模长漂移破坏正交性

    w, x, y, z = q.unbind(-1)

    # 与教科书公式逐项对应（# fmt: skip 保留三行三列的视觉排布，便于核对正负号）
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 四元数 (..., 4)（wxyz，且 w >= 0）。

    四个分量的幅值用对角元素的组合分别算出，各作主元构造候选，再选择幅值最大者对应的候选，
    避免 trace 接近 -1 时除以小量造成精度损失；最后归一化并翻到 w >= 0 半球。
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # w、x、y、z 的幅值（clamp 为开方与后续除法设置正下界）
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    # 分别以四个分量为主元构造候选，随后选择主元幅值最大的一组
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)  # 逐元素选主元
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)  # q 与 -q 同一旋转，统一取 w >= 0


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """轴角 -> 旋转矩阵：轴 (..., 3) 为固定单位向量，角度 (...)，输出 (..., 3, 3)。

    即 Rodrigues 旋转公式的显式展开：R = I + sin(t)[u]x + (1-cos(t))[u]x^2。
    """
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
    """旋转向量 (..., 3) -> 旋转矩阵 (..., 3, 3)，指数映射 exp([v]x)。

    小角度分支用泰勒展开替代 sin(t)/t 与 (1-cos(t))/t^2，保证 t -> 0 处光滑无 0/0。
    """
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)  # R = I + a·[v]x + b·[v]x^2


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 旋转向量 (..., 3)，对数映射。

    经由四元数中转：比直接对 trace 做 acos 在 0 与 pi 附近都更稳定。
    """
    q = mat_to_quat(R)
    return quat_to_rotvec(q)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """四元数 (..., 4) -> 旋转向量 (..., 3)（轴角表示，轴乘以角度）。

    用半角的 atan2 求角度；小角度走泰勒近似，避免除以接近 0 的 sin_half。
    """
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)  # 翻到 w >= 0 半球，保证取短弧
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])  # atan2 在 0 和 pi 附近都稳定
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def skew(v: torch.Tensor) -> torch.Tensor:
    """向量 (..., 3) -> 反对称矩阵 (..., 3, 3)，满足 [v]x u = v × u。"""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def rot_angle_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """两旋转之间的平方测地角 (...,)，可微，可直接作损失。

    角度由 tr(Ra^T Rb) = 1 + 2cos(t) 反解，acos 前先截断到 (-1, 1) 内以保证定义域与梯度。
    """
    tr = (Ra * Rb).sum((-1, -2))  # Ra^T Rb 的迹
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos) ** 2


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """从旋转矩阵 (..., 3, 3) 提取朝向（仅偏航）旋转 (..., 3, 3)。

    把 x 轴投影到水平面后归一化，即得绕 z 轴的朝向角（俯仰为 0）。
    """
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 6D 表示 (..., 6)：前两列展平。

    连续表示，常作网络输出目标；使用时需再经 Gram-Schmidt 正交化还原成旋转。
    """
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """四元数球面线性插值：单位四元数 q0, q1 为 (..., 4)，t 为 (...)，输出 (..., 4)。

    t 对应前置批量维度，函数内部会追加一维，不应额外传入末尾单例维度。
    dot < 0 时翻转 q1 取短弧；两向量近乎平行（sin(theta) ≈ 0）时退化为线性插值，避免除以 0。
    """
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # 取短弧：q 与 -q 表示同一旋转
    dot = dot.abs().clamp(max=1.0)  # acos 定义域保护
    theta = torch.acos(dot)
    s = torch.sin(theta)
    t = t[..., None]
    w0 = torch.where(s < 1e-6, 1 - t, torch.sin((1 - t) * theta) / (s + 1e-12))
    w1 = torch.where(s < 1e-6, t, torch.sin(t * theta) / (s + 1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)  # 出口再单位化，防止漂移
