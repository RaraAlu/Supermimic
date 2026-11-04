# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""世界模型、策略（动作编码器与共享解码器）和价值网络，见论文表 A.4。"""

from __future__ import annotations

import torch
import torch.nn as nn

from supermimic.tasks.tracking.mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from supermimic.utils.rotations import so3_exp, so3_log, yaw_mat
from supermimic.utils.distributed import global_moments


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    """按给定的隐藏层宽度构建多层感知机，可在激活函数前添加层归一化。"""
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """维护运行均值和方差，无梯度地更新，样本数达到 ``max_count`` 后冻结。"""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        """初始化均值、方差、样本数及冻结阈值。"""
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """合并各进程的样本统计量，并原位更新归一化缓冲区。"""
        x = x.reshape(-1, x.shape[-1]).float()
        # 在 GPU 上判断是否冻结，避免每个模拟步同步主机；原位写入以保持 CUDA 图使用的地址有效。
        active = self.count < self.max_count
        n, m, v = global_moments(x)
        tot = self.count + n
        # 合并新旧总体方差时额外计入均值差的平方项，不能直接平均两批方差。
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.mean.copy_(torch.where(active, mean, self.mean))
        self.var.copy_(torch.where(active, var, self.var))
        self.count.copy_(torch.where(active, tot, self.count))

    def forward(self, x):
        """使用当前均值和带数值稳定项的标准差归一化输入。"""
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        """返回加入数值稳定项后的标准差。"""
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# 仅对 MLP 矩阵乘法启用 bf16 自动混合精度；状态积分、正向运动学和奖励保持 fp32
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """按全局精度设置运行网络，并在混合精度下将输出转回 fp32。"""
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """确定性动力学模型 s_{t+1} = f(s_t, a_t, xi)，预测局部坐标系中的状态增量。"""

    def __init__(self, hidden=(2048, 2048, 2048)):
        """创建状态增量预测网络及其输入、输出归一化器。"""
        super().__init__()
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        # 输出层从零开始，使初始归一化增量为零，避免随机大幅状态跳变。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n):
        """拼接不变状态特征、动作和归一化的模拟参数。"""
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """计算从 S 到 S1 的局部坐标系增量，作为世界模型的回归目标。"""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        # 位置增量投到起点偏航系；旋转、线速度和角速度增量使用起点完整机体系。
        # 该约定必须与 step 中的逆变换一致。
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n):
        """预测并积分局部状态增量，返回下一状态。"""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        # 网络预测归一化增量，先还原物理尺度，再按各分量所属坐标系积分。
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        # 旋转增量通过指数映射右乘到起点姿态，保持旋转矩阵结构，而非直接相加矩阵元素。
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """以真实状态S为锚点的局部系状态误差，即从S到S_hat的增量。"""
        return WorldModel.targets(S, S_hat)


class Policy(nn.Module):
    """确定性策略 a = dec(enc(g), o_{t-9:t})；潜变量由 tanh 限制在有界范围内。"""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32):
        """创建目标编码器、共享动作解码器及观测和目标归一化器。"""
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        # 缩小动作输出层的初始权重，让策略初期动作接近默认关节目标，减少随机输出造成的扰动。
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """根据目标和带噪声的观测历史生成动作；历史形状为（B, hist, obs_dim），最新值在末尾。"""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    """对归一化输入估计标量状态价值。"""
    def __init__(self, in_dim: int):
        """创建价值网络及输入归一化器。"""
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        """返回输入批次中每个状态的价值估计。"""
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
