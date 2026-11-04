# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""固定大小的 GPU 经验回放缓冲区，按（时间、环境）排列模拟器转移。"""

from __future__ import annotations

import torch

from supermimic.tasks.tracking.mdp import NUM_J, STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (NUM_J, torch.float32),
    "a_prev": (NUM_J, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    """按时间循环覆盖转移数据，并支持采样连续窗口和想象推演起点。"""

    def __init__(self, T: int, N: int, device: str):
        """为 N 个环境分配 T 个时间步的存储空间。"""
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        """写入所有环境的一步转移，写满后覆盖最旧的时间步。"""
        for k in FIELDS:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """将按新旧排序的时间索引映射到循环缓冲区行号，0 表示最旧。"""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """采样 B 个长度为 k 的窗口，返回转移张量及形状为（B, k）的有效掩码。"""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        # 窗口内固定环境编号，仅沿时间前进；先构造逻辑时间，再映射到环形存储行号。
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        # 只有此前的转移均未结束回合，第 j 个转移才有效
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """采样想象推演的起点及其观测历史，历史按从旧到新排列。"""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # 历史偏移从 hist-1 递减到 0，按从旧到新排列
        # 历史不足时重复最早可用状态补齐长度，同时受回合起点和回放保留范围两重约束。
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # 不跨越回合边界
        back = torch.minimum(back, age.unsqueeze(-1))  # 不越过缓冲区中最旧的记录
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
