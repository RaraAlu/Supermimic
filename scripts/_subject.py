"""诊断脚本的被测源码树（--src），对所有版本提供同一套接口。

诊断脚本是测量工具，始终保持本仓库的版本；被测的supermimic包可以来自另一个提交的源码树。
凡是取决于被测版本的逻辑都集中在本模块：脚本只使用各版本都有的名称，supermimic本身也无需知道诊断脚本的存在。
旧源码树缺少的名称（GOAL_FRAME_DIM、soft_survival、较新的TrackingSim选项）在这里回退到旧树自身的算法。
"""

import inspect
import subprocess
import sys
from pathlib import Path

DEFAULT_SRC = str(Path(__file__).resolve().parents[1])
SRC_HELP = "source tree to import supermimic from (default: this repository; pass a checkpoint's commit tree to replay it)"


class Subject:
    def __init__(self, src: str, device: str):
        self.src = Path(src).resolve()
        sys.path.insert(0, str(self.src))
        import supermimic

        # 确认实际载入的是--src指定的源码树，而不是别处的同名包
        loaded = Path(supermimic.__file__).resolve().parents[1]
        if loaded != self.src:
            raise ImportError(f"supermimic was imported from {loaded}, not from --src {self.src}")
        self.device = device
        self.env_options = {}
        # dirty表示工作区有未提交的修改，此时commit不足以复现被测代码
        try:
            git = lambda *a: subprocess.check_output(["git", *a], cwd=self.src, text=True, stderr=subprocess.DEVNULL)  # noqa: E731
            self.commit = git("rev-parse", "HEAD").strip()
            self.dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        except (OSError, subprocess.CalledProcessError):
            self.commit = self.dirty = None

        from supermimic.tasks.tracking import mdp, motion

        self.STATE_DIM = mdp.STATE_DIM
        self.XI_DIM = mdp.XI_DIM
        self.GOAL_FRAME_DIM = getattr(motion, "GOAL_FRAME_DIM", 73)

    def provenance(self) -> dict:
        """被测的源码与仿真选项，写入各脚本的summary.json。"""
        return {"src": str(self.src), "commit": self.commit, "dirty": self.dirty, "env_options": self.env_options}

    # ---------------- 参考动作与MDP ----------------
    def reference(self, motion_source: str):
        """合并源文件的MotionLib及其上的MDP。"""
        from supermimic.tasks.tracking.mdp import MDP
        from supermimic.tasks.tracking.motion import MotionLib
        from supermimic.utils.kinematics import G1Kinematics

        kin = G1Kinematics(self.device)
        motion = MotionLib.from_source(motion_source, kin, self.device)
        return motion, MDP(kin, motion, kin.default_q)

    @staticmethod
    def survival_prob(mdp):
        """按源码树自身的算法给出每步存活概率：4293e90之前的源码树只有soft_termination，训练用1减去它。"""
        if hasattr(mdp, "soft_survival"):
            return mdp.soft_survival
        return lambda S, f: 1 - mdp.soft_termination(S, f)

    # ---------------- 网络 ----------------
    def set_amp(self, run_args: dict):
        """该run训练时用了bf16，则MLP同样用bf16 autocast。"""
        import torch

        import supermimic.tasks.tracking.rl.networks as networks

        networks.AMP_DTYPE = torch.bfloat16 if run_args.get("amp", "none") == "bf16" else None

    def goal_dim(self, run_args: dict) -> int:
        return self.GOAL_FRAME_DIM * run_args["goal_frames"]

    def policy(self, ck: dict):
        from supermimic.tasks.tracking.mdp import OBS_DIM
        from supermimic.tasks.tracking.rl.networks import Policy

        return self._load(Policy(OBS_DIM, self.goal_dim(ck["args"]), hist=ck["args"]["hist"]), ck["pi"])

    def world_model(self, ck: dict):
        from supermimic.tasks.tracking.rl.networks import WorldModel

        return self._load(WorldModel(), ck["wm"])

    def critic(self, ck: dict):
        from supermimic.tasks.tracking.mdp import CRITIC_EXTRA, INV_DIM
        from supermimic.tasks.tracking.rl.networks import Critic

        return self._load(Critic(INV_DIM + CRITIC_EXTRA + self.goal_dim(ck["args"])), ck["critic"])

    def _load(self, module, state):
        module = module.to(self.device)
        module.load_state_dict(state)
        return module.eval()  # 归一化统计量保持检查点中的值；训练时才会继续更新

    @staticmethod
    def bin_state(env) -> dict:
        """环境的重置分箱状态（分数、概率、访问数）：分箱移入BinSampler之前的源码树直接挂在TrackingSim上，
        更早的源码树没有概率与访问数，对应项为None。"""
        sampler = getattr(env, "sampler", None)
        if sampler is not None:
            return {"bin_G": sampler.scores, "bin_probs": sampler.probs, "bin_visits": sampler.visits}
        return {"bin_G": env.bin_G, "bin_probs": getattr(env, "bin_probs", None), "bin_visits": getattr(env, "bin_visits", None)}

    # ---------------- 仿真环境 ----------------
    def make_env(self, num_envs, motion, mdp, *, run_args=None, inherit=(), **options):
        """源码树中的TrackingSim。

        options是显式设定：源码树的TrackingSim不接受其中某项时直接报错，否则测量条件会与要求不符而无人察觉。
        inherit列出从检查点run_args继承的选项，仅当检查点有该项且源码树接受时才传入，其余保持源码树的默认值。
        实际传入的选项记在env_options中，由provenance()写出。
        """
        from supermimic.tasks.tracking.env import TrackingSim

        accepted = inspect.signature(TrackingSim.__init__).parameters
        unsupported = sorted(k for k in options if k not in accepted)
        if unsupported:
            raise TypeError(f"TrackingSim in {self.src} does not accept {', '.join(unsupported)}")
        inherited = {k: run_args[k] for k in inherit if run_args and k in run_args and k in accepted}
        skipped = [k for k in inherit if k not in inherited]
        if skipped:
            print(f"[subject] tree default kept for {', '.join(skipped)} (not in the checkpoint or not accepted by the tree)",
                  flush=True)
        kw = {**inherited, **options}
        self.env_options = dict(kw)
        return TrackingSim(num_envs, motion, mdp, self.device, **kw)
