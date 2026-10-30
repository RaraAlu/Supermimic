# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""基于 mjlab（MuJoCo Warp）的动作跟踪模拟器。

环境对 mjlab 的 Scene 和 Simulation 做轻量批量封装：应用关节目标、推进物理仿真、使用 mdp 中的解析奖励和终止条件标注转移，并按回报分箱采样参考状态来重置回合（论文附录 D.1）。这里不使用 mjlab 的管理器环境，因为训练需要直接访问状态、任意重置状态和自有经验回放缓冲区。"""

from __future__ import annotations

import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from supermimic.tasks.tracking.constants import FOOT_GEOMS, G1_JOINT_NAMES
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, MDP, NUM_J, XI_COM, XI_DIM, XI_JOINT, flatten_state
from supermimic.tasks.tracking.motion import CONTROL_DT, MotionLib
from supermimic.tasks.tracking.reset_sampling import BinSampler
from supermimic.utils.cuda_graph import GraphedCall
from supermimic.utils.rotations import mat_to_quat, quat_to_mat, so3_exp

NUM_SUBSTEPS = 4
PHYSICS_DT = CONTROL_DT / NUM_SUBSTEPS  # 5 ms


def make_sim_cfg() -> SimulationCfg:
    # 求解器迭代次数沿用 mjlab 的 G1 跟踪任务；接触与约束容量沿用参考实现，
    # 较小的缓冲区在跌倒时偶尔会溢出。
    """构建模拟配置，包括控制步长、求解器迭代次数和接触容量。"""
    return SimulationCfg(
        nconmax=128,
        njmax=500,
        mujoco=MujocoCfg(
            timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", iterations=10, ls_iterations=20
        ),
    )


class TrackingSim:
    """批量运行 G1 动作跟踪模拟。"""

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, device: str, gamma: float = 0.99,
                 eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True,
                 parallel=None, graph_reward: bool = False, bin_clip_end: str = "bootstrap",
                 bin_prior: str = "zero", bin_skip_steps: int = 50, reset_margin: int = 100, bin_clip_cap: float = 0.0):  # fmt: skip
        """bin_clip_end、bin_prior、bin_skip_steps、reset_margin与bin_clip_cap是重置分箱的选项，含义见reset_sampling.BinSampler。"""
        self.device = dev = device
        self.parallel = parallel if parallel is not None and parallel.enabled else None
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        # 先建分箱采样器：选项不合法时在建场景之前报错
        self.sampler = BinSampler(motion, dev, gamma, self.parallel, clip_end=bin_clip_end, prior=bin_prior,
                                  skip_steps=bin_skip_steps, reset_margin=reset_margin, max_steps=self.max_episode_steps,
                                  clip_cap=bin_clip_cap)
        scene_cfg = SceneCfg(
            num_envs=num_envs,
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
        )
        self.scene = Scene(scene_cfg, device=dev)
        self.sim = Simulation(num_envs=num_envs, cfg=make_sim_cfg(), spec=self.scene.spec, device=dev)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        # 各环境的摩擦系数和躯干质心（扩展会重新捕获 CUDA 图）
        self.sim.expand_model_fields(("geom_friction", "body_ipos"))
        idx = self.robot.indexing
        self._q_adr = idx.free_joint_q_adr
        self._v_adr = idx.free_joint_v_adr
        self._jq_adr = idx.joint_q_adr
        self._jv_adr = idx.joint_v_adr
        self._foot_geoms = idx.geom_ids[self.robot.find_geoms(FOOT_GEOMS)[0]]
        self._torso_body = int(idx.body_ids[self.robot.find_bodies("torso_link")[0][0]])

        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        # 每步奖励与终止包含数百个小型正向运动学内核，可按需合并为一个 CUDA 图重放
        self._reward_termination = GraphedCall(self._reward_termination_eager, enabled=graph_reward,
                                               thread_local=self.parallel is not None)
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        ids, _, scales = resolve_matching_names_values(G1_ACTION_SCALE, G1_JOINT_NAMES)
        if ids != list(range(len(G1_JOINT_NAMES))):
            raise RuntimeError("mjlab G1 action scales do not cover every motion-data joint")
        self.alpha = torch.tensor(scales, device=dev)

        # ---- 模拟上下文 xi = [摩擦系数, 关节偏移, 躯干质心偏移(3)] ----
        self.xi = torch.zeros(self.N, XI_DIM, device=dev)
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, XI_COM] = com
            default_com = self.sim.get_default_field("body_ipos")[self._torso_body]
            self.sim.model.body_ipos[:, self._torso_body] = default_com + com
            self.sim.recompute_constants(RecomputeLevel.set_const)
        # 足部几何体的接触优先级高于地面，因此接触摩擦只由足部决定，
        # 不会像参考实现那样与地面摩擦取最大值。
        self.sim.model.geom_friction[:, self._foot_geoms, 0] = self.xi[:, :1]

        # ---- 回合统计 ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, NUM_J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.start_f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        # 前 bin_skip_steps 步（重置瞬态）的折扣与未折扣奖励和
        self.ep_disc_head = torch.zeros(self.N, device=dev)
        self.ep_ret_head = torch.zeros(self.N, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.sampler.prepare(True)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ 状态
    def state(self):
        """直接从 qpos/qvel 读取根部状态，因此在物理步或重置后立即准确。"""
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        root = qpos[:, self._q_adr]
        rvel = qvel[:, self._v_adr]
        R = quat_to_mat(root[:, 3:7])  # mjlab 四元数按 wxyz 排列
        w = (R @ rvel[:, 3:6].unsqueeze(-1)).squeeze(-1)  # 自由关节角速度位于机体坐标系
        return (root[:, :3].clone(), R, rvel[:, :3].clone(), w, qpos[:, self._jq_adr].clone(), qvel[:, self._jv_adr].clone())

    # ------------------------------------------------------------------ 重置
    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        """将指定环境重置到给定或采样的参考帧，并初始化回合状态。"""
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sampler.sample(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        # 复制参考状态后才添加重置扰动，避免原位操作污染所有环境共用的动作库。
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # 按 mjlab 环境的重置顺序先清除 MuJoCo Warp 数据（预热起点、接触），再清除实体状态。
        # 后者会将执行器目标置零，因此两者都必须在写入新状态之前执行。
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self.robot.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        # 同步参考帧、动作历史及回合累计量；起始分箱保持到回合结束，用于将回报归给正确起点。
        self.f[env_ids] = frames
        self.start_f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.ep_disc_head[env_ids] = 0.0
        self.ep_ret_head[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, XI_JOINT] = torch.empty(n, NUM_J, device=dev).uniform_(-0.01, 0.01)

    def _reward_termination_eager(self, S1, f1, a, a_prev):
        """使用解析 MDP 计算下一状态的奖励及终止标志。"""
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        return r, terms, self.mdp.termination(S1, f1)

    def refresh(self):
        """写入状态后重新计算刚体位姿、接触等派生量。"""
        self.sim.forward()

    # ------------------------------------------------------------------ 推进
    def step(self, a: torch.Tensor, after_step=None) -> dict:
        """推进物理仿真，并可在重置等待主机之前排入转移处理。

        after_step 可以读取已分离的转移并更新训练统计，但不能修改环境、转移张量或随机数生成器状态，也不能按进程执行不同工作。回合汇总字段在回调之后添加。"""
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        # 策略输出先按关节尺度转成位置目标，再叠加默认姿态与本回合关节偏移；
        # 一个控制动作在多个物理子步内保持不变。
        target = self.default_q + xi[:, XI_JOINT] + self.alpha * a
        self.robot.set_joint_position_target(target)
        for _ in range(NUM_SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        # 读取重置前的真实后继状态，并与下一参考帧比较。
        # 失败终止与时间截断都结束回合，但只有前者代表跟踪失败。
        S1 = self.state()
        f1 = f + 1
        r, terms, term = self._reward_termination(S1, f1, a, a_prev)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        at_clip_end = f1 >= clip_end
        timeout = at_clip_end | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # 更新回合统计
        disc_r = (self.gamma ** ep_step.float()) * r
        self.ep_disc_ret += disc_r
        self.ep_ret += r
        head = ep_step < self.sampler.skip_steps
        self.ep_disc_head += torch.where(head, disc_r, 0.0)
        self.ep_ret_head += torch.where(head, r, 0.0)
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if after_step is not None:
            after_step(out)
        # 评估时只累积结束/失败标志，不自动抽取新起点；外层评估器负责屏蔽已结束环境。
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            T = self.ep_step[ids].float()
            self.sampler.record(self.start_f[ids], self.start_bin[ids], self.ep_disc_ret[ids], self.ep_ret[ids],
                                self.ep_disc_head[ids], self.ep_ret_head[ids], T, term[ids], at_clip_end[ids], MAX_STEP_REWARD)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
        # 先统一各卡课程概率再重置，即使本卡没有结束回合，也必须参与其他卡触发的同步。
        self.sampler.prepare(len(ids) > 0)
        self.reset(ids)
        return out
