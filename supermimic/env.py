# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Motion-tracking simulator on mjlab (MuJoCo Warp).

The environment is a thin batched wrapper over mjlab's ``Scene`` and ``Simulation``: it applies the
joint setpoint, steps MuJoCo Warp, labels the transition with the analytic reward/termination of
:mod:`supermimic.mdp`, and resets episodes to reference-motion states drawn by return-based bin
sampling (paper Appendix D.1). mjlab's manager-based env is not used, because SuperMimic needs
raw state access, arbitrary state resets and its own replay buffer.
"""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import FOOT_GEOMS, G1_JOINT_NAMES, action_scale, robot_cfg
from .rotations import mat_to_quat, quat_to_mat, so3_exp

NUM_SUBSTEPS = 4
PHYSICS_DT = CONTROL_DT / NUM_SUBSTEPS  # 5 ms


def make_sim_cfg() -> SimulationCfg:
    # Solver iterations follow mjlab's G1 tracking task; contact/constraint capacity follows the
    # reference implementation (rare overflows during falls with smaller buffers).
    return SimulationCfg(
        nconmax=128,
        njmax=500,
        mujoco=MujocoCfg(
            timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", iterations=10, ls_iterations=20
        ),
    )


class TrackingSim:
    """Batched G1 motion-tracking simulator."""

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, device: str, gamma: float = 0.99,
                 eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True,
                 self_collision: bool = False):  # fmt: skip
        self.device = dev = device
        scene_cfg = SceneCfg(
            num_envs=num_envs,
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg(self_collision)},
        )
        self.scene = Scene(scene_cfg, device=dev)
        self.sim = Simulation(num_envs=num_envs, cfg=make_sim_cfg(), spec=self.scene.spec, device=dev)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        # per-env friction and torso CoM (the expansion re-captures the CUDA graphs)
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
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, 33, device=dev)
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
            self.xi[:, 30:33] = com
            default_com = self.sim.get_default_field("body_ipos")[self._torso_body]
            self.sim.model.body_ipos[:, self._torso_body] = default_com + com
            self.sim.recompute_constants(RecomputeLevel.set_const)
        # Foot geoms have a higher contact priority than the ground, so their friction alone
        # sets the contact friction (no max-combination with the ground as in the reference).
        self.sim.model.geom_friction[:, self._foot_geoms, 0] = self.xi[:, :1]

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ state
    def state(self):
        """Root state read from qpos/qvel directly, so it is exact right after a step or reset."""
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        root = qpos[:, self._q_adr]
        rvel = qvel[:, self._v_adr]
        R = quat_to_mat(root[:, 3:7])  # mjlab quaternions are wxyz
        w = (R @ rvel[:, 3:6].unsqueeze(-1)).squeeze(-1)  # free-joint angular velocity is in the body frame
        return (root[:, :3].clone(), R, rvel[:, :3].clone(), w, qpos[:, self._jq_adr].clone(), qvel[:, self._jv_adr].clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
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
        # Same order as mjlab's env reset: clear the MuJoCo Warp data (warm starts, contacts), then the
        # entity state, which zeroes the actuator targets, so both run before the new state is written.
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self.robot.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)

    def refresh(self):
        """Recompute derived quantities (body poses, contacts) after writing states."""
        self.sim.forward()

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(target)
        for _ in range(NUM_SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
