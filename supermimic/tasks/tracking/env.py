# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Motion-tracking simulator on mjlab (MuJoCo Warp).

The environment is a thin batched wrapper over mjlab's ``Scene`` and ``Simulation``: it applies the
joint setpoint, steps MuJoCo Warp, labels the transition with the analytic reward/termination of
:mod:`supermimic.tasks.tracking.mdp`, and resets episodes to reference-motion states drawn by return-based bin
sampling (paper Appendix D.1). mjlab's manager-based env is not used, because SuperMimic needs
raw state access, arbitrary state resets and its own replay buffer.
"""

from __future__ import annotations

import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from supermimic.tasks.tracking.constants import FOOT_GEOMS, G1_JOINT_NAMES
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, MDP, flatten_state
from supermimic.tasks.tracking.motion import CONTROL_DT, MotionLib
from supermimic.utils.cuda_graph import GraphedCall
from supermimic.utils.distributed import update_bin_returns
from supermimic.utils.rotations import mat_to_quat, quat_to_mat, so3_exp

NUM_SUBSTEPS = 4
PHYSICS_DT = CONTROL_DT / NUM_SUBSTEPS  # 5 ms
TAIL_STEPS = 100  # resets this close to the clip end (2 s) are reported as tail resets
BIN_CLIP_END = ("bootstrap", "terminal")
BIN_PRIOR = ("zero", "mean")


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
                 parallel=None, graph_reward: bool = False, bin_clip_end: str = "bootstrap",
                 bin_prior: str = "zero"):  # fmt: skip
        """bin_clip_end: "bootstrap" extends episodes that reach the clip end with their mean reward and
        scores bins against the global maximum return; "terminal" ends the return at the clip end and
        scores each episode against the maximum attainable from its start frame (only the 500-step
        cap is bootstrapped). bin_prior: unvisited bins score 0 ("zero") or the mean of visited bins
        ("mean", whose first episode then replaces the prior instead of being blended into it)."""
        if bin_clip_end not in BIN_CLIP_END or bin_prior not in BIN_PRIOR:
            raise ValueError(f"bin_clip_end must be one of {BIN_CLIP_END} and bin_prior one of {BIN_PRIOR}")
        self.bin_clip_end, self.bin_prior = bin_clip_end, bin_prior
        self.device = dev = device
        self.parallel = parallel if parallel is not None and parallel.enabled else None
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
        # reward and termination of each step: hundreds of small FK kernels, replayed as one CUDA graph if asked
        self._reward_termination = GraphedCall(self._reward_termination_eager, enabled=graph_reward,
                                               thread_local=self.parallel is not None)
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        ids, _, scales = resolve_matching_names_values(G1_ACTION_SCALE, G1_JOINT_NAMES)
        if ids != list(range(len(G1_JOINT_NAMES))):
            raise RuntimeError("mjlab G1 action scales do not cover every motion-data joint")
        self.alpha = torch.tensor(scales, device=dev)

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
        self.start_f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        # per-bin score: discounted return divided by its maximum (G_max, or the start frame's attainable maximum)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_visits = torch.zeros(motion.num_bins, dtype=torch.long, device=dev)
        num_bins = motion.global_num_bins if self.parallel else motion.num_bins
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / num_bins, device=dev)
        self.bin_probs = torch.full((motion.num_bins,), 1.0 / num_bins, device=dev)
        self.reset_counts = torch.zeros(2, device=dev)  # sampled resets, of which tail resets
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        if self.parallel:
            self.prepare_sampling(True)
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
    def prepare_sampling(self, active):
        # Every rank participates, even when only another rank has completed episodes.
        from supermimic.utils.distributed import global_bin_probabilities

        if bool(self.parallel.reduce(int(active))):
            visits = self.bin_visits if self.bin_prior == "mean" else None
            self.bin_probs = global_bin_probabilities(self.bin_G, self.bin_w_avg, 1.0,
                                                      self.motion.global_num_bins, self.parallel, visits)

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if self.parallel:
            probs = self.bin_probs
        else:
            scores = self.bin_G
            if self.bin_prior == "mean":
                seen = self.bin_visits > 0
                scores = torch.where(seen, scores, (scores * seen).sum() / seen.sum().clamp(min=1))
            soft = torch.softmax(-25.0 * scores, 0)
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
            probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
            self.bin_probs = probs
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = m.bin_end[bins]  # stays >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        self.reset_counts[0] += n
        self.reset_counts[1] += (m.clip_end[m.bin_clip[bins]] - f < TAIL_STEPS).sum()
        return f, bins

    @torch.no_grad()
    def sampling_stats(self) -> torch.Tensor:
        """Local sums since the last call: [entropy terms, max prob, visited bins, resets, tail resets].

        The probabilities are global (they sum to 1 across ranks), so summing the entropy terms and
        visited bins and taking the max of the max over ranks gives the global values.
        """
        p = self.bin_probs.double()
        stats = torch.stack((-(p * (p + 1e-30).log()).sum(), p.max(), (self.bin_visits > 0).sum().double(),
                             *self.reset_counts.double()))
        self.reset_counts.zero_()
        return stats

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
        self.start_f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)

    def _reward_termination_eager(self, S1, f1, a, a_prev):
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        return r, terms, self.mdp.termination(S1, f1)

    def refresh(self):
        """Recompute derived quantities (body poses, contacts) after writing states."""
        self.sim.forward()

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor, after_step=None) -> dict:
        """Advance physics; optionally enqueue transition processing before the reset's host wait.

        after_step may read the detached transition and update training statistics. It must not
        mutate the environment, transition tensors or RNG state, or perform rank-conditional work.
        Episode summary fields are added later, after the callback.
        """
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
        r, terms, term = self._reward_termination(S1, f1, a, a_prev)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        at_clip_end = f1 >= clip_end
        timeout = at_clip_end | (ep_step + 1 >= self.max_episode_steps)
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
        if after_step is not None:
            after_step(out)
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin score; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            ended = term[ids] | (at_clip_end[ids] if self.bin_clip_end == "terminal" else False)
            boot = torch.where(ended, 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            G_max = self.G_max
            if self.bin_clip_end == "terminal":
                # steps from the start frame to the clip end; the 500-step cap bootstraps past it
                s = self.start_f[ids]
                left = (self.motion.clip_end[self.motion.clip_id[s]] - s).float()
                reach = left < self.max_episode_steps
                G_max = self.G_max * torch.where(reach, 1 - self.gamma**left, 1.0)
            update_bin_returns(self.bin_G, self.start_bin[ids], G / G_max, visits=self.bin_visits,
                               replace_first=self.bin_prior == "mean")
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
        if self.parallel:
            self.prepare_sampling(len(ids) > 0)
        self.reset(ids)
        return out
