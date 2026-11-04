"""因果actor的信息边界（docs/teleop/causal_intent_encoder.md第2、3节）：观测只取决于当前与已发布的参考前缀，
回合首帧的规则在采样、回放重建和想象推演中相同，167维排列与Teleopit一致。不启动仿真或GPU。

设置环境变量TELEOPIT_ROOT指向Teleopit源码时，另与其观测构建器逐项比较。"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

from supermimic.tasks.tracking.mdp import CAUSAL_OBS_DIM, CAUSAL_OBS_LAYOUT, MDP, NUM_J, OBS_DIM, XI_DIM, XI_JOINT, flatten_state, unflatten_state
from supermimic.tasks.tracking.motion import CONTROL_DT, MotionLib
from supermimic.tasks.tracking.rl.buffer import ReplayBuffer
from supermimic.tasks.tracking.rl.networks import CausalPolicy, Policy, build_policy
from supermimic.tasks.tracking.rl.trainer import Trainer
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.rotations import mat_to_6d, mat_to_quat, so3_log

CLIP_FRAMES = (60, 40)


def synthetic_clip(frames: int, seed: int) -> tuple:
    """平滑的根部平移、偏航与关节运动，各帧都不同；没有重复帧才能看出读错帧。"""
    t = np.arange(frames, dtype=np.float64) * CONTROL_DT
    rng = np.random.default_rng(seed)
    p = np.stack((0.8 * t, 0.1 * np.sin(2 * t), 0.75 + 0.02 * np.sin(3 * t)), -1)
    yaw = 0.7 * t + 0.1 * seed
    quat = np.stack((np.cos(yaw / 2), np.zeros(frames), np.zeros(frames), np.sin(yaw / 2)), -1)
    phase, rate = rng.uniform(0, 6, NUM_J), rng.uniform(1, 4, NUM_J)
    joints = 0.2 * np.sin(rate * t[:, None] + phase)
    return p.astype(np.float32), quat.astype(np.float32), joints.astype(np.float32)


def write_source(path: Path, clips) -> None:
    lengths = np.array([len(c[0]) for c in clips], dtype=np.int64)
    with h5py.File(path, "w") as out:
        out.attrs["format"] = "teleopit_motion_hdf5"
        out.attrs["version"] = 1
        out.attrs["fps"] = 50
        out["root_pos"] = np.concatenate([c[0] for c in clips])
        out["root_quat_w"] = np.concatenate([c[1] for c in clips])
        out["joint_pos"] = np.concatenate([c[2] for c in clips])
        out["source_clip_starts"] = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        out["source_clip_lengths"] = lengths
        out["source_clip_fps"] = np.full(len(clips), 50, dtype=np.int64)


class MotionFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.kin = G1Kinematics("cpu")
        cls.clips = [synthetic_clip(n, i) for i, n in enumerate(CLIP_FRAMES)]
        cls.motion = cls.load(cls.clips, "base.h5")
        cls.mdp = MDP(cls.kin, cls.motion, cls.kin.default_q, actor="causal")

    @classmethod
    def load(cls, clips, name):
        path = cls.root / name
        write_source(path, clips)
        return MotionLib.from_source(str(path), cls.kin, "cpu")

    def state(self, batch=2):
        """一个任意的机器人状态：朝向、速度都偏离参考，才能分清哪一项用的是机器人、哪一项用的是参考。"""
        torch.manual_seed(5)
        yaw = torch.linspace(-1.0, 1.0, batch)
        pitch = torch.full((batch,), 0.3)
        cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
        zero, one = torch.zeros(batch), torch.ones(batch)
        Rz = torch.stack((cy, -sy, zero, sy, cy, zero, zero, zero, one), -1).view(batch, 3, 3)
        Ry = torch.stack((cp, zero, sp, zero, one, zero, -sp, zero, cp), -1).view(batch, 3, 3)
        return (torch.randn(batch, 3), Rz @ Ry, torch.randn(batch, 3), torch.randn(batch, 3),
                self.kin.default_q + 0.1 * torch.randn(batch, NUM_J), torch.randn(batch, NUM_J))

    def obs(self, f, first=None, mdp=None, S=None):
        f = torch.as_tensor(f)
        S = self.state(len(f)) if S is None else S
        return (mdp or self.mdp).causal_obs(S, torch.ones(len(f), NUM_J), torch.zeros(len(f), XI_DIM), f, first)


def segments(o):
    out, start = {}, 0
    for name, dim, _ in CAUSAL_OBS_LAYOUT:
        out[name] = o[..., start:start + dim]
        start += dim
    return out


class CausalObsTests(MotionFixture):
    def test_layout_is_the_documented_167_dimensions(self):
        self.assertEqual(CAUSAL_OBS_DIM, 167)
        self.assertEqual(sum(dim for name, dim, _ in CAUSAL_OBS_LAYOUT if name.startswith("ref_")), 74)
        self.assertEqual(self.obs([10, 20]).shape, (2, 167))

    def test_each_segment_matches_its_definition(self):
        m, S, f = self.motion, self.state(), torch.tensor([10, 20])
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        a_prev = torch.arange(2 * NUM_J, dtype=torch.float32).view(2, NUM_J)
        xi = torch.zeros(2, XI_DIM)
        xi[:, XI_JOINT] = 0.01
        got = segments(self.mdp.causal_obs(S, a_prev, xi, f))
        ref_v = (m.root_p[f] - m.root_p[f - 1]) / CONTROL_DT
        ref_w = so3_log(m.root_R[f] @ m.root_R[f - 1].transpose(-1, -2)) / CONTROL_DT
        down = torch.tensor([0.0, 0.0, -1.0])
        expected = {
            "ref_joint_pos": m.q[f],
            "ref_joint_vel": (m.q[f] - m.q[f - 1]) / CONTROL_DT,
            "ref_anchor_ori_b": mat_to_6d(Rt @ m.root_R[f]),
            "base_ang_vel_b": (Rt @ w.unsqueeze(-1)).squeeze(-1),
            "joint_pos_rel": q - self.kin.default_q - 0.01,
            "joint_vel": qd,
            "prev_action": a_prev,
            "projected_gravity_b": Rt @ down,
            # 参考速度投影到机器人系，参考重力投影到参考系：两者不同，见Teleopit的builder
            "ref_anchor_lin_vel_b": (Rt @ ref_v.unsqueeze(-1)).squeeze(-1),
            "ref_anchor_ang_vel_b": (Rt @ ref_w.unsqueeze(-1)).squeeze(-1),
            "ref_projected_gravity_b": m.root_R[f].transpose(-1, -2) @ down,
            "ref_anchor_height": m.root_p[f][:, 2:3],
        }
        self.assertEqual(list(expected), list(got))
        for name, value in expected.items():
            torch.testing.assert_close(got[name], value, msg=name)

    def test_reference_velocity_is_a_backward_difference(self):
        # 中心差分的qd会读到f+1；这里必须只用f与f-1
        f = torch.tensor([10])
        got = segments(self.obs(f))["ref_joint_vel"]
        torch.testing.assert_close(got, (self.motion.q[f] - self.motion.q[f - 1]) / CONTROL_DT)
        self.assertGreater(float((got - self.motion.qd[f]).abs().max()), 1e-4)

    def test_first_frame_velocities_are_zero_even_mid_clip(self):
        f = torch.tensor([30, 30])
        got = segments(self.obs(f, torch.tensor([True, False])))
        for name in ("ref_joint_vel", "ref_anchor_lin_vel_b", "ref_anchor_ang_vel_b"):
            self.assertEqual(float(got[name][0].abs().max()), 0.0, name)
            self.assertGreater(float(got[name][1].abs().max()), 0.0, name)
        # 其他项不受影响
        torch.testing.assert_close(got["ref_joint_pos"][0], got["ref_joint_pos"][1])
        torch.testing.assert_close(got["ref_anchor_height"][0], got["ref_anchor_height"][1])

    def test_first_flag_guards_the_clip_boundary(self):
        second_clip = self.motion.clip_start[1:2]
        got = segments(self.obs(second_clip, torch.tensor([True])))
        self.assertEqual(float(got["ref_joint_vel"].abs().max()), 0.0)

    def test_first_none_means_no_frame_is_first(self):
        f = torch.tensor([30, 31])
        torch.testing.assert_close(self.obs(f), self.obs(f, torch.tensor([False, False])))

    def test_history_shape_uses_a_first_flag_per_frame(self):
        # 回放重建时f与first都是(B, hist)
        f = torch.tensor([[29, 30, 31], [29, 30, 31]])
        first = torch.tensor([[True, False, False], [False, False, False]])
        S = tuple(x.unsqueeze(1).expand(-1, 3, *x.shape[1:]) for x in self.state())
        o = self.mdp.causal_obs(S, torch.zeros(2, 3, NUM_J), torch.zeros(2, 3, XI_DIM), f, first)
        self.assertEqual(o.shape, (2, 3, 167))
        vel = segments(o)["ref_joint_vel"]
        self.assertEqual(float(vel[0, 0].abs().max()), 0.0)
        self.assertGreater(float(vel[1, 0].abs().max()), 0.0)

    def test_observation_ignores_the_future(self):
        # 第40帧之后完全不同的两段参考，到第40帧为止观测必须逐位相同
        T = 40
        p, quat, joints = self.clips[0]
        altered = [(np.concatenate((p[:T + 1], p[T + 1:] + 1.0)), quat.copy(),
                    np.concatenate((joints[:T + 1], joints[T + 1:] + 0.5))), self.clips[1]]
        motion = self.load(altered, "altered.h5")
        other = MDP(self.kin, motion, self.kin.default_q, actor="causal")
        for frame in (1, 20, T):
            f = torch.tensor([frame, frame])
            torch.testing.assert_close(self.obs(f), self.obs(f, mdp=other), rtol=0, atol=0)
        # 对照：未来帧目标确实看得到这种改变，说明上面的比较有区分力
        f = torch.tensor([T])
        self.assertGreater(float((self.motion.goal(f, 10) - motion.goal(f, 10)).abs().max()), 0.1)
        torch.testing.assert_close(self.motion.goal(torch.tensor([20]), 10), motion.goal(torch.tensor([20]), 10))

    def test_reference_offset_reads_the_next_frame_and_stops_at_the_clip_end(self):
        ahead = MDP(self.kin, self.motion, self.kin.default_q, actor="causal", reference_offset=1)
        f = torch.tensor([10, 30])
        S = self.state()
        # 只读到的参考前移一帧，机器人项不变
        torch.testing.assert_close(segments(self.obs(f, mdp=ahead, S=S))["ref_joint_pos"], self.motion.q[f + 1])
        torch.testing.assert_close(segments(self.obs(f, mdp=ahead, S=S))["ref_joint_vel"],
                                   (self.motion.q[f + 1] - self.motion.q[f]) / CONTROL_DT)
        torch.testing.assert_close(segments(self.obs(f, mdp=ahead, S=S))["joint_vel"], S[5])
        # 回合首帧同样取0，使C1f与C1只差一帧前瞻
        got = segments(self.obs(f, torch.tensor([True, False]), mdp=ahead, S=S))["ref_joint_vel"]
        self.assertEqual(float(got[0].abs().max()), 0.0)
        # 段末不读到下一段
        end = self.motion.clip_end[:1]
        torch.testing.assert_close(segments(self.obs(end, mdp=ahead))["ref_joint_pos"], self.motion.q[end])

    def test_offset_needs_the_causal_actor(self):
        with self.assertRaises(ValueError):
            MDP(self.kin, self.motion, self.kin.default_q, reference_offset=1)
        with self.assertRaises(ValueError):
            MDP(self.kin, self.motion, self.kin.default_q, actor="causal", reference_offset=2)
        with self.assertRaises(ValueError):
            MDP(self.kin, self.motion, self.kin.default_q, actor="teacher")

    def test_noise_follows_the_layout(self):
        o = self.obs([10, 20])
        torch.manual_seed(0)
        noisy = segments(self.mdp.add_noise(o))
        clean = segments(o)
        for name, _, width in CAUSAL_OBS_LAYOUT:
            error = (noisy[name] - clean[name]).abs()
            if width == 0:
                self.assertEqual(float(error.max()), 0.0, name)
            else:
                self.assertGreater(float(error.max()), 0.0, name)
                self.assertLessEqual(float(error.max()), width * 1.001, name)

    def test_future_actor_observation_is_unchanged(self):
        future = MDP(self.kin, self.motion, self.kin.default_q)
        S, f = self.state(), torch.tensor([10, 20])
        a, xi = torch.ones(2, NUM_J), torch.zeros(2, XI_DIM)
        torch.testing.assert_close(future.actor_obs(S, a, xi, f, torch.tensor([True, True])), future.obs(S, a, xi, f))
        self.assertEqual(future.actor_dim, OBS_DIM)
        self.assertEqual(self.mdp.actor_dim, CAUSAL_OBS_DIM)


class CausalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pi = CausalPolicy(CAUSAL_OBS_DIM, hist=10)
        self.hist = torch.randn(4, 10, CAUSAL_OBS_DIM)

    def test_output_shape_and_goal_is_ignored(self):
        a = self.pi(self.hist)
        self.assertEqual(a.shape, (4, NUM_J))
        torch.testing.assert_close(a, self.pi(self.hist, torch.randn(4, 730)))

    def test_encoder_matches_the_documented_structure(self):
        widths = [m.out_channels for m in self.pi.hist_enc if isinstance(m, torch.nn.Conv1d)]
        self.assertEqual(widths, [256, 128, 64])
        self.assertEqual({m.kernel_size for m in self.pi.hist_enc if isinstance(m, torch.nn.Conv1d)}, {(3,)})
        linear = [m.out_features for m in self.pi.head if isinstance(m, torch.nn.Linear)]
        self.assertEqual(linear, [2048, 1024, 512, 256, 128, NUM_J])
        self.assertEqual(self.pi.head[0].in_features, CAUSAL_OBS_DIM + 64)

    def test_current_and_past_frames_both_reach_the_action(self):
        hist = self.hist.clone().requires_grad_(True)
        self.pi(hist).sum().backward()
        per_frame = hist.grad.abs().sum((0, 2))
        self.assertTrue(bool((per_frame > 0).all()))
        # 当前帧有直通路径，影响最大
        self.assertEqual(int(per_frame.argmax()), 9)

    def test_initial_actions_are_small(self):
        self.pi.obs_norm.update(self.hist)
        self.assertLess(float(self.pi(self.hist).abs().max()), 0.5)

    def test_build_policy_selects_by_actor(self):
        self.assertIsInstance(build_policy("causal", 730, 10), CausalPolicy)
        future = build_policy("future", 730, 10)
        self.assertIsInstance(future, Policy)
        torch.manual_seed(1)
        a = build_policy("future", 730, 10)
        torch.manual_seed(1)
        b = Policy(OBS_DIM, 730, hist=10)
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            self.assertEqual(ka, kb)
            torch.testing.assert_close(va, vb)
        with self.assertRaises(ValueError):
            build_policy("teacher", 730, 10)


class PathConsistencyTests(MotionFixture):
    """采样、回放重建与想象推演必须给出同一份观测，这是线上离线一致的前提。"""

    def fill_buffer(self, start_frame=30, steps=6):
        """模拟一个从片段中途重置的回合，逐步写入回放，同时记下采样路径算出的观测（无噪声）。"""
        buffer = ReplayBuffer(steps, 1, "cpu")
        seen = []
        torch.manual_seed(3)
        for step in range(steps):
            S = self.state(1)
            S = (S[0] + step * 0.01, *S[1:])
            s = flatten_state(S)
            a_prev = torch.full((1, NUM_J), float(step))
            f = torch.tensor([start_frame + step])
            ep_step = torch.tensor([step], dtype=torch.int32)
            seen.append(self.mdp.actor_obs(unflatten_state(s), a_prev, torch.zeros(1, XI_DIM), f, ep_step == 0))
            buffer.add({"s": s, "s1": s, "a": torch.zeros(1, NUM_J), "a_prev": a_prev, "xi": torch.zeros(1, XI_DIM),
                        "r": torch.zeros(1), "f": f, "ep_step": ep_step, "done": torch.zeros(1, dtype=torch.bool)})
        return buffer, torch.cat(seen)

    def test_replay_rebuild_equals_sampling_including_the_first_frame(self):
        buffer, sampled = self.fill_buffer()
        with patch("supermimic.tasks.tracking.rl.buffer.torch.randint", side_effect=[torch.tensor([5]), torch.tensor([0])]):
            start, hist = buffer.sample_starts(1, 6)
        self.assertEqual(start["ep_step"].tolist(), [5])
        self.assertEqual(hist["ep_step"].tolist(), [[0, 1, 2, 3, 4, 5]])
        fake = SimpleNamespace(mdp=self.mdp)
        rebuilt = Trainer.replay_obs(fake, hist)
        torch.testing.assert_close(rebuilt[0], sampled, atol=1e-5, rtol=1e-5)
        self.assertEqual(float(segments(rebuilt)["ref_joint_vel"][0, 0].abs().max()), 0.0)

    def test_padded_history_repeats_the_first_frame_with_its_flag(self):
        buffer, sampled = self.fill_buffer(steps=3)
        with patch("supermimic.tasks.tracking.rl.buffer.torch.randint", side_effect=[torch.tensor([1]), torch.tensor([0])]):
            _, hist = buffer.sample_starts(1, 3)
        # 第1步只有两帧历史：最旧的位置重复回合首帧
        self.assertEqual(hist["ep_step"].tolist(), [[0, 0, 1]])
        rebuilt = Trainer.replay_obs(SimpleNamespace(mdp=self.mdp), hist)
        torch.testing.assert_close(rebuilt[0], sampled[[0, 0, 1]], atol=1e-5, rtol=1e-5)

    def imag_fake(self, S1):
        """只含_imag_step所需部件的假训练器：世界模型直接给出S1，不加噪声。"""
        mdp = SimpleNamespace(actor_obs=self.mdp.actor_obs, add_noise=lambda o: o)
        return SimpleNamespace(pi=lambda hist, g: torch.zeros(hist.shape[0], NUM_J),
                               wm=SimpleNamespace(step=lambda S, a, xi_n: S1), mdp=mdp, motion=self.motion, K=10)

    def test_imagination_uses_the_same_rule_as_sampling(self):
        S, S1, xi = self.state(), self.state(), torch.zeros(2, XI_DIM)
        f = torch.tensor([30, 10])
        hist = torch.zeros(2, 4, CAUSAL_OBS_DIM)
        out = Trainer._imag_step(self.imag_fake(S1), S, hist, torch.zeros(2, NUM_J), f, xi, xi, None)
        _, a, o_hist, f1, _ = out
        torch.testing.assert_close(f1, f + 1)
        sampled = self.mdp.actor_obs(S1, a, xi, f1, torch.tensor([False, False]))
        torch.testing.assert_close(o_hist[:, -1], sampled)
        # 想象里新生成的帧从不是回合首帧：即使起点是首帧，下一帧速度也来自差分
        self.assertGreater(float(segments(o_hist[:, -1])["ref_joint_vel"].abs().max()), 0.0)

    def test_collection_does_not_touch_the_future_goal(self):
        class Pi:
            obs_norm = SimpleNamespace(update=lambda o: None)

            def __call__(self, history, g):
                self.goal = g
                return torch.zeros(history.shape[0], NUM_J)

        pi = Pi()
        goal = Mock(side_effect=AssertionError("the causal actor must not read the future goal"))
        fake = SimpleNamespace(mdp=self.mdp, pi=pi, motion=SimpleNamespace(goal=goal), K=10, noise_scale=None,
                               args=SimpleNamespace(action_noise=0.0))
        S, f = self.state(), torch.tensor([30, 10])
        history = torch.ones(2, 4, CAUSAL_OBS_DIM) * 7.0
        action, new = Trainer._collect_action(fake, S, torch.zeros(2, NUM_J), torch.zeros(2, XI_DIM), f, torch.tensor([0, 3]), history)
        self.assertIsNone(pi.goal)
        self.assertEqual(action.shape, (2, NUM_J))
        # 回合首帧用当前观测填满历史；其余环境滑动窗口
        torch.testing.assert_close(new[0, 0], new[0, -1])
        self.assertEqual(float(new[1, 0].max()), 7.0)
        self.assertEqual(float(segments(new)["ref_joint_vel"][0, -1].abs().max()), 0.0)


class ResolveInitTests(unittest.TestCase):
    def args(self, **kw):
        return SimpleNamespace(**{"goal_frames": 10, "hist": 10, "actor": "future", "critic_warmup_iters": 0, "no_critic": False, **kw})

    def test_checkpoint_with_another_actor_is_rejected(self):
        from supermimic.scripts.train import resolve_init

        causal = {"args": {"goal_frames": 10, "hist": 10, "actor": "causal"}}
        older = {"args": {"goal_frames": 10, "hist": 10}}  # 旧检查点没有actor，按未来帧策略
        with self.assertRaisesRegex(ValueError, "--actor causal"):
            resolve_init(self.args(), causal, None)
        with self.assertRaisesRegex(ValueError, "--actor causal"):
            resolve_init(self.args(), None, causal)
        with self.assertRaisesRegex(ValueError, "--actor future"):
            resolve_init(self.args(actor="causal"), None, older)
        resolve_init(self.args(), older, None)
        resolve_init(self.args(actor="causal"), causal, None)


@unittest.skipUnless(os.environ.get("TELEOPIT_ROOT"), "TELEOPIT_ROOT is not set")
class TeleopitLayoutTests(MotionFixture):
    """用Teleopit自己的代码拼观测，逐项比较。锚点取pelvis时正向运动学的结果就是根部，所以不需要它的MuJoCo模型。"""

    @staticmethod
    def wxyz(R):
        return mat_to_quat(R).numpy().astype(np.float32)

    def builder(self):
        sys.path.insert(0, os.environ["TELEOPIT_ROOT"])
        module = importlib.import_module("teleopit.controllers.observation")
        base = module._VelCmdBaseObservationBuilder.__new__(module._VelCmdBaseObservationBuilder)
        base.num_actions = NUM_J
        base.default_dof_pos = self.kin.default_q.numpy().astype(np.float32)
        base.total_obs_size = NUM_J * 2 + 6 + 3 + NUM_J * 3
        base._anchor_body_id = 0
        pose = {}

        def run_fk(base_pos, base_quat, joint_pos):
            pose["pos"], pose["quat"] = np.asarray(base_pos, dtype=np.float32), np.asarray(base_quat, dtype=np.float32)

        base._run_fk = run_fk
        base._get_body_pos = lambda body: pose["pos"].copy()
        base._get_body_quat = lambda body: pose["quat"].copy()
        built = module.VelCmdObservationBuilder.__new__(module.VelCmdObservationBuilder)
        built._base, built.num_actions, built.total_obs_size = base, NUM_J, base.total_obs_size + 13
        return built

    def test_every_segment_matches_teleopit(self):
        builder = self.builder()
        m, S = self.motion, self.state(1)
        p, R, v, w, q, qd = S
        f = torch.tensor([20])
        a_prev = torch.linspace(-1, 1, NUM_J).unsqueeze(0)
        ours = self.mdp.causal_obs(S, a_prev, torch.zeros(1, XI_DIM), f)[0].numpy()
        pieces = segments(torch.from_numpy(ours))
        motion_qpos = np.concatenate((m.root_p[f][0].numpy(), self.wxyz(m.root_R[f])[0], m.q[f][0].numpy()))
        robot = SimpleNamespace(qpos=q[0].numpy(), qvel=qd[0].numpy(), quat=self.wxyz(R)[0],
                                ang_vel=pieces["base_ang_vel_b"].numpy(), base_pos=p[0].numpy())
        ref_v = ((m.root_p[f] - m.root_p[f - 1]) / CONTROL_DT)[0].numpy()
        ref_w = (so3_log(m.root_R[f] @ m.root_R[f - 1].transpose(-1, -2)) / CONTROL_DT)[0].numpy()
        theirs = builder.build(robot, motion_qpos, pieces["ref_joint_vel"].numpy(), a_prev[0].numpy(), ref_v, ref_w)
        self.assertEqual(theirs.shape, ours.shape)
        start = 0
        for name, dim, _ in CAUSAL_OBS_LAYOUT:
            np.testing.assert_allclose(ours[start:start + dim], theirs[start:start + dim], atol=2e-4, err_msg=name)
            start += dim


if __name__ == "__main__":
    unittest.main()
