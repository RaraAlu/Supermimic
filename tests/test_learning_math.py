# CPU checks of the learning math: replay windows, world-model stepping, reward/observation heads.
"""These lock the analytic MDP heads, the replay window/history semantics and the world-model
increment parameterisation without starting MuJoCo Warp or training."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from supermimic.tasks.tracking.constants import G1_JOINT_NAMES
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, STATE_DIM, XI_DIM
from supermimic.tasks.tracking.motion import MotionLib
from supermimic.tasks.tracking.rl.buffer import ReplayBuffer
from supermimic.tasks.tracking.rl.networks import WorldModel
from supermimic.utils.kinematics import G1Kinematics
from supermimic.utils.rotations import so3_exp


def _transition(age: int, done: bool) -> dict:
    s = torch.full((STATE_DIM,), float(age))
    return {
        "s": s, "s1": s + 1, "a": torch.zeros(29), "a_prev": torch.zeros(29),
        "xi": torch.zeros(XI_DIM), "r": torch.tensor(float(age)), "f": torch.tensor(age),
        "ep_step": torch.tensor(age, dtype=torch.int32), "done": torch.tensor(done),
    }  # fmt: skip


class ReplayWindowTests(unittest.TestCase):
    def make_buffer(self, dones, ep_steps=None):
        buffer = ReplayBuffer(len(dones), 1, "cpu")
        for age, done in enumerate(dones):
            tr = _transition(age, done)
            if ep_steps is not None:
                tr["ep_step"] = torch.tensor(ep_steps[age], dtype=torch.int32)
            buffer.add(tr)
        return buffer

    def test_window_mask_stops_after_done(self):
        buffer = self.make_buffer([False, False, False, True, False, False, False, False])
        starts = torch.tensor([1])
        envs = torch.tensor([0])
        with patch("supermimic.tasks.tracking.rl.buffer.torch.randint", side_effect=[starts, envs]):
            window = buffer.sample_windows(1, 4)
        # ages 1,2,3,4; the transition at age 3 is terminal, so later targets are invalid.
        self.assertEqual(window["mask"].tolist(), [[1.0, 1.0, 1.0, 0.0]])
        self.assertEqual(window["s"][0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_history_stays_inside_episode_and_buffer(self):
        buffer = self.make_buffer([False] * 8, ep_steps=[0, 1, 0, 1, 2, 3, 0, 1])
        age = torch.tensor([5])
        env = torch.tensor([0])
        with patch("supermimic.tasks.tracking.rl.buffer.torch.randint", side_effect=[age, env]):
            start, hist = buffer.sample_starts(1, 4)
        # age 5 has ep_step 3, so the four history rows are ages 5,4,3,2 (oldest first).
        self.assertEqual(start["s"][0, 0].item(), 5.0)
        self.assertEqual(hist["s"][0, :, 0].tolist(), [2.0, 3.0, 4.0, 5.0])

    def test_window_sampling_requires_room_for_k(self):
        buffer = self.make_buffer([False] * 8)
        with self.assertRaises(RuntimeError):
            with patch("supermimic.tasks.tracking.rl.buffer.torch.randint", side_effect=RuntimeError("no room")):
                buffer.sample_windows(1, 9)


class WorldModelMathTests(unittest.TestCase):
    def make_state(self, batch=2):
        generator = torch.Generator().manual_seed(0)
        p = torch.randn(batch, 3, generator=generator)
        R = so3_exp(0.1 * torch.randn(batch, 3, generator=generator))
        v = torch.randn(batch, 3, generator=generator)
        w = torch.randn(batch, 3, generator=generator)
        q = 0.1 * torch.randn(batch, 29, generator=generator)
        qd = 0.1 * torch.randn(batch, 29, generator=generator)
        return (p, R, v, w, q, qd)

    def test_zero_head_predicts_identity_step(self):
        wm = WorldModel(hidden=(32, 32))
        S = self.make_state()
        S1 = wm.step(S, torch.zeros(2, 29), torch.zeros(2, XI_DIM))
        for actual, expected in zip(S1, S):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_targets_and_state_error_round_trip(self):
        wm = WorldModel(hidden=(32, 32))
        S = self.make_state()
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in wm.net[-1].parameters():
                parameter.normal_()
        S1 = wm.step(S, 0.1 * torch.randn(2, 29), 0.1 * torch.randn(2, XI_DIM))
        targets = wm.targets(S, S1)
        self.assertEqual(tuple(targets.shape), (2, 70))
        error = wm.state_error(S1, S1)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-5, rtol=0)
        self.assertTrue(torch.isfinite(targets).all())


class AnalyticHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = G1Kinematics("cpu")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        frames = 61
        data = np.zeros((frames, 36), dtype="float32")
        data[:, 2] = 0.8
        data[:, 6] = 1.0  # quaternion xyzw -> (0,0,0,1)
        np.savetxt(root / "clip.csv", data, delimiter=",")
        self.motion = MotionLib([str(root / "clip.csv")], self.kin, "cpu", src_fps=30.0)
        self.mdp = MDP(self.kin, self.motion, self.kin.default_q)
        self.f = torch.tensor([10])

    def reference_state(self):
        return tuple(x.clone() if torch.is_tensor(x) else x for x in self.motion.ref_state(self.f))

    def test_reward_is_maximal_on_the_reference(self):
        S = self.reference_state()
        reward = self.mdp.reward(S, self.f, torch.zeros(1, 29), torch.zeros(1, 29))
        self.assertEqual(tuple(reward.shape), (1,))
        # Tracking terms are maximal; only the soft joint-limit penalty can remain at q = 0.
        q = S[4]
        limit = (torch.relu(q - self.mdp.soft_hi) + torch.relu(self.mdp.soft_lo - q)).sum()
        expected = (5.0 - 2.0 * float(limit)) * 0.02
        self.assertAlmostEqual(float(reward.item()), expected, places=5)
        self.assertGreater(float(reward.item()), 0.08)
        self.assertFalse(bool(self.mdp.termination(S, self.f).any()))
        self.assertLess(float(self.mdp.soft_termination(S, self.f).item()), 1e-3)

    def test_reward_never_exceeds_the_step_maximum_that_bounds_the_critic(self):
        torch.manual_seed(0)
        S0 = self.reference_state()
        for scale in (0.0, 0.05, 0.3, 1.0):
            S = tuple(x if i == 1 else x + scale * torch.randn_like(x) for i, x in enumerate(S0))  # index 1 is a rotation matrix
            r = self.mdp.reward(S, self.f, scale * torch.randn(1, 29), torch.zeros(1, 29))
            self.assertLessEqual(float(r.item()), MAX_STEP_REWARD + 1e-9)
        self.assertAlmostEqual(MAX_STEP_REWARD / (1 - 0.99), 10.0)  # train.py's critic value cap at gamma 0.99

    def test_termination_triggers_on_height_and_tilt(self):
        S = list(self.reference_state())
        S[0] = S[0] + torch.tensor([[0.0, 0.0, 0.3]])
        self.assertTrue(bool(self.mdp.termination(tuple(S), self.f).item()))
        S = list(self.reference_state())
        tilt = torch.eye(3).unsqueeze(0).clone()
        tilt[0] = so3_exp(torch.tensor([[2.0, 0.0, 0.0]]))[0]
        S[1] = tilt
        self.assertTrue(bool(self.mdp.termination(tuple(S), self.f).item()))

    def test_soft_survival_keeps_far_terminations_differentiable(self):
        S = self.reference_state()
        torch.testing.assert_close(self.mdp.soft_survival(S, self.f), 1 - self.mdp.soft_termination(S, self.f))
        # 0.6 m above the reference: (0.6 - 0.25) / 0.02 = 17.5, where 1 - sigmoid(x) is exactly 0 in fp32
        p = (S[0] + torch.tensor([[0.0, 0.0, 0.6]])).requires_grad_(True)
        keep = self.mdp.soft_survival((p,) + S[1:], self.f)
        self.assertEqual(float(1 - torch.sigmoid(torch.tensor(17.5))), 0.0)
        self.assertGreater(float(keep.item()), 0.0)
        torch.testing.assert_close(keep, torch.sigmoid(torch.tensor([-17.5])), rtol=1e-4, atol=0)
        keep.sum().backward()
        self.assertLess(float(p.grad[0, 2]), 0.0)  # higher still lowers survival: the gradient is not lost

    def test_observation_noise_and_goal_layout(self):
        S = self.reference_state()
        obs = self.mdp.obs(S, torch.zeros(1, 29), torch.zeros(1, XI_DIM), self.f)
        self.assertEqual(tuple(obs.shape), (1, OBS_DIM))
        noisy = self.mdp.add_noise(torch.zeros(2000, OBS_DIM))
        bound = torch.max(self.mdp.obs_noise).item()
        self.assertLessEqual(float(noisy.abs().max().item()), bound + 1e-6)
        goal = self.motion.goal(self.f, 10)
        self.assertEqual(tuple(goal.shape), (1, 730))
        future = self.motion.future_frames(self.f, 10)
        self.assertTrue(bool((future <= self.motion.clip_end[self.motion.clip_id[self.f]]).all()))
        critic = self.mdp.critic_features(S, self.f, torch.zeros(1, XI_DIM))
        self.assertEqual(critic.shape[1], 184)

    def test_reference_state_uses_named_joint_order(self):
        self.assertEqual(len(self.motion.q[0]), len(G1_JOINT_NAMES))


if __name__ == "__main__":
    unittest.main()
