"""片段末尾之后的目标延拓（--goal_tail）：真实帧与段内目标不变，只有越过段末的目标换成虚拟帧。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic.tasks.tracking.motion import CONTROL_DT, TAIL_BLEND, TAIL_STAND_Z, MotionLib, _hull_outside
from supermimic.utils.kinematics import G1Kinematics

K = 10


class GoalTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = G1Kinematics("cpu")
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        # 关节角为0时，把根放到最低的跟踪刚体离地2 cm处，脚才算着地
        eye = torch.eye(3).unsqueeze(0)
        z = torch.zeros(1, 3)
        bodies = cls.kin.forward(torch.tensor([[0.0, 0.0, 0.8]]), eye, z, z, torch.zeros(1, 29), torch.zeros(1, 29))[0]
        cls.stand_z = 0.8 - float(bodies[0, :, 2].min()) + 0.02
        frames = 61  # 30 fps的2 s，重采样为101帧
        t = np.arange(frames) / 30.0
        for name, x, height in (("a_run", 2.0 * t, cls.stand_z), ("b_still", 0 * t, cls.stand_z), ("c_low", 0 * t, 0.4)):
            data = np.zeros((frames, 36), dtype="float32")
            data[:, 0], data[:, 2], data[:, 6] = x, height, 1.0  # 四元数xyzw = (0, 0, 0, 1)
            np.savetxt(root / f"{name}.csv", data, delimiter=",")
        cls.libs = {mode: MotionLib.from_dir(str(root), cls.kin, "cpu", goal_tail=mode, tail_frames=K if mode != "clamp" else 0)
                    for mode in ("clamp", "hold", "stand", "auto")}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def real_frames(self, lib, c):
        return torch.arange(int(lib.clip_start[c]), int(lib.clip_end[c]) + 1)

    def test_clamp_keeps_the_previous_layout(self):
        lib = self.libs["clamp"]
        self.assertEqual(lib.num_frames, int(lib.clip_len.sum()))
        torch.testing.assert_close(lib.goal_end, lib.clip_end, rtol=0, atol=0)
        torch.testing.assert_close(lib.clip_start, torch.cumsum(lib.clip_len, 0) - lib.clip_len, rtol=0, atol=0)
        self.assertEqual(lib.description["goal_tail"], {"mode": "clamp", "frames": 0, "stand_clips": 0})

    def test_real_frames_bins_and_inside_goals_are_unchanged(self):
        base = self.libs["clamp"]
        for mode in ("hold", "stand", "auto"):
            lib = self.libs[mode]
            with self.subTest(mode=mode):
                self.assertEqual(lib.num_frames, base.num_frames + 3 * K)
                torch.testing.assert_close(lib.clip_len, base.clip_len, rtol=0, atol=0)
                for c in range(3):
                    old, new = self.real_frames(base, c), self.real_frames(lib, c)
                    for key in ("root_p", "root_quat", "root_v", "root_w", "q", "qd", "body_p", "body_quat"):
                        torch.testing.assert_close(getattr(lib, key)[new], getattr(base, key)[old], rtol=0, atol=1e-6)
                    torch.testing.assert_close(lib.frame_bin[new], base.frame_bin[old], rtol=0, atol=0)
                    # 未来K帧都在段内的目标逐位相同；段末前一步的目标换成延拓帧
                    inside = new[:-K]
                    torch.testing.assert_close(lib.goal(inside, K), base.goal(old[:-K], K), rtol=0, atol=1e-6)
                torch.testing.assert_close(lib.bin_start - lib.clip_start[lib.bin_clip],
                                           base.bin_start - base.clip_start[base.bin_clip], rtol=0, atol=0)
                torch.testing.assert_close(lib.bin_end - lib.clip_start[lib.bin_clip],
                                           base.bin_end - base.clip_start[base.bin_clip], rtol=0, atol=0)

    def test_goals_stop_at_the_clips_own_tail(self):
        for mode in ("hold", "auto"):
            lib = self.libs[mode]
            end = lib.clip_end[:2]  # 真实末帧的目标全部落在本段的虚拟帧上
            fut = lib.future_frames(end, K)
            torch.testing.assert_close(fut, end.unsqueeze(-1) + torch.arange(1, K + 1), rtol=0, atol=0)
            torch.testing.assert_close(fut[:, -1], lib.goal_end[:2], rtol=0, atol=0)
            torch.testing.assert_close(lib.future_frames(lib.goal_end[:2], K)[:, 0], lib.goal_end[:2], rtol=0, atol=0)
            self.assertTrue(bool((fut < lib.clip_start[1:3].unsqueeze(-1)).all()))
            self.assertTrue(bool((lib.clip_id[fut] == torch.arange(2).unsqueeze(-1)).all()))

    def test_hold_is_the_last_frame_at_rest(self):
        lib = self.libs["hold"]
        for c in range(3):
            end = int(lib.clip_end[c])
            tail = torch.arange(end + 1, end + 1 + K)
            torch.testing.assert_close(lib.root_p[tail], lib.root_p[end].expand(K, 3), rtol=0, atol=0)
            torch.testing.assert_close(lib.q[tail], lib.q[end].expand(K, 29), rtol=0, atol=0)
            for key in ("root_v", "root_w", "qd"):
                self.assertEqual(float(getattr(lib, key)[tail].abs().max()), 0.0)

    def test_stand_reaches_the_default_pose_at_the_capture_point(self):
        n = round(TAIL_BLEND / CONTROL_DT) + 5  # 过渡结束后再多几帧
        lib = MotionLib.from_dir(str(Path(self.temp.name)), self.kin, "cpu", goal_tail="stand", tail_frames=n)
        end = int(lib.clip_end[0])
        p, v = lib.root_p[end], lib.root_v[end]
        capture = p[:2] + v[:2] / torch.sqrt(9.81 / p[2])
        last = end + n
        torch.testing.assert_close(lib.q[last], self.kin.default_q, rtol=0, atol=1e-6)
        torch.testing.assert_close(lib.root_p[last], torch.cat((capture, torch.tensor([TAIL_STAND_Z]))), rtol=0, atol=1e-5)
        for key in ("root_v", "qd"):
            self.assertEqual(float(getattr(lib, key)[last].abs().max()), 0.0)
        # min-jerk起点：第一个虚拟帧只离末帧很近，且位置单调趋向终点
        self.assertLess(float((lib.root_p[end + 1] - p).norm()), 0.01)
        x = lib.root_p[end + 1:last + 1, 0]
        self.assertTrue(bool((x[1:] >= x[:-1] - 1e-6).all()))
        self.assertTrue(bool(lib.tail_stand.all()))

    def test_auto_stands_after_running_and_holds_still_or_low_endings(self):
        lib = self.libs["auto"]
        self.assertEqual(lib.tail_stand.tolist(), [True, False, False])
        self.assertEqual(lib.description["goal_tail"], {"mode": "auto", "frames": K, "stand_clips": 1})

    def test_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "goal_tail"):
            MotionLib.from_dir(str(Path(self.temp.name)), self.kin, "cpu", goal_tail="freeze", tail_frames=K)
        with self.assertRaisesRegex(ValueError, "tail_frames"):
            MotionLib.from_dir(str(Path(self.temp.name)), self.kin, "cpu", goal_tail="hold", tail_frames=0)

    def test_hull_outside_distance(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(_hull_outside(np.array([0.5, 0.5]), square), 0.0)
        self.assertAlmostEqual(_hull_outside(np.array([1.5, 0.5]), square), 0.5)
        self.assertAlmostEqual(_hull_outside(np.array([0.5, 1.0]), np.array([[0.0, 0.0], [1.0, 0.0]])), 1.0)  # 两点为线段
        self.assertAlmostEqual(_hull_outside(np.array([3.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])), 1.0)  # 共线
        self.assertEqual(_hull_outside(np.array([0.0, 0.0]), np.zeros((0, 2))), float("inf"))
        self.assertAlmostEqual(_hull_outside(np.array([3.0, 4.0]), np.zeros((3, 2))), 5.0)  # 重合点


if __name__ == "__main__":
    unittest.main()
