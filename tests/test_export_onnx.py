"""scripts/export_onnx.py：元数据里观测与目标的排列和MDP.obs、MotionLib.goal的拼接一致，导出的ONNX与PyTorch策略逐元素相同。"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
from mjlab.utils.lab_api.string import resolve_matching_names_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_onnx  # noqa: E402
from supermimic.tasks.tracking.constants import G1_JOINT_NAMES  # noqa: E402
from supermimic.tasks.tracking.mdp import MDP, OBS_DIM  # noqa: E402
from supermimic.tasks.tracking.motion import GOAL_FRAME_DIM, MotionLib  # noqa: E402
from supermimic.tasks.tracking.rl.networks import Policy  # noqa: E402
from supermimic.utils.kinematics import G1Kinematics  # noqa: E402
from supermimic.utils.rotations import so3_exp  # noqa: E402

HAVE_ONNX = all(importlib.util.find_spec(name) for name in ("onnx", "onnxscript"))


def segments(rows, x):
    """按layout把最后一维切成{段名: 张量}。"""
    return {row["name"]: x[..., row["offset"]:row["offset"] + row["dim"]] for row in export_onnx.layout(rows)[0]}


class ExportOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = G1Kinematics("cpu")
        cls.robot = export_onnx.robot_metadata()

    def test_robot_metadata_matches_training(self):
        torch.testing.assert_close(torch.tensor(self.robot["default_joint_pos"]), self.kin.default_q)
        _, _, scales = resolve_matching_names_values(G1_ACTION_SCALE, G1_JOINT_NAMES)  # 同TrackingSim.alpha
        self.assertEqual(self.robot["action_scale"], [float(s) for s in scales])
        for key in ("joint_stiffness", "joint_damping", "effort_limit", "armature"):
            self.assertTrue(all(v > 0 for v in self.robot[key]), key)

    def test_obs_layout_matches_mdp(self):
        g = torch.Generator().manual_seed(0)
        R, R_ref = so3_exp(torch.randn(2, 3, generator=g)), so3_exp(torch.randn(4, 3, generator=g))
        p, v, w = (torch.randn(2, 3, generator=g) for _ in range(3))
        q, qd, a_prev = (torch.randn(2, 29, generator=g) for _ in range(3))
        xi = torch.zeros(2, 33)
        f = torch.tensor([1, 3])
        mdp = MDP(self.kin, SimpleNamespace(root_R=R_ref), self.kin.default_q)
        o = segments(export_onnx.OBS_LAYOUT, mdp.obs((p, R, v, w, q, qd), a_prev, xi, f))
        Rt = R.transpose(-1, -2)
        torch.testing.assert_close(o["joint_pos_rel"], q - self.kin.default_q)
        torch.testing.assert_close(o["joint_vel"], qd)
        torch.testing.assert_close(o["base_ang_vel"], (Rt @ w.unsqueeze(-1)).squeeze(-1))
        torch.testing.assert_close(o["projected_gravity"], (Rt @ torch.tensor([0.0, 0.0, -1.0])))
        torch.testing.assert_close(o["last_action"], a_prev)
        M = Rt @ R_ref[f]  # rotation_6d：前两列按行展平
        torch.testing.assert_close(o["ref_root_rot_rel"], M[:, :, :2].reshape(2, 6))

    def test_goal_layout_matches_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = np.arange(61) / 30.0
            data = np.zeros((61, 36))
            data[:, 0], data[:, 2] = 0.5 * t, 0.8
            data[:, 5], data[:, 6] = np.sin(0.4 * t), np.cos(0.4 * t)  # xyzw，绕z转
            data[:, 7:] = 0.3 * np.sin(t[:, None] * 3 + np.arange(29))
            np.savetxt(Path(tmp) / "walk.csv", data, delimiter=",")
            motion = MotionLib.from_dir(tmp, self.kin, "cpu")
        K, n = 3, motion.num_frames
        f = torch.tensor([0, n - 2])  # 第二个在段末截断
        fut = motion.future_frames(f, K)
        goal = segments(export_onnx.GOAL_LAYOUT, motion.goal(f, K).reshape(2, K, GOAL_FRAME_DIM))  # 按帧拼接
        self.assertEqual(fut[1].tolist(), [n - 1] * K)
        torch.testing.assert_close(goal["joint_pos"], motion.q[fut])
        torch.testing.assert_close(goal["joint_vel_scaled"], 0.1 * motion.qd[fut])
        torch.testing.assert_close(goal["root_lin_vel"], motion.root_v_local[fut])
        torch.testing.assert_close(goal["root_ang_vel"], motion.root_w_local[fut])
        H = motion.root_R[f]  # 本例只绕z轴转，偏航部分即根朝向本身
        dp = (H.transpose(-1, -2).unsqueeze(1) @ (motion.root_p[fut] - motion.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(goal["root_pos"], dp, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(HAVE_ONNX, "needs onnx and onnxscript")
    def test_export_matches_policy_and_writes_metadata(self):
        import onnx

        torch.manual_seed(0)
        pi = Policy(OBS_DIM, GOAL_FRAME_DIM * 2, hist=3)
        with torch.no_grad():  # 非平凡的归一化统计量与输出层，确认都进了图
            for norm in (pi.obs_norm, pi.goal_norm):
                norm.mean.uniform_(-1, 1), norm.var.uniform_(0.5, 2)
            pi.dec[-1].weight.normal_(0, 0.05)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, out = Path(tmp) / "ckpt_000100.pt", Path(tmp) / "policy.onnx"
            torch.save({"pi": pi.state_dict(), "iteration": 100, "wall_time": 36.0, "sim_steps": 1, "saved_at": "now",
                        "args": {"goal_frames": 2, "hist": 3}, "reference": {"fingerprint": "abc"}}, ckpt)
            ckpt.with_name("ckpt_000100_eval.json").write_text(json.dumps({"norm_return": 0.5, "per_clip": [1]}))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = export_onnx.main([str(ckpt), str(out)])
            self.assertLess(summary["max_abs_diff"], 1e-4)
            model = onnx.load(str(out))
            meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        props = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(set(props), set(meta))
        self.assertEqual(json.loads(props["joint_names"]), G1_JOINT_NAMES)
        self.assertEqual(meta["inputs"], {"obs_hist": ["batch", 3, OBS_DIM], "goal": ["batch", 2 * GOAL_FRAME_DIM]})
        self.assertEqual([i.name for i in model.graph.input], ["obs_hist", "goal"])
        self.assertTrue(all(i.type.tensor_type.shape.dim[0].dim_param for i in model.graph.input))  # 批量可变
        self.assertEqual(meta["evaluation"], {"norm_return": 0.5})
        self.assertEqual((meta["checkpoint"]["iteration"], meta["reference"]), (100, {"fingerprint": "abc"}))
        self.assertEqual(meta["default_joint_pos"], self.robot["default_joint_pos"])


if __name__ == "__main__":
    unittest.main()
