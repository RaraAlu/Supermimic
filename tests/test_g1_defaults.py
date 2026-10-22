"""Offline source contracts for native G1 defaults, not simulator acceptance."""

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = REPO / "supermimic/tasks/tracking/env.py"
FK = REPO / "supermimic/utils/kinematics.py"
BUILTIN = "mjlab.asset_zoo.robots.unitree_g1.g1_constants"


class G1DefaultsTests(unittest.TestCase):
    def test_native_factory_is_called_without_robot_overrides(self):
        for path in (ENV, FK):
            with self.subTest(path=path.relative_to(REPO)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertTrue(any(
                    isinstance(node, ast.ImportFrom) and node.module == BUILTIN
                    and any(alias.name == "get_g1_robot_cfg" for alias in node.names)
                    for node in tree.body
                ))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "get_g1_robot_cfg"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].args, [])
                self.assertEqual(calls[0].keywords, [])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                        self.assertNotIn(node.attr, ("collisions", "init_state", "articulation", "spec_fn"))

    def test_no_local_robot_config_or_collision_option(self):
        self.assertFalse((REPO / "supermimic/asset_zoo").exists())
        self.assertFalse((REPO / "supermimic/tasks/tracking/config").exists())
        sources = list((REPO / "supermimic").rglob("*.py")) + list((REPO / "scripts").glob("*.py"))
        for path in sources:
            with self.subTest(path=path.relative_to(REPO)):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("self_collision", source)
                self.assertNotIn("def robot_cfg", source)

    def test_tracking_constants_are_only_motion_data_contracts(self):
        path = REPO / "supermimic/tasks/tracking/constants.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            self.assertIsInstance(node, ast.Assign)
            values[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(set(values), {"G1_JOINT_NAMES", "TRACKED_BODIES", "FOOT_GEOMS"})
        self.assertEqual(len(values["G1_JOINT_NAMES"]), 29)
        self.assertEqual(len(set(values["G1_JOINT_NAMES"])), 29)
        self.assertEqual(len(set(values["TRACKED_BODIES"])), 14)
        self.assertEqual(values["TRACKED_BODIES"][0], "pelvis")

    def test_pose_and_scale_mappings_read_builtin_values(self):
        for path, expected in (
            (FK, "resolve_matching_names_values(cfg.init_state.joint_pos, G1_JOINT_NAMES)"),
            (ENV, "resolve_matching_names_values(G1_ACTION_SCALE, G1_JOINT_NAMES)"),
        ):
            with self.subTest(path=path.relative_to(REPO)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                calls = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
                self.assertIn(expected, calls)


if __name__ == "__main__":
    unittest.main()
