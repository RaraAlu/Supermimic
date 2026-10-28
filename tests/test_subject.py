"""诊断脚本兼容层scripts/_subject.py的离线检查：用假的新旧源码树，不启动仿真。

Subject会把--src插到sys.path最前面并导入其中的supermimic，而测试进程已经导入了真正的包，
所以每个用例在子进程中运行。
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 旧源码树：没有GOAL_FRAME_DIM、soft_survival，TrackingSim也没有分箱选项
OLD_TREE = {
    "supermimic/tasks/tracking/mdp.py": "STATE_DIM = 71\nXI_DIM = 33\n",
    "supermimic/tasks/tracking/motion.py": "",
    "supermimic/tasks/tracking/env.py": textwrap.dedent("""
        class TrackingSim:
            def __init__(self, num_envs, motion, mdp, device, gamma=0.99, eval_mode=False):
                self.kw = {"gamma": gamma, "eval_mode": eval_mode}
    """),
}
NEW_TREE = {
    **OLD_TREE,
    "supermimic/tasks/tracking/motion.py": "GOAL_FRAME_DIM = 80\n",
    "supermimic/tasks/tracking/env.py": textwrap.dedent("""
        class TrackingSim:
            def __init__(self, num_envs, motion, mdp, device, gamma=0.99, eval_mode=False, bin_prior="zero"):
                self.kw = {"gamma": gamma, "eval_mode": eval_mode, "bin_prior": bin_prior}
    """),
}


def write_tree(root: Path, files: dict) -> Path:
    for package in ("supermimic", "supermimic/tasks", "supermimic/tasks/tracking"):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    for path, source in files.items():
        (root / path).write_text(source, encoding="utf-8")
    return root


def run(src: Path, body: str, extra_path: Path | None = None) -> dict:
    """在子进程里对src建Subject并执行body，body把结果写入字典result。"""
    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(REPO / "scripts")!r})
        {"sys.path.append(" + repr(str(extra_path)) + ")" if extra_path else ""}
        import _subject
        result = {{}}
        try:
            subject = _subject.Subject({str(src)!r}, "cpu")
    """) + textwrap.indent(textwrap.dedent(body), "    ") + textwrap.dedent("""
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result))
    """)
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=src.parent, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


ENV_CASES = """
result["goal_frame_dim"] = subject.GOAL_FRAME_DIM
result["state_dim"], result["xi_dim"] = subject.STATE_DIM, subject.XI_DIM
env = subject.make_env(8, None, None, run_args={"gamma": 0.9, "bin_prior": "mean"}, inherit=("gamma", "bin_prior", "lam"))
result["inherited"] = env.kw
result["provenance"] = subject.provenance()
try:
    subject.make_env(8, None, None, bin_prior="mean")
    result["explicit"] = "accepted"
except TypeError as exc:
    result["explicit"] = "TypeError"
mdp = type("OldMDP", (), {"soft_termination": staticmethod(lambda S, f: 0.25)})()
result["survival"] = subject.survival_prob(mdp)(None, None)
"""


class SubjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_tree_falls_back_and_keeps_its_defaults(self):
        result = run(write_tree(self.root / "old", OLD_TREE), ENV_CASES)
        self.assertNotIn("error", result)
        self.assertEqual(result["goal_frame_dim"], 73)
        self.assertEqual((result["state_dim"], result["xi_dim"]), (71, 33))
        # 旧树不接受bin_prior：继承时跳过并保留默认值，显式传入则报错
        self.assertEqual(result["inherited"], {"gamma": 0.9, "eval_mode": False})
        self.assertEqual(result["provenance"]["env_options"], {"gamma": 0.9})
        self.assertIsNone(result["provenance"]["commit"])
        self.assertIsNone(result["provenance"]["dirty"])
        self.assertEqual(result["explicit"], "TypeError")
        self.assertEqual(result["survival"], 0.75)

    def test_new_tree_takes_its_names_and_options(self):
        result = run(write_tree(self.root / "new", NEW_TREE), ENV_CASES)
        self.assertNotIn("error", result)
        self.assertEqual(result["goal_frame_dim"], 80)
        self.assertEqual(result["inherited"], {"gamma": 0.9, "eval_mode": False, "bin_prior": "mean"})
        self.assertEqual(result["provenance"]["env_options"], {"gamma": 0.9, "bin_prior": "mean"})
        self.assertEqual(result["explicit"], "accepted")

    def test_src_without_supermimic_is_rejected(self):
        # --src里没有supermimic时，会导入sys.path上别处的包；必须报错，不能静默测错代码
        empty = self.root / "empty"
        empty.mkdir()
        result = run(empty, "", extra_path=REPO)
        self.assertIn("ImportError", result.get("error", ""))
        self.assertIn("not from --src", result["error"])


if __name__ == "__main__":
    unittest.main()
