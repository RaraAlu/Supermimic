"""Launcher checks without loading ONNX, MuJoCo, or the local demo assets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
DIRNAME = shutil.which("dirname")
ASSETS = (
    "checkpoints/onnx/dance/dance_ckpt20000.onnx",
    "checkpoints/onnx/dance/dance_ckpt20000.json",
    "data/motions/dance/zhongzhuangyuan1_umr_50hz.npz",
    "scripts/infer_onnx_minimal.py",
)


@unittest.skipUnless(os.name == "posix" and BASH and DIRNAME, "requires POSIX, Bash, and dirname")
class DanceDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repository with spaces"
        self.script = self.repo / "scripts/run_dance_demo.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(REPO / "scripts/run_dance_demo.sh", self.script)
        for relative in ASSETS:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test fixture\n", encoding="utf-8")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "dirname").symlink_to(DIRNAME)
        self.uv = self.bin / "uv"
        self.uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
            "sys.exit(int(os.environ.get('DEMO_TEST_EXIT', '0')))\n",
            encoding="utf-8",
        )
        self.uv.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin), DISPLAY=":99")
        self.env.pop("WAYLAND_DISPLAY", None)

    def run_demo(self, *args):
        return subprocess.run([BASH, str(self.script), *args], cwd=self.root, env=self.env,
                              text=True, capture_output=True, timeout=10)

    def invocation(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_default_viewer_from_outside_repository_with_spaces(self):
        call = self.invocation(self.run_demo())
        self.assertEqual(call["cwd"], str(self.root))
        self.assertEqual(call["args"], [
            "run", "--project", str(self.repo), "--frozen", "--with", "onnxruntime==1.31.1",
            "python", "-B", str(self.repo / ASSETS[3]), str(self.repo / ASSETS[0]),
            str(self.repo / ASSETS[2]), "--viewer",
        ])

    def test_headless_and_forwarded_paths_need_no_display(self):
        self.env.pop("DISPLAY")
        call = self.invocation(self.run_demo("--headless", "--save", "saved motion.npz", "--xml", "robot model.xml"))
        self.assertNotIn("--viewer", call["args"])
        self.assertNotIn("--headless", call["args"])
        self.assertEqual(call["args"][-4:], ["--save", "saved motion.npz", "--xml", "robot model.xml"])

    def test_help_requires_neither_assets_nor_uv_nor_display(self):
        self.uv.unlink()
        self.env.pop("DISPLAY")
        for relative in ASSETS:
            (self.repo / relative).unlink()
        result = self.run_demo("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--headless", result.stdout)

    def test_missing_asset_reports_exact_path(self):
        for relative in ASSETS:
            with self.subTest(path=relative):
                path = self.repo / relative
                path.unlink()
                result = self.run_demo()
                self.assertEqual(result.returncode, 2)
                self.assertIn(str(path), result.stderr)
                path.write_text("test fixture\n", encoding="utf-8")

    def test_empty_asset_is_rejected(self):
        (self.repo / ASSETS[0]).write_bytes(b"")
        result = self.run_demo()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing or empty file", result.stderr)

    def test_missing_uv_is_reported(self):
        self.uv.unlink()
        result = self.run_demo()
        self.assertEqual(result.returncode, 2)
        self.assertIn("uv is not installed", result.stderr)

    def test_viewer_requires_display(self):
        self.env.pop("DISPLAY")
        result = self.run_demo()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--headless", result.stderr)

    def test_unknown_argument_is_rejected(self):
        result = self.run_demo("--typo")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown argument: --typo", result.stderr)

    def test_option_requires_value(self):
        for option in ("--save", "--xml"):
            for suffix in ((), ("",), ("--headless",)):
                with self.subTest(option=option, suffix=suffix):
                    result = self.run_demo(option, *suffix)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"{option} requires a path", result.stderr)

    def test_inference_exit_status_is_preserved(self):
        self.env["DEMO_TEST_EXIT"] = "7"
        self.assertEqual(self.run_demo().returncode, 7)


if __name__ == "__main__":
    unittest.main()
