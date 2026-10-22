"""Offline layout checks; never import simulator or training implementations."""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCES = sorted((REPO / "supermimic").rglob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
COMMANDS = ("supermimic/scripts/train.py", "supermimic/scripts/evaluate.py", "scripts/check_env.py")


def repo_namespace(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REPO" for t in node.targets)
    )
    namespace = {"__file__": str(path), "os": os, "Path": Path}
    namespace["REPO"] = eval(compile(ast.Expression(assignment.value), str(path), "eval"), namespace)
    return tree, namespace


class LayoutTests(unittest.TestCase):
    def test_python_sources_compile(self):
        for path in SOURCES:
            with self.subTest(path=path.relative_to(REPO)):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_package_initializers_have_no_side_effects(self):
        for path in (REPO / "supermimic").rglob("__init__.py"):
            with self.subTest(path=path.relative_to(REPO)):
                body = ast.parse(path.read_text(encoding="utf-8")).body
                self.assertTrue(all(
                    isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) for node in body
                ))

    def test_internal_imports_resolve(self):
        for path in SOURCES:
            package = ".".join(path.relative_to(REPO).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [importlib.util.resolve_name("." * node.level + (node.module or ""), package)]
                for name in names:
                    if name == "supermimic" or name.startswith("supermimic."):
                        with self.subTest(path=path.relative_to(REPO), module=name):
                            spec = importlib.util.find_spec(name)
                            self.assertIsNotNone(spec)
                            self.assertTrue(Path(spec.origin).resolve().is_relative_to(REPO / "supermimic"))

    def test_default_repository_paths(self):
        for relative in (*COMMANDS, "scripts/download_data.py"):
            path = REPO / relative
            tree, namespace = repo_namespace(path)
            with self.subTest(path=relative):
                self.assertEqual(Path(namespace["REPO"]).resolve(), REPO)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
                        continue
                    option = node.args[0].value
                    if option in ("--motion_dir", "--log_dir"):
                        default = next(k.value for k in node.keywords if k.arg == "default")
                        value = eval(compile(ast.Expression(default), str(path), "eval"), namespace)
                        if option == "--motion_dir":
                            self.assertIsNone(value)  # Resolved after restoring checkpoint selection.
                        else:
                            self.assertEqual(Path(value), REPO / "logs/run")

    def test_data_checker_help_without_dependencies(self):
        result = subprocess.run([sys.executable, "-B", str(REPO / "scripts/check_data.py"), "--help"],
                                cwd="/" if os.name != "nt" else tempfile.gettempdir(),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--verify_frames", result.stdout)

    def test_cli_help_by_path_outside_repository(self):
        with tempfile.TemporaryDirectory() as cwd:
            for relative in COMMANDS:
                with self.subTest(command=relative):
                    result = subprocess.run(
                        [sys.executable, "-B", str(REPO / relative), "--help"],
                        cwd=cwd, capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--device", result.stdout)

    def test_cli_help_by_module(self):
        for name in ("train", "evaluate"):
            with self.subTest(command=name):
                result = subprocess.run(
                    [sys.executable, "-B", "-m", f"supermimic.scripts.{name}", "--help"],
                    cwd=REPO, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--motion_dir", result.stdout)
                self.assertNotIn("--self_collision", result.stdout)

    def test_readme_paths_exist(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^)]+)\)", readme)
        code_paths = re.findall(r"`((?:supermimic|scripts|tests)/[^`]+)`", readme)
        for target in links + code_paths:
            if "://" not in target:
                with self.subTest(target=target):
                    self.assertTrue((REPO / target).exists())
        self.assertIn("python -m supermimic.scripts.train", readme)
        self.assertIn("python -m supermimic.scripts.evaluate", readme)


if __name__ == "__main__":
    unittest.main()
