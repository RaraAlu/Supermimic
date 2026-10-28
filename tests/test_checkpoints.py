"""rotate_states keeps earlier copies of a rolling checkpoint without touching the current file."""

import os
import tempfile
import unittest
from pathlib import Path

from supermimic.utils.checkpoints import rotate_states


def save(path, text):
    """The way train.py writes: a temporary file, then an atomic replace."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class RotateStatesTests(unittest.TestCase):
    def files(self, root):
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(root).iterdir())}

    def test_keeps_the_last_saves_under_stable_names(self):
        with tempfile.TemporaryDirectory() as root:
            last = Path(root) / "last.pt"
            for i in range(1, 6):
                rotate_states(str(last), 3)  # before the write, as in train.py
                save(last, f"s{i}")
            self.assertEqual(self.files(root), {"last.pt": "s5", "last_prev1.pt": "s4", "last_prev2.pt": "s3"})

    def test_two_files_and_one_file(self):
        with tempfile.TemporaryDirectory() as root:
            last = Path(root) / "last.pt"
            for i in range(1, 4):
                rotate_states(str(last), 2)
                save(last, f"s{i}")
            self.assertEqual(self.files(root), {"last.pt": "s3", "last_prev1.pt": "s2"})
        with tempfile.TemporaryDirectory() as root:
            last = Path(root) / "last.pt"
            for i in range(1, 4):
                rotate_states(str(last), 1)
                save(last, f"s{i}")
            self.assertEqual(self.files(root), {"last.pt": "s3"})

    def test_first_save_has_nothing_to_rotate(self):
        with tempfile.TemporaryDirectory() as root:
            rotate_states(str(Path(root) / "last.pt"), 3)
            self.assertEqual(self.files(root), {})


if __name__ == "__main__":
    unittest.main()
