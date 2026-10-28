"""Exercise evaluation batching without importing the simulator or running a CLI."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from supermimic.tasks.tracking.motion import check_reference

REPO = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FakePolicy:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def load_state_dict(self, state):
        pass

    def eval(self):
        pass

    def __call__(self, hist, goal):
        return torch.zeros(len(hist), 29)


class FakeEnv:
    def __init__(self, size, motion):
        self.N, self.motion = size, motion
        self.done_eval = torch.zeros(size, dtype=torch.bool)
        self.failed_eval = torch.zeros(size, dtype=torch.bool)
        self.a_prev = torch.zeros(size, 29)
        self.xi = torch.zeros(size, 1)
        self.batches = []

    def reset(self, ids, frames):
        self.f = frames
        self.clips = self.motion.clip_id[frames]
        self.batches.append(self.clips.tolist())

    def state(self):
        return (self.f,)

    def step(self, action):
        self.f = self.f + 1
        term = (self.clips == 1) & (self.f - self.motion.clip_start[self.clips] >= 2)
        self.failed_eval |= term & ~self.done_eval
        self.done_eval |= term | (self.f >= self.motion.clip_end[self.clips])
        return {"r": torch.full((self.N,), 0.1)}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        lengths = torch.tensor([2, 51, 4, 3, 6])
        starts = lengths.cumsum(0) - lengths
        self.motion = SimpleNamespace(names=[str(i) for i in range(5)], clip_len=lengths,
                                      clip_start=starts, clip_end=starts + lengths - 1,
                                      clip_id=torch.repeat_interleave(torch.arange(5), lengths),
                                      description={"fingerprint": "test"},
                                      goal=lambda f, k: torch.zeros(len(f), 73 * k))
        ns = {"torch": torch, "TrackingSim": FakeEnv, "motion": self.motion, "C": 5,
              "check_reference": check_reference, "dev": "cpu", "Policy": FakePolicy,
              "networks": SimpleNamespace(), "OBS_DIM": 1, "GOAL_FRAME_DIM": 73, "MAX_STEP_REWARD": 0.1,
              "mdp": SimpleNamespace(obs=lambda s, a, xi, f: torch.zeros(len(f), 1))}
        self.evaluate = load_function("supermimic/scripts/evaluate.py", "evaluate", ns)
        self.ck = {"args": {"goal_frames": 1, "hist": 2}, "pi": {},
                   "reference": self.motion.description}

    def test_padding_and_batch_independent_aggregation(self):
        single = self.evaluate(FakeEnv(5, self.motion), self.ck)
        for size in (1, 2, 3):
            with self.subTest(batch_size=size):
                env = FakeEnv(size, self.motion)
                batched = self.evaluate(env, self.ck)
                self.assertEqual(batched, single)
                self.assertEqual(batched["n_failed"], 1)
                self.assertEqual(batched["failed"], ["1"])
                self.assertAlmostEqual(batched["norm_return"], (4 + 0.04) / 5, places=6)
                self.assertAlmostEqual(batched["norm_return_framew"], 13 / 61, places=6)

    def test_batches_take_longest_clips_first(self):
        single = FakeEnv(5, self.motion)
        self.evaluate(single, self.ck)
        self.assertEqual(single.batches, [[0, 1, 2, 3, 4]])
        batched = FakeEnv(2, self.motion)
        self.evaluate(batched, self.ck)
        self.assertEqual(batched.batches, [[1, 4], [2, 3], [0, 0]])

    def test_checkpoint_mismatch_rejected(self):
        self.ck["reference"] = {"fingerprint": "different"}
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.evaluate(FakeEnv(2, self.motion), self.ck)

    def test_short_replay_sampling_is_bounded(self):
        ns = {"buffer": SimpleNamespace(sample_starts=lambda b, h: ({"f": torch.tensor([1])}, {})),
              "HZ": 32, "Hh": 1,
              "motion": SimpleNamespace(clip_end=torch.tensor([1]), clip_id=torch.tensor([0, 0]))}
        sample = load_function("supermimic/scripts/train.py", "sample_valid_starts", ns)
        with self.assertRaisesRegex(RuntimeError, "too few starts"):
            sample(1)


if __name__ == "__main__":
    unittest.main()
