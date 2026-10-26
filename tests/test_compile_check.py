"""Guard against a numerical comparison accidentally running the same kernel path twice."""

import copy
import unittest

import torch

from scripts.check_compile_equiv import compare_trees, kernel_scope, restore_training_state


class KernelScopeTests(unittest.TestCase):
    def test_frame_indices_must_be_exact_even_with_float_tolerances(self):
        self.assertFalse(compare_trees("frame indices", torch.tensor([100000]), torch.tensor([100001]), 1e-5, 1e-3))

    def test_optimizer_replay_does_not_mutate_saved_baseline(self):
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters())
        def update():
            opt.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            opt.step()
        update()
        saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
        results = []
        for _ in range(3):
            restore_training_state(model, opt, saved_model, saved_opt)
            update()
            results.append(model.weight.detach().clone())
            self.assertEqual(float(opt.state[model.weight]["step"]), 2)
        self.assertTrue(all(torch.equal(results[0], result) for result in results[1:]))
        self.assertEqual(float(next(iter(saved_opt["state"].values()))["step"]), 1)

    def test_shared_function_globals_switch_and_restore(self):
        namespace = {"inner": lambda: "eager"}
        exec("def step(): return inner()", namespace)
        step = namespace["step"]
        with kernel_scope(step.__globals__, {"inner": lambda: "compiled"}):
            self.assertEqual(step(), "compiled")
        self.assertEqual(step(), "eager")
        with self.assertRaises(RuntimeError):
            with kernel_scope(step.__globals__, {"inner": lambda: "compiled"}):
                raise RuntimeError("failure during comparison")
        self.assertEqual(step(), "eager")


if __name__ == "__main__":
    unittest.main()
