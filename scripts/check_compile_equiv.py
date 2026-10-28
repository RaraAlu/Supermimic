"""Compare eager and compiled updates from the same model, optimizer and RNG state.

Extra arguments are forwarded to supermimic.scripts.train.setup. The compiler uses fallback_random=True so that
both paths receive identical noise; this checks arithmetic, not equivalence of the two RNG
implementations. The normal training path keeps its default compiler RNG implementation.

    python scripts/check_compile_equiv.py --motion_source SOURCE --motion_limit 8 --amp none

Numerical tolerances are explicit (--atol/--rtol); a mismatch exits nonzero. The test also
reports errors in parameter *updates*, so the size of unchanged weights cannot hide drift.
"""

import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

import torch
from torch.utils._pytree import tree_flatten


@contextmanager
def kernel_scope(namespace, kernels):
    """在namespace（Trainer实例的属性字典）中临时换上另一组内核；各更新步骤都经它查找内核。"""
    original = {key: namespace[key] for key in kernels}
    namespace.update(kernels)
    try:
        yield
    finally:
        namespace.update(original)


def compare_trees(label, eager, compiled, atol, rtol):
    left, spec_left = tree_flatten(eager)
    right, spec_right = tree_flatten(compiled)
    if spec_left != spec_right:
        raise AssertionError(f"{label}: output structures differ")
    failures, maximum, max_relative = [], 0.0, 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if not torch.is_tensor(a):
            if a != b:
                failures.append(f"leaf {index}: {a!r} != {b!r}")
            continue
        diff = (a.detach().float() - b.detach().float()).abs()
        if diff.numel():
            maximum = max(maximum, float(diff.max()))
            relative = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a.detach().float()).clamp_min(atol)
            max_relative = max(max_relative, float(relative))
        try:
            floating = a.is_floating_point() or a.is_complex()
            torch.testing.assert_close(a, b, atol=atol if floating else 0, rtol=rtol if floating else 0)
        except AssertionError as exc:
            failures.append(f"leaf {index}: {exc}")
    print(f"{label}: max_abs={maximum:.3e}, max_relative_l2={max_relative:.3e}, pass={not failures}", flush=True)
    for failure in failures[:3]:
        print(failure, flush=True)
    return not failures


def restore_training_state(module, optimizer, saved_model, saved_optimizer):
    module.load_state_dict(saved_model)
    # load_state_dict can retain optimizer-state tensor references on the same device. Each replay
    # needs its own copy, or the warmup mutates the baseline's Adam step and moments.
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    options, training_args = parser.parse_known_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("this arithmetic comparison runs on one GPU; use check_cuda_graph.py for multi-GPU replay checks")
    from supermimic.scripts import train
    from supermimic.tasks.tracking.mdp import unflatten_state
    from supermimic.tasks.tracking.rl.trainer import set_requires_grad

    _, parallel, trainer = train.setup(["--num_envs", "512", "--buffer_steps", "100", "--wm_batch", "512",
                                        "--pi_batch", "512", "--warmup_steps", "40", "--motion_limit", "8",
                                        "--log_dir", os.path.join(REPO, "logs", "check_compile_equiv"), *training_args,
                                        "--iterations", "0", "--cuda_graph", "off", "--compile", "off"])
    kernels_namespace = vars(trainer)
    eager_kernels = {name: getattr(trainer, name) for name in ("wm_rollout", "wm_err", "imag_step", "reward_term", "critic_input_c")}
    compiled_kernels = {name: torch.compile(fn, options={"fallback_random": True}) for name, fn in eager_kernels.items()}
    print("compiler RNG: fallback_random=True (identical noise for arithmetic comparison)", flush=True)

    def compare(name, module, optimizer, inputs, pre=lambda: None, post=lambda: None):
        fn = getattr(trainer, f"_{name}")
        with kernel_scope(kernels_namespace, eager_kernels):
            pre()
            fn(*inputs)  # initialize Adam state before taking the baseline
            post()
        saved_model, saved_opt = copy.deepcopy(module.state_dict()), copy.deepcopy(optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        baseline_params = {key: value.detach().clone() for key, value in module.named_parameters()}

        def run(kernels):
            restore_training_state(module, optimizer, saved_model, saved_opt)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            trainer.nonfinite.zero_()
            with kernel_scope(kernels_namespace, kernels):
                pre()
                try:
                    outputs = fn(*inputs)
                finally:
                    post()
            torch.cuda.synchronize()
            updates = {key: value.detach().clone() - baseline_params[key] for key, value in module.named_parameters()}
            return outputs, copy.deepcopy(module.state_dict()), updates, copy.deepcopy(optimizer.state_dict())

        run(compiled_kernels)  # compilation/warmup may execute updates; run() restores before both comparisons
        left = run(eager_kernels)
        right = run(compiled_kernels)
        results = [compare_trees(f"{name}/{label}", a, b, options.atol, options.rtol)
                   for label, a, b in zip(("outputs", "model", "parameter_updates", "optimizer"), left, right)]
        return all(results)

    w = trainer.buffer.sample_windows(trainer.layout.local_wm_batch, trainer.args.wm_window)
    ok = compare("wm_step", trainer.wm, trainer.opt_wm, (w["s"], w["a"], w["xi"], w["s1"], w["mask"]))
    s, h = trainer.sample_valid_starts(trainer.layout.local_pi_batch)
    with torch.no_grad():
        history = trainer.mdp.add_noise(trainer.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]))
    wm, critic = trainer.wm, trainer.critic
    ok &= compare("policy_step", trainer.pi, trainer.opt_pi, (s["s"], s["xi"], s["f"], s["a_prev"], history),
                  pre=lambda: (set_requires_grad(wm, False), set_requires_grad(critic, False)),
                  post=lambda: (set_requires_grad(wm, True), set_requires_grad(critic, True)))
    print(f"RESULT {'PASS' if ok else 'FAIL'} (atol={options.atol}, rtol={options.rtol})", flush=True)
    trainer.close()
    parallel.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
