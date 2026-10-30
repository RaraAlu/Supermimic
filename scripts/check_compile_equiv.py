"""从相同模型、优化器和随机数状态比较普通执行与编译后的更新。

额外参数转交 train.setup。编译器设置 fallback_random=True 以提供相同噪声，仅检查算术结果，
不检查两种随机数实现的等价性；正常训练仍使用默认编译器随机数实现。
--atol 和 --rtol 指定数值容差，任一不匹配使程序非零退出。另比较参数更新量，避免未变化权重掩盖偏移。"""

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
    """逐叶比较嵌套输出，按给定容差检查张量并报告最大绝对与相对误差。"""
    left, spec_left = tree_flatten(eager)
    right, spec_right = tree_flatten(compiled)
    if spec_left != spec_right:
        raise AssertionError(f"{label}: output structures differ")
    # 既检查嵌套结构与非张量值，也检查张量误差；整数和布尔张量必须完全一致。
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
    """恢复模型及独立的优化器状态副本，避免预热污染比较基准。"""
    module.load_state_dict(saved_model)
    # load_state_dict 可能保留同一设备上的优化器状态张量引用；每次重放
    # 都需独立副本，避免预热修改基准 Adam 步数及动量。
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))


def main():
    """建立单卡训练状态，对普通与编译内核逐项比较并以退出码报告结果。"""
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
        """从一致的训练与随机数状态执行两种计算路径，比较输出及参数更新。"""
        fn = getattr(trainer, f"_{name}")
        with kernel_scope(kernels_namespace, eager_kernels):
            pre()
            fn(*inputs)  # 保存比较基准前先初始化 Adam 状态
            post()
        saved_model, saved_opt = copy.deepcopy(module.state_dict()), copy.deepcopy(optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        # 保留更新前参数，用增量单独比较，避免大权重本身让相对容差掩盖微小但重要的更新差异。
        baseline_params = {key: value.detach().clone() for key, value in module.named_parameters()}

        def run(kernels):
            """恢复基准状态和随机数，执行指定内核并返回输出、状态及参数增量。"""
            restore_training_state(module, optimizer, saved_model, saved_opt)
            # 每次都恢复 CPU 与 CUDA 随机数状态，使编译和普通执行使用相同样本噪声。
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

        run(compiled_kernels)  # 编译和预热可能执行更新；run() 在两次比较前都会恢复状态
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
