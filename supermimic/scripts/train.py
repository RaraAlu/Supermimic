# 移植自 zita-ch/supermimic_vibed（MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang）；详见 THIRD_PARTY_NOTICES.md。
"""SuperMimic 训练（算法 A.1）：采样、拟合世界模型，再通过 BPTT 更新策略和价值网络。

本模块负责命令行、分布式进程、数据加载和运行目录；训练状态与更新步骤位于 supermimic.tasks.tracking.rl.trainer。检查脚本可通过 setup() 获取预热后的 Trainer，无需运行训练循环。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--wm_update", type=str, default="steps", choices=["steps", "single"],
                    help="one world-model update per iteration: --wm_batches optimizer steps of --wm_batch windows "
                         "each (steps), or one step over wm_batches x wm_batch windows (single); paper Table A.5 "
                         "'10 batches of 2048 windows per update' with one policy update per world-model update admits both")
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--clean_action_frac", type=float, default=0.0,
                    help="fraction of each rank's environments collected without action noise (not in the paper; 0 keeps "
                         "the paper's behavior). The policy imagines with its deterministic actions, while a buffer of only "
                         "noisy actions leaves the world model without data along them; also writes clean_log.csv")
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--critic_cap", type=str, default="on", choices=["on", "off"],
                    help="bound the critic's value by the best possible return MAX_STEP_REWARD / (1 - gamma) wherever it is "
                         "used (GAE bootstrap and the policy objective's terminal value); the regression still fits the "
                         "raw output, so an overshoot is pulled back while the policy cannot chase it. Never changes a "
                         "correct critic")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--term_log_every", type=int, default=500,
                    help="每这么多次迭代写一次终止集中情况（termwin_log.csv与term_bins.csv，见README）；0为不写")
parser.add_argument("--term_top", type=int, default=20, help="term_bins.csv每次记录终止最多的前几个片段")
parser.add_argument("--keep_states", type=int, default=3,
                    help="full training states to keep: last.pt plus the previous saves as last_prev1.pt, ...; a collapse "
                         "then leaves the state from before it for keep - 1 more saves (1: only last.pt)")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--tensorboard", type=str, default="auto", choices=["auto", "on", "off"],
                    help="also write every log row as TensorBoard scalars to LOG_DIR/tensorboard (auto: only if tensorboard is installed)")
source = parser.add_mutually_exclusive_group()
source.add_argument("--motion_dir", type=str, default=None)
source.add_argument("--motion_index", type=str, default=None)
source.add_argument("--motion_source", type=str, default=None)
parser.add_argument("--motion_root", type=str, default=None)
parser.add_argument("--motion_limit", type=int, default=None, help="explicit development subset, never an implicit memory fallback")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--fused_adam", type=str, default="on", choices=["on", "off"],
                    help="fused CUDA Adam kernels for all three optimizers (same update up to float rounding)")
parser.add_argument("--compile", type=str, default="auto", choices=["auto", "on", "off"], help="torch.compile (auto: only if triton is installed)")
parser.add_argument("--cuda_graph", type=str, default="auto", choices=["auto", "on", "off"],
                    help="replay the world-model step and the policy imagination+BPTT step as CUDA graphs "
                         "(auto: single CUDA process without SM_PROFILE, including torch.compile; multi-GPU needs on)")
parser.add_argument("--cuda_graph_critic", type=str, default="on", choices=["on", "off"],
                    help="with CUDA graphs on, also replay the critic update (GAE targets and its minibatch steps)")
parser.add_argument("--cuda_graph_env", type=str, default="on", choices=["on", "off"],
                    help="with CUDA graphs on, also replay the simulator step's reward and termination")
parser.add_argument("--cuda_graph_collect", type=str, default="on", choices=["on", "off"],
                    help="with CUDA graphs on, replay collection observations, normalizers, policy and action noise")
parser.add_argument("--collect_overlap", type=str, default="on", choices=["on", "off"],
                    help="enqueue replay writes and world-model statistics before the simulator reset waits for CUDA")
parser.add_argument("--bin_clip_end", type=str, default="bootstrap", choices=["bootstrap", "terminal"],
                    help="reset-bin scores: bootstrap episodes that reach the clip end with their mean reward and "
                         "normalize by the global maximum, or end the return there and normalize by the maximum "
                         "attainable from each start frame (paper Appendix D.1 leaves this open)")
parser.add_argument("--bin_prior", type=str, default="zero", choices=["zero", "mean"],
                    help="score of reset bins without a finished episode: 0, or the mean of visited bins")
parser.add_argument("--bin_skip_steps", type=int, default=50,
                    help="score reset bins from this step of each episode on: the reset transient (reset noise, zero "
                         "previous action, repeated observation history) costs reward for about 1 s regardless of the "
                         "bin and would pull resets onto clip ends; 0 scores from the start as before")
parser.add_argument("--reset_margin", type=int, default=100,
                    help="no reset starts within this many steps of a clip's end (a shorter clip starts only at its "
                         "first frame), so every scored episode outlasts the transient; 0 allows every start")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_scope", type=str, default="global", choices=["global", "per_gpu"],
                    help="--num_envs/--wm_batch/--pi_batch are global totals split across ranks, or per-GPU sizes; "
                         "paper Table A.5 explicitly labels only environments per GPU")
parser.add_argument("--lr_scaling", type=str, default="none", choices=["none", "sqrt"],
                    help="multiply --lr by sqrt(WORLD_SIZE); the paper uses sqrt(2) x for two GPUs")
parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)


def tensorboard_enabled(args) -> bool:
    """根据命令行设置及 TensorBoard 是否可用决定是否启用日志。"""
    import importlib.util

    return args.tensorboard == "on" or (args.tensorboard == "auto" and importlib.util.find_spec("tensorboard") is not None)


def setup(argv: list[str] | None = None):
    """解析参数、建立进程组、加载参考数据并写运行元数据，返回(args, parallel, 预热后的Trainer)。

    参数解析先于torch等重型导入，--help不必加载它们。
    """
    args = parser.parse_args(argv)

    import torch

    from supermimic.tasks.tracking.motion import MotionLib, check_reference, resolve_motion_args
    import supermimic.tasks.tracking.rl.networks as networks
    from supermimic.tasks.tracking.rl.trainer import Trainer, make_layout
    from supermimic.utils.distributed import DistributedContext
    from supermimic.utils.kinematics import G1Kinematics

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    parallel = DistributedContext(args.device)
    dev = parallel.device
    with parallel.coordinated("argument validation"):
        resume_ck = torch.load(args.resume, map_location="cpu") if args.resume else None
        resolve_motion_args(args, resume_ck["args"] if resume_ck else None)
        layout = make_layout(args, parallel)
    parallel.require_equal({k: v for k, v in vars(args).items() if k != "local_rank"}, "training arguments")

    # 每卡载入自己的完整片段分片；先确认存在足够长的片段，再进入需要固定时域的训练。
    with parallel.coordinated("reference loading"):
        kin = G1Kinematics(dev)
        motion = MotionLib.from_args(args, kin, dev, parallel.rank, parallel.world_size)
        if resume_ck:
            check_reference(motion.global_description, resume_ck.get("reference"))
        if not bool((motion.clip_len > args.horizon).any()):
            raise ValueError("this reference shard has no complete imagination horizon; use fewer ranks or a larger selection")
    parallel.require_equal(motion.global_description["fingerprint"], "global reference fingerprint")
    # 主进程写全局运行信息，各卡随后单独记录本地分片和实际批量，便于复现实验资源配置。
    with parallel.coordinated("run metadata"):
        if parallel.primary:
            os.makedirs(args.log_dir, exist_ok=True)
            reference_path = Path(args.log_dir) / "reference.json"
            if reference_path.exists():
                check_reference(motion.global_description, json.loads(reference_path.read_text(encoding="utf-8")))
            reference_path.write_text(json.dumps(motion.global_description, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(os.path.join(args.log_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)
            (Path(args.log_dir) / "shards" / f"world_{parallel.world_size}").mkdir(parents=True, exist_ok=True)
    with parallel.coordinated("shard manifest"):
        shard_path = Path(args.log_dir) / "shards" / f"world_{parallel.world_size}" / f"rank_{parallel.rank:04d}.json"
        local_reference = {**motion.description, "runtime": {
            "device": dev, "local_envs": layout.local_envs, "local_wm_batch": layout.local_wm_batch,
            "wm_step_batch": layout.wm_step_batch, "wm_steps": layout.wm_steps, "local_pi_batch": layout.local_pi_batch,
            "batch_scope": args.batch_scope, "global_envs": layout.global_envs, "base_lr": layout.base_lr}}
        shard_path.write_text(json.dumps(local_reference, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rank {parallel.rank}/{parallel.world_size} on {dev}: {len(motion.names)} full clips, "
          f"{motion.num_frames} local frames, reference {motion.description['reference_tensor_bytes'] / 1024**3:.3f} GiB", flush=True)
    if tensorboard_enabled(args):
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401  （启用时若缺少依赖，在预热前就报错）

    trainer = Trainer(args, parallel, layout, kin, motion, resume_ck)
    # 经验回放缓冲区不写入检查点，因此恢复训练后也需要采集新的转移。
    trainer.collect(args.warmup_steps)
    return args, parallel, trainer


def main(argv: list[str] | None = None):
    """初始化训练状态，循环执行训练、记录日志并保存检查点。"""
    args, parallel, trainer = setup(argv)
    from supermimic.tasks.tracking.rl.trainer import CLEAN_TABLE, LOG_TABLES
    from supermimic.utils.run_log import RunLog

    # train/sampling/diag各一份CSV，后两者单独成文件，恢复旧run时train_log.csv的列不变；TensorBoard用purge_step
    # 隐藏上次运行在恢复点之后写下的事件（恢复点及之前的行仍有效）
    start_it = trainer.start_it
    tables = LOG_TABLES | (CLEAN_TABLE if args.clean_action_frac > 0 else {})
    log = RunLog(args.log_dir, tables, tensorboard=tensorboard_enabled(args),
                 purge_step=start_it + 1 if start_it else None) if parallel.primary else None
    trainer.start_clock()
    for it in range(start_it, args.iterations):
        lr = trainer.iteration(it)
        if (it + 1) % args.log_every == 0:
            rows = trainer.log_rows(it, lr)
            if log:
                log.write(rows)
                r = rows["train"]
                print(f"it {it + 1} | {r[1]:.3f} h | wm {r[3]:.4f} e8 {r[4]:.4f} | pi {r[5]:.3f} v {r[6]:.4f} imag_r {r[7]:.3f} surv {r[9]:.3f} "
                      f"| sim_r {r[10]:.3f} ep_len {r[11]:.1f} ep_ret {r[12]:.3f} term {r[13]:.2f} | {r[14]:.2f} it/s", flush=True)  # fmt: skip
                if "clean" in rows:
                    c = rows["clean"]
                    print(f"   clean sim_r {c[1]:.3f} ep_ret {c[4]:.3f} term {c[5]:.2f} | noisy sim_r {c[2]:.3f} ep_ret {c[7]:.3f} "
                          f"term {c[8]:.2f}", flush=True)
        if args.term_log_every and (it + 1) % args.term_log_every == 0:
            summary, lines = trainer.term_rows(it)  # 集合通信，所有rank都调用
            if log:
                log.write(summary)
                log.write_lines("term_bins", lines)
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            with parallel.coordinated("checkpoint write"):
                bins = trainer.gather_bins()  # 集合通信：各rank的课程状态汇到主进程
                if parallel.primary:
                    trainer.save(it, args.log_dir, args.keep_states, bins)
    if log:
        log.close()
    # 先释放可能包含 NCCL 操作的捕获图，再销毁进程组，保持资源依赖的释放顺序。
    trainer.close()
    parallel.close()


if __name__ == "__main__":
    main()
