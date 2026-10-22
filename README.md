# Supermimic

复现论文SuperMimic（arXiv:2610.28480，CoRL 2026 Oral）。

论文用一个可微世界模型训练G1全身动作追踪策略。
作者未公开代码；本仓库为独立复现，`supermimic/`下的部分模块移植自第三方MIT实现
[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)（Isaac Lab 3 → mjlab），
许可与出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前状态

- 阶段：实现v0.1.0（训练闭环、评估、数据/自检脚本），规划v2.2。
- 目标层级：T2，即在129,514段基准上复现主结果；T3除实机外全部排期。
- 算力：8卡H100服务器负责训练与评估；RTX 5080与RTX 4070 Ti SUPER负责开发。
- 依赖版本：以`pyproject.toml`与`uv.lock`为准（mjlab 1.6.0、mujoco-warp 3.11.0、warp-lang 1.18.0、torch 2.14.1+cu130），规划D4已同步。
- 数据：复用Bote_Teleopit的HDF5格式与MotionLib；观测与奖励按论文重写（D2已定）。
- 数据/检查点/日志均不入库。

## 快速开始

```bash
# 数据：下载40段G1重定向LAFAN1（30 fps CSV）
uv run --frozen python scripts/download_data.py

# 环境自检：FK与仿真一致性、状态回环、参考回放奖励
uv run --frozen python scripts/check_env.py --num_envs 64

# 冒烟训练（1024 envs × 300 iters，约2.5 min，单卡）
uv run --frozen python -m supermimic.scripts.train \
  --num_envs 1024 --iterations 300 --buffer_steps 200 \
  --warmup_steps 16 --wm_only_iters 50 --wm_batch 1024 --pi_batch 1024 \
  --ckpt_every 300 --log_every 25 --log_dir logs/smoke

# 评估（附录F.1协议，逐检查点；--watch 常驻监视）
uv run --frozen python -m supermimic.scripts.evaluate --run_dir logs/smoke --watch
```

## 实现

目录职责参照 [mjlab](https://github.com/mujocolab/mjlab/tree/main/src/mjlab)：
任务放在`tasks/tracking/`，依赖该任务MDP的缓冲与网络放在任务内的`rl/`，
数学工具放在`utils/`，命令入口放在`scripts/`。
G1资产直接复用 [mjlab内置配置](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py)，
仿真和FK直接调用`get_g1_robot_cfg()`，不维护本地资产、机器人配置或碰撞覆盖。
`tasks/tracking/constants.py`只记录动作数据的关节顺序、跟踪刚体与足部名称；
默认姿态和动作缩放从mjlab内置值按名称读取。
mjlab默认开启自碰撞，不再提供`--self_collision`开关；旧关闭自碰撞的检查点
在当前默认环境下的评估结果，不应与原环境结果直接比较。
当前仍是自定义世界模型训练循环，不是mjlab的`ManagerBasedRlEnv`或PPO任务注册。
保留现有`uv`非安装模式，不引入`src/`、打包后端或新的依赖。

```text
supermimic/
  tasks/tracking/
    constants.py
    env.py
    mdp.py
    motion.py
    rl/
      buffer.py
      networks.py
  utils/
    kinematics.py
    rotations.py
  scripts/
    train.py
    evaluate.py
scripts/
  download_data.py
  check_env.py
tests/
  test_layout.py
  test_g1_defaults.py
docs/planning/
  plan.md
  rubric.json
```

| 路径 | 内容 |
| --- | --- |
| `supermimic/scripts/train.py` | 算法A.1：采样→拟合世界模型→想象推演BPTT→critic更新 |
| `supermimic/tasks/tracking/env.py` | mjlab TrackingSim：50 Hz控制、4×5 ms子步、域随机化、回合重置 |
| `supermimic/tasks/tracking/mdp.py` | 93维观测、表A.2奖励、终止、软存活权重、critic特征 |
| `supermimic/tasks/tracking/rl/networks.py` | 世界模型(MultiHead)、策略(编码器+共享解码器)、critic、RunningNorm |
| `supermimic/tasks/tracking/rl/buffer.py` | 每环境1000步FIFO回放缓冲、k步窗口与起点采样 |
| `supermimic/tasks/tracking/motion.py` | LAFAN1 G1 CSV→50 Hz动作库、目标帧、1 s重置分箱 |
| `supermimic/utils/kinematics.py` | MuJoCo前向运动学（身体位姿/速度） |
| `supermimic/tasks/tracking/constants.py` | 动作数据关节顺序与跟踪目标，不定义机器人配置 |
| `supermimic/scripts/evaluate.py` | 附录F.1评估：逐原始段首帧播放、失败计数、两种归一化回报 |
| `scripts/` | 数据下载与仿真自检 |
| `tests/` | 无GPU和第三方依赖的结构回归 |

在仓库根目录运行上述命令；也可用`python supermimic/scripts/train.py`或
`python supermimic/scripts/evaluate.py`直接启动。默认数据仍在根目录`data/`，
日志仍在`logs/`；显式传入的相对路径仍相对于当前工作目录。
内部导入路径已调整为上述分层，旧的平铺模块路径不再保留。

离线结构检查（不下载数据，不启动仿真或训练）：

```bash
python3 -B -m unittest discover -s tests -v
```

## 文档

| 文件 | 内容 |
| --- | --- |
| [docs/planning/plan.md](docs/planning/plan.md) | 主张、证据帐本、依赖、显存估算、验收标准、里程碑 |
| [docs/planning/rubric.json](docs/planning/rubric.json) | 加权评分树，叶子引用验收标准与帐本ID |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方代码许可与出处 |

## 使用约束

- 数据集与检查点不入库。
- BONES-SEED需在Hugging Face同意许可后下载。
- LAFAN1采用CC BY-NC-ND 4.0。
- AMASS：需注册，仅限非商用研究，不得再分发。
- 论文项目页的网页演示受专有许可约束。
  它只能人工浏览，不能下载、抓取或用作对照。
