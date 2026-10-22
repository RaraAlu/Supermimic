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
- 数据：支持 LAFAN1 CSV 与索引白名单选择的原生 50 Hz G1 HDF5；本仓库 MotionLib 生成参考速度和 FK。
- 数据/检查点/日志均不入库。

## 平台

Windows与Linux（x86_64，NVIDIA GPU）共用同一份`pyproject.toml`、`uv.lock`与`.python-version`（Python 3.12）；
`uv sync --frozen`按平台自动选择mjlab、MuJoCo Warp与torch（cu130）的对应轮子，代码不分平台。

| 项 | Windows | Linux |
| --- | --- | --- |
| 环境 | `uv sync --frozen` | `uv sync --frozen` |
| `torch.compile` | 关闭（PyTorch在Windows无Triton，`--compile auto`自动跳过） | 开启（Triton随torch安装） |
| 已验证 | RTX 4070 Ti SUPER，CUDA | RTX 5080，CUDA，`--compile off`；LAFAN1小规模回归通过 |

Linux本次回归使用Python 3.12.3，运行64个环境、4次迭代，验证训练、检查点保存与加载、40条LAFAN1动作评估，以及中文目录中的UTF-8 JSON/CSV读写。
该结果仅验证运行兼容性，不代表训练质量达标；`--compile auto/on`开启编译的路径与H100尚未验证。

同一目录被两个系统共享（例如虚拟机挂载宿主机磁盘）时，两边不能共用`.venv`：
`.venv`内的解释器路径与目录结构按系统生成，在一边执行`uv sync`或`uv run`会重建另一边的环境。
此时由Windows使用默认`.venv`，Linux把环境放到共享目录之外：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/supermimic"
```

分别克隆的仓库各自使用默认`.venv`即可。

## 快速开始

以下命令在bash与PowerShell中写法相同，均在仓库根目录运行。

```bash
# 数据：下载40段G1重定向LAFAN1（30 fps CSV）
uv run --frozen python scripts/download_data.py

# 环境自检：FK与仿真一致性、状态回环、参考回放奖励
uv run --frozen python scripts/check_env.py --num_envs 64

# 冒烟训练（1024 envs × 300 iters，约2.5 min，单卡）
uv run --frozen python -m supermimic.scripts.train --num_envs 1024 --iterations 300 --buffer_steps 200 --warmup_steps 16 --wm_only_iters 50 --wm_batch 1024 --pi_batch 1024 --ckpt_every 300 --log_every 25 --log_dir logs/smoke

# 评估（附录F.1协议，逐检查点；--watch 常驻监视）
uv run --frozen python -m supermimic.scripts.evaluate --run_dir logs/smoke --watch
```

无GPU时各命令加`--device cpu`，只用于检查流程能否跑通，速度不具参考意义。

## 索引 HDF5 参考数据

`--motion_index` 只将 paired index 当作动作身份白名单：根据 `g1_h5` 和
`g1_source_clip_index` 定位源库，使用源库 `source_clip_starts/lengths` 的完整动作。
不使用配对窗口的 `start/length`，也不使用源库的 `clip_*` 存储窗口或旧 precomputed 缓存。
不删除或改写任何源文件；这是逻辑筛选，不会生成物理裁剪后的 HDF5 副本。

索引旁必须保留原 `run.json`，用于绑定源库清单、分片校验和与 G1 关节顺序。
默认从其中的 `g1_root` 读取源库；搬迁后可通过 `--motion_root` 指定新位置，内容必须仍匹配。
索引的 `training_ready=false` 属于 human/G1 配对门禁，本管线只消费 G1 参考，不据此宣称人机时间对齐。

```bash
# INDEX 替换为 paired_50hz/index.h5 的绝对路径；不需要 GPU
uv run --frozen python scripts/check_data.py --motion_index INDEX --verify_hashes --verify_frames

# 明确只取前 8 段做开发冒烟，训练仍保留完整动作与默认 32 步想象长度
uv run --frozen python -m supermimic.scripts.train --motion_index INDEX --motion_limit 8 --num_envs 32 --iterations 4 --buffer_steps 32 --warmup_steps 16 --wm_only_iters 1 --wm_batches 1 --wm_batch 32 --pi_batch 32 --critic_epochs 1 --critic_minibatches 1 --compile off --amp none --ckpt_every 4 --log_every 1 --log_dir logs/index_smoke

# 自动继承检查点的数据选择；评估按 3+3+2 分批，不为每段同时创建环境
uv run --frozen python -m supermimic.scripts.evaluate --run_dir logs/index_smoke --ckpt logs/index_smoke/ckpt_000004.pt --eval_batch_size 3
```

原生 50 Hz 数据不重采样、不按窗口裁短。每段独立计算根/关节速度与根角速度，
再用 mjlab 内置默认 G1 做 FK；差分不跨原动作边界，未来目标帧只在真实段尾钳制。
训练仍按 1 s 分箱重置，回放缓冲的世界模型窗口和策略想象长度不改变源动作表示。
不足 33 帧的动作仍保留，但不能提供完整的默认 32 步想象起点；没有合法起点时会明确报错，不无限重采样。

训练在 `reference.json` 和检查点保存数据指纹，恢复与评估时验证一致性。
评估默认最多同时运行 256 个环境，逐检查点结果保存为 `*_eval.json`；
轨迹录制仅支持单个评估批次，需显式设置足够的 `--eval_batch_size`。
已有 `eval.csv` 若没有 `eval_reference.json`，需归档旧结果后再运行，不能混入未标识数据集的历史分数。

当前参考张量仍完整驻留在所选设备，不是流式 GPU 缓存。全量约 50.08 GiB，
还不含仿真、网络、优化器和回放缓冲；显存不足会在分配前报错，不会自动缩减数据。
开发卡用显式 `--motion_limit`，全量训练前仍需在目标设备检查总显存预算。

2026-10-22 Linux 验证：本次索引选中 114,406 段、39,652,996 帧，源库中另外 551 段不参与；
全量 SHA256、帧数值和单位四元数检查通过，8 段不足 33 帧且未删除。
该索引子集不是论文完整基准，冒烟结果不代表论文指标复现。
新增 HDF5 路径尚未在 Windows 或 H100 上验收。

索引接入时的本地验收：25 项离线测试通过；RTX 5080 上 8 段/32 环境/4 次迭代、
检查点恢复至第 5 次迭代、3+3+2 分批评估及原 CSV 小规模训练均跑通。
未充分训练的策略评估为 8/8 失败，只验证流程；全量参考显存保护也已验证。
本地 `uv sync --frozen`、锁检查和 145 个已安装包的兼容性检查通过，
锁文件中 14 个按 Python 版本区分的同名多版本条目保留，无可操作依赖冲突。
局域网 `data:3141` 健康检查超时，因此未完成资源缓存同步验收；未改动服务配置。

## 多卡自动分片

使用 [torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html) 启动，训练入口从
`WORLD_SIZE/RANK/LOCAL_RANK` 自动识别参与进程及本地 GPU。
`--nproc-per-node=gpu` 使用所有可见 GPU；改为整数只启用相应数量。
普通 `python -m supermimic.scripts.train` 仍为单进程，不会自动占用其它卡。

```bash
# INDEX 替换为索引绝对路径；所有可见 GPU 参与，自动分片，无需另建数据配置
uv run --frozen torchrun --standalone --nproc-per-node=gpu -m supermimic.scripts.train --motion_index INDEX --log_dir logs/full

# 限定 4 张可见 GPU 参与；实际分片数自动变成 4
uv run --frozen torchrun --standalone --nproc-per-node=4 -m supermimic.scripts.train --motion_index INDEX --log_dir logs/four_gpu

# 只规划 8 个分片的数量/显存，不读取所有帧，也不启动训练
uv run --frozen python scripts/check_data.py --motion_index INDEX --num_shards 8
```

完整动作按帧数从长到短确定性装入当前最小分片，不复制、不丢弃、不拆段；
每个进程只读取自己动作的帧、计算其速度/FK，并在分配前检查本卡参考显存。
当前全量 8 分片为每卡 4,956,608～4,956,633 帧，参考张量估算每卡约 6.26 GiB，
不含仿真、回放缓冲、模型、优化器、梯度同步临时张量和 CUDA 工作区。
CSV 也支持相同的分片规则，但会先在 CPU 解析/重采样所选 CSV。

`--num_envs`、`--wm_batch`、`--pi_batch` 始终是**全局总量**，除法余数按 rank 顺序分配，
严格保持总量，梯度按实际本地样本数加权；总量小于进程数时直接报错。
默认 8 卡对应每卡 512 环境、256 世界模型样本、512 策略样本；3/5/6/7 卡也无需手工整除批大小。
每卡动作数和帧数无需整除；动作不足以覆盖各进程，或某分片没有完整想象起点时会协调报错。
`--motion_limit` 在分片前作用于全局选择，不是每卡各取该数量。

每卡各自运行仿真和回放缓冲，三类模型使用相同初始权重；每次反向传播后通过
`torch.distributed` 显式平均梯度，再做裁剪与优化器更新。没有把自定义 `wm.step`/BPTT
循环简单包进 DDP，也不是各卡训练独立策略。归一化聚合全局样本数、均值和方差，
分箱 softmax 和 EMA 使用全局归一化，避免误把各分片的局部方差/权重直接平均。
CUDA 使用 NCCL，CPU 多进程验证使用 Gloo；不更改内置 G1 配置。
多 GPU 路径面向 Linux/NCCL；Windows 单卡入口保留，Windows 多 GPU 不在本次支持/验收范围。

采样边界：每卡固定分片和环境数，在自己的分片内按全局分箱权重的条件分布重置。
因此模型更新是同步的，但不承诺与单卡全库自适应采样、回放轨迹或随机序列逐步等价，
也不将该多卡数据分配方式宣称为论文原实现的等价证明。

仅主进程写共享训练日志和检查点；日志汇总各进程，回合统计按结束回合数加权。
各进程的动作索引和帧数清单在日志目录的 `shards/world_N/rank_XXXX.json`。
全局数据指纹与分片数无关，恢复时可更换卡数并重新分片，仍校验原全局数据选择；
恢复会重新填充回放缓冲，仿真状态、随机数状态和分箱课程不做逐步恢复。
检查点保持无 DDP 前缀，可用原单进程评估命令加载；评估仍需在其设备加载完整的所选参考库。
CPU/Gloo 验证不能替代多 GPU NCCL/H100 的吞吐、显存峰值或数值验收。

2026-10-22 本地验证：30 项测试通过，含真实双进程 Gloo 加权梯度/优化器一致性、
全局归一化、分箱概率及单 rank 错误协调；真实索引双进程训练、5 样本的 3+2 拆分、
critic 非整除小批次及 2→1→2 进程恢复均跑通。
RTX 5080 单卡回归、`--nproc-per-node=gpu` 自动启动和多进程检查点的单卡评估通过。
检查点评估仅为流程验收，短跑策略仍有失败；未运行全量长训、H100/NCCL 或新多卡编译路径。
本次证据位于忽略目录 `logs/auto_shard_validation_20261022/`，没有改写源数据或提交 Git。

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
保留现有`uv`非安装模式，不引入`src/`或打包后端；HDF5 读取新增 `h5py` 依赖。

```text
supermimic/
  tasks/tracking/
    constants.py
    env.py
    mdp.py
    motion.py
    motion_hdf5.py
    rl/
      buffer.py
      networks.py
  utils/
    distributed.py
    kinematics.py
    rotations.py
  scripts/
    train.py
    evaluate.py
scripts/
  download_data.py
  check_env.py
  check_data.py
tests/
  test_layout.py
  test_g1_defaults.py
  test_motion_hdf5.py
  test_evaluation.py
  test_distributed.py
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
| `supermimic/tasks/tracking/motion.py` | CSV/HDF5→50 Hz完整参考、独立差分与FK、目标帧、1 s重置分箱 |
| `supermimic/tasks/tracking/motion_hdf5.py` | 索引身份白名单、完整源段读取与源库绑定检查 |
| `supermimic/utils/kinematics.py` | MuJoCo前向运动学（身体位姿/速度） |
| `supermimic/utils/distributed.py` | 进程/设备绑定、协调错误、梯度平均与全局统计同步 |
| `supermimic/tasks/tracking/constants.py` | 动作数据关节顺序与跟踪目标，不定义机器人配置 |
| `supermimic/scripts/evaluate.py` | 附录F.1评估：逐原始段首帧播放、失败计数、两种归一化回报 |
| `scripts/` | 数据下载、离线数据验证与显式仿真自检 |
| `tests/` | 无GPU的结构、HDF5/FK数值、数据选择与分批评估回归；数值测试使用项目依赖 |

在仓库根目录运行上述命令；也可用`python supermimic/scripts/train.py`或
`python supermimic/scripts/evaluate.py`直接启动。默认数据仍在根目录`data/`，
日志仍在`logs/`；显式传入的相对路径仍相对于当前工作目录。
内部导入路径已调整为上述分层，旧的平铺模块路径不再保留。

离线结构与 CPU 数值检查（不下载数据，不启动仿真或训练）：

```bash
uv run --frozen python -B -m unittest discover -s tests -v
```

## 文档

| 文件 | 内容 |
| --- | --- |
| [docs/planning/plan.md](docs/planning/plan.md) | 主张、证据帐本、依赖、显存估算、验收标准、里程碑 |
| [docs/planning/rubric.json](docs/planning/rubric.json) | 加权评分树，叶子引用验收标准与帐本ID |
| [docs/analysis/algorithm.md](docs/analysis/algorithm.md) | 论文算法分析：问题设定、世界模型、一阶策略梯度、critic、加速来源与风险 |
| [docs/analysis/training_pipeline.md](docs/analysis/training_pipeline.md) | 训练流程图与算法细节：采样、重置采样、世界模型拟合、想象推演BPTT、critic、评估、实现与论文差异 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方代码许可与出处 |

## 使用约束

- 数据集与检查点不入库。
- BONES-SEED需在Hugging Face同意许可后下载。
- LAFAN1采用CC BY-NC-ND 4.0。
- AMASS：需注册，仅限非商用研究，不得再分发。
- 论文项目页的网页演示受专有许可约束。
  它只能人工浏览，不能下载、抓取或用作对照。
