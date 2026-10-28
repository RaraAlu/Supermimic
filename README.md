# Supermimic

复现论文SuperMimic（arXiv:2610.28480，CoRL 2026 Oral）。

论文用一个可微世界模型训练G1全身动作追踪策略。
作者未公开代码；本仓库为独立复现，`supermimic/`下的部分模块移植自第三方MIT实现
[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)（Isaac Lab 3 → mjlab），
许可与出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前状态

- 阶段：实现v0.1.0（训练闭环、评估、数据/自检脚本）。
- 目标：在129,514段基准上复现论文主结果。
- 算力：8卡H100服务器负责训练与评估；2卡A100 SXM4 80GB（NVLink）服务器用于当前训练与全量诊断；RTX 5080与RTX 4070 Ti SUPER负责开发。
- 2026-10-27：修正重置分箱后，在2卡A100上重新开始全量训练（全局8192环境、世界模型10×2048、策略4096、学习率2e-4不乘√2）。
  此前一个run日志中为负的`ep_ret_norm`来自尾部短分箱缺陷，不代表策略变差，见[重置采样分析](docs/analysis/reset_sampling.md)。
- 依赖版本：以`pyproject.toml`与`uv.lock`为准（mjlab 1.6.0、mujoco-warp 3.11.0、warp-lang 1.18.0、torch 2.14.1+cu130）。
- 数据：支持 LAFAN1 CSV、索引白名单 HDF5 与单文件合并源 HDF5；后两者读取原生 50 Hz 完整动作，由 MotionLib 生成参考速度和 FK。
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
该结果仅验证运行兼容性，不代表训练质量达标。后续A100上已验证compile与CUDA graph的组合及双卡NCCL图重放；H100尚未实测。

训练默认用CUDA graph重放世界模型更新和策略的想象推演加BPTT（`--cuda_graph auto`：单进程、CUDA、未设`SM_PROFILE`时启用，可与`torch.compile`组合），
三个优化器默认用fused Adam（`--fused_adam on`）。两者都只改变执行方式，不改变算法；实测与开关见
[docs/analysis/optimization.md](docs/analysis/optimization.md)第6至8节。用`scripts/profile_train.py`剖析时请加`--cuda_graph off`。
图启用时，critic更新与环境每步的奖励和终止也用图重放（`--cuda_graph_critic`、`--cuda_graph_env`，默认on）。
采样侧的观测、策略前向与动作噪声也可用图重放（`--cuda_graph_collect on`）；`--collect_overlap on`在reset等待GPU前提交回放写入与世界模型统计。
两项均有独立开关，完整训练A/B工具为`scripts/benchmark_collect.py`。Linux上`auto`现在可同时开启compile与图；显式复现实验可加`--compile on --cuda_graph on`。
多卡（torchrun）时`auto`也不启用CUDA graph；加`--cuda_graph on`可在多卡下启用，梯度all-reduce也在图内执行。
这条路径已在A100双卡（NCCL）上逐位验证；换机器后，正式训练前先在目标机器上运行
`torchrun --standalone --nproc-per-node=2 scripts/check_cuda_graph.py --motion_limit 8 --batch_scope per_gpu`，
各rank都输出“bitwise equal True”后再使用。

多个环境在同一仿真步结束于同一个动作bin时，先对该bin的回报取均值，再更新一次EMA，避免重复索引覆盖。
此修复会改变旧版本在重复bin时未定义的更新结果；性能对照必须在同一修复版本中切换开关。
`scripts/check_compile_equiv.py`另外检查eager与compile的数值差异；它使用相同噪声、独立恢复的模型及Adam状态，
默认严格门槛为`--atol 1e-5 --rtol 1e-3`，不把CUDA graph的逐位一致解释为compile与eager逐位一致。

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

训练在 `reference.json` 和检查点保存数据指纹，恢复与评估时比对已有指纹。
兼容边界：`check_reference`遇到缺失或空的旧`reference`会跳过该次比对，不会因此拒绝旧检查点；
这类检查点不能仅凭加载成功证明数据一致，现有run目录元数据的校验仍独立执行。
评估默认每批最多 4096 个环境（论文每卡环境数）；多于一批时按段长从长到短分批，
使同批动作长度接近，减少等待最长段的空转。逐检查点结果保存为 `*_eval.json`；
轨迹录制仅支持单个评估批次，需显式设置足够的 `--eval_batch_size`。
已有 `eval.csv` 若没有 `eval_reference.json`，需归档旧结果后再运行，不能混入未标识数据集的历史分数。

当前参考张量仍完整驻留在所选设备，不是流式 GPU 缓存。根与身体朝向存为四元数，按帧索引时才转成旋转矩阵，
每帧 1,052 字节（存矩阵时为 1,356）。本索引全量约 38.85 GiB，论文全量 BONES-SEED（264.56 h）约 46.66 GiB，
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

## 合并源 HDF5 参考数据

`--motion_source SOURCE.h5`直接读取单个`teleopit_motion_hdf5` v1源文件，不需要paired index或`run.json`。
文件须声明50 Hz，包含float32的`root_pos`、`root_quat_w`（wxyz）、`joint_pos`，
以及连续覆盖整个帧轴的`source_clip_starts/lengths/fps`；每段至少2帧且均为50 Hz。
可选`clips/source_relative_path`提供段名，否则生成`clip_000000`等名称。完整格式见[训练流程第2.1节](docs/analysis/training_pipeline.md)。

```bash
uv run --frozen python -m supermimic.scripts.train --motion_source SOURCE.h5 --log_dir logs/merged
```

`--motion_dir`、`--motion_index`、`--motion_source`互斥；`--motion_root`只适用于索引入口。
合并源的指纹绑定文件大小、总帧数和所选段的起点/长度，不计算姿态内容SHA256；
相同大小且边界不变的姿态替换可能无法被识别。上面的`scripts/check_data.py`只支持索引入口，不能传`--motion_source`。

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
当前全量 8 分片为每卡 4,956,608～4,956,633 帧，参考张量估算每卡约 4.86 GiB，
不含仿真、回放缓冲、模型、优化器、梯度同步临时张量和 CUDA 工作区。
CSV 也支持相同的分片规则，但会先在 CPU 解析/重采样所选 CSV。

默认 `--batch_scope global`：`--num_envs`、`--wm_batch`、`--pi_batch` 是**全局总量**，除法余数按 rank 顺序分配，
严格保持总量，梯度按实际本地样本数加权；总量小于进程数时直接报错。
默认 8 卡对应每卡 512 环境、256 世界模型样本、512 策略样本；3/5/6/7 卡也无需手工整除批大小。

论文表 A.5 明确规定每卡4096环境、双卡学习率乘√2；世界模型批量2048与策略批量4096仅标为
“per update”，PDF及arXiv的LaTeX原始文件都没有进一步标注全局或每卡。不能将三者全部按每卡解释当作作者已确认的设置。
已测方案A按“每次全局更新”解释：`--batch_scope global --num_envs 8192 --wm_batch 2048 --pi_batch 4096 --lr_scaling sqrt`。
方案B按每卡批量解释，其全局世界模型/策略批量为4096/8192。两种解读每卡都运行4096环境。
同行源码与√2学习率的线索更支持方案B，但不是作者确认；两轮实测外推32万次更新分别为33.24/50.89小时。
精度使用`--amp none`，保留FP32张量并允许TF32矩阵乘法；`--wm_only_iters 0`取消额外的500次世界模型单独训练。
论文未规定初始replay填充量，此处保留64个采样步；正式迭代从第一轮起同时更新世界模型、策略与critic。

```bash
# 全量数据短测：两种批量解读各用两卡，每组200次，第二轮交换GPU对。
# 该入口限制为最多1000次迭代，没有长训模式；输出目录必须是新目录。
uv run --frozen python scripts/benchmark_paper_two_gpu.py --motion_source SOURCE \
  --gpus 0 1 4 5 --output_dir logs/paper_two_gpu --iterations 200 --skip 50
```

表A.5原始证据、每卡实际批量及时间估算见[双卡短测记录](docs/analysis/paper_two_gpu.md)。

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

恢复训练使用`--resume RUN/last.pt`：它包含三类模型、优化器、迭代与计时状态。
`ckpt_*.pt`只保存策略、世界模型及元数据，用于评估，不能作为训练恢复文件。
未显式选择数据入口时，恢复会继承检查点的数据路径、子集与源根；其余训练参数仍取当前命令行或默认值，
不会自动继承旧run的批量、学习率、精度或采样选项，续训时需明确提供。
加载后会按当前fused/CUDA graph模式重新设置Adam标志、`step`所在设备和学习率类型，随后按当前调度计算学习率。

### TensorBoard

`--tensorboard {auto, on, off}`（默认`auto`：装有tensorboard时开启，它随mjlab安装）让主进程把每条日志同时写成TensorBoard标量，
位于`LOG_DIR/tensorboard`：`train/<列名>`与`sampling/<列名>`，列名与`train_log.csv`、`sampling_log.csv`相同，步数为迭代数。
只在记日志时写入，不增加GPU同步。恢复训练时，旧run在恢复点之后写下的事件会被隐藏（`purge_step`），曲线不会重叠；
CSV仍照旧追加。没有开启TensorBoard的旧run可用`scripts/csv_to_tensorboard.py`从CSV补写同名标量，只读CSV，可在训练进行中运行、可重复运行。

```bash
uv run --frozen tensorboard --logdir logs
```

2026-10-22 本地验证：30 项测试通过，含真实双进程 Gloo 加权梯度/优化器一致性、
全局归一化、分箱概率及单 rank 错误协调；真实索引双进程训练、5 样本的 3+2 拆分、
critic 非整除小批次及 2→1→2 进程恢复均跑通。
RTX 5080 单卡回归、`--nproc-per-node=gpu` 自动启动和多进程检查点的单卡评估通过。
检查点评估仅为流程验收，短跑策略仍有失败；未运行全量长训、H100/NCCL 或新多卡编译路径。
本次证据位于忽略目录 `logs/auto_shard_validation_20261022/`，没有改写源数据或提交 Git。

## 重置采样与诊断

附录D.1的按回报重置采样按1 s分箱。多分箱片段的最后一箱若不足25个起点，并入前一箱：
否则从片段末尾起步的回合只有1–4步，首步（上一动作为0）的平滑惩罚摊不开，外推后这些分箱显得极难，
课程重置被集中到片段末尾，训练日志按回合平均的`ep_ret_norm`也随之为负。
全量BONES-SEED上，修正把不超过5步的回合从52%降到2.7%。排查、对照与未解决问题见[重置采样分析](docs/analysis/reset_sampling.md)。

当前默认跳过重置后前50个转移的分箱评分，并限制段末100步内的重置起点；
只能从首帧起步的短片段及撑不过过渡期的分箱，只分配均匀采样份额。
这些调整只改课程信号与重置起点，训练奖励和`ep_ret_norm`仍包含整个回合。
四轮排查及历史验证见[课程修正记录](docs/troubleshooting/reset_curriculum.md)。可配置项：

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `--bin_clip_end {bootstrap, terminal}` | `bootstrap` | 到达片段末尾的回合用平均奖励外推；或不外推、按起点可达最大回报归一化（不跳过过渡期的历史对照中会放大短回合偏差） |
| `--bin_prior {zero, mean}` | `zero` | 未访问且可评分分箱取0，或取已访问可评分分箱的全局均值 |
| `--bin_skip_steps N` | `50` | 不计前N个转移；首个计分奖励为`r[N+1]`，折扣从此重新起算；过渡期内终止计0分，过渡期内截断不更新 |
| `--reset_margin N` | `100` | 普通起点须满足`f < clip_end - N`；不足的片段仍可从首帧起步，但不参与自适应份额 |
| `--wm_update {steps, single}` | `steps` | 每次迭代世界模型走10步各2048窗口，或1步20480窗口（表A.5两种读法） |

`--bin_clip_end`的均值、长度与可达最大回报均按跳过过渡期后的区间计算；`--bin_skip_steps 0 --reset_margin 0`恢复尾箱合并后的旧评分与起点范围。
训练目录另写`sampling_log.csv`（全局分箱权重的熵、最大权重、已访问分箱比例、距片段末尾2 s内的重置比例），`train_log.csv`的列不变。
熵除以`log(max(全局分箱数, 2))`，访问率除以全部分箱数；禁止重置的分箱也在分母内，因此访问率上限可能小于100%。
多卡各自在本地分片内重新归一化后抽样，所以`samp_top1`是全局权重的最大值，不是实际重置频率；
`reset_tail_frac`则按实际抽出的重置计数。全库不可评分时，`e0c923c`已修复退化路径，直接在可重置分箱上均匀分配，返回权重有限且总和为1，见[训练流程第4节](docs/analysis/training_pipeline.md)。

想象推演使用`MDP.soft_survival`直接计算每步存活概率：采用`sigmoid(-x)`，存活权重与GAE直接使用该值，
避免`1 - sigmoid(x)`和`1 - (1 - keep)`在float32下抵消为0（`4293e90`）。

`--critic_cap {on, off}`（默认`on`）把critic估值在GAE自举和策略目标中限制到`MAX_STEP_REWARD / (1 - gamma)`以内（`gamma`为0.99时为10，
每步奖励不超过`MAX_STEP_REWARD`）：回归仍拟合原始输出，高估会被拉回而策略追不上它，对正确的critic没有影响。
重置分箱的分数在softmax前限制到[0, 1]，非有限分数不更新分箱，原因见[问题与解决记录](docs/troubleshooting/training_metrics.md)第11节。

下列诊断中的策略重放脚本只读检查点与源数据，向`--out`写结果，`--src`可指向某提交的源码树目录。
`diag_train_episodes.py`未显式传入的采样选项取该源码树的`TrackingSim`默认值，不自动继承检查点；
`diag_wm_error.py`仅继承`gamma`、`bin_clip_end`与`bin_prior`，过渡期和末尾范围也取源码默认值。
`--action_noise`在这两个脚本中独立默认0.1；使用旧检查点时要核对这些条件。

| 脚本 | 用途 |
| --- | --- |
| `scripts/make_motion_subset.py` | 从HDF5源随机抽取完整片段，生成小规模子集（CPU） |
| `scripts/diag_first_step.py` | 附录F.1评估，并记录每段首步的平滑与限位惩罚、去掉首步后的归一化回报 |
| `scripts/diag_train_episodes.py` | 按训练条件重放，记录每个回合的长度、`ep_ret_norm`及其奖励/惩罚分解，导出分箱状态 |
| `scripts/diag_bin_difficulty.py` | 从每个1 s起步位置按评估条件跑2 s，生成与分箱方案无关的难度标签 |
| `scripts/diag_compare_bins.py` | 汇总多次重放：短回合、片段末尾重置、难位置倾斜、采样集中度 |
| `scripts/diag_wm_error.py` | 用检查点的策略按训练条件收集新rollout，测世界模型k步预测误差：物理单位逐分量，及几种候选标称误差下的归一化值 |
| `scripts/diag_critic_bias.py` | 用若干完整训练状态（含critic）交叉评估：各状态的策略收集新rollout，各状态的critic对其估值，报告真实/想象回报偏差、输入标准化超标比例与输入梯度范数 |
| `scripts/watch_train.py` | 监督训练日志，出现异常、到达里程碑或训练结束时退出 |
| `scripts/csv_to_tensorboard.py` | 把run的`train_log.csv`与`sampling_log.csv`写成TensorBoard标量，标签与`--tensorboard`相同 |

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
  check_compile_equiv.py
  check_cuda_graph.py
  profile_train.py
  benchmark_collect.py
  benchmark_paper_two_gpu.py
  make_motion_subset.py
  diag_first_step.py
  diag_train_episodes.py
  diag_bin_difficulty.py
  diag_compare_bins.py
  diag_wm_error.py
  diag_critic_bias.py
  watch_train.py
  csv_to_tensorboard.py
  smrun.sh
tests/
  test_layout.py
  test_g1_defaults.py
  test_motion_hdf5.py
  test_merged_source.py
  test_evaluation.py
  test_distributed.py
  test_bin_returns.py
  test_learning_math.py
  test_compile_check.py
  test_tensorboard_log.py
```

| 路径 | 内容 |
| --- | --- |
| `supermimic/scripts/train.py` | 算法A.1：采样→拟合世界模型→想象推演BPTT→critic更新 |
| `supermimic/tasks/tracking/env.py` | mjlab TrackingSim：50 Hz控制、4×5 ms子步、域随机化、回合重置 |
| `supermimic/tasks/tracking/mdp.py` | 99维观测（93维本体观测加6维参考相对朝向）、表A.2奖励、硬/软终止、critic特征 |
| `supermimic/tasks/tracking/rl/networks.py` | 预测70维状态增量的世界模型、策略(编码器+共享解码器)、critic、RunningNorm；奖励/观测/终止由MDP解析计算 |
| `supermimic/tasks/tracking/rl/buffer.py` | 每环境1000步FIFO回放缓冲、k步窗口与起点采样 |
| `supermimic/tasks/tracking/motion.py` | CSV/HDF5→50 Hz完整参考、独立差分与FK、目标帧、1 s重置分箱（尾箱合并） |
| `supermimic/tasks/tracking/motion_hdf5.py` | 索引身份白名单与合并源两种读取器、完整源段读取、源库绑定与结构指纹 |
| `supermimic/utils/kinematics.py` | MuJoCo前向运动学（身体位姿/速度） |
| `supermimic/utils/distributed.py` | 进程/设备绑定、协调错误、梯度平均与全局统计同步 |
| `supermimic/tasks/tracking/constants.py` | 动作数据关节顺序与跟踪目标，不定义机器人配置 |
| `supermimic/scripts/evaluate.py` | 附录F.1评估：逐原始段首帧播放、失败计数、两种归一化回报 |
| `scripts/` | 数据下载、离线数据验证、仿真自检、compile/CUDA graph一致性检查、训练剖析与基准、重置采样诊断与训练监督 |
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
| [docs/analysis/algorithm.md](docs/analysis/algorithm.md) | 论文算法分析：问题设定、世界模型、一阶策略梯度、critic、加速来源与风险 |
| [docs/analysis/training_pipeline.md](docs/analysis/training_pipeline.md) | 训练流程图与算法细节：采样、重置采样、世界模型拟合、想象推演BPTT、critic、评估、实现与论文差异 |
| [docs/analysis/formulas.md](docs/analysis/formulas.md) | 公式精讲：每个公式按直觉、拆分、符号、数值算例、代码位置逐项讲解 |
| [docs/analysis/markdown_math.md](docs/analysis/markdown_math.md) | GitHub公式排版约定：行内公式、独立公式、多行对齐与界面验收 |
| [docs/analysis/optimization.md](docs/analysis/optimization.md) | 性能优化分析：剖析方法与实测、瓶颈（kernel发射）、按优先级的优化项与验证顺序 |
| [docs/analysis/reset_sampling.md](docs/analysis/reset_sampling.md) | 按回报重置采样：负`ep_ret_norm`的排查、尾部分箱修正、分箱选项、全量对照、开源实现对照与未解决问题 |
| [docs/analysis/paper_two_gpu.md](docs/analysis/paper_two_gpu.md) | 双卡短测：表A.5批量口径的两种解读与实测耗时 |
| [docs/analysis/doc_code_audit.md](docs/analysis/doc_code_audit.md) | 2026-10-28文档与代码逐项核对：差异、实现依据、修正与验证边界 |
| [docs/troubleshooting/README.md](docs/troubleshooting/README.md) | 问题与解决记录：重置采样四轮课程跑偏、训练指标判读与监督、性能优化中的正确性问题、平台与远程运维 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方代码许可与出处 |
| [CONTRIBUTORS.md](CONTRIBUTORS.md) | 贡献者名单、AI 编码代理署名与提交模板约定 |

## 使用约束

- 数据集与检查点不入库。
- BONES-SEED需在Hugging Face同意许可后下载。
- LAFAN1采用CC BY-NC-ND 4.0。
- AMASS：需注册，仅限非商用研究，不得再分发。
- 论文项目页的网页演示受专有许可约束。
  它只能人工浏览，不能下载、抓取或用作对照。

## 贡献

贡献者与 AI 编码代理的署名方式见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。提交模板 `.gitmessage` 默认带
`Co-authored-by` 尾注，安装：`git config commit.template .gitmessage`。
