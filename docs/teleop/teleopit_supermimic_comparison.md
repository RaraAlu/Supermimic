# Teleopit与Supermimic：训练速度、跟踪精度与可复用设计

更新：2026-11-03；2026-11-04复核了mjlab 1.4.0上游实现与Teleopit论文。本文是源代码对比，不是新跑出的训练基准。它支持[实时跟踪与统一扰动恢复设计](causal_intent_encoder.md)中的架构选择。

## 1. 对比范围与证据

本次通过SSH只读检查rara上的工作树；两个工作树检查时均无未提交改动。

| 对象 | 路径 | 检查版本 |
| --- | --- | --- |
| Teleopit | `rara:D:\Train\Bote_Teleopit` | `4023a6b83021f51fc032c88bd6cf06fec0552a53` |
| Supermimic | `rara:D:\Train\Supermimic` | `07dee2758b4235de43e7da1b10246f0161f34011` |
| 本地设计所在仓库 | 当前Supermimic工作树 | 修订前`1fc30fe` |

rara与本地Supermimic的`networks.py`、`motion.py`、`mdp.py`、`env.py`相同；`trainer.py`及训练、评估入口存在版本差异。下面的远程实现描述以表中提交为准，设计的改动位置以本地工作树为准。

“Teleopit训练也很快、跟踪精度较差”是用户提供的使用观察。本次在所检查的rara标准日志目录中未找到可配对的完整训练记录，也没有启动训练或实机控制。因此不能从本次检查得出速度倍率、MPJPE差值，或把精度差归因于某一个模块。

Teleopit论文给出其自身协议下的背景数字：8块A800约50 h、40k iteration（所检查代码的默认值为`max_iterations=30_000`）；mocap验证集成功率91.7%、MPJPE 0.040 m，同表SONIC为0.037 m、HoloMotion为0.034 m（[Teleopit论文§4.1](https://arxiv.org/html/2608.01834v1#S4.SS1)）。即成功率最高而MPJPE并非最低，方向与上述观察一致。这些数字来自论文的数据、硬件和指标定义，不能与本仓库结果直接换算。

## 2. 已确认的实现差异

| 项目 | Teleopit | Supermimic | 对当前目标的意义 |
| --- | --- | --- | --- |
| 策略学习 | RSL-RL PPO；每轮每环境采集24步，5个epoch、4个minibatch | 拟合可微世界模型；默认32步想象推演，经回报反传更新策略 | PPO与解析梯度都需要进入相同条件下的速度—精度对照 |
| 参考输入 | `window_steps=(0,)`；当前参考显式进入观测 | `f+1…f+10`，73维/帧，共730维目标 | 现有Supermimic不能直接用于零前瞻遥操作 |
| actor结构 | 当前167维观测直通；10帧观测经Conv1d压到64维；MLP头 | 未来目标编码为32维；与10×99维观测展平拼接后解码 | Teleopit的当前帧直通应优先借鉴，避免当前目标也被迫通过小潜变量 |
| 隐藏层 | actor/critic主干`2048,1024,512,256,128`；CNN通道`256,128,64` | 目标编码器/动作解码器`2048,1024,512,256`；另训练世界模型与critic | 网络和更新工作量不同，不能按iteration直接比较 |
| 失败采样 | 失败后以0.8概率回到同片段，向前回退25–75个控制步，即0.5–1.5 s | 按分箱回报自适应采样；有尾箱、重置过渡期等修正 | rewind可作为低成本难段课程候选；它本身不是物理恢复 |
| 扰动 | 每4–6 s修改根速度；摩擦、质心、惯性、关节零位随机化 | 当前环境随机化摩擦、质心、关节零位；没有同等周期推扰流程 | 训练任务难度不同；公平对比必须统一扰动配置 |
| 跟踪目标 | 根/刚体/关节位置速度，另有手腕点（1.0）；存活项3.0、动作变化项−0.5、关节限位项−10、自碰撞项−0.1、脚部加速度项−2.5e−6 | 6项根/刚体跟踪；动作变化项−0.1、关节限位项−2 | 原始reward与归一化return不能直接跨实现比较 |
| 参考锚点 | `torso_link` | 物理根状态为pelvis | 即使观测维度相同，也不能直接互换模型、特征和误差 |
| 失效终止 | 锚点高度、倾斜及手腕/脚踝高度误差超限 | 根高度与倾斜误差超限 | 两者都需要扩展，才能训练不中断的扰动恢复 |
| 数据入口 | 预计算FK/速度，支持原动作切成重叠窗口 | 本仓库按原始完整片段载入并计算FK/速度 | 必须按原始片段身份统一数据划分，不能比较不同窗口集合的成功率 |

Teleopit的倾斜终止实际比较的是投影重力z分量之差；配置中的`threshold=1.0`不能解释为1 rad。论文把该阈值写作1.0 rad，是这一混淆的来源；以代码为准。Supermimic也使用相应旋转矩阵元素之差。这些训练终止规则应与独立的角度误差评测分开。

## 3. 为什么两者都可能训练很快

Teleopit每次更新直接使用GPU并行仿真采样，没有世界模型拟合及多步模型反向传播成本；历史CNN把1670维历史压为64维，并保留当前观测直通。rewind还增加了失败前难段的训练频率。这些都是值得复用或验证的设计，不能仅凭它使用PPO就判断其训练慢。

Supermimic的不同之处是：采集到的转移可反复训练共享动力学模型，策略在便宜的模型推演中获得解析梯度。当前默认`utd=0.4`意味着平均每轮采集2.5个向量化物理控制步；同一轮还含默认10次世界模型更新和32步策略推演。它的一轮与Teleopit的24步rollout不是相同工作单位。

本地Supermimic还有`torch.compile`、CUDA graph及批量奖励计算。Teleopit所检查的自有PPO runner中没有对应的显式训练图重放流程；这不代表底层MuJoCo Warp/mjlab没有图优化。比较时须记录操作系统、GPU、精度、编译与图开关、全局环境数及显存占用。

应画三条曲线：

1. 固定硬件下的物理采样吞吐与完整更新耗时，解释执行成本。
2. 同一评测误差/成功率随墙钟时间和GPU-hours的变化，解释达到目标的速度。
3. 加入扰动恢复后，同一联合验收门槛的达标时间，防止只把“很快会站立”算作完成训练。

现有论文和本仓库报告只能作为各自协议下的背景。SuperMimic的训练集覆盖率不等于遥操作跟踪精度，也不等于扰动恢复成功率。[官方实验口径](https://supermimic.github.io/)

## 4. 精度差异的排查顺序

### 4.1 参考信息量不同

Supermimic看到未来10帧，Teleopit主路径只有当前参考和历史；前者可以提前准备突发动作。要比较优化器能力，应让PPO与世界模型策略使用相同的因果输入。原版未来帧Supermimic保留为信息更充分的参照，不把它称作因果策略必然能达到的上限。

### 4.2 名义当前帧不等于全链路严格因果

Teleopit预计算`joint_vel`、刚体线速度和角速度时使用中心差分；训练读取这些预计算量。其在线参考处理则由当前与上一份参考计算速度，并可做因果EMA。Supermimic原始参考速度也使用中心差分。

因此，两份实现都需要为因果actor单独建立与线上一致的参考特征：后向差分、时间戳、因果滤波、段首补齐。干净中心差分可以保留给训练奖励与critic，不能混进部署actor。`window_steps=(0,)`只证明没有显式未来帧窗口，不能证明离线训练输入没有未来信息。

### 4.3 奖励时间相位不同

Teleopit按mjlab管理器顺序先推进物理，再计算奖励/终止，之后推进motion command，故其训练奖励比较`state[t+1]`与`reference[t]`；该顺序也写在仓库训练文档中。Teleopit的`pyproject.toml`锁定`mjlab==1.4.0`；其上游`src/mjlab/envs/manager_based_rl_env.py`的`step`依次为物理子步、termination、reward、reset、`sim.forward()`、command更新、事件与观测，与上述文档一致。2026-11-04复核时，rara上Teleopit的`.venv`已不可用（解释器路径失效，site-packages中没有mjlab），无法读取实际安装版本；复现原生基线前需重建环境，并在实验manifest中锁定版本、复核顺序。

Supermimic的`TrackingSim.step`明确用`f1=f+1`计算后继状态的奖励。在50 Hz下，两种约定差一个20 ms控制步。这是需要对齐的设计差异，不是已经证明的精度损失原因。现有Supermimic actor的目标输入含`f+1`，所以该约定对它不构成外推；改为只读当前参考的因果actor后，同一约定要求额外外推一步，见[设计稿§2.3](causal_intent_encoder.md#23-奖励与评测时间约定)。统一实验采用真实时间戳和同一奖励相位，保留原生配置对照；离线平移轨迹只能作为延迟诊断，不能替换主精度指标。

### 4.4 目标权衡、采样与指标不同

Teleopit的存活奖励、较强动作变化惩罚和额外末端终止可能影响“稳定完成”与“紧贴动作”的取舍；Supermimic则存在模型误差被策略利用的风险。上述解释都需要消融，不能直接称为根因。

两者的机器人XML、PD参数、锚点、数据窗口、初始化、失败阈值、指标对齐方式和失败后的计分长度也必须统一。尤其不能只对成功段计算精度，或把较早结束的失败段从均值中隐去。

## 5. 对设计稿的具体影响

| 决策 | 处理方式 |
| --- | --- |
| 主要actor候选 | 借鉴Teleopit：当前观测直通＋历史CNN；保留当前关节目标、参考高度/速度/朝向的显式通路 |
| 主要训练候选 | Supermimic的世界模型与解析梯度闭环；同时建立相同actor/任务的PPO对照 |
| 蒸馏 | 从必经三阶段改为可选初始化；教师成本计入，教师动作只在其有效的跟踪状态用于监督 |
| 采样 | 比较rewind与当前分箱课程；统一保留正常跟踪、难段和受扰状态的数据份额，防止难例挤占正常动作 |
| 因果性 | 重做部署特征契约，覆盖速度、插值、滤波与缓存；用改变未来片段的测试验证actor前缀不变 |
| 恢复 | 在同一个控制接口内学习受扰后的恢复与接回；rewind仅用于失败回合结束后的训练起点选择 |
| 优化器选择 | 依据同机、同数据、同因果观测下的达标时间选择；不预先保证世界模型或PPO在所有规模下更快 |

首轮实验只依次改变一件事：原生基线 → 严格因果特征/统一时序 → 历史架构 → 优化器 → 恢复课程。奖励和采样的候选改动单独做消融，避免一次修改后无法判断收益来自哪里。完整实验矩阵与验收见[设计稿第8节](causal_intent_encoder.md#8-评测与验收)。

## 6. 源码定位

以下Teleopit路径均相对`rara:D:\Train\Bote_Teleopit`、提交`4023a6b`；Supermimic链接指向本仓库，对照时以第1节的版本为准。

| 结论 | 源码位置 |
| --- | --- |
| PPO参数、CNN配置 | `train_mimic/tasks/tracking/config/rl.py:13–71` |
| 当前帧、10帧历史、torso锚点 | `train_mimic/tasks/tracking/config/env.py:99–115,230–280` |
| 当前观测与历史编码拼接 | `train_mimic/tasks/tracking/rl/temporal_cnn_model.py:118`起；`conv1d_encoder.py:21–60` |
| rollout与更新 | `train_mimic/tasks/tracking/rl/runner.py:264–296`；`distributed_ppo.py` |
| rewind | `train_mimic/tasks/tracking/mdp/commands.py:918–951,1244–1246` |
| 速度预计算 | `train_mimic/data/motion_fk.py:36–48,104,220`；`dataset_lib.py:804` |
| 在线参考速度 | `teleopit/controllers/reference_processing.py:152`起；`teleopit/sim/reference_timeline.py` |
| 奖励、推扰、终止 | `train_mimic/tasks/tracking/tracking_env_cfg.py:168–305`；自碰撞、脚部加速度与手腕点奖励在`config/env.py:162–210`；`mdp/terminations.py:27–86` |
| 奖励与参考推进顺序 | `docs/docs/tutorials/training.md:556–575`；mjlab 1.4.0 `src/mjlab/envs/manager_based_rl_env.py`的`ManagerBasedRlEnv.step`（`pyproject.toml`锁定版本） |
| Teleopit误差汇总 | `train_mimic/benchmarking.py:185–279`；`mdp/metrics.py` |
| Supermimic输入、模型与更新 | [motion.py](../../supermimic/tasks/tracking/motion.py)、[networks.py](../../supermimic/tasks/tracking/rl/networks.py)、[trainer.py](../../supermimic/tasks/tracking/rl/trainer.py) |
| Supermimic奖励、终止与时序 | [mdp.py](../../supermimic/tasks/tracking/mdp.py)、[env.py](../../supermimic/tasks/tracking/env.py) |

后续实验需随结果保存两侧提交、实际配置、数据指纹、模型与检查点哈希、硬件/依赖版本和评测代码版本。本文没有把源代码推断写成已经测得的性能结论。
