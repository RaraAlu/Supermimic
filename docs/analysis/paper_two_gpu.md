# 双A100全量数据短测：论文设置与批量口径

本页是2026-10-27的历史短测。当前`4b4d27a`新增了默认过渡期评分与末尾起点限制，启动器未显式覆盖这些选项，
因此用当前源码重跑下列命令会采用当前采样默认值；复现原数字须使用`paper_two_gpu_results.json`记录的源码版本与配置。

## 原始证据与尚未明确的部分

核对版本：[SuperMimic arXiv:2610.28480v1 PDF](https://arxiv.org/pdf/2610.28480v1)，
以及[同版本LaTeX源文件](https://arxiv.org/src/2610.28480v1)中的`sections/appendix.tex`第207、214、227、235–236行。

表A.5将环境数明确标为每GPU 4096；世界模型写作“10 batches of 2048 windows per update”，
策略写作“4096 imagined rollouts per update”。源文件没有额外的全局/每卡说明。
因此分别测量全局批量与每卡批量两种解释；两者均不能冒充作者另行确认。
双卡学习率为 $`\alpha_2=\sqrt{2}\,(2\times10^{-4})\approx2.8284\times10^{-4}`$；32万次迭代、25万次余弦衰减至0.03倍、UTD=0.4。
图5说明世界模型训练使用TF32，故本次关闭BF16 autocast，FP32张量的CUDA矩阵乘法允许TF32。

| 项目 | 方案A：全局批量 | 方案B：每卡批量 |
| --- | --- | --- |
| 每卡环境数 | 4096 | 4096 |
| 全局环境数 | 8192 | 8192 |
| 每卡世界模型小批量 | 1024窗口 | 2048窗口 |
| 全局世界模型小批量 | 2048窗口 | 4096窗口 |
| 每次迭代的世界模型更新 | 10次，每窗口8步 | 相同 |
| 每卡策略批量 | 2048条想象轨迹 | 4096条想象轨迹 |
| 全局策略批量 | 4096条想象轨迹 | 8192条想象轨迹 |
| 想象长度 | 32步 | 32步 |
| 学习率 | 0.0002828427起 | 相同 |

两种解读都使用全局计数显式传参，避免环境数与优化器批量被一个开关同时改变。
使用全量可用HDF5，未设置`motion_limit`。当前数据114,406段、39,631,040帧、220.17小时；
论文是129,514段、264.56小时，数据集仍有差异。

## 短测边界

`scripts/benchmark_paper_two_gpu.py`每组200次迭代，排除前50次，日志间隔10次；第二轮交换GPU对。
两种解读分别使用两张A100-SXM4-80GB，初始GPU对为0/1、4/5，NVLink相连。
本次同时运行两个双卡任务，共占用4卡；测得的每组吞吐仅由该组2卡产生。
脚本也支持仅提供`--gpus 0 1`，按顺序运行两种解读；这是默认模式。
源码保留两项采样优化，明确开启compile和CUDA graph。

`wm_only_iters=0`：从第一次正式迭代起同时更新世界模型与策略。开始前保留64个采样步填充replay，
这时不做优化器更新；该初始化数量是实现选择，论文没有规定。
短测仍使用25万次的学习率衰减日程，未把它缩短到200次。

脚本拒绝超过1000次迭代及超过两轮的请求，没有长训模式。
每组最后保存一次checkpoint；稳定速度来自训练循环中的区间日志，最后的checkpoint写入不包含在该速度中。
`process_wall_seconds`包含加载、初始化、编译、训练和最后的checkpoint写入。
源码HEAD、diff、实际命令、启动脚本副本以及每个rank的实际批量保存在输出目录。

复现命令（输出目录必须是新目录）：

```bash
.venv/bin/python scripts/benchmark_paper_two_gpu.py \
  --motion_source /train_result/log/train_data/data/boneseed_filtered_114406.h5 \
  --gpus 0 1 4 5 --output_dir /train_result/log/supermimic/paper_two_gpu_new \
  --iterations 200 --skip 50 --rounds 2
```

本次日志：`/train_result/log/supermimic/codex_opt/paper_two_gpu`；
tmux日志：`/train_result/log/supermimic/codex_opt/tmux/codex-paper-two-gpu.log`。

稳定耗时记为 $`t_{\mathrm{iter}}`$（秒/迭代），换算公式为：

```math
T_{\mathrm{hours}} = \frac{320{,}000\,t_{\mathrm{iter}}}{3600}.
```

这仅估算完成指定迭代数，不预测达到论文零失败结果的时间；
完整checkpoint评估另计，短测也不能代替长期收敛验证。


## 两轮实测结果（2026-10-27）

02:06:59启动，02:23:17结束（UTC+8），启动器退出码0；四组均完成200次，所有CSV值有限。
逐项核对args与rank清单，批量、学习率及完整数据分片均符合上表；结束后8张GPU显存占用均为0。
实际使用备份目录内的`.venv`与HDF5，验证了备份环境能够完成真实双卡训练与保存checkpoint。

| 批量解释 | 第一轮秒/迭代 | 第二轮秒/迭代 | 两轮均值/中位数 | 32万次更新迭代 |
| --- | ---: | ---: | ---: | ---: |
| 全局2048 / 4096 | 0.374962 | 0.372910 | 0.373936 | **33.24小时** |
| 每卡2048 / 4096 | 0.572887 | 0.572097 | 0.572492 | **50.89小时** |

两轮外推范围分别为33.15–33.33小时、50.85–50.92小时；这只是两次观测范围，不是统计置信区间。
每组进程总耗时分别为450.12 / 336.12秒、586.26 / 390.37秒；训练循环分别为168.71 / 91.35秒、304.07 / 131.13秒。
前者包含加载、预填充、编译、保存与退出，后者包含训练循环内的首次编译/捕捉；稳定速度已排除前50次。
最后一次checkpoint写入及训练后评估不在稳定速度内。排期可另留10–15%余量，但余量不是实测结果。

数据见[汇总与实际rank配置](paper_two_gpu_results.json)及`paper_two_gpu_logs/`的四份原始CSV。
本测没有对旧版进行相同TF32双卡A/B，不能将历史单卡BF16的2.65%提速直接套到这里。
严格FP32 eager/compile数值比较仍存在[optimization.md第8.3节](optimization.md#83-验证结果与边界)记录的误差，短测跑通不消除该差异。

## 同行核对与解释（检索日期2026-10-27）

- [CuboiLeo独立复现的训练循环](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/training/loop.py)
  每rank保留4096环境及自己的buffer，学习率乘sqrt(world)；
  [世界模型](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/wm_trainer.py)
  和[策略训练器](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/imagination/policy_trainer.py)
  直接抽取配置的2048窗口/4096起点，没有除以world_size。因此其默认训练批量按每卡理解。
- [zita-ch的实现](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/train.py)
  直接使用2048/4096，但所查入口是单进程，不能据此解决双卡口径。
  [btx0424实现](https://github.com/btx0424/rethinking-supermimic/blob/main/src/supermimic/algo.py)也是同样的批量数值，未见该入口按world_size切分。
- [Isaac Lab官方多卡说明](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html)
  明确环境数按每GPU计，比较扩展时保持每rank环境数。
  这支持每卡采样规模不变的做法，但不能证明SuperMimic的优化器批量。
- [DreamerV3官方JAX实现](https://github.com/danijar/dreamerv3/blob/main/embodied/jax/agent.py)
  将数据批维分片到训练设备；单主机配置批量可以分给多GPU，多主机还乘process_count。
  所以不存在所有世界模型项目都把batch解释为每GPU的统一惯例。
- [PyTorch DDP官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
  明确DDP不会自动切分输入；具体批量口径由应用代码决定。
- [Adam缩放规则论文](https://arxiv.org/abs/2205.10287)及
  [作者说明](https://sadhikamalladi.github.io/blog/2024/01/22/SDEs-ScalingRules/)
  将学习率sqrt(k)与批量扩大k倍联系起来；完整理论规则还涉及beta与epsilon。
  此处仅将sqrt(2)当作批量可能翻倍的线索，不擅自修改SuperMimic规定的Adam参数。

**判断：两种分配都能训练；若目的是复现论文双卡方案，建议优先采用每卡2048/4096，并以约51小时排期。**
这是结合最接近的公开复现、sqrt(2)学习率和每卡4096环境作出的推断，不是作者确认。
在本实现中，保持每卡训练批量也保持训练样本与新增仿真数据的比例；全局批量固定而环境翻倍会使该比例减半。
约33小时仍是全局批量方案的有效测量，不能因为它更接近H100的9.8小时就证明其口径更正确。

还需区分“迭代”的定义：本报告32万次指同时执行世界模型、策略和critic更新的外层循环，平均每轮采集2.5个向量化仿真步。
论文算法标题说明两个模型每次iteration都更新，支持这种解释；CuboiLeo的复现计划另将32万次按仿真步解释，
相当于约12.8万次更新，其文档也承认存在歧义。本报告没有采用其较短训练长度，亦未将33/51小时宣称为达到零失败所需时间。
当前训练数据114,406段仍少于论文129,514段，固定分片的采样方式也未证明与作者实现等价。
