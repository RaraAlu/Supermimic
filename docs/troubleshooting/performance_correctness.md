# 性能优化中遇到的正确性问题

- 范围：为提速引入`torch.compile`、CUDA graph、fused Adam与多卡后遇到的问题及解法。
- 测量与收益见[optimization.md](../analysis/optimization.md)；本文只按“问题→原因→解决→验证”整理。

## 1. 起点：kernel发射瓶颈

A100上世界模型段10次更新约280 ms，其中只有约45 ms随批量变化，其余是kernel发射、Python调度与优化器开销（估计）。
compile与CUDA graph叠加后单卡每次迭代从1.457 s降到0.512 s（2.8×）。双卡只带来约1.35倍速度：每卡的发射数不变。

## 2. 问题清单

| 问题 | 原因 | 解决 | 验证 / 提交 |
| --- | --- | --- | --- |
| 图内不能做主机分支，非有限梯度原先被静默跳过 | 策略梯度非有限时只跳过`optimizer.step`，训练继续 | 世界模型、critic、策略统一记入GPU上的`nonfinite`标志，每次迭代写日志与检查点前检查，非有限即中止，坏的一步不会被保存 | `42db6e6`，新增9项学习数学单测 |
| `cumprod`反向触发主机同步 | 启用图时存活权重的`cumprod`反向会同步 | 改为逐步连乘，前向逐位一致 | `373fb54` |
| Adam不能进图 | 学习率是Python标量 | 世界模型与策略的Adam改为capturable，学习率为设备张量并原地更新 | `373fb54` |
| 多进程捕获失效 | NCCL watchdog线程的事件查询使捕获失效 | 多进程捕获用`capture_error_mode="thread_local"` | `e5c8739` |
| 梯度平均有主机往返 | 要先判断“哪些参数有梯度” | `average_gradients(..., total_count=)`：固定全局样本数时跳过判断，可在图内执行，与通用路径逐位一致 | `e5c8739`，两进程gloo测试 |
| `RunningNorm`每次更新同步4次 | 冻结判断`count >= max_count`在主机上做 | 改为GPU上`torch.where`选新旧值，缓冲原地写入（图读取的地址不变） | `e6db3e8`，跨冻结点30次更新逐位一致 |
| 加载检查点后fused Adam失效 | `load_state_dict`用检查点里的优化器标志覆盖当前设置 | 加载后重设`fused/foreach/capturable`，step与学习率类型按当前模式恢复，见下一项 | `213bcbd`、`38c9425` |
| 旧检查点恢复训练，第一次世界模型更新即CUDA报错 | 非fused、非capturable检查点的Adam step计数在CPU上；加载后改成fused/capturable却没搬step，fused内核把CPU地址当显存读 | 恢复时按当前模式放置：fused或capturable时step放参数所在设备，否则放CPU；学习率在capturable时为设备张量 | `38c9425`；Rara上旧↔新四种组合恢复后都训练到第20次迭代，未修时复现`CUDA error: invalid argument` |
| 软终止在fp32下被抵消成0 | `1 - sigmoid(x)`在 $`x`$ 超过约17时精确为0、梯度也为0；返回`d = 1 - alive`后存活权重再算`1 - d`，极小的存活概率被舍成0（Barron等，[arXiv:2610.31642](https://arxiv.org/abs/2610.31642)所说的抵消） | `MDP.soft_survival`写作`sigmoid(-x)`，存活权重与GAE直接用每步存活概率 | `4293e90`；图重放与直接执行逐位一致，新增单测覆盖 $`x=17.5`$ |
| 同一步结束于同一分箱的回合，分箱更新结果不确定 | 对重复索引`index_put_`会互相覆盖，结果未定义；固定输入重复40次得到39种结果 | 先按稳定顺序分组取均值，再做一次 $`0.8\cdot\text{old}+0.2\cdot\text{mean}`$ 更新 | `9852fb4`，CUDA重复40次逐位一致 |
| 进程退出时卡住 | 包含采样归一化collective的图在销毁NCCL进程组后才释放 | 销毁进程组前先释放图 | 见optimization.md第8.1节 |
| eager与compile不逐位一致 | 编译后的内核数值不同 | 不追求逐位；严格比较门槛`atol=1e-5, rtol=1e-3`下世界模型参数更新最大差`1.647e-5`，记为未通过；CUDA graph与同一编译路径逐位一致 | `check_compile_equiv.py`，见optimization.md第8.3节 |
| 每步12次多余同步 | `x[ids] = 0.0`这种写法在torch 2.14.1上每次同步 | 改`index_fill_`后每步同步从13.1次降到1.0次，但只快0.4%（这些同步都在`done.nonzero()`之后，GPU队列已空），**未采用** | optimization.md第7.5节 |

## 3. 验证工具

- `scripts/check_cuda_graph.py`：逐rank比较图重放与直接执行的输出、参数、缓冲是否逐位一致（单卡与双卡NCCL）。
- `scripts/check_compile_equiv.py`：eager与compiled各自从独立快照恢复后比较，避免编译预热污染起点。
- 每项优化都附固定种子的短程训练对照，曲线落在基线两次运行的波动内才保留。

## 4. 经验

- 图化只省CPU发射，不省GPU计算；分段计时前后有同步，分段数字不能直接当作可省的时间。
- 定义了新语义的修复（如重复分箱的分组均值）不能声称与旧的未定义行为逐位一致，要单独说明。
- 每次把一部分放进图，都要先确认其中没有主机读值、没有变化的形状、随机数走同一生成器。
