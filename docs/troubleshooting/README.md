# 问题与解决记录

复现过程中遇到的困难、原因与解决办法。每条只写结论与依据，细节放在分册里；原理与公式仍以`docs/analysis/`为准。
记录截至2026-10-27（分支`fix/merge-short-tail-bins`，提交`a3c41e3`）。
现行行为于2026-10-28按main的`4b4d27a`复核；历史run状态与服务器记录仍以原日期为准，没有重新查询远端训练状态。

| 分册 | 内容 |
| --- | --- |
| [reset_curriculum.md](reset_curriculum.md) | 按回报重置采样的四轮课程跑偏：尾部极短分箱、首步惩罚假设、重置过渡期、短片段首帧 |
| [training_metrics.md](training_metrics.md) | 指标判读与监督：`ep_ret_norm`、学习率口径、`wm_err8`、短暂崩溃、覆盖率、监督阈值、critic归一化、评估显存 |
| [performance_correctness.md](performance_correctness.md) | compile、CUDA graph、fused Adam与多卡带来的正确性问题 |
| [platform_ops.md](platform_ops.md) | Windows/Linux共用仓库、远程服务器连接与传输、代码同步、运行管理 |

## 总表

状态：**已解决**＝修正已提交并验证；**验证中**＝记录当时已提交、完整训练尚在进行；**未解决**＝尚无验证过的修正；**已取代**＝后续方案替换；**绕行**＝仅有操作上的替代办法。

| 问题 | 解决办法 | 状态 | 详见 |
| --- | --- | --- | --- |
| `ep_ret_norm`为负，课程被片段末尾极短分箱吸走 | 少于25个起点的尾箱并入前一箱（`fc9b5e8`） | 已解决 | [reset_curriculum.md](reset_curriculum.md)第2节 |
| 以为首步平滑惩罚压低末尾分箱评分 | `--bin_first_step`实测无效，已被取代（`85d7f82`） | 已取代 | 第3节 |
| 重置过渡期（约1–2 s）压低短回合评分，单分箱占42%重置 | 评分跳过前50步，末尾100步不起步，撑不过过渡期的分箱只分均匀份额（`a0695e5`） | 已解决 | 第4节 |
| 只能从首帧起步的短片段在训练中占43–48%重置 | 这类首帧不评分，只分均匀份额（`a3c41e3`） | 验证中 | 第5节 |
| 真实失败位置上的集中度（子集上单分箱14%） | 属于论文设计；份额上限只作消融 | 未解决 | 第6节 |
| global批量实验的学习率口径 | 该次实验选`--lr_scaling none`；论文训练批量口径仍未确认 | 未解决 | [training_metrics.md](training_metrics.md)第2节 |
| `wm_err8`与论文图3(c)不可比 | `diag_wm_error.py`按物理单位与候选标称误差测量；论文标称值未知 | 未解决 | 第3节 |
| 课程转向后`wm_err8`升到0.45 | 固定分布诊断确认是数据分布变化；监督阈值放宽 | 已解决 | 第4、7节 |
| 2.23万次迭代短暂崩溃 | 自行恢复；原因未查明 | 未解决 | 第5节 |
| 覆盖率停在78.4% | 等于可重置分箱占比，不是卡住 | 已解决 | 第6节 |
| critic输入归一化约1,480次迭代冻结 | 只分析，未修改 | 未解决 | 第8节 |
| 全量评估不能与训练同时跑 | 拉回Rara子集评估，或等GPU空闲 | 未解决 | 第9节 |
| 非有限策略梯度被静默跳过 | 统一`nonfinite`标志，写检查点前中止（`42db6e6`） | 已解决 | [performance_correctness.md](performance_correctness.md) |
| CUDA graph捕获中的主机同步、NCCL捕获失效、Adam不能进图 | 逐步连乘、`thread_local`捕获、capturable Adam、固定样本数梯度平均 | 已解决 | 同上 |
| 恢复训练后Adam模式、step设备或学习率类型与当前执行方式不符 | 加载后按当前fused/图捕获模式重新设置（`38c9425`） | 已解决 | 同上 |
| 重复分箱更新结果不确定（40次39种结果） | 分组均值后一次更新（`9852fb4`） | 已解决 | 同上 |
| eager与compile不逐位一致 | 记录差异，CUDA graph与同一编译路径逐位一致 | 未解决 | 同上 |
| Linux侧`uv run`删掉Windows的`.venv` | 执行一律经`ssh Rara` | 已解决 | [platform_ops.md](platform_ops.md)第1节 |
| Windows编码、路径断言、`from_args`装饰器回归 | UTF-8、`as_posix()`、补回`@classmethod` | 已解决 | 同上 |
| 服务器连接：无TTY、主机密钥、MaxStartups、绕过主连接 | 终端标签页交互登录、核对指纹、重试、一律用别名 | 已解决 | 第2节 |
| 主连接约4 h失效、大文件传输断开 | 监督脚本失败后提示重登；诊断放服务器、小文件限速传输 | 绕行 | 第2、3节 |
| 两台机器的提交哈希不一致 | `format-patch`加`am --committer-date-is-author-date`，同一身份 | 已解决 | 第4节 |

## 论文未规定、因而造成困难的细节

| 细节 | 本实现的选择 | 详见 |
| --- | --- | --- |
| 表A.5批量是全局还是每卡 | 两种都测；当前用global，学习率不乘 $`\sqrt{2}`$ | [paper_two_gpu.md](../analysis/paper_two_gpu.md) |
| 世界模型每次迭代10步×2048还是1步×20480 | 默认10步，`--wm_update single`为另一读法 | [reset_sampling.md](../analysis/reset_sampling.md)第7节 |
| 截断回合与片段末尾的分箱评分 | 本回合平均奖励外推；`--bin_clip_end terminal`可选 | [reset_curriculum.md](reset_curriculum.md) |
| 未访问分箱的初值 | 0；`--bin_prior mean`可选 | 同上 |
| 重置后的过渡期与起点范围 | 跳过50步评分，末尾100步不起步 | 同上 |
| 8步预测误差的标称值 | 未知，按物理单位报告 | [training_metrics.md](training_metrics.md)第3节 |
| critic输入归一化 | 累计平均，2e8样本后冻结 | 同上第8节 |

其余差异见[training_pipeline.md](../analysis/training_pipeline.md)第11节。

## 通用经验

1. **先量化，再修正。** 第二轮“首步惩罚”看起来能解释现象，实测却不是主因；用回合长度分布等直接证据定位后再改代码。
2. **尖锐的softmax会放大任何系统性偏差。** 课程每次被吸走，都是某类分箱的评分被结构性地压低；看`samp_top1`与`reset_tail_frac`最快。
3. **离线验证通过不等于训练中不会出问题。** 验证检查点失败多时，偏差会被真实失败分摊；要在训练中持续监控，并在可重置分箱全部访问过后重点观察。
4. **指标先问“在什么数据上算的”。** `wm_err8`算在回放数据上，`ep_ret_norm`按回合平均，它们的变化先要排除数据分布与统计口径。
5. **异常要能从日志定位到时间点。** 每50次迭代的日志、采样快照与每2000次迭代的检查点，让崩溃和课程跑偏都能对上具体迭代。
6. **远端操作留档。** run目录保存启动脚本、`console.log`、`user-stop.json`与`exit.json`；会话名与目录同名，事后能复原发生了什么。
