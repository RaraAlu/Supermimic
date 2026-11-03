# 实时遥操作

目标是在G1上结合Supermimic的训练效率与精度、Teleopit式无未来帧输入，以及受扰后自主恢复并接回实时reference的能力。

2026-11-03修订：已只读比较rara上的Supermimic `07dee275`与Teleopit `4023a6b`，本地实现基线为`1fc30fe`。下面的组合方案仍是设计稿；源码对比不代表已经完成同条件训练基准、恢复实现或实机验证。2026-11-04另核对了Any2Track官方代码OpenTrack `cb9b751`、mjlab 1.4.0上游实现与所引论文。

| 文件 | 内容 |
| --- | --- |
| [causal_intent_encoder.md](causal_intent_encoder.md) | 实时跟踪与统一扰动恢复：因果特征、当前观测直通＋历史CNN、世界模型训练、恢复与重接、部署契约、对照实验及里程碑 |
| [teleopit_supermimic_comparison.md](teleopit_supermimic_comparison.md) | 两侧源码对比：更新工作量、采样、参考信息、速度差分、奖励时序、精度排查与可复用设计；区分已确认事实和待验证解释 |
