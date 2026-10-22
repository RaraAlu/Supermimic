# Supermimic

复现论文SuperMimic（arXiv:2610.28480，CoRL 2026 Oral）。

论文用一个可微世界模型训练G1全身动作追踪策略。
作者未公开代码。

## 当前状态

- 阶段：复现规划v2.2，尚无实现代码。
- 目标层级：T2，即在129,514段基准上复现主结果；T3除实机外全部排期。
- 算力：8卡H100服务器负责训练与评估；RTX 5080与RTX 4070 Ti SUPER负责开发。
- 依赖版本：mjlab 1.4.0、mujoco-warp 3.8.1、warp-lang 1.15.0。
- 数据：复用Bote_Teleopit的HDF5格式与MotionLib；观测与奖励按论文重写（D2已定）。

## 文档

| 文件 | 内容 |
| --- | --- |
| [plan/plan.md](plan/plan.md) | 主张、证据帐本、依赖、显存估算、验收标准、里程碑 |
| [plan/rubric.json](plan/rubric.json) | 加权评分树，叶子引用验收标准与帐本ID |

## 使用约束

- 数据集与检查点不入库。
- BONES-SEED需在Hugging Face同意许可后下载。
- LAFAN1采用CC BY-NC-ND 4.0。
- AMASS：需注册，仅限非商用研究，不得再分发。
- 论文项目页的网页演示受专有许可约束。
  它只能人工浏览，不能下载、抓取或用作对照。
