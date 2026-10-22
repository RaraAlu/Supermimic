# Supermimic

复现论文SuperMimic（arXiv:2610.28480，CoRL 2026 Oral）。

论文用一个可微世界模型训练G1全身动作追踪策略。
作者未公开代码。

## 当前状态

- 阶段：复现规划，尚无实现代码。
- 目标层级：T2，即在129,514段基准上复现主结果。
- 技术栈：未定。仿真暂时只能用上游MuJoCo Warp。

## 文档

| 文件 | 内容 |
| --- | --- |
| [plan/plan.md](plan/plan.md) | 主张、证据帐本、依赖、显存估算、验收标准、里程碑 |
| [plan/rubric.json](plan/rubric.json) | 加权评分树，叶子引用验收标准与帐本ID |

## 使用约束

- 数据集与检查点不入库。
- BONES-SEED需在Hugging Face同意许可后下载。
- LAFAN1采用CC BY-NC-ND 4.0。
- 论文项目页的网页演示受专有许可约束。
  它只能人工浏览，不能下载、抓取或用作对照。
