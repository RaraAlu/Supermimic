# 实时遥操作

Supermimic跟踪策略与Teleopit（PICO输入、重定向、sim2real运行时）对接的设计与记录。
截至Supermimic `c00864c`、Teleopit `4023a6b`，本目录只有设计稿，没有实现或测量结果。

| 文件 | 内容 |
| --- | --- |
| [causal_intent_encoder.md](causal_intent_encoder.md) | 方案D：因果意图编码器替换未来帧目标编码器；两侧接口、备选方案比较、训练三阶段、数据与噪声、ONNX与Teleopit对接、评测、风险与里程碑 |
