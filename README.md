# CLVA

**CLVA** (Combined Learning for Visio Action model) 是一个占位型研究项目骨架，用于搭建三阶段训练流程：

1. IL：先对 VA 模型（暂定 DiT）做模仿学习预训练。
2. DreamerV3：训练世界模型，预测一个 chunk 之后的图像。
3. RL：在仿真中进行强化学习后训练。

核心评分逻辑（计划）：

- VA 在仿真执行一个 chunk 后得到真实图像；
- DreamerV3 对同一 chunk 终点给出预测图像；
- 基于两者一致性构建分数/奖励。

## 项目结构

```text
chunkrl/
├── agents/          # 顶层 Agent 相关占位模块
├── policy/          # 策略占位模块
├── data/            # 数据集目录
├── util/            # 工具脚本（IL 预训练、RL 后训练、Dreamer 训练）
├── scripts/         # 各阶段运行脚本
└── README.md
```

## 当前状态

- 目前以占位代码为主，便于后续逐步填充实现。
- 已提供 IL、DreamerV3、RL 三个阶段的最小入口脚本。
- 已预留 Agent 与 Policy 的基础接口文件。

## 运行占位入口

```bash
python scripts/run_il.py
python scripts/run_dreamer.py
python scripts/run_rl.py
```
