# Zbot 任务重构重置计划

本次重置按用户明确要求执行：删除现有 Zbot 任务与训练设置，只保留机器人资产和项目永久记忆；将官方 MJLab G1 velocity task 复制为本地只读参考；之后从最小骨架重新构建 Zbot 任务。

保留：
- `assets/`
- `docs/project-memory.md`
- `docs/zbot-coordinate-system.md`
- `docs/reset-plan.md`
- Git 元数据

删除：
- 旧 `src/` Zbot 任务实现
- 旧训练、评估、回放和可视化脚本
- 旧 logs、outputs 和缓存产物

参考：
- `g1_reference/velocity/config/g1/`
- `g1_reference/velocity/velocity_env_cfg.py`
- `g1_reference/velocity/mdp/`

新任务重构顺序：资产编译 → Base 坐标和速度测量单测 → G1 任务结构复制 → Zbot 传感器/动作/奖励逐项替换 → headless smoke → Viewer 验证。

## 新入口约定

- `train.sh`：调用 `mjlab.scripts.train`，默认 task 为 `Mjlab-Zbot-6dof-Bipedal-Walking`。
- `run.sh`：调用 `mjlab.scripts.play`，默认 native viewer，支持 `./run.sh 32 --checkpoint path/to/model.pt`。
- 任务参数使用 `--task stepping`（周期踏步）或 `--task walking`（带前进奖励的行走任务）。
- `ZBOT_TASK_ID`、`ZBOT_LOG_ROOT`、`ZBOT_NUM_ENVS` 可覆盖默认值。
- 当前入口会在新 Zbot 包尚未创建时明确退出；不会回退运行 G1 或旧任务。

## 新预览入口

- `preview_train.sh`：监测周期踏步任务训练进程和新 checkpoint，后台刷新 Native Viewer。
- `preview_training.sh`：兼容旧命名的转发入口。
- 默认 task 为 `Mjlab-Zbot-6dof-Periodic-Stepping`；可用 `TRAIN_PID` 精确绑定训练进程。
- 周期踏步任务当前目标频率为 `1.5 Hz`；频率归一化值已加入 actor/critic 观测，频率相位接触匹配奖励记录为 `frequency_tracking`。
