# 蛇足双向变形：iter599（600轮）

任务：`Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional`。
来源：`logs/rsl_rl/zbot_transform_snake_biped_bidirectional/2026-10-17_18-30-25/model_599.pt`。
从零训练的37维Actor / 45维Critic，3维镜像Beta动作，蛇身纵轴随机侧滚−180°～+180°。

## 验证结果

- 自动训练规则：32环境，两个方向各16个源初态，每回合16s，seed42。严格站立成功12/16、爬下16/16。全部16次站立都曾达到base高度0.23m以上；没有低位躺着摆双足的终态。四次站立在约1.5～1.84s触发跌倒保护，已原样保留失败记录。
- 连续键盘指令：8环境，随机侧滚蛇形开始，站立16s→爬下16s→站立16s，三个阶段均8/8达到并最终保持严格成功。自动reset共0次，按键回调不改root姿态、关节角、执行器目标或回合计数。
- 手动序列末尾的最短连续保持时间依次为12.06s、13.20s、9.94s。两个协议分别记录，不能将手动控制成功率当作训练跌倒规则下的全角度100%成功率。

详细结果：`evaluation.json`、`manual_evaluation.json`；简表：`evaluation_summary.json`。
以上为无外力仿真样本验证，未测试硬件或抗扰动表现。

## 播放

在仓库根目录执行：

```bash
python play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --checkpoint-file bests/transform_snake_biped_bidirectional_2026-10-17_iter599/model_599.pt \
  --env.scene.num-envs 1
```

点击MuJoCo窗口获得键盘焦点，**PageUp站立、PageDown爬下**。首次按键后关闭自动reset，保持当前状态执行新指令；Enter为主动reset并恢复自动演示。未按键时默认每16s重新采样。
可加`--env.seed 42`复现随机种子；加`--env.commands.transform.mode stand`可固定从随机侧滚蛇形开始。

## 保存内容

模型包含Actor/Critic、优化器、训练计数和最终奖励metadata；保留原始启动`env.yaml`/`agent.yaml`、`final_effective_weights.json`、TensorBoard事件、训练Git差异、当前项目源码及机器人资产快照、两个评估脚本和SHA256校验和。

五项权重：构型12、高度1、首次上升速度8、脚底朝下1、失败−10，无`goal_motion`。
构型项含完整base重力朝向误差；速度衰减结合构型、高度和朝向，并保留0.23m最高高度锁存关闭。跌倒从5cm启用，要求至少1cm回落；手动play不自动reset。

`source_snapshot/`用于追溯当时的定义，不是完整打包环境；外部MJLab和Python环境版本在`metadata.json`。未来改变动作/观测/物理定义时需重新核验归档的可播放性。
