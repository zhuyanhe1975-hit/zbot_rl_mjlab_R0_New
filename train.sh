source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 1、按照指定频率迈步+基本平地行走
# python train.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 600

# 2、按照指定频率迈步+平地行走+扰动力和力矩
# python train.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 600 \

# python train.py Mjlab-Zbot-6dof-Velocity-Tracking-Walking \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 1500 \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     # --env.test-frequency 1.0 \

#新建一个实验，跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python train.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500

#继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化，保留速度跟踪、无频率跟踪和能量奖励配置。
# python train.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500

#全方位行走实验
# python train.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500

#实机部署实验
# --disturbance	基座受推
# --domain-randomization	足底摩擦、编码器零偏、基座质心
# --mass-randomization	质量和惯量
# --gain-randomization	PD 增益
# --observation-noise	actor 观测噪声和偏置
# --action-delay [N]	动作延迟 0 到 N 步,默认 2,只在训练时生效
# --blind-actor	actor 看不到 base_lin_vel、yaw_error
# --deployment	以上全部
python train.py Mjlab-Zbot-6dof-Deploy-Flat-Foot-Walking \
    --deployment \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 500 \
    --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)"
