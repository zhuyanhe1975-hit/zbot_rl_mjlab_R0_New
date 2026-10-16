source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 1、按照指定频率迈步+基本平地行走
python play.py Mjlab-Zbot-6dof-Walking \
    --env.step-frequency-min 0.3 \
    --env.step-frequency-max 1.0 \
    --env.scene.num-envs 8 \
    --checkpoint-file bests/zbot_walking/model_599.pt

# 2、按照指定频率迈步+平地行走+扰动力和力矩
# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.5 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/zbot_disturbed_walking/model_599.pt

# 3、跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_energy_tracking_2026-10-10_iter499/model_499.pt

# 4、继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_energy_disturbed_2026-10-10_iter499/model_499.pt

# 5、全方位行走实验
# python play.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
#     --checkpoint-file bests/omni_velocity_energy_2026-10-10_iter499/model_499.pt \
#     --env.scene.num-envs 1 \

# 6、实机部署实验-1 （扰动 + 域随机化）
# python play.py Mjlab-Zbot-6dof-Deploy-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.4)" \
#     --env.scene.num-envs 16 \
#     --checkpoint-file bests/deploy_walking_2026-10-11_iter1499/model_1499.pt
