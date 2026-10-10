source /home/yhzhu/AI/mjlab/.venv/bin/activate

#变频率变速度的平地行走
# python play.py Mjlab-Zbot-6dof-Velocity-Tracking-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     --env.scene.num-envs 8 \
#     --checkpoint-file bests/velocity_energy_disturbed_2026-10-10_iter499/model_499.pt

#指定频率、速度的平地行走
# python play.py Mjlab-Zbot-6dof-Velocity-Tracking-Walking \
#     --frequency 0.5 \
#     --target-speed 0.2 \
#     --checkpoint-file bests/velocity_energy_disturbed_2026-10-10_iter499/model_499.pt

# 基本平地行走
# python play.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8 \
#     --checkpoint-file bests/bests/zbot6_walking_2026-10-09_13-43-52/model_599.pt

# 带扰动力的平地行走
# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 4 \
#     --checkpoint-file bests/zbot6_disturbed_walking_2026-10-09_14-25-33/model_599.pt

#新建一个实验，跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 

#继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化，保留速度跟踪、无频率跟踪和能量奖励配置。
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 

#全方位行走实验
python play.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
    --env.scene.num-envs 1

