#!/usr/bin/env bash
set -euo pipefail
source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 左: 原生 MuJoCo + PyTorch 推理；右: MJLab 推理。省略 --checkpoint 时取 logs/ 下最新的 model_*.pt。
python viewer_compare.py \
    --task Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
    --target-speed 0.4 \
    --force-scale 0 \
    --torque-scale 0 \
    "$@"
