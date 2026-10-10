"""Velocity tracking without phase tracking, with energy efficiency shaping."""

from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from .velocity_tracking import velocity_tracking_cfg, velocity_tracking_ppo_cfg

VELOCITY_ENERGY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"


def velocity_energy_tracking_cfg(play=False):
    cfg = velocity_tracking_cfg(play=play)
    cfg.rewards.pop("frequency_tracking", None)
    cfg.rewards["joint_acceleration"].weight = -0.0001
    cfg.rewards["lateral_velocity"].weight = -0.5
    cfg.rewards["joint_energy"] = RewardTermCfg(
        func=mdp.joint_energy_penalty,
        weight=-0.5,
        params={"energy_scale": 10.0},
    )
    return cfg


def velocity_energy_tracking_ppo_cfg():
    cfg = velocity_tracking_ppo_cfg()
    cfg.experiment_name = "zbot_velocity_energy_tracking"
    return cfg
