"""Flat-ground velocity walking recipe."""

from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from ..env_cfg import WALKING_TASK_ID, env_cfg, ppo_cfg


def walking_cfg(play=False):
    cfg = env_cfg(play=play)
    cfg.scene.num_envs = 8192 if not play else 1
    cfg.step_frequency_min = 0.5
    cfg.step_frequency_max = 1.5
    cfg.rewards["frequency_tracking"].weight = 1.5
    cfg.rewards["forward_velocity"] = RewardTermCfg(
        func=mdp.forward_velocity_reward, weight=2.0
    )
    return cfg


def walking_ppo_cfg():
    cfg = ppo_cfg()
    cfg.experiment_name = "zbot_walking"
    return cfg


__all__ = ["WALKING_TASK_ID", "walking_cfg", "walking_ppo_cfg"]
