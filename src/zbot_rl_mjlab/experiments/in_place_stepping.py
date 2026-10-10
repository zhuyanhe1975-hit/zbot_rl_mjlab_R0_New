"""Periodic in-place stepping task entry point."""

from ..env_cfg import TASK_ID, env_cfg, ppo_cfg


def stepping_cfg(play=False):
    return env_cfg(play=play)


def stepping_ppo_cfg():
    cfg = ppo_cfg()
    cfg.experiment_name = "zbot_periodic_stepping"
    return cfg


__all__ = ["TASK_ID", "stepping_cfg", "stepping_ppo_cfg"]
