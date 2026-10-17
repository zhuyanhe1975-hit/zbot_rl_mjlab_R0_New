"""Periodic in-place stepping: the shared Zbot environment with no changes."""

from ..env_cfg import TASK_ID
from .base import Experiment


class PeriodicStepping(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_periodic_stepping"


__all__ = ["TASK_ID", "PeriodicStepping"]
