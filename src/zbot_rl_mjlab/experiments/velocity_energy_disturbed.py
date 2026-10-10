"""Energy-aware velocity tracking with disturbance randomization."""

from .common import add_base_disturbance, add_domain_randomization
from .velocity_energy_tracking import (
    velocity_energy_tracking_cfg,
    velocity_energy_tracking_ppo_cfg,
)

VELOCITY_ENERGY_DISTURBED_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking"


def velocity_energy_disturbed_cfg(play=False):
    cfg = velocity_energy_tracking_cfg(play=play)
    add_base_disturbance(cfg)
    add_domain_randomization(cfg)
    return cfg


def velocity_energy_disturbed_ppo_cfg():
    cfg = velocity_energy_tracking_ppo_cfg()
    cfg.experiment_name = "zbot_velocity_energy_disturbed"
    return cfg
