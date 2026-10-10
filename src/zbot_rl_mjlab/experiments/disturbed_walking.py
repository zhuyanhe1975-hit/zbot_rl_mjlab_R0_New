"""Walking experiment with randomized disturbances and domain randomization."""

from ..env_cfg import (
    DISTURBED_WALKING_TASK_ID,  # noqa: F401
    ppo_cfg,
)
from .common import add_base_disturbance, add_domain_randomization
from .flat_walking import walking_cfg as flat_walking_cfg


def disturbed_walking_cfg(play=False):
    cfg = flat_walking_cfg(play=play)
    add_base_disturbance(cfg)
    add_domain_randomization(cfg)
    return cfg


def disturbed_walking_ppo_cfg():
    cfg = ppo_cfg()
    cfg.experiment_name = "zbot_disturbed_walking"
    return cfg
