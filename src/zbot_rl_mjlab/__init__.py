from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import TASK_ID
from .experiments.disturbed_walking import (
    DISTURBED_WALKING_TASK_ID,
    disturbed_walking_cfg,
    disturbed_walking_ppo_cfg,
)
from .experiments.flat_walking import WALKING_TASK_ID, walking_cfg, walking_ppo_cfg
from .experiments.in_place_stepping import stepping_cfg, stepping_ppo_cfg
from .experiments.omni_velocity_energy import (
    OMNI_VELOCITY_ENERGY_TASK_ID,
    omni_velocity_energy_cfg,
    omni_velocity_energy_ppo_cfg,
)
from .experiments.velocity_energy_disturbed import (
    VELOCITY_ENERGY_DISTURBED_TASK_ID,
    velocity_energy_disturbed_cfg,
    velocity_energy_disturbed_ppo_cfg,
)
from .experiments.velocity_energy_tracking import (
    VELOCITY_ENERGY_TRACKING_TASK_ID,
    velocity_energy_tracking_cfg,
    velocity_energy_tracking_ppo_cfg,
)
from .experiments.velocity_tracking import (
    VELOCITY_TRACKING_TASK_ID,
    velocity_tracking_cfg,
    velocity_tracking_ppo_cfg,
)

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=stepping_cfg(),
    play_env_cfg=stepping_cfg(play=True),
    rl_cfg=stepping_ppo_cfg(),
)
register_mjlab_task(
    task_id=WALKING_TASK_ID,
    env_cfg=walking_cfg(),
    play_env_cfg=walking_cfg(play=True),
    rl_cfg=walking_ppo_cfg(),
)
register_mjlab_task(
    task_id=DISTURBED_WALKING_TASK_ID,
    env_cfg=disturbed_walking_cfg(),
    play_env_cfg=disturbed_walking_cfg(play=True),
    rl_cfg=disturbed_walking_ppo_cfg(),
)
register_mjlab_task(
    task_id=VELOCITY_TRACKING_TASK_ID,
    env_cfg=velocity_tracking_cfg(),
    play_env_cfg=velocity_tracking_cfg(play=True),
    rl_cfg=velocity_tracking_ppo_cfg(),
)
register_mjlab_task(
    task_id=VELOCITY_ENERGY_TRACKING_TASK_ID,
    env_cfg=velocity_energy_tracking_cfg(),
    play_env_cfg=velocity_energy_tracking_cfg(play=True),
    rl_cfg=velocity_energy_tracking_ppo_cfg(),
)
register_mjlab_task(
    task_id=VELOCITY_ENERGY_DISTURBED_TASK_ID,
    env_cfg=velocity_energy_disturbed_cfg(),
    play_env_cfg=velocity_energy_disturbed_cfg(play=True),
    rl_cfg=velocity_energy_disturbed_ppo_cfg(),
)
register_mjlab_task(
    task_id=OMNI_VELOCITY_ENERGY_TASK_ID,
    env_cfg=omni_velocity_energy_cfg(),
    play_env_cfg=omni_velocity_energy_cfg(play=True),
    rl_cfg=omni_velocity_energy_ppo_cfg(),
)
