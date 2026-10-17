"""Flat-ground walking at a commanded stepping frequency."""

from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from ..env_cfg import WALKING_TASK_ID
from .base import Experiment, RewardEdit


class FlatWalking(Experiment):
    task_id = WALKING_TASK_ID
    experiment_name = "zbot_walking"
    train_envs = 8192
    step_frequency = (0.25, 1.0)

    def reward_edit(self):
        return RewardEdit(
            weights={"frequency_tracking": 2.0},
            terms={
                "forward_velocity": RewardTermCfg(
                    func=mdp.forward_velocity_reward, weight=2.0
                )
            },
        )


__all__ = ["WALKING_TASK_ID", "FlatWalking"]
