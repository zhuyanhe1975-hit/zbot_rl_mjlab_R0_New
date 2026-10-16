"""Velocity tracking without phase tracking, with energy efficiency shaping."""

from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from .base import RewardEdit
from .velocity_tracking import VelocityTracking

VELOCITY_ENERGY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"


class VelocityEnergyTracking(VelocityTracking):
    task_id = VELOCITY_ENERGY_TRACKING_TASK_ID
    experiment_name = "zbot_velocity_energy_tracking"

    def reward_edit(self):
        return RewardEdit(
            remove=("frequency_tracking",),
            weights={"joint_acceleration": -0.0001, "lateral_velocity": -0.5},
            terms={
                "joint_energy": RewardTermCfg(
                    func=mdp.joint_energy_penalty,
                    weight=-0.5,
                    params={"energy_scale": 10.0},
                )
            },
        )
