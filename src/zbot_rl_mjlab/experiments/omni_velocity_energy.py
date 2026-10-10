"""Omnidirectional walking: track forward, lateral and yaw-rate commands."""

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from .velocity_energy_tracking import (
    velocity_energy_tracking_cfg,
    velocity_energy_tracking_ppo_cfg,
)
from .velocity_tracking import velocity_command_cfg

OMNI_VELOCITY_ENERGY_TASK_ID = "Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking"

# Command ranges: forward m/s, left m/s, yaw rate rad/s (counter-clockwise positive).
LIN_VEL_X = (-0.3, 0.5)
LIN_VEL_Y = (-0.2, 0.2)
ANG_VEL_Z = (-0.5, 0.5)


def velocity_command(env):
    # The command term keeps the "target_speed" key so play.py and sim2sim overrides
    # keep working; here it carries all three components.
    return env.command_manager.get_command("target_speed")


def track_lateral_speed(env, std=0.1):
    error = mdp.base_lin_vel(env)[:, 1] - velocity_command(env)[:, 1]
    return torch.exp(-error.square() / std**2)


def track_yaw_rate(env, std=0.25):
    # base_ang_vel's third axis points along gravity (down), so a counter-clockwise
    # turn seen from above is negative there; see docs/zbot-coordinate-system.md.
    error = -mdp.base_ang_vel(env)[:, 2] - velocity_command(env)[:, 2]
    return torch.exp(-error.square() / std**2)


def omni_velocity_energy_cfg(play=False):
    cfg = velocity_energy_tracking_cfg(play=play)
    cfg.commands["target_speed"] = velocity_command_cfg(
        lin_vel_x=LIN_VEL_X,
        lin_vel_y=LIN_VEL_Y,
        ang_vel_z=ANG_VEL_Z,
        cfg_cls=mdp.VelocityCommandVizCfg,
        debug_vis=play,
        full_scale=tuple(max(map(abs, r)) for r in (LIN_VEL_X, LIN_VEL_Y, ANG_VEL_Z)),
    )
    # The 3D command arrows replace the forward-only speed arrows used in play.
    cfg.commands.pop("forward_velocity_viz", None)
    for group in cfg.observations.values():
        group.terms = dict(group.terms)
        # Heading drift against the initial direction is meaningless when turning.
        group.terms.pop("yaw_error")
        group.terms.pop("target_speed")
        group.terms["velocity_command"] = ObservationTermCfg(func=velocity_command)
    # Lateral speed and yaw drift were penalties; now they are commanded.
    cfg.rewards.pop("lateral_velocity")
    cfg.rewards.pop("yaw_drift")
    cfg.rewards["track_lateral_speed"] = RewardTermCfg(
        func=track_lateral_speed, weight=1.0, params={"std": 0.1}
    )
    cfg.rewards["track_yaw_rate"] = RewardTermCfg(
        func=track_yaw_rate, weight=1.0, params={"std": 0.25}
    )
    return cfg


def omni_velocity_energy_ppo_cfg():
    cfg = velocity_energy_tracking_ppo_cfg()
    cfg.experiment_name = "zbot_omni_velocity_energy"
    return cfg
