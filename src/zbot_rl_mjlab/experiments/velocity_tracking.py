"""Undisturbed walking with commanded speed and random stepping frequency."""

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from .. import mdp
from .flat_walking import walking_cfg, walking_ppo_cfg

VELOCITY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Tracking-Walking"


class SequentialVelocityCommand(UniformVelocityCommandCfg):
    """Command config that samples forward speeds by environment index."""

    def build(self, env):
        return _SequentialVelocityCommand(self, env)


class _SequentialVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        values = torch.linspace(
            self.cfg.ranges.lin_vel_x[0],
            self.cfg.ranges.lin_vel_x[1],
            self.num_envs,
            device=self.device,
        )
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = values[env_ids]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


def target_speed(env):
    return env.command_manager.get_command("target_speed")[:, :1]


def track_forward_speed(env, std=0.05):
    """Gaussian tracking reward; backward motion and overspeed increase error."""
    error = mdp.base_lin_vel(env)[:, 0] - target_speed(env)[:, 0]
    return torch.exp(-error.square() / std**2)


def velocity_command_cfg(
    lin_vel_x=(0.0, 0.5),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
    sequential=False,
    cfg_cls=None,
    debug_vis=False,
    **kwargs,
):
    """Velocity command in the Zbot frame: forward, left, yaw rate (CCW positive)."""
    cfg_cls = cfg_cls or (
        SequentialVelocityCommand if sequential else UniformVelocityCommandCfg
    )
    return cfg_cls(
        entity_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z, heading=None
        ),
        debug_vis=debug_vis,
        **kwargs,
    )


def velocity_tracking_cfg(play=False):
    cfg = walking_cfg(play=play)
    cfg.step_frequency_min = 0.25
    cfg.step_frequency_max = 1.0
    cfg.test_frequency = None
    cfg.frequency_sampling = "random"
    cfg.commands["target_speed"] = velocity_command_cfg(sequential=play)
    for group in cfg.observations.values():
        group.terms = dict(group.terms)
        group.terms["target_speed"] = ObservationTermCfg(func=target_speed)
    cfg.rewards.pop("forward_velocity")
    cfg.rewards["track_forward_speed"] = RewardTermCfg(
        func=track_forward_speed, weight=2.0, params={"std": 0.1}
    )
    return cfg


def velocity_tracking_ppo_cfg():
    cfg = walking_ppo_cfg()
    cfg.experiment_name = "zbot_velocity_tracking"
    return cfg
