"""Compact full-state teacher: 36 actor inputs and 44 critic inputs.

Teacher actors do not receive exact external pushes or reset history. Privileged
critic inputs improve value estimation without becoming required student sensors.
"""

import math
from copy import deepcopy

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .mdp import _base, body_masses, commanded_target, projected_gravity
from .robust import ROBUST_TASK_ID, fall_retry_state
from .snake_to_biped import _sole_direction_cosine, base_height, base_vertical_velocity

TEACHER_TASK_ID = ROBUST_TASK_ID + "-Teacher"
TEACHER_ACTOR_DIM = 36
TEACHER_CRITIC_DIM = 44
STUDENT_PROPRIO_DIM = 24


def joint_goal_error(env):
    position = env.scene["robot"].data.joint_pos
    return position - position.new_tensor(env.cfg.standing_joint_pos)


def joint_velocity(env):
    return env.scene["robot"].data.joint_vel


def independent_target(env):
    return commanded_target(env)[:, :3]


def independent_last_action(env):
    return env.action_manager.get_term("joint_pos").raw_action


def base_angular_velocity(env):
    data, index = _base(env)
    return quat_apply_inverse(data.body_link_quat_w[:, index], data.body_link_ang_vel_w[:, index])


def base_linear_velocity(env):
    data, index = _base(env)
    return quat_apply_inverse(data.body_link_quat_w[:, index], data.body_link_lin_vel_w[:, index])


def base_origin_height(env):
    return base_height(env).unsqueeze(-1)


def base_world_vertical_velocity(env):
    return base_vertical_velocity(env).unsqueeze(-1)


def ground_contacts(env):
    feet = env.scene["feet_ground_contact"].data.found
    base = env.scene["base_ground_contact"].data.found
    if feet is None or base is None:
        raise ValueError("Teacher observations require foot and base contact sensors")
    return torch.cat(((feet > 0).reshape(env.num_envs, 2), (base > 0).reshape(env.num_envs, -1).any(-1, keepdim=True)), dim=-1).float()


def sole_heights(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    return robot.data.site_pos_w[:, ids, 2] - env.scene.env_origins[:, 2, None]


def com_relative_to_base(env):
    data, index = _base(env)
    masses = body_masses(env)
    com = (data.body_com_pos_w * masses[..., None]).sum(-2) / masses.sum(-1, keepdim=True).clamp_min(1e-6)
    return quat_apply_inverse(data.body_link_quat_w[:, index], com - data.body_link_pos_w[:, index])


def normalized_external_wrench(env):
    data, index = _base(env)
    wrench = data.body_external_wrench[:, index]
    params = env.cfg.events["base_disturbance"].params
    force_scale = max(max(abs(v) for v in params["force_range"]), 1e-3)
    torque_scale = max(max(abs(v) for v in params["torque_range"]), 1e-3)
    quaternion = data.body_link_quat_w[:, index]
    return torch.cat((quat_apply_inverse(quaternion, wrench[:, :3]) / force_scale, quat_apply_inverse(quaternion, wrench[:, 3:]) / torque_scale), dim=-1)


def reset_memory(env):
    """Read-only reset state; never expose future push vectors/times to the actor."""
    state = fall_retry_state(env)
    height_drop = (state.peak_base_height - base_height(env)).clamp_min(0)
    return torch.stack((
        state.armed.float(), state.low_time / max(env.cfg.fall_retry_hold_s, 1e-6),
        state.motion_started.float(), state.had_base_contact.float(), state.base_left_ground.float(),
        height_drop / 0.1, state.still_time / max(env.cfg.stagnation_hold_s, 1e-6),
        env.episode_length_buf.float() / max(env.max_episode_length, 1),
    ), dim=-1)


def teacher_reference_observation_groups():
    # Clip is applied before scale by ObservationManager. All scaled values are
    # bounded at approximately +/-5; no second running normalizer is required.
    actor = {
        "joint_goal_error": ObservationTermCfg(func=joint_goal_error, scale=1 / math.pi, clip=(-4 * math.pi, 4 * math.pi)),
        "joint_velocity": ObservationTermCfg(func=joint_velocity, scale=0.5, clip=(-10, 10)),
        "independent_target": ObservationTermCfg(func=independent_target, scale=1 / math.pi, clip=(-4 * math.pi, 4 * math.pi)),
        "independent_last_action": ObservationTermCfg(func=independent_last_action, clip=(-1, 1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, clip=(-1, 1)),
        "base_angular_velocity": ObservationTermCfg(func=base_angular_velocity, scale=0.5, clip=(-10, 10)),
        "base_height": ObservationTermCfg(func=base_origin_height, scale=4.0, clip=(-0.25, 1.25)),
        "base_linear_velocity": ObservationTermCfg(func=base_linear_velocity, scale=5.0, clip=(-1, 1)),
        "base_world_vertical_velocity": ObservationTermCfg(func=base_world_vertical_velocity, scale=5.0, clip=(-1, 1)),
        "ground_contacts": ObservationTermCfg(func=ground_contacts),
        "sole_heights": ObservationTermCfg(func=sole_heights, scale=4.0, clip=(-0.25, 1.25)),
        "sole_down_cosine": ObservationTermCfg(func=_sole_direction_cosine, clip=(-1, 1)),
    }
    critic = deepcopy(actor)
    critic.update({
        "com_relative_to_base": ObservationTermCfg(func=com_relative_to_base, scale=4.0, clip=(-1.25, 1.25)),
        "external_wrench": ObservationTermCfg(func=normalized_external_wrench, clip=(-5, 5)),
        "reset_memory": ObservationTermCfg(func=reset_memory, clip=(0, 5)),
    })
    return {name: ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=False) for name, terms in (("actor", actor), ("critic", critic))}


def teacher_actor_state(env):
    """Collect/scale actor features once, shared by Actor and Critic this frame."""
    stamp = (env.common_step_counter, getattr(env, "_observation_generation", 0))
    cached = getattr(env, "_teacher_actor_cache", None)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    robot = env.scene["robot"]
    data = robot.data
    metadata = getattr(env, "_teacher_observation_metadata", None)
    if metadata is None:
        base_index = robot.find_bodies("base")[0][0]
        sites, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
        metadata = (
            base_index, torch.tensor(sites, device=env.device, dtype=torch.long),
            data.joint_pos.new_tensor(env.cfg.standing_joint_pos),
            data.joint_pos.new_tensor([0, 0, -1]).expand(env.num_envs, 3),
        )
        env._teacher_observation_metadata = metadata
    base_index, sites, goal, gravity = metadata
    quaternion = data.body_link_quat_w[:, base_index]
    linear = data.body_link_lin_vel_w[:, base_index]
    angular = data.body_link_ang_vel_w[:, base_index]
    position = data.body_link_pos_w[:, base_index]
    action = env.action_manager.get_term("joint_pos")
    foot_found = env.scene["feet_ground_contact"].data.found
    base_found = env.scene["base_ground_contact"].data.found
    if foot_found is None or base_found is None:
        raise ValueError("Teacher observations require foot and base contacts")
    contacts = torch.cat(((foot_found > 0).reshape(env.num_envs, 2), (base_found > 0).reshape(env.num_envs, -1).any(-1, keepdim=True)), dim=-1).float()
    sole_quat = data.site_quat_w[:, sites]
    normals = (1 - 2 * (sole_quat[..., 1].square() + sole_quat[..., 2].square())) * sole_quat.new_tensor([1, -1])
    normals = torch.where(torch.isfinite(sole_quat).all(-1), normals.clamp(-1, 1), float("nan"))
    heights = data.site_pos_w[:, sites, 2] - env.scene.env_origins[:, 2, None]
    value = torch.cat((
        (data.joint_pos - goal).clamp(-4 * math.pi, 4 * math.pi) / math.pi,
        data.joint_vel.clamp(-10, 10) * 0.5,
        action.commanded_target[:, :3].clamp(-4 * math.pi, 4 * math.pi) / math.pi,
        action.raw_action.clamp(-1, 1),
        quat_apply_inverse(quaternion, gravity).clamp(-1, 1),
        quat_apply_inverse(quaternion, angular).clamp(-10, 10) * 0.5,
        (position[:, 2:3] - env.scene.env_origins[:, 2:3]).clamp(-0.25, 1.25) * 4,
        quat_apply_inverse(quaternion, linear).clamp(-1, 1) * 5,
        linear[:, 2:3].clamp(-1, 1) * 5,
        contacts, heights.clamp(-0.25, 1.25) * 4, normals,
    ), dim=-1)
    env._teacher_actor_cache = (stamp, value)
    return value


def teacher_critic_state(env):
    return torch.cat((teacher_actor_state(env), reset_memory(env).clamp(0, 5)), dim=-1)


def teacher_observation_groups():
    return {
        "actor": ObservationGroupCfg(terms={"teacher_state": ObservationTermCfg(func=teacher_actor_state)}, enable_corruption=False),
        "critic": ObservationGroupCfg(terms={"teacher_critic_state": ObservationTermCfg(func=teacher_critic_state)}, enable_corruption=False),
    }


def requested_target_speed_rms(env):
    action = env.action_manager.get_term("joint_pos")
    return action.raw_action.square().mean(-1).sqrt() * action.rate_cfg.max_target_speed


def target_speed_rms(env):
    return env.action_manager.get_term("joint_pos").commanded_target_velocity.square().mean(-1).sqrt()


def joint_speed_rms(env):
    return env.scene["robot"].data.joint_vel.square().mean(-1).sqrt()


def joint_speed_peak(env):
    return env.scene["robot"].data.joint_vel.abs().amax(-1)


def actuator_torque_peak(env):
    return env.scene["robot"].data.actuator_force.abs().amax(-1)
