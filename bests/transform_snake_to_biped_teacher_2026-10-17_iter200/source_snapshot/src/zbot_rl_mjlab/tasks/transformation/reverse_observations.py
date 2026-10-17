"""Optional, read-only support feedback for the reverse transformation task."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

SUPPORT_OBSERVATION_COLUMNS = (
    "base_angular_velocity_x_div_10",
    "base_angular_velocity_y_div_10",
    "base_angular_velocity_z_div_10",
    "foot_0_contact",
    "foot_1_contact",
    "sole_0_height_div_0_1",
    "sole_1_height_div_0_1",
    "sole_0_down_cosine",
    "sole_1_down_cosine",
    "support_armed",
    "support_violation_fraction",
    "standing_stage",
    "standing_ramp_fraction",
)


def reserved_hold_observation(env):
    """Keep the legacy one-column slot fixed at zero; success timers stay private."""
    return torch.zeros(env.num_envs, 1, device=env.device)


def support_observation(env):
    """Return zero columns by default or the 13 columns listed above.

    Signed angular rates (base frame, 10 rad/s scale), heights (0.1 m scale)
    and downward cosines are clipped to [-1, 1]. Flags, elapsed violation
    fraction, stage (1 -> 0, 2 -> 1) and ramp fraction are in [0, 1].
    Reading observations never creates or advances support protection state.
    """
    if not env.cfg.support_observations:
        return torch.zeros(env.num_envs, 0, device=env.device)
    from .snake_to_biped import _sole_direction_cosine

    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    data = robot.data
    angular_velocity = (
        quat_apply_inverse(
            data.body_link_quat_w[:, base_id], data.body_link_ang_vel_w[:, base_id]
        )
        / 10.0
    )
    found = env.scene["feet_ground_contact"].data.found
    if found is None:
        raise ValueError("feet_ground_contact sensor must provide found data")
    contact = (torch.isfinite(found) & (found > 0)).to(angular_velocity.dtype)
    site_ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    heights = (
        data.site_pos_w[:, site_ids, 2] - env.scene.env_origins[:, 2, None]
    ) / 0.1
    cosine = _sole_direction_cosine(env)
    support = getattr(env, "_support_protection_state", None)
    flags = angular_velocity.new_zeros(env.num_envs, 2)
    if support is not None:
        flags[:, 0] = support.support_armed
        flags[:, 1] = support.support_violation_s / max(
            env.cfg.support_failure_hold_s, 1e-6
        )
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    curriculum = angular_velocity.new_zeros(env.num_envs, 2)
    if stage == 2:
        curriculum[:, 0] = 1.0
        start = getattr(env, "_standing_stage_start_step", None)
        curriculum[:, 1] = (
            1.0
            if start is None or env.cfg.standing_stage_ramp_steps == 0
            else (env.common_step_counter - start) / env.cfg.standing_stage_ramp_steps
        )
    signed = torch.cat((angular_velocity, heights, cosine), dim=-1)
    signed = torch.nan_to_num(signed).clamp(-1.0, 1.0)
    flags = torch.nan_to_num(flags).clamp(0.0, 1.0)
    curriculum = torch.nan_to_num(curriculum).clamp(0.0, 1.0)
    return torch.cat((signed[:, :3], contact, signed[:, 3:], flags, curriculum), dim=-1)
