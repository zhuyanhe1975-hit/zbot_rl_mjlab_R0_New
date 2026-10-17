"""Reverse task: rise from a grounded straight snake into the standing joint pose."""

import math
from dataclasses import dataclass, fields, replace
from functools import lru_cache

import mujoco
import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from ...robot import DEFAULT_JOINT_POS, JOINT_NAMES, get_spec
from . import mdp
from .actions import JointTargetRateActionCfg
from .env_cfg import TransformationEnvCfg, transformation_env_cfg
from .reverse_observations import reserved_hold_observation, support_observation
from .snake_pose import SNAKE_JOINT_POS, SNAKE_ROOT_POS, SNAKE_ROOT_QUAT

TASK_ID = "Mjlab-Zbot-6dof-Transform-Snake-To-Biped"
BETA_TASK_ID = TASK_ID + "-Beta"
ARC_TASK_ID = BETA_TASK_ID + "-Arc"


@dataclass
class SnakeToBipedEnvCfg(TransformationEnvCfg):
    support_observations: bool = False
    require_sole_support_for_positive_rewards: bool = True
    initial_joint_pos: tuple[float, ...] = SNAKE_JOINT_POS
    standing_joint_pos: tuple[float, ...] = DEFAULT_JOINT_POS
    joint_standing_error_scale: float = 0.5
    standing_base_height: float = 0.264
    base_height_tolerance: float = 0.02
    sole_height_tolerance: float = 0.005
    sole_alignment_height_std: float = 0.03
    sole_alignment_angle_std_deg: float = 30.0
    sole_angle_tolerance_deg: float = 5.0
    base_z_roll_limit_deg: float = 179.0
    base_gravity_alignment_weight: float = 0.0
    survival_weight: float = 0.0
    curriculum_start_stage: int = 1
    sole_alignment_stage_threshold: float = 1.0
    curriculum_support_hold_s: float = 0.5
    curriculum_support_ready_fraction: float = 0.8
    curriculum_joint_error_threshold: float | None = None
    curriculum_prepare_pose: bool = False
    first_stage_initial_pose_weight: float = -0.02
    first_stage_joint_weight: float = 0.1
    first_stage_prepared_joint_weight: float = 0.0
    standing_stage_joint_weight: float = 1.0
    standing_stage_height_weight: float = 0.0
    standing_stage_velocity_weight: float = 0.0
    standing_stage_ramp_steps: int = 2400
    base_support_distance_weight: float = -0.2
    base_support_distance_std: float = 0.05
    support_angle_limit_deg: float = 25.0
    support_height_limit: float = 0.01
    support_failure_hold_s: float = 0.1
    feet_contact_weight: float = 0.5
    end_joint_flip_weight: float = -1.0
    end_joint_allowed_error: float = 1.0
    first_stage_roll_rate_weight: float = -0.05
    reward_hot_reload_path: str | None = None


def _joint_error(env):
    target = torch.tensor(env.cfg.standing_joint_pos, device=env.device)
    return env.scene["robot"].data.joint_pos - target


def initial_joint_pose_cost(env):
    """Mean squared displacement from this experiment's reset joint pose."""
    positions = env.scene["robot"].data.joint_pos
    initial = positions.new_tensor(env.cfg.initial_joint_pos)
    cost = (positions - initial).square().mean(-1)
    return torch.nan_to_num(cost, nan=100.0, posinf=100.0).clamp(max=100.0)


def end_joint_flip_cost(env):
    """Discourage the +/-pi end-joint shortcut without restricting middle joints."""
    if not math.isfinite(env.cfg.end_joint_allowed_error) or env.cfg.end_joint_allowed_error <= 0:
        raise ValueError("end_joint_allowed_error must be finite and positive")
    excess = (_joint_error(env)[:, (0, 5)].abs() - env.cfg.end_joint_allowed_error).clamp_min(0.0)
    return torch.nan_to_num(excess.square().mean(-1), nan=100.0, posinf=100.0)


def axial_roll_rate_cost(env):
    """Squared actual base angular speed about its longitudinal local Z axis."""
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    velocity = robot.data.body_link_ang_vel_w[:, base_id]
    local_z = torch.zeros_like(velocity)
    local_z[:, 2] = 1.0
    axis = quat_apply(robot.data.body_link_quat_w[:, base_id], local_z)
    rate = (velocity * axis).sum(-1)
    return torch.nan_to_num(rate.square(), nan=100.0, posinf=100.0)


@lru_cache(maxsize=16)
def standing_roll_reference(standing_joint_pos):
    """Base roll in the upright-root standing target, derived from native FK."""
    model = get_spec().compile()
    # MuJoCo's runtime exports are missing from the installed typing metadata.
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, position in zip(JOINT_NAMES, standing_joint_pos, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return math.atan2(rotation[2, 0], rotation[2, 1])


@lru_cache(maxsize=16)
def standing_gravity_reference(standing_joint_pos):
    """Full local gravity direction in the upright-root target, from CPU FK."""
    model = get_spec().compile()
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, position in zip(JOINT_NAMES, standing_joint_pos, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return tuple(float(-component) for component in rotation[2])


def base_gravity_alignment_cost(env):
    """Full gravity mismatch, including the component omitted by axial roll."""
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    quaternion = robot.data.body_link_quat_w[:, base_id]
    gravity = torch.zeros_like(quaternion[:, :3])
    gravity[:, 2] = -1.0
    local = quat_apply_inverse(quaternion, gravity)
    norm = local.norm(dim=-1, keepdim=True)
    unit = local / norm.clamp_min(1e-6)
    reference = local.new_tensor(
        standing_gravity_reference(tuple(env.cfg.standing_joint_pos))
    )
    reference = reference / reference.norm().clamp_min(1e-6)
    cost = 1.0 - (unit * reference).sum(-1)
    valid = torch.isfinite(quaternion).all(-1) & (quaternion.norm(dim=-1) > 1e-6)
    valid &= torch.isfinite(local).all(-1) & torch.isfinite(norm).squeeze(-1) & (norm.squeeze(-1) > 1e-6)
    return torch.where(valid, torch.nan_to_num(cost, nan=2.0, posinf=2.0, neginf=2.0).clamp(0.0, 2.0), 2.0)


def axial_roll_angle(base_quat, reference_angle=0.0):
    """Actual base roll about local Z, with local -Y down at zero roll.

    The zero-joint snake reset of ZBOT6.xml has gravity along base -Y.
    Projecting gravity into base XY removes world yaw without cancelling any
    rotation caused by joints. A vertical base Z has undefined axial roll.
    """
    gravity = torch.zeros_like(base_quat[..., :3])
    gravity[..., 2] = -1.0
    local_gravity = quat_apply_inverse(base_quat, gravity)
    angle = torch.atan2(-local_gravity[..., 0], -local_gravity[..., 1])
    angle = torch.remainder(angle - reference_angle + math.pi, 2.0 * math.pi) - math.pi
    valid = torch.isfinite(base_quat).all(-1)
    valid &= torch.isfinite(local_gravity).all(-1)
    valid &= local_gravity[..., :2].norm(dim=-1) > 1e-4
    return torch.where(valid, angle, torch.full_like(angle, math.pi))


def base_z_roll_angle(env):
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    reference = standing_roll_reference(tuple(env.cfg.standing_joint_pos))
    return axial_roll_angle(robot.data.body_link_quat_w[:, base_id], reference)


def _roll_limit(env):
    if not 0.0 < env.cfg.base_z_roll_limit_deg < 180.0:
        raise ValueError("base_z_roll_limit_deg must be between 0 and 180")
    return math.radians(env.cfg.base_z_roll_limit_deg)


def base_z_roll_cost(env):
    return (base_z_roll_angle(env) / _roll_limit(env)).square()


def base_z_roll_exceeded(env):
    return base_z_roll_angle(env).abs() > _roll_limit(env) + 1e-6


def base_z_roll_degrees(env):
    return torch.rad2deg(base_z_roll_angle(env).abs())


def reset_failure_cost(env):
    # TerminationManager runs before rewards; its union mask charges once even
    # when several safety conditions trigger. Compensate RewardManager's dt.
    return env.termination_manager.terminated.float() / env.step_dt


def survival_reward(env):
    """Constant normal-step offset; genuine failures receive no survival reward."""
    return (~env.termination_manager.terminated).float()


def joint_standing_reward(env):
    """Negative Smooth L1 pose error, with linear feedback for distant joints.

    Positive curriculum weights scale this penalty. Support loss never masks
    its negative feedback; foot support protection remains enforced separately.
    """
    scale = env.cfg.joint_standing_error_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("joint_standing_error_scale must be finite and positive")
    error = _joint_error(env).abs()
    cost = torch.where(error < scale, 0.5 * error.square() / scale, error - 0.5 * scale)
    return -torch.nan_to_num(cost, nan=100.0, posinf=100.0).mean(-1)


def base_height(env):
    """Base coordinate origin height above the task's flat ground plane."""
    data, base_id = mdp._base(env)
    return data.body_link_pos_w[:, base_id, 2] - env.scene.env_origins[:, 2]


def base_height_progress_reward(env):
    """Base origin height progress, subject to stage-two foot support protection."""
    initial_height = SNAKE_ROOT_POS[2]
    span = env.cfg.standing_base_height - initial_height
    if span <= 0:
        raise ValueError("standing_base_height must exceed the initial snake height")
    progress = (base_height(env) - initial_height) / span
    return constrain_standing_reward(env, torch.nan_to_num(progress, nan=-1.0).clamp(-1.0, 1.0))


def sole_height_cost(env):
    """Mean absolute distance of both sole sites to the task's flat ground plane."""
    if env.cfg.sole_alignment_height_std <= 0:
        raise ValueError("sole_alignment_height_std must be positive")
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    sole_height = robot.data.site_pos_w[:, ids, 2] - env.scene.env_origins[:, 2, None]
    cost = sole_height.abs() / env.cfg.sole_alignment_height_std
    return torch.nan_to_num(cost, nan=4.0, posinf=4.0).clamp(max=4.0).mean(-1)


def base_support_distance(env):
    """Horizontal distance from the base coordinate origin to the sole segment."""
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    base_id = robot.find_bodies("base")[0][0]
    base_xy = robot.data.body_link_pos_w[:, base_id, :2]
    feet = robot.data.site_pos_w[:, ids, :2]
    start = feet[:, 0]
    segment = feet[:, 1] - start
    length_sq = segment.square().sum(-1)
    fraction = ((base_xy - start) * segment).sum(-1) / length_sq.clamp_min(1e-12)
    fraction = fraction.clamp(0.0, 1.0)
    closest = start + fraction[..., None] * segment
    return (base_xy - closest).norm(dim=-1)


def base_support_distance_cost(env):
    """Normalized support-segment distance cost; use a negative reward weight."""
    if not math.isfinite(env.cfg.base_support_distance_std) or env.cfg.base_support_distance_std <= 0:
        raise ValueError("base_support_distance_std must be finite and positive")
    distance = base_support_distance(env)
    cost = distance / env.cfg.base_support_distance_std
    return torch.nan_to_num(cost, nan=4.0, posinf=4.0).clamp(max=4.0)


def _sole_direction_cosine(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    quat = robot.data.site_quat_w[:, ids]
    normal_z = 1.0 - 2.0 * (quat[..., 1].square() + quat[..., 2].square())
    # foot_0's physical outward sole normal is -site_Z; foot_1's is +site_Z.
    cosine = (normal_z * quat.new_tensor([1.0, -1.0])).clamp(-1.0, 1.0)
    return torch.where(torch.isfinite(quat).all(-1), cosine, float("nan"))


def _sole_ground_alignment(env, angle_power: float = 2.0):
    """Per-foot signed sole direction score, independent of foot height."""
    if env.cfg.sole_alignment_angle_std_deg <= 0:
        raise ValueError("sole_alignment_angle_std_deg must be positive")
    if not math.isfinite(angle_power) or angle_power <= 0:
        raise ValueError("angle_power must be finite and positive")
    cosine = _sole_direction_cosine(env)
    angle = torch.acos(cosine)
    sigma = math.radians(env.cfg.sole_alignment_angle_std_deg)
    # Power 1 retains a nonzero one-sided slope at flat; power 2 is legacy.
    score = 0.5 * (cosine - 1.0) + torch.exp(-(angle / sigma).pow(angle_power))
    return torch.nan_to_num(score, nan=0.0)


def sole_ground_alignment_reward(env, angle_power: float = 2.0):
    # Keep feedback from both feet while giving the worse foot extra influence.
    scores = _sole_ground_alignment(env, angle_power=angle_power)
    return 0.5 * (scores.mean(-1) + scores.amin(-1))


def sole_angles_degrees(env):
    """Per-foot angle from downward; invalid orientations count as 180 degrees."""
    cosine = _sole_direction_cosine(env)
    return torch.nan_to_num(torch.rad2deg(torch.acos(cosine)), nan=180.0)


def sole_angle_degrees(env, foot_index):
    return sole_angles_degrees(env)[:, foot_index]


def both_soles_within_10deg(env):
    return (sole_angles_degrees(env).amax(-1) < 10.0).float()


def sole_support_valid(env):
    """Both sole sites must face down and remain near the ground plane."""
    cfg = env.cfg
    if not math.isfinite(cfg.support_angle_limit_deg) or not 0 < cfg.support_angle_limit_deg < 90:
        raise ValueError("support_angle_limit_deg must be between 0 and 90")
    if not math.isfinite(cfg.support_height_limit) or cfg.support_height_limit <= 0:
        raise ValueError("support_height_limit must be finite and positive")
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    heights = robot.data.site_pos_w[:, ids, 2] - env.scene.env_origins[:, 2, None]
    cosine = _sole_direction_cosine(env)
    return (
        both_feet_ground_contact(env)
        & torch.isfinite(cosine).all(-1)
        & torch.isfinite(heights).all(-1)
        & (cosine.amin(-1) >= math.cos(math.radians(cfg.support_angle_limit_deg)))
        & (heights.abs().amax(-1) <= cfg.support_height_limit)
    )


def both_feet_ground_contact(env):
    """Actual foot-body contacts against terrain, independently for both feet."""
    found = env.scene["feet_ground_contact"].data.found
    if found is None:
        raise ValueError("feet_ground_contact sensor must provide found data")
    return (found > 0).all(-1)


def both_feet_contact_metric(env):
    return both_feet_ground_contact(env).float()


def constrain_standing_reward(env, reward):
    if not env.cfg.require_sole_support_for_positive_rewards:
        return reward
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    if stage == 1:
        return reward
    # Preserve every negative term when support is lost; gate only positive gain.
    return torch.where(sole_support_valid(env), reward, reward.clamp(max=0.0))


@dataclass
class SupportProtectionState:
    support_armed: torch.Tensor
    support_violation_s: torch.Tensor
    support_hold_s: torch.Tensor
    last_episode_ready: torch.Tensor


def support_protection_state(env):
    if not hasattr(env, "_support_protection_state"):
        env._support_protection_state = SupportProtectionState(
            support_armed=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            support_violation_s=torch.zeros(env.num_envs, device=env.device),
            support_hold_s=torch.zeros(env.num_envs, device=env.device),
            last_episode_ready=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
    return env._support_protection_state


def reset_reverse_transition(env, env_ids):
    mdp.reset_transformation(env, env_ids)
    task = support_protection_state(env)
    ids = slice(None) if env_ids is None else env_ids
    task.support_armed[ids] = False
    task.support_violation_s[ids] = 0.0
    task.support_hold_s[ids] = 0.0


def update_support_protection(env, fresh):
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    task = support_protection_state(env)
    valid = sole_support_valid(env)
    held = torch.where(valid, task.support_hold_s + env.step_dt, 0.0)
    task.support_hold_s.copy_(torch.where(fresh, held, task.support_hold_s))
    if stage == 1:
        return
    armed = task.support_armed | valid
    elapsed = torch.where(armed & ~valid, task.support_violation_s + env.step_dt, 0.0)
    task.support_armed.copy_(torch.where(fresh, armed, task.support_armed))
    task.support_violation_s.copy_(torch.where(fresh, elapsed, task.support_violation_s))


def foot_support_lost(env):
    if not math.isfinite(env.cfg.support_failure_hold_s) or env.cfg.support_failure_hold_s <= 0:
        raise ValueError("support_failure_hold_s must be finite and positive")
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    task = support_protection_state(env)
    return (
        (stage == 2)
        & task.support_armed
        & (task.support_violation_s >= env.cfg.support_failure_hold_s - 1e-6)
    )


def sole_support_valid_metric(env):
    return sole_support_valid(env).float()


def base_vertical_velocity(env):
    """World Z linear velocity of the base coordinate origin, not its COM."""
    data, base_id = mdp._base(env)
    return data.body_link_lin_vel_w[:, base_id, 2]


def base_ascent_velocity_reward(env, target_speed=0.03, pose_error_range: tuple[float, float] | None = None):
    if target_speed <= 0:
        raise ValueError("target_speed must be positive")
    velocity = base_vertical_velocity(env)
    # Saturate upward progress at the target speed, without an overspeed penalty.
    score = 1.0 - ((velocity / target_speed).clamp(max=1.0) - 1.0).square()
    score = torch.where(torch.isfinite(velocity), score, -4.0)
    score = torch.nan_to_num(score, nan=-4.0, posinf=-4.0, neginf=-4.0).clamp(-4.0, 1.0)
    if pose_error_range is not None:
        zero_error, full_error = pose_error_range
        if not (math.isfinite(zero_error) and math.isfinite(full_error) and 0 <= zero_error < full_error):
            raise ValueError("pose_error_range must satisfy 0 <= zero < full, both finite")
        error = torch.nan_to_num(_joint_error(env).abs(), nan=math.inf).amax(-1)
        fraction = ((error - zero_error) / (full_error - zero_error)).clamp(0.0, 1.0)
        # Smoothstep cancels both ascent gain and descent cost near the pose goal.
        score = score * fraction.square() * (3.0 - 2.0 * fraction)
    return constrain_standing_reward(env, score)


def final_condition(env, joint_tolerance=None):
    tolerance = env.cfg.joint_tolerance if joint_tolerance is None else joint_tolerance
    robot = env.scene["robot"]
    data = robot.data
    base_id = robot.find_bodies("base")[0][0]
    site_ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    soles = data.site_pos_w[:, site_ids, 2] - env.scene.env_origins[:, 2, None]
    return (
        (_joint_error(env).abs().amax(-1) < tolerance)
        & both_feet_ground_contact(env)
        & torch.isfinite(data.body_link_pos_w[:, base_id]).all(-1)
        & (soles.abs().amax(-1) < env.cfg.sole_height_tolerance)
        & (
            _sole_direction_cosine(env).amin(-1)
            > math.cos(math.radians(env.cfg.sole_angle_tolerance_deg))
        )
        & ~base_z_roll_exceeded(env)
        & (data.joint_vel.square().mean(-1).sqrt() < env.cfg.joint_speed_tolerance)
        & (
            data.body_link_lin_vel_w[:, base_id].norm(dim=-1)
            < env.cfg.base_speed_tolerance
        )
    )


def advance(env):
    if hasattr(env, "_standing_curriculum_stage"):
        apply_standing_stage_weights(env)
    task = mdp.state(env)
    fresh = task.last_step != env.common_step_counter
    update_support_protection(env, fresh)
    held = torch.where(final_condition(env), task.hold_time + env.step_dt, 0.0)
    task.hold_time.copy_(torch.where(fresh, held, task.hold_time))
    task.last_step.fill_(env.common_step_counter)


def update_state(env):
    advance(env)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def success(env):
    return (
        mdp.state(env).hold_time >= env.cfg.success_hold_s - 1e-5
    ) & final_condition(env)


def joint_standing_error_metric(env):
    return _joint_error(env).abs().mean(-1)


def apply_standing_stage_weights(env):
    """Ramp live reward weights every control step after promotion."""
    cfg = env.cfg
    if cfg.standing_stage_ramp_steps < 0:
        raise ValueError("standing_stage_ramp_steps must be nonnegative")
    if not math.isfinite(cfg.first_stage_prepared_joint_weight) or cfg.first_stage_prepared_joint_weight < 0:
        raise ValueError("first_stage_prepared_joint_weight must be finite and nonnegative")
    stage = env._standing_curriculum_stage
    scale = 0.0
    if stage == 2:
        if cfg.standing_stage_ramp_steps == 0:
            scale = 1.0
        else:
            elapsed = env.common_step_counter - env._standing_stage_start_step
            scale = min(max(elapsed / cfg.standing_stage_ramp_steps, 0.0), 1.0)
    for name, weight in (
        ("base_height_progress", cfg.standing_stage_height_weight),
        ("base_ascent_velocity", cfg.standing_stage_velocity_weight),
    ):
        env.reward_manager.get_term_cfg(name).weight = weight * scale
    starting_joint_weight = cfg.first_stage_joint_weight
    if cfg.curriculum_prepare_pose and (
        stage == 2 or getattr(env, "_standing_pose_prepared", False)
    ):
        starting_joint_weight = cfg.first_stage_prepared_joint_weight
    env.reward_manager.get_term_cfg("joint_standing").weight = (
        starting_joint_weight
        + (cfg.standing_stage_joint_weight - starting_joint_weight) * scale
    )
    env.reward_manager.get_term_cfg("base_support_distance").weight = cfg.base_support_distance_weight
    env.reward_manager.get_term_cfg("base_gravity_alignment").weight = cfg.base_gravity_alignment_weight
    env.reward_manager.get_term_cfg("survival").weight = cfg.survival_weight
    env.reward_manager.get_term_cfg("initial_joint_pose").weight = (
        cfg.first_stage_initial_pose_weight if stage == 1 else 0.0
    )
    env.reward_manager.get_term_cfg("end_joint_flip").weight = cfg.end_joint_flip_weight
    env.reward_manager.get_term_cfg("axial_roll_rate").weight = (
        cfg.first_stage_roll_rate_weight if stage == 1 else 0.0
    )
    return scale


def alignment_to_standing_curriculum(env, env_ids):
    """Promote the whole run once completed episodes exceed the alignment gate.

    Called before reset clears RewardManager's dt-integrated episode sums.
    The gate uses exactly the normalization of Episode_Reward logs, not the
    unweighted score (which cannot exceed one) or the dt-scaled step reward.
    Stage two stays active across all subsequent episode resets.
    """
    cfg = env.cfg
    if cfg.curriculum_start_stage not in (1, 2):
        raise ValueError("curriculum_start_stage must be 1 or 2")
    if not math.isfinite(cfg.curriculum_support_hold_s) or cfg.curriculum_support_hold_s <= 0:
        raise ValueError("curriculum_support_hold_s must be finite and positive")
    if not 0 < cfg.curriculum_support_ready_fraction <= 1:
        raise ValueError("curriculum_support_ready_fraction must be in (0, 1]")
    pose_threshold = cfg.curriculum_joint_error_threshold
    if pose_threshold is not None and (
        isinstance(pose_threshold, bool)
        or not math.isfinite(pose_threshold)
        or pose_threshold <= 0
    ):
        raise ValueError("curriculum_joint_error_threshold must be finite and positive or None")
    alignment_weight = env.reward_manager.get_term_cfg("sole_ground_alignment").weight
    if not 0.0 <= cfg.sole_alignment_stage_threshold < alignment_weight:
        raise ValueError("sole alignment stage threshold must be reachable with its reward weight")
    stage = getattr(env, "_standing_curriculum_stage", cfg.curriculum_start_stage)
    ids = slice(None) if env_ids is None else env_ids
    completed = env.episode_length_buf[ids] > 0
    # RewardManager exposes no public accessor for episode sums. Keep this
    # dependency local and read it before the manager's reset() clears it.
    scores = env.reward_manager._episode_sums["sole_ground_alignment"][ids]
    scores = scores[completed] / env.max_episode_length_s
    alignment = scores.mean().item() if scores.numel() else 0.0
    support = support_protection_state(env)
    held = support.support_hold_s[ids]
    failed = env.termination_manager.terminated[ids]
    ready = (held >= cfg.curriculum_support_hold_s - 1e-6) & ~failed
    if pose_threshold is not None:
        ready &= _joint_error(env)[ids].abs().amax(-1) < pose_threshold
    support.last_episode_ready[ids] = torch.where(completed, ready, support.last_episode_ready[ids])
    # Keep the latest completed outcome for every world. A tiny asynchronous
    # reset batch must not promote all worlds based on a single lucky episode.
    ready_fraction = support.last_episode_ready.float().mean().item()
    alignment_wait_steps = 0
    changed_step = getattr(env, "_standing_alignment_hot_changed_step", None)
    if changed_step is not None:
        alignment_wait_steps = max(
            0, env.max_episode_length - (env.common_step_counter - changed_step)
        )
    if cfg.curriculum_prepare_pose and (
        stage == 2 or (alignment > cfg.sole_alignment_stage_threshold and alignment_wait_steps == 0)
    ):
        env._standing_pose_prepared = True
    if (stage == 1 and alignment > cfg.sole_alignment_stage_threshold
            and ready_fraction >= cfg.curriculum_support_ready_fraction
            and alignment_wait_steps == 0):
        stage = 2
        env._standing_stage_start_step = env.common_step_counter
    elif stage == 2 and not hasattr(env, "_standing_stage_start_step"):
        # Explicit stage-two starts (including playback) use the full objective.
        env._standing_stage_start_step = env.common_step_counter - cfg.standing_stage_ramp_steps
    env._standing_curriculum_stage = stage
    ramp = apply_standing_stage_weights(env)
    return {"stage": float(stage), "alignment": alignment, "ramp": ramp,
            "support_ready_fraction": ready_fraction,
            "joint_error_threshold": -1.0 if pose_threshold is None else float(pose_threshold),
            "pose_prepared": float(getattr(env, "_standing_pose_prepared", False)),
            "alignment_hot_wait_steps": float(alignment_wait_steps)}


def snake_to_biped_env_cfg(play=False):
    base = transformation_env_cfg(play=play)
    cfg = SnakeToBipedEnvCfg(
        **{f.name: getattr(base, f.name) for f in fields(base) if f.init}
    )
    # Playback evaluates the full standing objective; training starts at stage 1.
    cfg.curriculum_start_stage = 2 if play else 1
    for group in cfg.observations.values():
        group.terms["hold_time"] = ObservationTermCfg(func=reserved_hold_observation)
        group.terms["support_state"] = ObservationTermCfg(func=support_observation)
    standing = cfg.curriculum_start_stage == 2
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
        ),
    )
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(
        robot.init_state,
        pos=SNAKE_ROOT_POS,
        rot=SNAKE_ROOT_QUAT,
        joint_pos=dict(zip(JOINT_NAMES, cfg.initial_joint_pos, strict=True)),
        joint_vel={".*": 0.0},
    )
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointTargetRateActionCfg):
        raise TypeError("reverse transformation requires rate-controlled joint targets")
    action.joint_target_bounds = {
        JOINT_NAMES[i]: (cfg.standing_joint_pos[i] - cfg.end_joint_allowed_error,
                         cfg.standing_joint_pos[i] + cfg.end_joint_allowed_error)
        for i in (0, 5)
    }
    cfg.rewards = {
        "end_joint_flip": RewardTermCfg(
            func=end_joint_flip_cost,
            weight=cfg.end_joint_flip_weight,
        ),
        "axial_roll_rate": RewardTermCfg(
            func=axial_roll_rate_cost,
            weight=0.0 if standing else cfg.first_stage_roll_rate_weight,
        ),
        "initial_joint_pose": RewardTermCfg(
            func=initial_joint_pose_cost,
            weight=0.0 if standing else cfg.first_stage_initial_pose_weight,
        ),
        "joint_standing": RewardTermCfg(
            func=joint_standing_reward,
            weight=cfg.standing_stage_joint_weight if standing else cfg.first_stage_joint_weight,
        ),
        "base_height_progress": RewardTermCfg(
            func=base_height_progress_reward,
            weight=cfg.standing_stage_height_weight if standing else 0.0,
        ),
        "base_ascent_velocity": RewardTermCfg(
            func=base_ascent_velocity_reward,
            weight=cfg.standing_stage_velocity_weight if standing else 0.0,
            params={"target_speed": 0.1},
        ),
        "base_support_distance": RewardTermCfg(
            func=base_support_distance_cost,
            weight=cfg.base_support_distance_weight,
        ),
        "joint_symmetry": RewardTermCfg(
            func=mdp.joint_symmetry_cost,
            weight=-1.0
        ),
        "sole_ground_alignment": RewardTermCfg(
            func=sole_ground_alignment_reward,
            weight=2.0,
        ),
        "both_feet_contact": RewardTermCfg(func=both_feet_contact_metric, weight=cfg.feet_contact_weight),
        "sole_height": RewardTermCfg(func=sole_height_cost, weight=-1.0),
        "base_z_roll": RewardTermCfg(func=base_z_roll_cost, weight=-1.0),
        "base_gravity_alignment": RewardTermCfg(func=base_gravity_alignment_cost, weight=cfg.base_gravity_alignment_weight),
        "reset_failure": RewardTermCfg(func=reset_failure_cost, weight=-10.0),
        "survival": RewardTermCfg(func=survival_reward, weight=cfg.survival_weight),
    }
    cfg.terminations["update_state"] = TerminationTermCfg(func=update_state)
    cfg.terminations["base_z_roll"] = TerminationTermCfg(func=base_z_roll_exceeded)
    cfg.terminations["foot_support_lost"] = TerminationTermCfg(func=foot_support_lost)
    cfg.events["reset_transition"].func = reset_reverse_transition
    cfg.curriculum = {
        "alignment_to_standing": CurriculumTermCfg(func=alignment_to_standing_curriculum)
    }
    cfg.metrics["joint_error"] = MetricsTermCfg(func=joint_standing_error_metric)
    cfg.metrics["base_height"] = MetricsTermCfg(func=base_height)
    cfg.metrics["base_vertical_velocity"] = MetricsTermCfg(func=base_vertical_velocity)
    cfg.metrics["base_support_distance"] = MetricsTermCfg(func=base_support_distance)
    cfg.metrics["sole_support_valid"] = MetricsTermCfg(func=sole_support_valid_metric)
    cfg.metrics["both_feet_contact"] = MetricsTermCfg(func=both_feet_contact_metric)
    cfg.metrics["success"] = MetricsTermCfg(func=success, reduce="last")
    cfg.metrics["base_z_roll_deg"] = MetricsTermCfg(
        func=base_z_roll_degrees, reduce="max"
    )
    for foot in (0, 1):
        cfg.metrics[f"sole_{foot}_angle_deg"] = MetricsTermCfg(
            func=sole_angle_degrees, params={"foot_index": foot}, reduce="mean"
        )
    cfg.metrics["both_soles_within_10deg"] = MetricsTermCfg(
        func=both_soles_within_10deg, reduce="mean"
    )
    return cfg
