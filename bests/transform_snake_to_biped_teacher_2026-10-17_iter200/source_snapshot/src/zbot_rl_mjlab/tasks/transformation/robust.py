"""Checkpoint-compatible force/torque fine-tuning of snake-to-biped."""

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from mjlab.envs import mdp as common_mdp

from ...mdp import PartialBaseImpulse
from . import mdp as transformation_mdp
from .hot_rewards import SnakeToBipedHotRewardRunner
from .mdp import commanded_target
from .mdp import final_condition as snake_final_condition
from .mirror_actions import (
    MirroredJointTargetRateAction,
    MirroredJointTargetRateActionCfg,
    migrate_beta_mirror_head,
)
from .roll_reset import randomize_initial_snake_roll
from .snake_to_biped import (
    BETA_TASK_ID,
    SnakeToBipedEnvCfg,
    base_ascent_velocity_reward,
    base_height,
    base_vertical_velocity,
    reset_reverse_transition,
)
from .snake_to_biped import (
    final_condition as standing_final_condition,
)
from .snake_to_biped import (
    update_state as update_standing_state,
)

ROBUST_TASK_ID = BETA_TASK_ID + "-Robust"
PRETRAINED_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "bests/transform_snake_to_biped_beta_2026-10-17_iter2550/model_2550.pt"
)


@dataclass
class RobustSnakeToBipedEnvCfg(SnakeToBipedEnvCfg):
    compact_standing_rewards: ClassVar[bool] = True
    pretrained_checkpoint: str | None = None
    disturbance_initial_scale: float = 1.0
    disturbance_warmup_steps: int = 0
    disturbance_ramp_steps: int = 0
    ascent_reward_stop_height: float = 0.23
    fall_retry_arm_height: float = 0.18
    fall_retry_height: float = 0.10
    fall_retry_hold_s: float = 0.10
    recovery_target_speed: float = 0.5
    recovery_snake_hold_s: float = 0.5
    recovery_target_tolerance: float = 0.02
    initial_roll_range_deg: tuple[float, float] = (-90.0, 90.0)
    initial_roll_clearance: float = 0.0001
    early_contact_motion_threshold: float = 0.01
    early_contact_drop_threshold: float = 0.001
    early_contact_min_down_speed: float = 0.03
    stagnation_height_tolerance: float = 0.03
    stagnation_joint_tolerance: float = 0.25
    stagnation_joint_speed: float = 0.03
    stagnation_base_speed: float = 0.01
    stagnation_angular_speed: float = 0.05
    stagnation_hold_s: float = 1.0
    stagnation_grace_s: float = 0.3


class FallRetryState:
    def __init__(self, env):
        self.armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.low_time = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.recovering = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.snake_time = torch.zeros(env.num_envs, device=env.device)
        self.motion_started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.had_base_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.base_left_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.early_base_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.peak_base_height = torch.full((env.num_envs,), -math.inf, device=env.device)
        self.contact_last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.still_time = torch.zeros(env.num_envs, device=env.device)
        self.still_last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)


def fall_retry_state(env):
    if not hasattr(env, "_fall_retry_state"):
        env._fall_retry_state = FallRetryState(env)
    return env._fall_retry_state


def fallen_after_standing(env):
    """Latch a risen base before detecting a sustained low base, without foot gates."""
    cfg = env.cfg
    if not (math.isfinite(cfg.fall_retry_arm_height) and math.isfinite(cfg.fall_retry_height)
            and 0 < cfg.fall_retry_height < cfg.fall_retry_arm_height):
        raise ValueError("fall retry heights must be finite and satisfy 0 < fall < arm")
    if not math.isfinite(cfg.fall_retry_hold_s) or cfg.fall_retry_hold_s <= 0:
        raise ValueError("fall_retry_hold_s must be finite and positive")
    state = fall_retry_state(env)
    fresh = state.last_step != env.common_step_counter
    height = base_height(env)
    armed = state.armed | (~state.recovering & (height >= cfg.fall_retry_arm_height))
    low_time = torch.where(armed & (height < cfg.fall_retry_height), state.low_time + env.step_dt, 0.0)
    state.armed.copy_(torch.where(fresh, armed, state.armed))
    state.low_time.copy_(torch.where(fresh, low_time, state.low_time))
    state.last_step.fill_(env.common_step_counter)
    return ~state.recovering & state.armed & (state.low_time >= cfg.fall_retry_hold_s - 1e-6)


def base_ground_contact(env):
    found = env.scene["base_ground_contact"].data.found
    return found.reshape(env.num_envs, -1).bool().any(-1)


def early_base_ground_contact(env):
    """Reset a small early drop/contact without requiring a prior high stand.

    Initial settling is not a fall. After commanded motion starts, an observed
    contact -> separation -> contact sequence is a fall at any lift height.
    A small downward impact is also detected when soft contact never separates.
    """
    cfg = env.cfg
    for name in ("early_contact_motion_threshold", "early_contact_drop_threshold", "early_contact_min_down_speed"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    state = fall_retry_state(env)
    fresh = state.contact_last_step != env.common_step_counter
    contact = base_ground_contact(env)
    height = base_height(env)
    motion = state.motion_started | (commanded_target(env).abs().amax(-1) >= cfg.early_contact_motion_threshold)
    separated = state.base_left_ground | (motion & state.had_base_contact & ~contact)
    peak = torch.maximum(state.peak_base_height, height)
    recontact = motion & separated & contact
    impact = (
        motion & contact
        & (peak - height >= cfg.early_contact_drop_threshold)
        & (base_vertical_velocity(env) <= -cfg.early_contact_min_down_speed)
    )
    fallen = state.early_base_fallen | ((recontact | impact) & ~state.recovering)
    for current, updated in (
        (state.motion_started, motion), (state.had_base_contact, state.had_base_contact | contact),
        (state.base_left_ground, separated), (state.peak_base_height, peak),
        (state.early_base_fallen, fallen),
    ):
        current.copy_(torch.where(fresh, updated, current))
    state.contact_last_step.fill_(env.common_step_counter)
    return state.early_base_fallen


def abandoned_exploration(env):
    """End a quiet/stuck attempt unless both height and joint pose are near goal."""
    cfg = env.cfg
    for name in ("stagnation_height_tolerance", "stagnation_joint_tolerance", "stagnation_joint_speed", "stagnation_base_speed", "stagnation_angular_speed", "stagnation_hold_s"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.stagnation_grace_s) or cfg.stagnation_grace_s < 0:
        raise ValueError("stagnation_grace_s must be finite and nonnegative")
    robot = env.scene["robot"]
    data = robot.data
    base_id = robot.find_bodies("base")[0][0]
    target = data.joint_pos.new_tensor(cfg.standing_joint_pos)
    near_goal = ((base_height(env) - cfg.standing_base_height).abs() < cfg.stagnation_height_tolerance)
    near_goal &= (data.joint_pos - target).abs().amax(-1) < cfg.stagnation_joint_tolerance
    quiet = data.joint_vel.square().mean(-1).sqrt() < cfg.stagnation_joint_speed
    quiet &= data.body_link_lin_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_base_speed
    quiet &= data.body_link_ang_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_angular_speed
    state = fall_retry_state(env)
    eligible = env.episode_length_buf * env.step_dt >= cfg.stagnation_grace_s
    stopped = eligible & ~near_goal & quiet & ~state.recovering
    fresh = state.still_last_step != env.common_step_counter
    elapsed = torch.where(stopped, state.still_time + env.step_dt, 0.0)
    state.still_time.copy_(torch.where(fresh, elapsed, state.still_time))
    state.still_last_step.fill_(env.common_step_counter)
    return state.still_time > cfg.stagnation_hold_s + 1e-6


def snake_recovery_ready(env):
    if not math.isfinite(env.cfg.recovery_target_tolerance) or env.cfg.recovery_target_tolerance <= 0:
        raise ValueError("recovery_target_tolerance must be finite and positive")
    data = env.scene["robot"].data
    return (
        snake_final_condition(env)
        & (commanded_target(env).abs().amax(-1) < env.cfg.recovery_target_tolerance)
        & (data.body_link_ang_vel_w.norm(dim=-1).amax(-1) < env.cfg.joint_speed_tolerance)
    )


def update_recovery_phase(env):
    state = fall_retry_state(env)
    fresh = state.last_step != env.common_step_counter
    fallen = fallen_after_standing(env) & fresh
    state.recovering |= fallen
    state.armed[fallen] = False
    state.low_time[fallen] = 0
    if not math.isfinite(env.cfg.recovery_snake_hold_s) or env.cfg.recovery_snake_hold_s <= 0:
        raise ValueError("recovery_snake_hold_s must be finite and positive")
    ready = state.recovering & snake_recovery_ready(env)
    held = torch.where(ready, state.snake_time + env.step_dt, 0.0)
    state.snake_time.copy_(torch.where(fresh, held, state.snake_time))
    complete = fresh & state.recovering & (state.snake_time >= env.cfg.recovery_snake_hold_s - 1e-6)
    state.recovering[complete] = False
    state.snake_time[complete] = 0
    ids = complete.nonzero(as_tuple=False).flatten()
    if ids.numel():
        # Logical counters only: this helper never writes robot/root state.
        reset_reverse_transition(env, ids)
        env.episode_length_buf[ids] = 0


def update_robust_state(env):
    update_standing_state(env)
    update_recovery_phase(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def recovery_time_out(env):
    return common_mdp.time_out(env) & ~fall_retry_state(env).recovering


def recovery_active_metric(env):
    return fall_retry_state(env).recovering.float()


@dataclass(kw_only=True)
class RecoveryJointTargetRateActionCfg(MirroredJointTargetRateActionCfg):
    def build(self, env):
        return RecoveryJointTargetRateAction(self, env)


class RecoveryJointTargetRateAction(MirroredJointTargetRateAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        speed = env.cfg.recovery_target_speed
        if not math.isfinite(speed) or not 0 < speed <= cfg.max_target_speed:
            raise ValueError("recovery_target_speed must be positive and within max_target_speed")
        self._recovery_speed = speed

    def process_actions(self, actions):
        previous = self.commanded_target.clone()
        super().process_actions(actions)
        step = self._recovery_speed * self._env.step_dt
        target = previous + (-previous).clamp(-step, step)
        target = self._clamp_target(target)
        recovering = fall_retry_state(self._env).recovering
        self._processed_actions.copy_(torch.where(recovering[:, None], target, self._processed_actions))


def reset_fall_retry(env, env_ids):
    randomize_initial_snake_roll(env, env_ids)
    if env.cfg.compact_standing_rewards:
        transformation_mdp.reset_transformation(env, env_ids)
    else:
        reset_reverse_transition(env, env_ids)
    apply_fixed_standing_weights(env)
    state = fall_retry_state(env)
    ids = slice(None) if env_ids is None else env_ids
    state.armed[ids] = False
    state.low_time[ids] = 0
    state.last_step[ids] = env.common_step_counter
    state.recovering[ids] = False
    state.snake_time[ids] = 0
    state.motion_started[ids] = False
    state.had_base_contact[ids] = False
    state.base_left_ground[ids] = False
    state.early_base_fallen[ids] = False
    state.peak_base_height[ids] = -math.inf
    state.contact_last_step[ids] = env.common_step_counter
    state.still_time[ids] = 0
    state.still_last_step[ids] = env.common_step_counter
    if hasattr(env, "_standing_goal_valid"):
        env._standing_goal_valid[ids] = False


def disturbance_scale(env):
    cfg = env.cfg
    if not math.isfinite(cfg.disturbance_initial_scale) or not 0 <= cfg.disturbance_initial_scale <= 1:
        raise ValueError("disturbance_initial_scale must be finite and in [0, 1]")
    if cfg.disturbance_warmup_steps < 0 or cfg.disturbance_ramp_steps < 0:
        raise ValueError("disturbance warmup/ramp steps must be nonnegative")
    elapsed = env.common_step_counter - cfg.disturbance_warmup_steps
    progress = float(elapsed >= 0) if cfg.disturbance_ramp_steps == 0 else min(1.0, max(0.0, elapsed / cfg.disturbance_ramp_steps))
    return cfg.disturbance_initial_scale + (1 - cfg.disturbance_initial_scale) * progress


def disturbance_scale_metric(env):
    return torch.full((env.num_envs,), disturbance_scale(env), device=env.device)


def initial_ascent_velocity_reward(env, target_speed=0.15, pose_error_range=None):
    """Pay for the first rise only; peak-height memory prevents reactivation."""
    threshold = env.cfg.ascent_reward_stop_height
    if not math.isfinite(threshold) or not 0 < threshold <= env.cfg.standing_base_height:
        raise ValueError("ascent_reward_stop_height must be positive and no higher than the standing goal")
    state = fall_retry_state(env)
    state.peak_base_height.copy_(torch.maximum(state.peak_base_height, base_height(env)))
    score = base_ascent_velocity_reward(env, target_speed=target_speed, pose_error_range=pose_error_range)
    return torch.where(state.peak_base_height >= threshold, 0.0, score)


def apply_fixed_standing_weights(env):
    """Only the three core dense terms; failure weight remains independently editable."""
    for name, field in (
        ("joint_standing", "standing_stage_joint_weight"),
        ("base_height_progress", "standing_stage_height_weight"),
        ("base_ascent_velocity", "standing_stage_velocity_weight"),
    ):
        env.reward_manager.get_term_cfg(name).weight = getattr(env.cfg, field)


def update_fixed_standing_state(env):
    task = transformation_mdp.state(env)
    fresh = task.last_step != env.common_step_counter
    valid = standing_final_condition(env)
    held = torch.where(valid, task.hold_time + env.step_dt, 0.0)
    task.hold_time.copy_(torch.where(fresh, held, task.hold_time))
    task.last_step.fill_(env.common_step_counter)
    env._standing_goal_valid = valid
    env._standing_goal_step = env.common_step_counter
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def standing_success_metric(env):
    valid = getattr(env, "_standing_goal_valid", None)
    if valid is None or getattr(env, "_standing_goal_step", -1) != env.common_step_counter:
        valid = standing_final_condition(env)
    return (transformation_mdp.state(env).hold_time >= env.cfg.success_hold_s - 1e-5) & valid


class RampedBaseImpulse(PartialBaseImpulse):
    """Reuse the impulse engine with scaled amplitudes and clean episode resets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._disabled = cfg.params.get("disturbed_fraction", 0.75) == 0 or (
            not any(cfg.params["force_range"]) and not any(cfg.params["torque_range"])
        )
        if self._disabled:
            self._disturbed_mask.zero_()

    @property
    def disturbed_mask(self):
        return self._disturbed_mask

    def __call__(self, env, env_ids, force_range, torque_range, **kwargs):
        if self._disabled:
            return
        scale = disturbance_scale(env)
        super().__call__(
            env, env_ids,
            force_range=tuple(value * scale for value in force_range),
            torque_range=tuple(value * scale for value in torque_range),
            **kwargs,
        )
        recovering = fall_retry_state(env).recovering.nonzero(as_tuple=False).flatten()
        if recovering.numel():
            self._active[recovering] = False
            self._time_remaining[recovering] = 0
            self._interval_time_left[recovering] = self._sample_cooldown(len(recovering))
            zeros = torch.zeros((len(recovering), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=recovering, body_ids=self._body_ids)

    def reset(self, env_ids=None):
        ids = torch.arange(self._num_envs, device=self._device)
        if env_ids is not None:
            ids = ids[env_ids]
        self._active[ids] = False
        self._time_remaining[ids] = 0
        self._interval_time_left[ids] = self._sample_cooldown(len(ids))
        zeros = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
        self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self._body_ids)


class RobustSnakeToBipedRunner(SnakeToBipedHotRewardRunner):
    """Warm-start a new run; regular --agent.resume uses full checkpoint restore."""

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        full_load = load_cfg is None or all(load_cfg.get(key, False) for key in ("actor", "critic", "optimizer", "iteration"))
        env = self.env.unwrapped
        if not isinstance(env.cfg, RobustSnakeToBipedEnvCfg):
            raise TypeError("robust runner requires RobustSnakeToBipedEnvCfg")
        # A resumed older run must not restore the superseded foot-first recipe.
        recipe_config = {name: getattr(env.cfg, name) for name in (
            "standing_stage_joint_weight", "standing_stage_height_weight", "standing_stage_velocity_weight",
        )}
        recipe_rewards = {name: env.reward_manager.get_term_cfg(name).weight for name in ("reset_failure", "sole_ground_alignment")}
        payload = torch.load(path, map_location=map_location or self.device, weights_only=False) if Path(path).is_file() else None
        migrated = False
        if payload is not None:
            actor, migrated = migrate_beta_mirror_head(payload["actor_state_dict"])
            if migrated:
                payload["actor_state_dict"] = actor
        if migrated:
            # Optimizer tensors and iteration schedules from a six-action run are
            # incompatible with this new policy. Transfer weights as a fresh run.
            effective_load = {"actor": True, "critic": full_load or bool(load_cfg and load_cfg.get("critic")), "optimizer": False, "iteration": False}
            with tempfile.TemporaryDirectory(prefix="zbot-mirror-") as temporary:
                converted = Path(temporary) / "checkpoint.pt"
                torch.save(payload, converted)
                infos = super().load(str(converted), effective_load, strict, map_location)
            self._mirror_migration = {"source": str(Path(path).resolve()), "old_action_dim": 6, "new_action_dim": 3, "optimizer_and_iteration_restored": False}
            print("[INFO] Migrated six-action Beta head to three mirrored actions; optimizer and iteration remain fresh.")
        else:
            infos = super().load(path, load_cfg, strict, map_location)
        if full_load:
            for name, value in recipe_config.items():
                setattr(env.cfg, name, value)
            for name, weight in recipe_rewards.items():
                env.reward_manager.get_term_cfg(name).weight = weight
            apply_fixed_standing_weights(env)
        return infos

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        cfg = env.unwrapped.cfg
        if not isinstance(cfg, RobustSnakeToBipedEnvCfg):
            raise TypeError("robust runner requires RobustSnakeToBipedEnvCfg")
        source = cfg.pretrained_checkpoint
        if log_dir is None or train_cfg.get("resume", False) or source is None:
            return
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained snake-to-biped checkpoint not found: {path}")
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False}
        self.load(str(path), load_cfg=load_cfg, strict=True, map_location=device)
        record = {
            "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "load_cfg": load_cfg,
            "initial_iteration": self.current_learning_iteration,
            "initial_common_step_counter": env.unwrapped.common_step_counter,
            "learning_rate": self.alg.learning_rate,
            "mirror_migration": getattr(self, "_mirror_migration", None),
        }
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "pretrained_initialization.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[INFO] Warm-started actor/critic from {path}; optimizer and training counters are fresh.")
