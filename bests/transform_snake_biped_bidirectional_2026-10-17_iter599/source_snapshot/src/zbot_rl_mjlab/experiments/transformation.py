"""Registry adapter for an independent transformation environment."""

import math
from dataclasses import fields, replace

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from ..robot import JOINT_NAMES
from ..tasks.transformation.actions import JointTargetRateActionCfg
from ..tasks.transformation.arc_pose import ARC_JOINT_POS, ARC_ROOT_POS, ARC_ROOT_QUAT
from ..tasks.transformation.biped_to_snake import TASK_ID, transformation_env_cfg
from ..tasks.transformation.mirror_actions import (
    MirroredJointTargetRateActionCfg,
    mirrored_last_action,
)
from ..tasks.transformation.robust import (
    ROBUST_TASK_ID,
    RampedBaseImpulse,
    RobustSnakeToBipedEnvCfg,
    RobustSnakeToBipedRunner,
    abandoned_exploration,
    early_base_ground_contact,
    fallen_after_standing,
    initial_ascent_velocity_reward,
    reset_fall_retry,
    standing_success_metric,
    update_fixed_standing_state,
)
from ..tasks.transformation.snake_to_biped import (
    ARC_TASK_ID,
    BETA_TASK_ID,
    snake_to_biped_env_cfg,
)
from ..tasks.transformation.snake_to_biped import (
    TASK_ID as REVERSE_TASK_ID,
)
from ..tasks.transformation.teacher_observations import (
    TEACHER_TASK_ID,
    joint_speed_peak,
    teacher_observation_groups,
)
from .base import Experiment
from .common import add_base_disturbance


class BipedToSnake(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_transform_biped_to_snake"
    supports_walking_addons = False

    def build_env(self, play=False):
        cfg = transformation_env_cfg(play=play)
        if self.train_envs is not None and not play:
            cfg.scene.num_envs = self.train_envs
        return cfg

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
            "std_range": (0.1, 1.0),
        }
        cfg.algorithm.entropy_coef = 0.005
        return cfg


class SnakeToBiped(BipedToSnake):
    task_id = REVERSE_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped"

    def register(self):
        from ..tasks.transformation.hot_rewards import SnakeToBipedHotRewardRunner

        register_mjlab_task(
            task_id=self.task_id,
            env_cfg=self.build_env(),
            play_env_cfg=self.build_env(play=True),
            rl_cfg=self.build_ppo(),
            runner_cls=SnakeToBipedHotRewardRunner,
        )

    def build_env(self, play=False):
        cfg = snake_to_biped_env_cfg(play=play)
        if self.train_envs is not None and not play:
            cfg.scene.num_envs = self.train_envs
        return cfg


class SnakeToBipedBeta(SnakeToBiped):
    """Explicit bounded-Beta candidate; original Gaussian task stays available."""

    task_id = BETA_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_beta"
    train_envs = 8192
    max_iterations = 4000

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "BetaDistribution",
            "action_range": (-1.0, 1.0),
        }
        return cfg

    def build_env(self, play=False):
        cfg = super().build_env(play=play)
        cfg.curriculum_prepare_pose = True
        cfg.first_stage_joint_weight = 0.6
        cfg.first_stage_prepared_joint_weight = 3.0
        cfg.standing_stage_joint_weight = 6.0
        cfg.curriculum_joint_error_threshold = 1.0
        cfg.rewards["joint_symmetry"].weight = -0.2
        cfg.rewards["joint_standing"].weight = 6.0 if play else 0.6
        return cfg


class SnakeToBipedArc(SnakeToBipedBeta):
    """Explore from a low arc without commanding the opposite angle branch."""

    task_id = ARC_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_arc"

    def build_env(self, play=False):
        cfg = super().build_env(play=play)
        cfg.initial_joint_pos = ARC_JOINT_POS
        robot = cfg.scene.entities["robot"]
        robot.init_state = replace(
            robot.init_state,
            pos=ARC_ROOT_POS,
            rot=ARC_ROOT_QUAT,
            joint_pos=dict(zip(JOINT_NAMES, ARC_JOINT_POS, strict=True)),
        )
        action = cfg.actions["joint_pos"]
        if not isinstance(action, JointTargetRateActionCfg):
            raise TypeError("arc transformation requires rate-controlled joint targets")
        bounds = action.joint_target_bounds
        for name, target in zip(JOINT_NAMES, cfg.standing_joint_pos, strict=True):
            low, high = bounds.get(name, (-math.pi, math.pi))
            bounds[name] = (max(0.0, low), min(math.pi, high)) if target > 0 else (max(-math.pi, low), min(0.0, high))
        return cfg


class SnakeToBipedRobust(SnakeToBipedBeta):
    """Train mirrored snake-to-biped from scratch under randomized wrenches."""

    task_id = ROBUST_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_robust"
    max_iterations = 2000

    def register(self):
        register_mjlab_task(
            task_id=self.task_id,
            env_cfg=self.build_env(),
            play_env_cfg=self.build_env(play=True),
            rl_cfg=self.build_ppo(),
            runner_cls=RobustSnakeToBipedRunner,
        )

    def build_env(self, play=False):
        base = super().build_env(play=play)
        cfg = RobustSnakeToBipedEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        # A confirmed fall retries from the randomized-roll snake reset pose.
        cfg.terminations = {
            name: cfg.terminations[name] for name in ("update_state", "time_out")
        }
        cfg.terminations["update_state"] = TerminationTermCfg(func=update_fixed_standing_state)
        cfg.terminations["fallen_after_standing"] = TerminationTermCfg(func=fallen_after_standing)
        cfg.terminations["early_base_ground_contact"] = TerminationTermCfg(func=early_base_ground_contact)
        cfg.terminations["abandoned_exploration"] = TerminationTermCfg(func=abandoned_exploration)
        cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
            name="base_ground_contact",
            primary=ContactMatch(mode="body", pattern="base", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce",
        ))
        action = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = MirroredJointTargetRateActionCfg(**{f.name: getattr(action, f.name) for f in fields(action) if f.init})
        cfg.actions["joint_pos"].max_target_speed = 2 * math.pi
        for group in cfg.observations.values():
            group.terms["last_action"].func = mirrored_last_action
        cfg.events["reset_transition"].func = reset_fall_retry
        cfg.curriculum_start_stage = 2
        cfg.curriculum_prepare_pose = False
        cfg.require_sole_support_for_positive_rewards = False
        cfg.curriculum = {}
        cfg.standing_stage_ramp_steps = 0
        cfg.standing_stage_height_weight = 1.0
        cfg.standing_stage_velocity_weight = 8.0
        cfg.feet_contact_weight = 0.0
        cfg.base_support_distance_weight = 0.0
        cfg.first_stage_joint_weight = 3.0
        cfg.standing_stage_joint_weight = 12.0
        cfg.rewards["joint_standing"].weight = 12.0
        cfg.rewards["initial_joint_pose"].weight = 0.0
        cfg.rewards["axial_roll_rate"].weight = 0.0
        cfg.rewards["base_height_progress"].weight = 1.0
        cfg.rewards["base_ascent_velocity"].weight = 8.0
        cfg.rewards["base_ascent_velocity"].func = initial_ascent_velocity_reward
        cfg.rewards["base_ascent_velocity"].params.update(target_speed=0.15, pose_error_range=(0.2, 0.8))
        cfg.rewards["sole_ground_alignment"].weight = 1.0
        cfg.rewards["sole_ground_alignment"].params.update(angle_power=1.0)
        cfg.sole_alignment_angle_std_deg = 10.0
        for name in ("sole_height", "both_feet_contact", "base_support_distance"):
            cfg.rewards[name].weight = 0.0
        add_base_disturbance(
            cfg,
            force_range=(0.0, 0.0),
            torque_range=(0.0, 0.0),
            duration_s=(0.08, 0.20),
            cooldown_s=(0.8, 2.0),
            disturbed_fraction=0.0,
        )
        cfg.events["base_disturbance"].func = RampedBaseImpulse
        cfg.rewards = {name: cfg.rewards[name] for name in ("joint_standing", "base_height_progress", "base_ascent_velocity", "sole_ground_alignment", "reset_failure")}
        cfg.metrics = {
            "joint_error": cfg.metrics["joint_error"],
            "base_height": cfg.metrics["base_height"],
            "base_vertical_velocity": cfg.metrics["base_vertical_velocity"],
            "success": MetricsTermCfg(func=standing_success_metric, reduce="last"),
            "joint_speed_peak": MetricsTermCfg(func=joint_speed_peak, reduce="max"),
        }
        cfg.scene.sensors = tuple(replace(sensor, fields=("found",)) if isinstance(sensor, ContactSensorCfg) else sensor for sensor in cfg.scene.sensors)
        if play:
            cfg.seed = -1  # Randomize across launches; --env.seed can still reproduce a run.
            cfg.disturbance_initial_scale = 1.0
        return cfg

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.resume = False
        cfg.algorithm.learning_rate = 1e-4
        cfg.algorithm.entropy_coef = 0.001
        return cfg


class SnakeToBipedTeacher(SnakeToBipedRobust):
    """Full-state teacher, trained from scratch before a future sensor-only student."""

    task_id = TEACHER_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_teacher"

    def build_env(self, play=False):
        cfg = super().build_env(play=play)
        cfg.observations = teacher_observation_groups()
        action = cfg.actions["joint_pos"]
        if not isinstance(action, MirroredJointTargetRateActionCfg):
            raise TypeError("teacher requires mirrored target-rate actions")
        action.max_target_speed = 2 * math.pi
        return cfg

    def build_ppo(self):
        cfg = super().build_ppo()
        # Physical unit scaling is explicit; avoid amplifying nearly constant
        # binary contacts/timer features with a second running normalization.
        cfg.actor.obs_normalization = False
        cfg.critic.obs_normalization = False
        cfg.algorithm.learning_rate = 3e-4
        cfg.algorithm.entropy_coef = 0.005
        return cfg
