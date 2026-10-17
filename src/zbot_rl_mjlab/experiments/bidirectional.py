"""Independent snake/biped bidirectional command experiment."""

from dataclasses import fields

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from ..tasks.transformation import commanded
from .transformation import SnakeToBipedTeacher


class SnakeBipedBidirectional(SnakeToBipedTeacher):
    """One command-conditioned teacher for both directions and in-place switching."""

    task_id = commanded.TASK_ID
    experiment_name = "zbot_transform_snake_biped_bidirectional"

    def register(self):
        register_mjlab_task(
            task_id=self.task_id, env_cfg=self.build_env(),
            play_env_cfg=self.build_env(play=True), rl_cfg=self.build_ppo(),
            runner_cls=commanded.robust.RobustSnakeToBipedRunner,
        )

    def build_env(self, play=False):
        base = super().build_env(play=play)
        cfg = commanded.BidirectionalEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        cfg.pretrained_checkpoint = None
        cfg.initial_roll_range_deg = (-180.0, 180.0)
        cfg.early_contact_drop_threshold = 0.01
        cfg.fall_retry_arm_height = 0.05
        cfg.episode_length_s = 16.0
        cfg.commands = {"transform": commanded.TransformCommandCfg(resampling_time_range=(1e9, 1e9))}
        cfg.observations = {
            "actor": ObservationGroupCfg(terms={"command_teacher_state": ObservationTermCfg(func=commanded.actor_state)}, enable_corruption=False),
            "critic": ObservationGroupCfg(terms={"command_teacher_critic_state": ObservationTermCfg(func=commanded.critic_state)}, enable_corruption=False),
        }
        cfg.events["reset_transition"].func = commanded.reset_command_transition
        for name, func in (("joint_standing", commanded.pose_reward), ("base_height_progress", commanded.height_reward), ("base_ascent_velocity", commanded.velocity_reward), ("sole_ground_alignment", commanded.sole_reward)):
            cfg.rewards[name].func = func
        for name, func in (("update_state", commanded.update_state), ("fallen_after_standing", commanded.fallen), ("early_base_ground_contact", commanded.early_contact), ("abandoned_exploration", commanded.abandoned_exploration)):
            cfg.terminations[name].func = func
        if play:
            cfg.terminations["time_out"].func = commanded.play_time_out
            for name in ("fallen_after_standing", "early_base_ground_contact", "abandoned_exploration"):
                term = cfg.terminations[name]
                term.params = {"condition": term.func}
                term.func = commanded.play_failure
        cfg.metrics["joint_error"].func = commanded.joint_error
        cfg.metrics["success"].func = commanded.success
        cfg.metrics["transform_command"] = MetricsTermCfg(func=commanded.command_metric, reduce="last")
        cfg.metrics["stand_env_fraction"] = MetricsTermCfg(func=commanded.stand_env_fraction)
        return cfg
