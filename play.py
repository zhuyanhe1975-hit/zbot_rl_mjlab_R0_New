"""Direct MJLab-style CLI for Zbot checkpoint playback."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.env_cfg import ZbotEnvCfg
from zbot_rl_mjlab.experiments.common import latest_checkpoint
from zbot_rl_mjlab.viewer import FollowCameraViewer


@dataclass
class PlayConfig:
    env: ZbotEnvCfg
    checkpoint_file: str | None = None
    frequency: float | None = None
    target_speed: float | None = None
    log_root: str = "logs/rsl_rl"
    device: str = "cuda:0"
    viewer: str = "native"


def main():
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [t for t in list_tasks() if t.startswith("Mjlab-Zbot")]
        ),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayConfig,
        args=remaining,
        default=PlayConfig(env=load_env_cfg(task, play=True)),
        config=mjlab.TYRO_FLAGS,
    )
    cfg.env.scene.env_spacing = 1.0
    cfg.env.viewer.enable_shadows = True
    cfg.env.viewer.enable_reflections = True
    if cfg.frequency is not None:
        if cfg.frequency <= 0:
            raise ValueError("frequency must be positive")
        cfg.env.test_frequency = cfg.frequency
    if cfg.target_speed is not None:
        if cfg.target_speed < 0:
            raise ValueError("target-speed must be non-negative")
        command = cfg.env.commands.get("target_speed")
        if command is None:
            raise ValueError("target-speed requires the velocity-tracking task")
        command.ranges.lin_vel_x = (cfg.target_speed, cfg.target_speed)
    if not 0 < cfg.env.step_frequency_min <= cfg.env.step_frequency_max:
        raise ValueError("Require 0 < step-frequency-min <= step-frequency-max")
    agent = load_rl_cfg(task)
    checkpoint = (
        Path(cfg.checkpoint_file)
        if cfg.checkpoint_file
        else latest_checkpoint(agent.experiment_name, cfg.log_root)
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Frequency: {cfg.env.step_frequency_min}–{cfg.env.step_frequency_max} Hz")
    if cfg.frequency is not None:
        print(f"Fixed frequency: {cfg.frequency} Hz")
    if cfg.target_speed is not None:
        print(f"Fixed target speed: {cfg.target_speed} m/s")
    # Keep frequency encoding consistent with training even for fixed tests.
    if cfg.env.step_frequency_min == cfg.env.step_frequency_max:
        cfg.env.test_frequency = cfg.env.step_frequency_min
        training_env = load_env_cfg(task)
        cfg.env.step_frequency_min = training_env.step_frequency_min
        cfg.env.step_frequency_max = training_env.step_frequency_max
    if cfg.viewer not in ("native", "viser"):
        raise ValueError("viewer must be native or viser")
    env = ManagerBasedRlEnv(cfg.env, device=cfg.device)
    try:
        # MuJoCo's default shadow clip is only 1 m.  Extend it so shadows do
        # not disappear when the robot walks away from the initial origin.
        model = getattr(env.sim, "_mj_model", None)
        if model is not None:
            model.vis.map.shadowclip = 10.0
            model.vis.map.shadowscale = 0.9
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=cfg.device,
        )
        policy = runner.get_inference_policy(device=cfg.device)
        viewer = FollowCameraViewer if cfg.viewer == "native" else ViserPlayViewer
        viewer(wrapped, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
