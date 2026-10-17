"""Direct MJLab-style CLI for Zbot checkpoint playback."""

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViserPlayViewer

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.cli import CliError, apply_cli
from zbot_rl_mjlab.env_cfg import ZbotEnvCfg
from zbot_rl_mjlab.experiments.common import latest_checkpoint
from zbot_rl_mjlab.viewer import FollowCameraViewer


@dataclass
class PlayConfig:
    env: ManagerBasedRlEnvCfg
    checkpoint_file: str | None = None
    frequency: float | None = None
    target_speed: float | None = None
    log_root: str = "logs/rsl_rl"
    device: str = "cuda:0"
    viewer: str = "native"


def main():
    try:
        sys.argv[1:] = apply_cli()
    except CliError as error:
        sys.exit(f"error: {error}")
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
        if not isinstance(cfg.env, ZbotEnvCfg):
            raise ValueError("frequency requires a periodic walking task")
        if cfg.frequency <= 0:
            raise ValueError("frequency must be positive")
        cfg.env.test_frequency = cfg.frequency
    if cfg.target_speed is not None:
        if cfg.target_speed < 0:
            raise ValueError("target-speed must be non-negative")
        command = cfg.env.commands.get("target_speed")
        if not isinstance(command, UniformVelocityCommandCfg):
            raise ValueError("target-speed requires the velocity-tracking task")
        command.ranges.lin_vel_x = (cfg.target_speed, cfg.target_speed)
    periodic = cfg.env if isinstance(cfg.env, ZbotEnvCfg) else None
    if periodic is not None and not (
        0 < periodic.step_frequency_min <= periodic.step_frequency_max
    ):
        raise ValueError("Require 0 < step-frequency-min <= step-frequency-max")
    agent = load_rl_cfg(task)
    checkpoint = (
        Path(cfg.checkpoint_file)
        if cfg.checkpoint_file
        else latest_checkpoint(agent.experiment_name, cfg.log_root)
    )
    print(f"Checkpoint: {checkpoint}")
    if periodic is not None:
        print(
            f"Frequency: {periodic.step_frequency_min}–{periodic.step_frequency_max} Hz"
        )
    if cfg.frequency is not None:
        print(f"Fixed frequency: {cfg.frequency} Hz")
    if cfg.target_speed is not None:
        print(f"Fixed target speed: {cfg.target_speed} m/s")
    # Keep frequency encoding consistent with training even for fixed tests.
    if (
        periodic is not None
        and periodic.step_frequency_min == periodic.step_frequency_max
    ):
        periodic.test_frequency = periodic.step_frequency_min
        training_env = load_env_cfg(task)
        if not isinstance(training_env, ZbotEnvCfg):
            raise TypeError("periodic task requires ZbotEnvCfg for training and play")
        periodic.step_frequency_min = training_env.step_frequency_min
        periodic.step_frequency_max = training_env.step_frequency_max
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
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent), device=cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=cfg.device,
        )
        policy = runner.get_inference_policy(device=cfg.device)
        roll_offsets = getattr(env, "_initial_snake_roll_offset", None)
        if roll_offsets is not None:
            angles = [round(angle * 180 / math.pi, 2) for angle in roll_offsets[:8].tolist()]
            print(f"Initial snake roll offsets (degrees, first {len(angles)} envs): {angles}")
        if "transform" in cfg.env.commands:
            transform = env.command_manager.get_command("transform")
            if transform is None:
                raise ValueError("Transform command is configured but unavailable")
            values = transform[:, 0]
            stand_count = int((values > 0).sum().item())
            snake_count = int((values < 0).sum().item())
            print(f"Initial directions: snake -> stand {stand_count}, biped -> snake {snake_count}")
            print(f"Commands (first 16 envs; +1 stand, -1 snake): {values[:16].tolist()}")
            print("Transform controls: PageUp = stand (+1), PageDown = snake (-1). Keyboard control disables automatic resets; no pose reset.")
        viewer = FollowCameraViewer if cfg.viewer == "native" else ViserPlayViewer
        viewer(wrapped, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
