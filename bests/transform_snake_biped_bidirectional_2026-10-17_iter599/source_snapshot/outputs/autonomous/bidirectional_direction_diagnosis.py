"""Score each source direction once, without PPO updates or rendering."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.transformation import commanded, mdp, robust
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as standing

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default="logs/rsl_rl/zbot_transform_snake_biped_bidirectional/2026-10-17_17-48-28/model_100.pt")
parser.add_argument("--output", default="outputs/autonomous/bidirectional_direction_diagnosis.json")
args = parser.parse_args()
cfg = load_env_cfg(commanded.TASK_ID)
cfg.scene.num_envs = 32
cfg.seed = 42
cfg.auto_reset = False
cfg.episode_length_s = 16
env = ManagerBasedRlEnv(cfg, device="cuda:0")
checkpoint = args.checkpoint
try:
    agent = load_rl_cfg(commanded.TASK_ID)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = robust.RobustSnakeToBipedRunner(wrapper, asdict(agent), device=env.device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device=env.device)
    initial_command = commanded.command(env)[:, 0].clone()
    initial_roll = torch.rad2deg(env._initial_snake_roll_offset).clone()
    episodes = {}
    trace = []
    observations = wrapper.get_observations()
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    peak = standing.base_height(env).clone()
    final_height = torch.zeros_like(peak)
    final_error = torch.zeros_like(peak)
    final_q_abs = torch.zeros_like(peak)
    final_angle = torch.zeros_like(peak)
    succeeded = torch.zeros_like(finished)
    durations = torch.zeros_like(peak)
    with torch.inference_mode():
        for step in range(env.max_episode_length + 1):
            observations, reward, done, _ = wrapper.step(policy(observations))
            active = ~finished
            peak[active] = torch.maximum(peak[active], standing.base_height(env)[active])
            succeeded[active] |= commanded.success(env)[active]
            complete = done.bool() & active
            final_height[complete] = standing.base_height(env)[complete]
            final_error[complete] = commanded.joint_error(env)[complete]
            final_q_abs[complete] = env.scene["robot"].data.joint_pos[complete].abs().mean(-1)
            final_angle[complete] = standing.sole_angles_degrees(env)[complete].amax(-1)
            durations[complete] = (step + 1) * env.step_dt
            for i in complete.nonzero().flatten().tolist():
                episodes[i] = {"environment": i, "initial_command": float(initial_command[i]), "initial_roll_offset_deg": float(initial_roll[i]), "ever_strict_success": bool(succeeded[i]), "final_strict_success": bool(commanded.success(env)[i]), "duration_s": float(durations[i]), "continuous_hold_s": float(mdp.state(env).hold_time[i]), "final_base_height_m": float(final_height[i]), "final_joint_error_mean_rad": float(final_error[i]), "final_joint_error_max_rad": float((env.scene["robot"].data.joint_pos[i] - commanded.goal_joints(env)[i]).abs().max()), "final_worst_sole_angle_deg": float(final_angle[i]), "joint_positions_rad": env.scene["robot"].data.joint_pos[i].tolist(), "termination_reasons": [name for name in cfg.terminations if bool(env.termination_manager.get_term(name)[i])]}
            if not finished[0] and step % 10 == 0:
                trace.append({"time_s": (step+1)*env.step_dt, "base_height_m": float(standing.base_height(env)[0]), "joint_error_rad": float(commanded.joint_error(env)[0]), "sole_angles_deg": standing.sole_angles_degrees(env)[0].tolist(), "joint_positions_rad": env.scene["robot"].data.joint_pos[0].tolist()})
            finished |= complete
            if finished.all():
                break
            ids = done.nonzero().flatten()
            if len(ids):
                env.reset(env_ids=ids)
                observations = wrapper.get_observations()
    report = {"checkpoint": checkpoint, "training_updates": 0, "seed": cfg.seed, "episode_seconds": cfg.episode_length_s, "initial_roll_range_degrees": cfg.initial_roll_range_deg, "episodes": [episodes[i] for i in sorted(episodes)], "trace_env_0": trace}
    for name, select in (("snake_to_biped", initial_command > 0), ("biped_to_snake", initial_command < 0)):
        report[name] = {"episodes": int(select.sum()), "ever_strict_success": int(succeeded[select].sum()), "mean_final_base_height_m": float(final_height[select].mean()), "mean_goal_joint_error_rad": float(final_error[select].mean()), "mean_abs_joint_position_rad": float(final_q_abs[select].mean()), "mean_final_worst_sole_angle_deg": float(final_angle[select].mean()), "mean_duration_s": float(durations[select].mean()), "episodes_ever_above_0_23m": int((peak[select] >= .23).sum()), "lying_target_pose_count": int(((final_error[select] < .2) & (final_height[select] < .2) & (final_angle[select] > 45)).sum())}
    destination = Path(args.output)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
finally:
    env.close()
