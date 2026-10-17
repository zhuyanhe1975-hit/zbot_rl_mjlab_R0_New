"""Evaluate keyboard-controlled stand/snake/stand without automatic resets or PPO."""

import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab import viewer
from zbot_rl_mjlab.tasks.transformation import commanded, mdp, robust
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as standing

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
cfg = load_env_cfg(commanded.TASK_ID, play=True)
cfg.scene.num_envs = 8
cfg.seed = 42
cfg.commands["transform"].mode = "stand"
env = ManagerBasedRlEnv(cfg, device="cuda:0")
try:
    agent = load_rl_cfg(commanded.TASK_ID)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = robust.RobustSnakeToBipedRunner(wrapped, asdict(agent), device=env.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device=env.device)
    control = object.__new__(viewer.FollowCameraViewer)
    control.env = SimpleNamespace(unwrapped=env)
    control._actions = deque()
    control.user_key_callback = control._on_key
    data = env.scene["robot"].data
    report = {"checkpoint": args.checkpoint, "environments": env.num_envs, "seed": 42, "initial_roll_offsets_deg": torch.rad2deg(env._initial_snake_roll_offset).tolist(), "seconds_per_command": 16., "training_updates": 0, "stages": [], "trace_env_0": []}
    callbacks_preserved = True
    auto_resets = 0
    for stage, (target, key) in enumerate((("stand", viewer.KEY_PAGE_UP), ("snake", viewer.KEY_PAGE_DOWN), ("stand", viewer.KEY_PAGE_UP))):
        before = (data.root_link_pose_w.clone(), data.joint_pos.clone(), env.action_manager.get_term("joint_pos").commanded_target.clone(), env.episode_length_buf.clone())
        control._safe_key_callback(key)
        action, payload = control._actions.popleft()
        control._handle_custom_action(action, payload)
        for old, new in zip(before, (data.root_link_pose_w, data.joint_pos, env.action_manager.get_term("joint_pos").commanded_target, env.episode_length_buf), strict=True):
            callbacks_preserved &= torch.equal(old, new)
        ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_success = torch.full((env.num_envs,), -1., device=env.device)
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for step in range(800):
                obs, reward, done, _ = wrapped.step(policy(obs))
                assert torch.isfinite(reward).all()
                auto_resets += int(done.sum())
                assert env.command_manager.get_term("transform").manual_control.all()
                valid = commanded.success(env)
                first_success[valid & ~ever] = (step + 1) * env.step_dt
                ever |= valid
                if step % 20 == 0:
                    report["trace_env_0"].append({"stage": stage, "command": target, "time_in_stage_s": (step+1)*env.step_dt, "base_height_m": float(standing.base_height(env)[0]), "goal_joint_error_rad": float(commanded.joint_error(env)[0]), "sole_angles_deg": standing.sole_angles_degrees(env)[0].tolist()})
        report["stages"].append({"command": target, "ever_strict_success_count": int(ever.sum()), "final_strict_success_count": int(commanded.success(env).sum()), "first_success_seconds": first_success.tolist(), "final_base_heights_m": standing.base_height(env).tolist(), "final_mean_joint_errors_rad": commanded.joint_error(env).tolist(), "final_joint_positions_rad": data.joint_pos.tolist(), "final_sole_angles_deg": standing.sole_angles_degrees(env).tolist(), "final_hold_seconds": mdp.state(env).hold_time.tolist()})
    report["keyboard_callbacks_preserve_pose_targets_and_episode_counters"] = callbacks_preserved
    report["automatic_reset_count"] = auto_resets
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace_env_0"}, indent=2))
finally:
    env.close()
