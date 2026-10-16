"""Replay a checkpoint at fixed forward speeds and count falls.

    python fall_check.py <TASK> <checkpoint.pt> [speed ...]

16 environments x 1000 control steps (one full 20 s episode) per speed. A checkpoint
qualifies for bests/ when falls == 0 and every environment times out.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab import mdp

NUM_ENVS, STEPS, WARMUP = 16, 1000, 250


def check(task, checkpoint, speed, device="cuda:0"):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.commands["target_speed"].ranges.lin_vel_x = (speed, speed)
    env = ManagerBasedRlEnv(cfg, device=device)
    agent = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    obs = wrapped.reset()[0]
    falls = timeouts = 0
    speeds = []
    with torch.inference_mode():
        for step in range(STEPS):
            obs, _, dones, extras = wrapped.step(policy(obs))
            timed_out = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~timed_out).sum())
            timeouts += int((dones.bool() & timed_out).sum())
            if step >= WARMUP:
                speeds.append(mdp.base_lin_vel(env)[:, 0].mean().item())
    env.close()
    return sum(speeds) / len(speeds), falls, timeouts


def main():
    task, checkpoint = sys.argv[1], sys.argv[2]
    for speed in [float(s) for s in sys.argv[3:]] or [0.1, 0.25, 0.45]:
        measured, falls, timeouts = check(task, checkpoint, speed)
        print(
            f"RESULT vx={speed:.2f} measured={measured:+.3f} "
            f"falls={falls} timeouts={timeouts}"
        )


if __name__ == "__main__":
    main()
