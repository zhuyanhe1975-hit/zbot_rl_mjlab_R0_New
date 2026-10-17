"""How hard does a policy work the RS00 joints?

    python actuator_usage.py <TASK> <checkpoint.pt> [--speed 0.25]

Rolls out 16 environments for 1000 control steps at a fixed forward command (first 250
steps dropped, 15 s kept) with the actuator model in robot.py, and reports per joint:

* torque: peak, 99th percentile and rms of |torque| against the 14 N*m peak and the
  5 N*m rated (continuous) torque;
* speed: 99th percentile and peak of |joint velocity| against the 100 rpm point where the
  datasheet T-N curve starts to fall and the 315 rpm no-load speed;
* the simulator applies only the constant 14 N*m cap, not the datasheet's falling T-N
  curve, so it can ask more of the motor than the hardware gives.  "over curve" is the
  share of samples where the torque in the direction of motion exceeds what the datasheet
  curve allows at that speed (robot.available_torque).  Anything above 0 means the
  simulated gait uses torque the real motor cannot deliver at that speed;
* a thermal check from the manual's rotating-overload table (robot.RS00_OVERLOAD): the
  largest rms torque over any window of 3, 7, 10 and 15 s, against the load the motor can
  hold for that long.  The table is for a cold motor on a 90x85 mm heatsink, so it is a
  rough guide for the real module.

Falls are counted too, so this also shows whether a policy trained under other actuator
limits still walks under the RS00 limits.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab import robot

NUM_ENVS, STEPS, WARMUP = 16, 1000, 250
WINDOWS_S = (3.0, 7.0, 10.0, 15.0)


def curve_limit(velocity):
    """Datasheet T-N torque limit at each joint speed, elementwise over an array."""
    return np.vectorize(robot.available_torque)(velocity)


def hold_limit(seconds):
    """Load (N*m) the manual says the motor can hold for this long (rotating)."""
    times, loads = zip(*robot.RS00_OVERLOAD)
    if seconds >= times[-1]:
        return robot.RS00_RATED_TORQUE
    return float(np.interp(seconds, times, loads))


def window_rms(torque, steps):
    """Largest rms of torque over any window of `steps` control steps, per joint.

    torque has shape (time, envs, joints); the maximum is over windows and envs.
    """
    square = np.cumsum(np.square(torque), axis=0)
    square = np.concatenate([np.zeros_like(square[:1]), square])
    mean = (square[steps:] - square[:-steps]) / steps
    return np.sqrt(mean).max(axis=(0, 1))


def collect(task, checkpoint, speed, device="cuda:0"):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.commands["target_speed"].ranges.lin_vel_x = (speed, speed)
    env = ManagerBasedRlEnv(cfg, device=device)
    agent = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    data = env.scene["robot"].data
    torque, velocity = [], []
    falls = 0
    obs = wrapped.reset()[0]
    with torch.inference_mode():
        for step in range(STEPS):
            obs, _, dones, extras = wrapped.step(policy(obs))
            timed_out = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~timed_out).sum())
            if step >= WARMUP:
                torque.append(data.actuator_force.cpu().numpy())
                velocity.append(data.joint_vel.cpu().numpy())
    dt = env.step_dt
    env.close()
    return np.stack(torque), np.stack(velocity), falls, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    torque, velocity, falls, dt = collect(args.task, args.checkpoint, args.speed)

    peak, rated = robot.RS00_PEAK_TORQUE, robot.RS00_RATED_TORQUE
    corner, top = robot.RS00_CORNER_SPEED, robot.RS00_NO_LOAD_SPEED
    print(
        f"RESULT falls={falls}  (peak {peak} N*m, rated {rated} N*m, T-N curve falls "
        f"from {corner:.1f} rad/s, no-load {top:.1f} rad/s)"
    )
    # Torque pushing along the motion is the case the T-N curve limits; braking is not.
    driving = torque * velocity > 0
    over_curve = driving & (np.abs(torque) > curve_limit(velocity) + 1e-6)
    flat = lambda a: a.reshape(-1, a.shape[-1])
    t_all, v_all, over = flat(torque), flat(velocity), flat(over_curve)
    limits = {w: hold_limit(w) for w in WINDOWS_S}
    heat = {w: window_rms(torque, round(w / dt)) for w in WINDOWS_S}
    for joint in range(torque.shape[-1]):
        t, v = np.abs(t_all[:, joint]), np.abs(v_all[:, joint])
        rms = float(np.sqrt(np.mean(t_all[:, joint] ** 2)))
        print(
            f"RESULT joint{joint + 1}: torque peak {t.max():5.2f} p99 {np.percentile(t, 99):5.2f}"
            f" rms {rms:4.2f}{' >rated' if rms > rated else ''} | speed p99 "
            f"{np.percentile(v, 99):5.1f} peak {v.max():5.1f} rad/s"
            f"{' >100rpm' if v.max() > corner else ''}{' >no-load' if v.max() > top else ''}"
            f" | over curve {100 * over[:, joint].mean():5.2f}% of samples"
        )
    for window in WINDOWS_S:
        worst = heat[window]
        flag = " OVER" if (worst > limits[window]).any() else ""
        print(
            f"RESULT thermal {window:4.0f} s window: max rms per joint "
            + " ".join(f"{x:5.2f}" for x in worst)
            + f" | allowed {limits[window]:5.2f}{flag}"
        )


if __name__ == "__main__":
    main()
