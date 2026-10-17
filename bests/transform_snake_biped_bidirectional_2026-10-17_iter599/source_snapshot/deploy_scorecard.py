"""Deployment scorecard: smoothness, energy, naturalness and robustness in one table.

    python deploy_scorecard.py --model NAME TASK CHECKPOINT [--model ...] [--speed 0.25]

Every model is rolled out (16 environments x 1000 control steps) at a fixed forward command,
once clean ("nominal") and once per perturbation.  Perturbations touch only what the policy
sees or what the actuators do, one at a time:

* sensor noise on the actor observations,
* a constant IMU tilt in the projected-gravity term (+/- x, +/- y),
* base_lin_vel withheld (only for models whose actor reads it),
* the action reaching the actuators 1, 2 or 3 control steps late,
* PD gains x0.7 / x1.3,
* the base COM shifted 3 cm along +/- x and +/- y of the base body.

Training-time randomization events are removed first so the scenarios are controlled.
Observation slices come from the task's actor term order, so blind and non-blind actors
are handled alike.  All of this is simulation: it is a proxy for hardware, not a measurement.
"""

import argparse
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from gait_metrics import touchdown_stats
from zbot_rl_mjlab import mdp

NUM_ENVS, STEPS, WARMUP = 16, 1000, 250
GRAVITY = 9.81
TERM_DIMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "yaw_error": 1,
    "joint_pos": 6,
    "joint_vel": 6,
    "actions": 6,
    "phase": 2,
    "frequency": 1,
    "target_speed": 1,
    "velocity_command": 3,
}
TILT = 0.05  # about 3 degrees
COM_SHIFT = 0.03  # m

SCENARIOS = {
    "nominal": {},
    "sensor noise": {
        "noise": {
            "base_lin_vel": 0.05,
            "base_ang_vel": 0.05,
            "projected_gravity": 0.02,
            "joint_pos": 0.01,
            "joint_vel": 0.15,
        }
    },
    "IMU tilt +x": {"bias": {"projected_gravity": (TILT, 0.0, 0.0)}},
    "IMU tilt -x": {"bias": {"projected_gravity": (-TILT, 0.0, 0.0)}},
    "IMU tilt +y": {"bias": {"projected_gravity": (0.0, TILT, 0.0)}},
    "IMU tilt -y": {"bias": {"projected_gravity": (0.0, -TILT, 0.0)}},
    "base_lin_vel withheld": {"zero": ["base_lin_vel"]},
    "action delay 1 step": {"delay": 1},
    "action delay 2 steps": {"delay": 2},
    "action delay 3 steps": {"delay": 3},
    "PD gains x0.7": {"gain": 0.7},
    "PD gains x1.3": {"gain": 1.3},
    "base COM +3 cm x": {"com": (COM_SHIFT, 0.0, 0.0)},
    "base COM -3 cm x": {"com": (-COM_SHIFT, 0.0, 0.0)},
    "base COM +3 cm y": {"com": (0.0, COM_SHIFT, 0.0)},
    "base COM -3 cm y": {"com": (0.0, -COM_SHIFT, 0.0)},
}


def term_slices(actor_terms):
    slices, start = {}, 0
    for name in actor_terms:
        slices[name] = slice(start, start + TERM_DIMS[name])
        start += TERM_DIMS[name]
    return slices


def needs(scenario):
    """Actor terms a scenario cannot run without (noise just skips missing terms)."""
    return set(scenario.get("bias", {})) | set(scenario.get("zero", []))


def run(task, checkpoint, speed, scenario, device="cuda:0"):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.commands["target_speed"].ranges.lin_vel_x = (speed, speed)
    for name in [n for n in cfg.events if n != "reset_scene_to_default"]:
        cfg.events.pop(name)  # controlled scenarios: no training-time randomization
    slices = term_slices(cfg.observations["actor"].terms)

    effort_limit = 0.0
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        gain = scenario.get("gain", 1.0)
        actuator.stiffness *= gain
        actuator.damping *= gain
        effort_limit = actuator.effort_limit
    if "com" in scenario:
        shift = scenario["com"]
        cfg.events["scorecard_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "operation": "add",
                "ranges": {axis: (shift[axis], shift[axis]) for axis in range(3)},
            },
        )

    env = ManagerBasedRlEnv(cfg, device=device)
    agent = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    robot = env.scene["robot"]
    total_mass = float(env.sim.mj_model.body_mass.sum())

    delay = scenario.get("delay", 0)
    pending = deque(
        [torch.zeros(NUM_ENVS, 6, device=device) for _ in range(delay)],
        maxlen=delay or 1,
    )
    obs = wrapped.reset()[0]
    previous = torch.zeros(NUM_ENVS, 6, device=device)
    falls = timeouts = 0
    speeds, rate, acc, power, torque_peak, contact = [], [], [], [], [], []
    with torch.inference_mode():
        for step in range(STEPS):
            seen = obs["actor"].clone()
            for name, std in scenario.get("noise", {}).items():
                if name in slices:
                    block = seen[:, slices[name]]
                    seen[:, slices[name]] = block + std * torch.randn_like(block)
            for name, value in scenario.get("bias", {}).items():
                seen[:, slices[name]] += torch.tensor(value, device=device)
            for name in scenario.get("zero", []):
                seen[:, slices[name]] = 0.0
            obs_in = obs.clone()
            obs_in["actor"] = seen
            action = policy(obs_in)
            sent = action
            if delay:
                pending.append(action)
                sent = pending.popleft()
            obs, _, dones, extras = wrapped.step(sent)
            timed_out = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~timed_out).sum())
            timeouts += int((dones.bool() & timed_out).sum())
            if step >= WARMUP:
                speeds.append(mdp.base_lin_vel(env)[:, 0].mean().item())
                rate.append((action - previous).square().mean().item())
                acc.append(robot.data.joint_acc.square().mean().item())
                torque = robot.data.actuator_force
                power.append(
                    (torque * robot.data.joint_vel).abs().sum(-1).mean().item()
                )
                torque_peak.append(
                    (torque.abs().amax(-1) > 0.9 * effort_limit).float().mean().item()
                )
                contact.append(env.scene["feet_ground_contact"].data.found > 0)
            previous = action
    dt = env.step_dt
    env.close()

    mean_speed = float(np.mean(speeds))
    mean_power = float(np.mean(power))
    periods, offsets = touchdown_stats(torch.stack(contact).cpu().numpy(), dt)
    return {
        "falls": falls,
        "timeouts": timeouts,
        "speed": mean_speed,
        "action_rate_rms": float(np.sqrt(np.mean(rate))),
        "joint_acc_rms": float(np.sqrt(np.mean(acc))),
        "power_w": mean_power,
        "cost_of_transport": mean_power
        / (total_mass * GRAVITY * max(mean_speed, 0.05)),
        "torque_saturation": float(np.mean(torque_peak)),
        "period_cv": float(periods.std() / periods.mean()),
        "alternation_phase": float(offsets.mean()),
        "alternation_std": float(offsets.std()),
    }


def evaluate(name, task, checkpoint, speed):
    actor_terms = load_env_cfg(task, play=True).observations["actor"].terms
    results = {}
    for label, scenario in SCENARIOS.items():
        if not needs(scenario) <= set(actor_terms):
            results[label] = None  # the actor has no such input
            continue
        results[label] = run(task, checkpoint, speed, scenario)
        r = results[label]
        print(
            f"RESULT {name} | {label:24s} falls={r['falls']:4d} "
            f"timeouts={r['timeouts']:3d} speed={r['speed']:+.3f}",
            flush=True,
        )
    return results


NOMINAL_ROWS = (
    ("falls", "falls", "{:d}"),
    ("speed (cmd)", "speed", "{:.3f}"),
    ("action rate rms (smooth: low)", "action_rate_rms", "{:.3f}"),
    ("joint acc rms rad/s2 (smooth: low)", "joint_acc_rms", "{:.1f}"),
    ("mechanical power W", "power_w", "{:.2f}"),
    ("cost of transport (low)", "cost_of_transport", "{:.2f}"),
    ("torque near limit, fraction", "torque_saturation", "{:.3f}"),
    ("step period cv", "period_cv", "{:.3f}"),
    ("alternation phase (0.50)", "alternation_phase", "{:.2f}"),
    ("alternation std", "alternation_std", "{:.3f}"),
)


def print_table(models, speed):
    names = list(models)
    width = max(14, *(len(n) + 2 for n in names))
    print(f"\nSCORECARD (forward command {speed} m/s)")
    print(f"{'nominal':38s}" + "".join(f"{n:>{width}s}" for n in names))
    for title, key, fmt in NOMINAL_ROWS:
        cells = [fmt.format(models[n]["nominal"][key]) for n in names]
        print(f"{title:38s}" + "".join(f"{c:>{width}s}" for c in cells))
    print(
        f"\n{'falls under perturbation':38s}" + "".join(f"{n:>{width}s}" for n in names)
    )
    for label in list(SCENARIOS)[1:]:
        cells = []
        for n in names:
            r = models[n][label]
            cells.append("n/a" if r is None else f"{r['falls']}/{r['speed']:.2f}")
        print(f"{label:38s}" + "".join(f"{c:>{width}s}" for c in cells))
    print("(each cell: falls / mean forward speed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "TASK", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    models = {
        name: evaluate(name, task, checkpoint, args.speed)
        for name, task, checkpoint in args.model
    }
    print_table(models, args.speed)


if __name__ == "__main__":
    main()
