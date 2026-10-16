# Best checkpoints

Layout: `bests/<experiment>_<date>_iter<N>/{model_<N>.pt, env.yaml, agent.yaml, SHA256SUMS}`.

A checkpoint is kept here only if it loads strictly into the current task and
runs a full 20 s episode without falling (checked with 16 parallel envs).
When the environment, observations or robot parameters change, re-check every
entry and delete the ones that no longer replay.

**Sensor note (2026-10-11):** every checkpoint below except `deploy_walking_*` was trained while the
foot height scan was attached to the foot *bodies*. foot_1's body origin sits ~54 mm above its sole,
so its `swing_clearance` / `swing_foot_excess_height` readings were offset and the two feet were
rewarded asymmetrically. The scan now starts at the sole sites (`src/zbot_rl_mjlab/env_cfg.py`).
Replay is unaffected (height is not an observation), but new training runs are not directly
comparable with the older entries.

**Which one to deploy:** `deploy_walking_2026-10-11_iter1499` (see the scorecard below). It is the
only entry whose actor does not read `base_lin_vel` / `yaw_error`; the others cannot run on hardware
without a velocity estimator.

| Directory | Task | Notes |
|---|---|---|
| `deploy_walking_2026-10-11_iter1499` | `Mjlab-Zbot-6dof-Deploy-Walking` | 1500 iterations, 8192 envs. 28-dim actor without base_lin_vel / yaw_error; trained with observation noise and IMU bias, 0-2 step action delay, PD gain, mass/inertia, friction, COM and base-push randomization. 0 falls in every scorecard scenario (scorecard was run by the user; this checkpoint is assumed to be the one scored). **Not yet tried on hardware.** |
| `velocity_energy_disturbed_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking` | 500 iterations, 8192 envs. Also replays in `...Velocity-Energy-Tracking-Walking`. 0 falls in 16 envs x 1000 steps. |
| `velocity_energy_tracking_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking` | 500 iterations. 0 falls in 16 envs x 1000 steps. |
| `omni_velocity_energy_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking` | 500 iterations. 0 falls under 6 fixed commands, 16 envs x 1000 steps. Tracks vx/vy well; **pure yaw is weak** (see below). |
| `flat_foot_velocity_energy_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Walking` | 500 iterations, 8192 envs. **Fall check not run yet** (`python fall_check.py <TASK> <ckpt>`); archived on request. Did **not** flatten the feet (see below). |

## Deployment scorecard (forward 0.25 m/s, `python deploy_scorecard.py`, 16 envs x 1000 steps)

`baseline` = `velocity_energy_tracking_2026-10-10_iter499`, `deploy` = the deploy_walking entry,
`deploy_flat` = `Deploy-Flat-Foot-Walking` (same randomization plus flat-foot penalties; not archived,
it was worse on every axis below).

| nominal | baseline | deploy | deploy_flat |
|---|---|---|---|
| falls | 0 | 0 | 0 |
| speed (cmd 0.25) | 0.247 | 0.249 | 0.247 |
| action rate rms (lower = smoother) | 0.154 | 0.181 | 0.215 |
| joint acc rms rad/s2 | 110.1 | 113.3 | 118.9 |
| mechanical power W | 2.01 | 2.07 | 2.18 |
| cost of transport | 0.19 | 0.20 | 0.21 |
| step period cv | 0.054 | 0.032 | 0.041 |
| alternation phase (0.50 ideal) | 0.50 | 0.51 | 0.61 |

| falls / speed under perturbation | baseline | deploy | deploy_flat |
|---|---|---|---|
| IMU tilt +x (0.05, ~3 deg) | **127** / 0.16 | 0 / 0.25 | 0 / 0.24 |
| IMU tilt -y | 5 / 0.23 | 0 / 0.26 | 0 / 0.25 |
| base_lin_vel withheld | **464** / - | n/a | n/a |
| action delay 3 steps | 0 / 0.26 | 0 / 0.22 | 0 / 0.19 |
| base COM +-3 cm x, y | 0 | 0 | 0 |
| sensor noise, PD gains x0.7 / x1.3, delay 1-2 steps | 0 | 0 | 0 |

The scorecard is a simulation proxy at 0.25 m/s forward; it does not cover in-place stepping, the case
that failed on hardware.

## Flat-foot checkpoint gait metrics (vx = 0.25, `python gait_metrics.py`)

| Metric | velocity-energy-tracking (baseline) | flat-foot |
|---|---|---|
| foot0 pitch / roll, mean (p90) deg | 3.4 (7.5) / 3.9 (6.8) | 2.9 (8.9) / 3.4 (7.8) |
| foot1 pitch / roll, mean (p90) deg | 3.2 (6.8) / 3.7 (6.5) | 3.1 (8.6) / 2.7 (6.2) |
| period / cv | 1.08 s / 0.05 | 1.25 s / 0.05 |
| alternation phase (0.50 ideal) | 0.50, std 0.02 | 0.46, std 0.01 |
| mirror error joint1/6, 2/5, 3/4 | 0.68, 0.34, 0.29 | 1.28, 0.18, 0.68 |

Sole tilt p90 did not improve (6.5-7.5 deg to 6.2-8.9 deg) and the joint1/6 mirror error got worse.
The added flatness penalties at weights -0.1 / -0.4 / -0.1 were too weak to change the stance, while
the earlier -0.5 / -2.0 / -0.5 run collapsed (episode length ~30 steps).

## Omni checkpoint tracking (mean over steps 250-1000, 16 envs, `(vx, vy, wz)` command)

| Command | Measured |
|---|---|
| (+0.30, 0, 0) | (+0.294, +0.005, -0.037) |
| (-0.20, 0, 0) | (-0.197, -0.002, +0.036) |
| (0, +0.15, 0) | (-0.010, +0.119, -0.004) |
| (0, 0, +0.40) | (-0.002, -0.001, +0.005) |
| (+0.30, +0.10, +0.30) | (+0.293, +0.085, +0.190) |
| (0, 0, 0) | (+0.000, +0.000, -0.008) |

A pure yaw command alone barely turns the robot; yaw only appears when combined with forward motion.

Replay: `python play.py <TASK> --checkpoint-file bests/<dir>/model_<N>.pt`
Sim-to-sim: `./sim2sim.sh --checkpoint bests/<dir>/model_<N>.pt`
