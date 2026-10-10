# Best checkpoints

Layout: `bests/<experiment>_<date>_iter<N>/{model_<N>.pt, env.yaml, agent.yaml, SHA256SUMS}`.

A checkpoint is kept here only if it loads strictly into the current task and
runs a full 20 s episode without falling (checked with 16 parallel envs).
When the environment, observations or robot parameters change, re-check every
entry and delete the ones that no longer replay.

| Directory | Task | Notes |
|---|---|---|
| `velocity_energy_disturbed_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking` | 500 iterations, 8192 envs. Also replays in `...Velocity-Energy-Tracking-Walking`. 0 falls in 16 envs x 1000 steps. |
| `velocity_energy_tracking_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking` | 500 iterations. 0 falls in 16 envs x 1000 steps. |
| `omni_velocity_energy_2026-10-10_iter499` | `Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking` | 500 iterations. 0 falls under 6 fixed commands, 16 envs x 1000 steps. Tracks vx/vy well; **pure yaw is weak** (see below). |

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
