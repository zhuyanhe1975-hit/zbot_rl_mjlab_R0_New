# Task and experiment structure

Tasks are organized by behavior category. Walking uses the shared configuration
described below. Transformation tasks have their own environment under
`src/zbot_rl_mjlab/tasks/transformation/`: they change support contacts and body
configuration, so walking phases, rewards, and fall criteria do not apply.
The category includes biped-to-snake, snake-to-biped, and an independent
command-conditioned snake/biped bidirectional experiment.
Both categories use the existing task registry, training, and playback entrypoints.

The shared task contract lives in `src/zbot_rl_mjlab/env_cfg.py`: scene, robot,
sensors, observations, actions, terminations, common rewards, and PPO defaults.

Training recipes live under `src/zbot_rl_mjlab/experiments/`:

- `in_place_stepping.py` registers the periodic stepping task.
- `flat_walking.py` changes only the Walking experiment frequency and forward-speed recipe.
- `common.py` provides shared walking regularizers and domain randomization.
- `disturbed_walking.py` registers the disturbance-trained Walking task.
- `velocity_tracking.py` adds commanded forward speed (0.05–0.25 m/s) to
  undisturbed flat walking, with fixed 0.5 Hz stepping. Commands resample every
  5–10 seconds; actor and critic each receive a target-speed observation.
  The forward reward is replaced by `exp(-(speed-target)^2 / 0.05^2)`.
  Train with `./train_velocity_tracking.sh`. Its actor has 32 observations;
  existing 31-observation Walking checkpoints cannot be resumed directly.

The robot model remains isolated in `robot.py` and `assets/`. Adding a new
experiment should compose the shared task configuration rather than duplicating
the environment definition.
