"""Walking with commanded stepping frequency, as its own experiment for reward tuning.

This derives from ``FlatWalking`` (same observations, rewards and weights) but is
registered as a separate task with its own experiment name, so editing the table below only
changes this task and never the checkpoints, logs or recipes of ``Mjlab-Zbot-6dof-Walking``.

Every reward weight and every reward parameter lives in ``REWARD_WEIGHTS`` / ``REWARD_PARAMS``.
The values are the ones ``Walking`` has now; ``tests/test_walking_tuning.py`` checks that this
task starts identical to ``Walking`` apart from the frequency range and the training schedule,
so a failing test after you edit the tables just means you have tuned something on purpose.

The stepping frequency is the commanded one: the policy sees the phase (sin, cos) and the
frequency, and ``frequency_tracking`` rewards matching the left/right support force to that
phase.  The frequency range below replaces the 0.5-1.5 Hz of the base recipe with the
0.25-1.0 Hz range that ``train.sh`` passes on the command line.
"""

from .base import RewardEdit
from .flat_walking import FlatWalking

WALKING_TUNING_TASK_ID = "Mjlab-Zbot-6dof-Walking-Tuning"

# Commanded stepping frequency range (Hz) and the training schedule.
STEP_FREQUENCY_RANGE = (0.25, 1.0)
NUM_ENVS = 8192
MAX_ITERATIONS = 600

# Reward weights.  Positive terms are rewards, negative terms are penalties.
REWARD_WEIGHTS = {
    "frequency_tracking": 1.5,  # left/right support force follows the commanded phase
    "swing_clearance": 0.5,  # swing foot reaches the target height
    "forward_velocity": 2.0,  # bounded "go forward" reward, no target speed
    "lateral_velocity": -0.2,  # sideways speed
    "upright": 0.2,  # body stays upright
    "yaw_drift": -0.2,  # heading drift from the start direction
    "support_foot_slip": -5.0,  # a loaded foot sliding or twisting
    "soft_landing": -0.02,  # peak landing force
    "foot_nonparallel_ground": -5.0,  # loaded sole not parallel to the ground
    "action_rate": -0.05,  # change of action between steps
    "joint_limits": -0.1,  # joint position limit violations
    "joint_acceleration": -0.0005,  # sum of absolute joint accelerations
}

# Reward shaping parameters, only for terms that have tunable ones.  The values written
# here equal the function defaults, so listing them does not change behaviour; terms not
# listed keep the parameters of the base recipe.
REWARD_PARAMS = {
    "forward_velocity": {"scale": 0.2},  # speed (m/s) at which the reward reaches 63 %
    "swing_clearance": {"target": 0.05},  # swing foot target height (m), width 0.04 m
    "foot_nonparallel_ground": {"std": 0.1},  # tilt tolerance of the flatness penalty
}


class WalkingTuning(FlatWalking):
    task_id = WALKING_TUNING_TASK_ID
    experiment_name = "zbot_walking_tuning"
    train_envs = NUM_ENVS
    step_frequency = STEP_FREQUENCY_RANGE
    max_iterations = MAX_ITERATIONS

    def reward_edit(self):
        # Applied after FlatWalking's own edit, so these tables have the last word.
        return RewardEdit(weights=REWARD_WEIGHTS, params=REWARD_PARAMS)

    def edit(self, cfg, play):
        missing = set(REWARD_WEIGHTS) - set(cfg.rewards)
        unknown = set(cfg.rewards) - set(REWARD_WEIGHTS)
        if missing or unknown:
            raise ValueError(
                f"REWARD_WEIGHTS must list exactly the inherited rewards; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
