"""Velocity + energy walking that lands and stands on the whole sole.

Replaces the old foot_nonparallel_ground term, whose 1-|nz| form was ~theta^4 and
gave almost no gradient below 10 degrees, with a sin^2-based cost active in three
phases: while loaded, at touchdown, and just before touchdown.

The three weights follow a curriculum.  At the full weights from iteration 0 the net
reward was negative, so the policy learned to fall quickly (episode length ~30 steps)
and never recovered; at a fixed small weight (-0.1 / -0.4 / -0.1) it walked but the
soles stayed as tilted as before.  So the weights start small and grow once a gait
exists.
"""

from mjlab.envs.mdp import reward_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from .. import mdp
from .base import RewardEdit
from .velocity_energy_tracking import VelocityEnergyTracking

FLAT_FOOT_VELOCITY_ENERGY_TASK_ID = "Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Walking"

# Sole tilt is penalized through sin^2(tilt) / SIGMA^2; 0.05 is about 3 degrees.
FLAT_SIGMA = 0.05
PRE_LANDING_HEIGHT = 0.03  # sole height (m) below which a swing foot must be flat

# env.common_step_counter advances once per vector step and one PPO iteration is
# num_steps_per_env of them, so a stage "from iteration N" starts at N * this.
STEPS_PER_ITERATION = 24
# (first iteration, stance, touchdown, pre-landing weight).  The last row equals the
# weights that collapsed when used from iteration 0.
WEIGHT_STAGES = (
    (0, -0.05, -0.2, -0.05),
    (150, -0.15, -0.6, -0.15),
    (300, -0.3, -1.2, -0.3),
    (400, -0.5, -2.0, -0.5),
)
_TERMS = ("foot_flat_stance", "foot_flat_touchdown", "foot_flat_pre_landing")


class FlatFootVelocityEnergy(VelocityEnergyTracking):
    task_id = FLAT_FOOT_VELOCITY_ENERGY_TASK_ID
    experiment_name = "zbot_flat_foot_velocity_energy"
    weight_stages = WEIGHT_STAGES
    """(first iteration, stance, touchdown, pre-landing weight) rows of the curriculum."""

    def reward_edit(self):
        first = self.weight_stages[0]
        params = {"sigma": FLAT_SIGMA}
        return RewardEdit(
            remove=("foot_nonparallel_ground",),
            terms={
                "foot_flat_stance": RewardTermCfg(
                    func=mdp.stance_flat_penalty, weight=first[1], params=params
                ),
                "foot_flat_touchdown": RewardTermCfg(
                    func=mdp.touchdown_flat_penalty, weight=first[2], params=params
                ),
                "foot_flat_pre_landing": RewardTermCfg(
                    func=mdp.pre_landing_flat_penalty,
                    weight=first[3],
                    params={**params, "height": PRE_LANDING_HEIGHT},
                ),
            },
        )

    def edit(self, cfg, play):
        for column, name in enumerate(_TERMS, start=1):
            cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
                func=reward_curriculum,
                params={
                    "reward_name": name,
                    "stages": [
                        {"step": row[0] * STEPS_PER_ITERATION, "weight": row[column]}
                        for row in self.weight_stages
                    ],
                },
            )
