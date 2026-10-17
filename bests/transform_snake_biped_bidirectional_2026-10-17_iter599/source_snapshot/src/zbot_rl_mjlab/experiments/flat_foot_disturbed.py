"""Flat-foot velocity + energy walking with base disturbances and domain randomization.

Same disturbance and randomization as velocity_energy_disturbed (random base force and
torque impulses on a quarter of the environments; foot friction, encoder bias and base
COM randomized at startup), on top of the curriculum flat-foot rewards.  Meant to be
fine-tuned from a flat-foot checkpoint: the resumed step counter keeps the penalty
curriculum on its final weights.
"""

from .addons import DISTURBED
from .flat_foot_velocity_energy import FlatFootVelocityEnergy

FLAT_FOOT_DISTURBED_TASK_ID = (
    "Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Disturbed-Walking"
)


class FlatFootDisturbed(FlatFootVelocityEnergy):
    task_id = FLAT_FOOT_DISTURBED_TASK_ID
    experiment_name = "zbot_flat_foot_velocity_energy_disturbed"
    addons = DISTURBED
