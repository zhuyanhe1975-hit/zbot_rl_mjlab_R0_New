"""Energy-aware velocity tracking with disturbance randomization."""

from .addons import DISTURBED
from .velocity_energy_tracking import VelocityEnergyTracking

VELOCITY_ENERGY_DISTURBED_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking"


class VelocityEnergyDisturbed(VelocityEnergyTracking):
    task_id = VELOCITY_ENERGY_DISTURBED_TASK_ID
    experiment_name = "zbot_velocity_energy_disturbed"
    addons = DISTURBED
