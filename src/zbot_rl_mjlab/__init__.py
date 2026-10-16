"""Zbot tasks for mjlab.  Importing this package registers every experiment.

Each experiment is a class in ``experiments/`` that derives from its parent experiment and
states only its own changes (see ``experiments/base.py``).  To add one, write the class and
list it here.
"""

from .experiments.deploy import DeployFlatFoot, DeployWalking
from .experiments.disturbed_walking import DisturbedWalking
from .experiments.flat_foot_disturbed import FlatFootDisturbed
from .experiments.flat_foot_velocity_energy import FlatFootVelocityEnergy
from .experiments.flat_walking import FlatWalking
from .experiments.in_place_stepping import PeriodicStepping
from .experiments.omni_velocity_energy import OmniVelocityEnergy
from .experiments.velocity_energy_disturbed import VelocityEnergyDisturbed
from .experiments.velocity_energy_tracking import VelocityEnergyTracking
from .experiments.velocity_tracking import VelocityTracking
from .experiments.walking_tuning import WalkingTuning

EXPERIMENTS = (
    PeriodicStepping,
    FlatWalking,
    DisturbedWalking,
    VelocityTracking,
    VelocityEnergyTracking,
    VelocityEnergyDisturbed,
    OmniVelocityEnergy,
    FlatFootVelocityEnergy,
    FlatFootDisturbed,
    DeployWalking,
    DeployFlatFoot,
    WalkingTuning,
)

for experiment in EXPERIMENTS:
    experiment().register()
