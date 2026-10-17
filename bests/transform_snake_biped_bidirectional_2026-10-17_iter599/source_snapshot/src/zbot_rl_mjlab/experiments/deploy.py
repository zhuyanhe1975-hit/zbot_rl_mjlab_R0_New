"""Walking trained for real-robot deployment, with and without the flat-foot penalties.

Every setting here exists to narrow the sim-to-real gap; nothing is tuned for sim score.
Both tasks use the ``DEPLOYMENT`` add-on preset (``addons.py``), so compared with the plain
velocity+energy tasks:

* the actor no longer sees ``base_lin_vel`` or ``yaw_error`` (a real robot has no direct
  measurement of either); the critic still does, as privileged information;
* actor observations get Gaussian noise, and the IMU-derived terms also get a bias that is
  constant within an episode (gyro zero offset, IMU mounting / calibration tilt);
* joint position targets reach the actuators 0..2 control steps late;
* PD gains, link mass/inertia, foot friction, encoder zero offset and base COM are
  randomized, and random impulses push the base.

Two variants share all of that and differ only in the foot reward, so the benefit of the
flat-foot penalties can be measured (``deploy_scorecard.py``) instead of assumed:

* ``Deploy-Walking``: the velocity+energy rewards unchanged;
* ``Deploy-Flat-Foot-Walking``: plus the curriculum flat-foot penalties.

Noise and action delay apply to training only (the play config turns them off).  The actor
input is 28-dimensional, so neither task can warm-start from a checkpoint of the older tasks.
"""

from .addons import DEPLOYMENT
from .flat_foot_velocity_energy import FlatFootVelocityEnergy
from .velocity_energy_tracking import VelocityEnergyTracking

DEPLOY_WALKING_TASK_ID = "Mjlab-Zbot-6dof-Deploy-Walking"
DEPLOY_FLAT_FOOT_TASK_ID = "Mjlab-Zbot-6dof-Deploy-Flat-Foot-Walking"

# The flat-foot penalty curriculum, stretched to the 1500-iteration run (iteration,
# stance, touchdown, pre-landing weights).  The last row is the full penalty.
DEPLOY_WEIGHT_STAGES = (
    (0, -0.05, -0.2, -0.05),
    (300, -0.15, -0.6, -0.15),
    (600, -0.3, -1.2, -0.3),
    (800, -0.5, -2.0, -0.5),
)
MAX_ITERATIONS = 1500


class DeployWalking(VelocityEnergyTracking):
    task_id = DEPLOY_WALKING_TASK_ID
    experiment_name = "zbot_deploy_walking"
    max_iterations = MAX_ITERATIONS
    addons = DEPLOYMENT


class DeployFlatFoot(FlatFootVelocityEnergy):
    task_id = DEPLOY_FLAT_FOOT_TASK_ID
    experiment_name = "zbot_deploy_flat_foot"
    max_iterations = MAX_ITERATIONS
    addons = DEPLOYMENT
    weight_stages = DEPLOY_WEIGHT_STAGES
