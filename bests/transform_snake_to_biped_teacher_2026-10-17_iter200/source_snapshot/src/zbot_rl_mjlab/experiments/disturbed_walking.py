"""Walking experiment with randomized disturbances and domain randomization."""

from ..env_cfg import DISTURBED_WALKING_TASK_ID
from .addons import DISTURBED
from .flat_walking import FlatWalking


class DisturbedWalking(FlatWalking):
    task_id = DISTURBED_WALKING_TASK_ID
    experiment_name = "zbot_disturbed_walking"
    addons = DISTURBED
