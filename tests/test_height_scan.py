from mjlab.tasks.registry import load_env_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.env_cfg import WALKING_TASK_ID


def test_height_scan_rays_start_at_the_soles():
    # The sole sites sit on the ground-contact plane of both feet; the foot_1 body
    # origin does not (it is ~54 mm above the sole), so the frames must be sites.
    cfg = load_env_cfg(WALKING_TASK_ID)
    sensor = next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")
    assert [(f.type, f.name) for f in sensor.frame] == [
        ("site", "foot_0_sole"),
        ("site", "foot_1_sole"),
    ]
