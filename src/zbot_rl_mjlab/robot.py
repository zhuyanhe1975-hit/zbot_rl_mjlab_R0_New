import math
from functools import partial
from itertools import pairwise
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ASSET_PATH = (
    # Path(__file__).resolve().parents[2] / "assets/zbot_6dof/robot_geometry_inertia.xml"
    Path(__file__).resolve().parents[2] / "assets/zbot_6/ZBOT6.xml"
)
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_JOINT_POS = (0.312, 0.837, -2.02, 2.02, -0.837, -0.312)

# RobStride RS00 (Lingzu Times) integrated quasi-direct-drive joint, output shaft.
# Source: RS00使用说明书 (260713), manual pages: 1.2-1.4 (pp.6-9), 4.1 (pp.31-32),
# 4.3.1 (p.41).  See docs/rs00-actuator.md for what is and is not modeled.
RS00_PEAK_TORQUE = 14.0  # N*m, "peak load"; NOT applied in the simulator, see below
RS00_RATED_TORQUE = 5.0  # N*m, continuous, measured at 100 rpm on a 90x85 mm heatsink
RS00_NO_LOAD_SPEED = 315.0 * 2.0 * math.pi / 60.0  # rad/s, 315 rpm +/-10 %, about 33.0
RS00_KP_RANGE = (0.0, 500.0)  # N*m/rad, MIT-mode command encoding (p.32)
RS00_KD_RANGE = (0.0, 5.0)  # N*m*s/rad, MIT-mode command encoding (p.32)

# T-N curve at 48 V (manual p.7, read off the figure, +/-0.5 N*m): (rpm, N*m).  The
# simulator does NOT apply it (see docs/rs00-actuator.md); it is the reference that
# actuator_usage.py compares the joint speeds and torques a policy asks for against.
RS00_TN_CURVE = (
    (100.0, 14.0),
    (150.0, 12.0),
    (200.0, 9.5),
    (250.0, 6.5),
    (280.0, 4.0),
    (300.0, 2.0),
    (315.0, 0.5),
)
RS00_CORNER_SPEED = 100.0 * 2.0 * math.pi / 60.0  # rad/s, flat peak torque up to here

# Rotating overload limits (manual p.8, same heatsink): the longest a load can be held.
# (seconds, N*m); 5 N*m is continuous.
RS00_OVERLOAD = (
    (3.0, 14.0),
    (7.0, 13.0),
    (10.0, 12.0),
    (20.0, 10.0),
    (53.0, 8.0),
    (94.0, 7.0),
    (285.0, 6.0),
)

# The gains are the MIT-mode Kp and Kd the deployment sends, and are inside the MIT range.
# The rotor inertia is not given in the manual, so the armature, and any gearbox friction
# or backlash, remain unverified.
STIFFNESS = 15.0  # N*m/rad
DAMPING = 1.5  # N*m*s/rad
ARMATURE = 0.001
# What the simulator actually allows.  This is the value every checkpoint so far was
# trained with; the hardware peak is RS00_PEAK_TORQUE (14), so simulated gaits may use
# torque the real motor cannot give.  actuator_usage.py measures how much.
SIM_EFFORT_LIMIT = 20.0


def available_torque(speed):
    """Torque (N*m) the T-N curve allows at this joint speed (rad/s), by interpolation.

    Flat at the peak up to 100 rpm, then the datasheet points, which end at 0.5 N*m at
    the 315 rpm no-load speed; zero beyond the last point.  Reference for checks only;
    the simulator does not apply it.
    """
    rpm = abs(speed) * 60.0 / (2.0 * math.pi)
    if rpm <= RS00_TN_CURVE[0][0]:
        return RS00_PEAK_TORQUE
    for (r0, t0), (r1, t1) in pairwise(RS00_TN_CURVE):
        if rpm <= r1:
            return t0 + (t1 - t0) * (rpm - r0) / (r1 - r0)
    return 0.0


def get_spec(contact_dimension=3):
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # Match G1's collision profile: only feet use 3D frictional
            # contacts; internal/self-collision geoms use condim=1.
            is_foot = geom.name.startswith(("foot_0_", "foot_1_"))
            geom.condim = contact_dimension if is_foot else 1
            if is_foot:
                geom.friction = (0.6, geom.friction[1], geom.friction[2])
    return spec


def robot_cfg():
    return EntityCfg(
        spec_fn=partial(get_spec),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # Kept at the original actuator so every checkpoint trained so far
                # still replays: implicit-integration position servo, 15 / 1.5 gains,
                # 20 N*m limit.  The RS00 constants above are NOT applied here (the
                # real peak is 14 N*m); a switch to a datasheet-faithful actuator broke
                # replay of existing models and is not made until they are retrained.
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=STIFFNESS,
                    damping=DAMPING,
                    effort_limit=SIM_EFFORT_LIMIT,
                    armature=ARMATURE,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, -0.06, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0},
        ),
    )
