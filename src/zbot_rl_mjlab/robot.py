from functools import partial
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
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=15.0,
                    damping=1.5,
                    effort_limit=20.0,
                    armature=0.001,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, -0.06, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0},
        ),
    )
