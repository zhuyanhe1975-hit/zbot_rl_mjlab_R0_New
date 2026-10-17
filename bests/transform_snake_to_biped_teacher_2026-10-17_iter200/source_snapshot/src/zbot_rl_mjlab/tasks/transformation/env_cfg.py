"""Joint-zeroing and COM descent, without contact sensors or support estimation."""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from ...robot import get_spec, robot_cfg
from . import mdp
from .actions import JointTargetRateActionCfg


@dataclass
class TransformationEnvCfg(ManagerBasedRlEnvCfg):
    success_hold_s: float = 2.0
    joint_tolerance: float = 0.10
    final_max_body_height: float = 0.10
    joint_speed_tolerance: float = 0.10
    base_speed_tolerance: float = 0.02
    max_downward_com_speed: float = 0.30
    workspace_radius: float = 1.5


def full_body_spec():
    spec = get_spec()
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.condim = 3
            geom.friction = (0.6, geom.friction[1], geom.friction[2])
    return spec


def transformation_env_cfg(play=False):
    robot = robot_cfg()
    robot.spec_fn = full_body_spec
    actor = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_positions),
        "joint_vel": ObservationTermCfg(func=common_mdp.joint_vel_rel),
        "commanded_target": ObservationTermCfg(func=mdp.commanded_target),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "com_state": ObservationTermCfg(func=mdp.com_state),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "hold_time": ObservationTermCfg(func=mdp.hold_observation),
    }
    return TransformationEnvCfg(
        seed=42,
        decimation=4,
        episode_length_s=16.0,
        scene=SceneCfg(
            num_envs=1 if play else 1024,
            env_spacing=4.0,
            terrain=TerrainEntityCfg(
                terrain_type="plane",
                geoms=(
                    GeomCfg(
                        geom_names_expr=("terrain$",), friction=(0.6, 0.005, 0.0001)
                    ),
                ),
            ),
            entities={"robot": robot},
            sensors=(),
        ),
        observations={
            name: ObservationGroupCfg(
                actor, concatenate_terms=True, enable_corruption=False
            )
            for name in ("actor", "critic")
        },
        actions={
            "joint_pos": JointTargetRateActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                use_default_offset=False,
            )
        },
        rewards={
            "joint_zero": RewardTermCfg(func=mdp.joint_zero_reward, weight=3.0),
            "com_descent_velocity": RewardTermCfg(
                func=mdp.com_descent_velocity_reward,
                weight=1.0,
                params={"target_speed": 0.03},
            ),
            "joint_symmetry": RewardTermCfg(func=mdp.joint_symmetry_cost, weight=-0.5),
        },
        terminations={
            "update_state": TerminationTermCfg(func=mdp.update_state),
            "unsafe": TerminationTermCfg(func=mdp.unsafe),
            "time_out": TerminationTermCfg(func=common_mdp.time_out, time_out=True),
        },
        events={
            "reset_scene": EventTermCfg(
                func=common_mdp.reset_scene_to_default, mode="reset"
            ),
            "reset_transition": EventTermCfg(
                func=mdp.reset_transformation, mode="reset"
            ),
        },
        metrics={
            "joint_error": MetricsTermCfg(func=mdp.joint_error_metric),
            "com_height": MetricsTermCfg(func=mdp.com_height),
            "success": MetricsTermCfg(func=mdp.success, reduce="last"),
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
            njmax=500,
        ),
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base",
            distance=1.3,
            elevation=-20.0,
            azimuth=135.0,
            lookat=(0.0, 0.0, 0.15),
            origin_type=ViewerConfig.OriginType.WORLD,
        ),
    )
