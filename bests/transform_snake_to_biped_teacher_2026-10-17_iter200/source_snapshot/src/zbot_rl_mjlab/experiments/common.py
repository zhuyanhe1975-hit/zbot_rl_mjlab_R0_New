"""Shared configuration pieces used by walking experiments."""

from pathlib import Path

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef

from .. import mdp


def latest_checkpoint(experiment_name, log_root="logs/rsl_rl"):
    """Newest numbered model_*.pt under log_root/experiment_name."""
    root = Path(log_root) / experiment_name
    candidates = [p for p in root.rglob("model_*.pt") if p.stem[6:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def add_base_disturbance(
    cfg,
    force_range=(-5.0, 5.0),
    torque_range=(-0.5, 0.5),
    duration_s=(0.08, 0.20),
    cooldown_s=(0.8, 2.0),
    disturbed_fraction=0.25,
):
    """Random base force/torque impulses on a fraction of the environments."""
    cfg.events["base_disturbance"] = EventTermCfg(
        func=mdp.PartialBaseImpulse,
        mode="step",
        params={
            "force_range": force_range,
            "torque_range": torque_range,
            "duration_s": duration_s,
            "cooldown_s": cooldown_s,
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            "body_point_offset": (0.0, 0.0, 0.0),
            "disturbed_fraction": disturbed_fraction,
        },
    )
    return cfg


def add_walking_regularizers(cfg):
    """Add shared swing-foot, contact, energy, and acceleration terms."""
    foot_accelerometers = tuple(
        BuiltinSensorCfg(
            name=f"foot_{index}_lin_acc",
            sensor_type="framelinacc",
            obj=ObjRef(type="body", name=f"foot_{index}", entity="robot"),
        )
        for index in range(2)
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *foot_accelerometers)
    cfg.rewards.update(
        {
            "swing_foot_acceleration": RewardTermCfg(
                func=mdp.swing_foot_acceleration_penalty,
                weight=-0.1,
                params={
                    "acceleration_sensors": tuple(
                        sensor.prefixed_name for sensor in foot_accelerometers
                    ),
                    "acceleration_scale": 50.0,
                },
            ),
            "swing_foot_excess_height": RewardTermCfg(
                func=mdp.swing_foot_excess_height_penalty,
                weight=-20.0,
                params={"max_height": 0.05, "height_scale": 0.05},
            ),
            "joint_energy": RewardTermCfg(
                func=mdp.joint_energy_penalty,
                weight=-0.2,
                params={"energy_scale": 10.0},
            ),
            "feet_self_collision": RewardTermCfg(
                func=mdp.foot_self_collision_penalty,
                weight=-0.5,
                params={"sensor_name": "feet_self_contact", "scale": 10.0},
            ),
            "foot_nonparallel_ground": RewardTermCfg(
                func=mdp.foot_nonparallel_ground_penalty,
                weight=-10.0,
                params={"sensor_name": "feet_ground_contact", "std": 0.05},
            ),
            "joint_acceleration": RewardTermCfg(
                func=mdp.joint_acceleration_penalty, weight=-0.001
            ),
        }
    )
    return cfg


def add_domain_randomization(
    cfg,
    friction_range=(0.3, 1.2),
    encoder_bias_range=(-0.015, 0.015),
    base_com_ranges=None,
):
    """Startup randomization of foot friction, encoder zero offset and base COM."""
    if base_com_ranges is None:
        base_com_ranges = {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}
    cfg.events.update(
        {
            "randomize_foot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", geom_names=("a1_geom", "b6_geom")
                    ),
                    "operation": "abs",
                    "ranges": friction_range,
                    "shared_random": True,
                },
            ),
            "randomize_encoder_bias": EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "bias_range": encoder_bias_range,
                },
            ),
            "randomize_base_com": EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                    "operation": "add",
                    "ranges": base_com_ranges,
                },
            ),
        }
    )
    return cfg
