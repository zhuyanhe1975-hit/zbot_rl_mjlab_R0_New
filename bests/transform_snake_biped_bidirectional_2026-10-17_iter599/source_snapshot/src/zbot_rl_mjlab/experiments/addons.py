"""Optional add-ons that make any experiment robust or deployment-oriented.

An add-on is a small object with parameters.  An experiment lists the ones it wants in its
``addons`` attribute and they are applied after every inherited reward and environment edit,
so the same add-on works on any experiment:

    class DisturbedWalking(FlatWalking):
        addons = DISTURBED                       # a preset: disturbance + randomization

    class MyOmni(OmniVelocityEnergy):
        addons = (BaseDisturbance(disturbed_fraction=1.0), ActionDelay(max_lag=3))

Add-ons accumulate down the inheritance chain.  If a subclass lists an add-on of a type its
parent already has, the subclass's one replaces it, so a child can retune a parent's
disturbance without repeating the rest.  Each add-on has a fixed ``order``, so the result
never depends on how they are listed.  The order matters in one place: ``MassRandomization``
rewrites the body COM from its defaults, so it runs before ``DomainRandomization``, which
shifts the base COM.

Training-only effects (observation noise, action delay) are switched off in the play
configuration, like ``enable_corruption`` does for the actor observations.
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    UniformNoiseCfg,
)

from ..actions import DelayedJointPositionActionCfg
from .common import add_base_disturbance, add_domain_randomization

# Observation terms a real robot cannot measure directly; only the critic may see them.
PRIVILEGED_TERMS = ("base_lin_vel", "yaw_error")

# Per-step Gaussian noise std on actor observations.
NOISE_STD = {
    "base_ang_vel": 0.05,  # rad/s, gyro
    "projected_gravity": 0.02,  # unit vector components, accelerometer / filter
    "joint_pos": 0.01,  # rad, encoder
    "joint_vel": 0.15,  # rad/s, differentiated encoder
}
# Bias drawn once per episode, uniform in +/- value.
NOISE_BIAS = {
    "base_ang_vel": 0.02,  # rad/s, gyro zero offset
    "projected_gravity": 0.05,  # ~3 degrees of IMU tilt on each axis
}

MAX_ACTION_LAG = 2  # control steps (20 ms each)
GAIN_SCALE = (0.7, 1.3)  # PD stiffness and damping, per environment per episode
MASS_ALPHA = (-0.07, 0.07)  # mass and inertia scale by exp(2 * alpha), about +/-15 %


@dataclass(frozen=True)
class Addon:
    """One optional change to an experiment's environment configuration."""

    order: ClassVar[int]
    """Position in the fixed application order (lower first)."""

    def apply(self, cfg, play):
        raise NotImplementedError


@dataclass(frozen=True)
class BlindActor(Addon):
    """Hide measurements a real robot lacks from the actor; the critic keeps them."""

    order: ClassVar[int] = 10
    hidden: tuple[str, ...] = PRIVILEGED_TERMS

    def apply(self, cfg, play):
        actor = cfg.observations["actor"]
        actor.terms = {
            name: term for name, term in actor.terms.items() if name not in self.hidden
        }


@dataclass(frozen=True)
class ObservationNoise(Addon):
    """Gaussian noise on actor observations, plus a per-episode bias for some terms."""

    order: ClassVar[int] = 20
    std: dict[str, float] = field(default_factory=lambda: dict(NOISE_STD))
    bias: dict[str, float] = field(default_factory=lambda: dict(NOISE_BIAS))

    def apply(self, cfg, play):
        actor = cfg.observations["actor"]
        known = set(actor.terms) | set(cfg.observations["critic"].terms)
        unknown = (set(self.std) | set(self.bias)) - known
        if unknown:
            raise KeyError(f"no observation term named {sorted(unknown)}")
        if set(self.bias) - set(self.std):
            raise ValueError("bias is only added on top of terms that also have std")
        actor.terms = {
            name: self._noisy(term, name) if name in self.std else term
            for name, term in actor.terms.items()
        }

    def _noisy(self, term, name):
        noise = GaussianNoiseCfg(std=self.std[name])
        if name in self.bias:
            noise = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=noise,
                bias_noise_cfg=UniformNoiseCfg(
                    n_min=-self.bias[name], n_max=self.bias[name]
                ),
            )
        return replace(term, noise=noise)


@dataclass(frozen=True)
class ActionDelay(Addon):
    """Joint position targets reach the actuators 0..max_lag control steps late."""

    order: ClassVar[int] = 30
    max_lag: int = MAX_ACTION_LAG

    def apply(self, cfg, play):
        joint_pos = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = DelayedJointPositionActionCfg(
            entity_name=joint_pos.entity_name,
            actuator_names=joint_pos.actuator_names,
            scale=joint_pos.scale,
            offset=joint_pos.offset,
            use_default_offset=joint_pos.use_default_offset,
            clip=joint_pos.clip,
            max_lag=0 if play else self.max_lag,
        )


@dataclass(frozen=True)
class MassRandomization(Addon):
    """Link mass and inertia scaled together (physically consistent) at startup."""

    order: ClassVar[int] = 40
    alpha_range: tuple[float, float] = MASS_ALPHA

    def apply(self, cfg, play):
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={"alpha_range": self.alpha_range},
        )


@dataclass(frozen=True)
class GainRandomization(Addon):
    """PD stiffness and damping scaled per environment at every reset."""

    order: ClassVar[int] = 45
    scale: tuple[float, float] = GAIN_SCALE

    def apply(self, cfg, play):
        cfg.events["randomize_pd_gains"] = EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "kp_range": self.scale,
                "kd_range": self.scale,
                "operation": "scale",
            },
        )


@dataclass(frozen=True)
class BaseDisturbance(Addon):
    """Random force and torque impulses on the base, for a fraction of the environments."""

    order: ClassVar[int] = 50
    force_range: tuple[float, float] = (-5.0, 5.0)  # N
    torque_range: tuple[float, float] = (-0.5, 0.5)  # N*m
    duration_s: tuple[float, float] = (0.08, 0.20)
    cooldown_s: tuple[float, float] = (0.8, 2.0)
    disturbed_fraction: float = 0.25

    def apply(self, cfg, play):
        add_base_disturbance(
            cfg,
            force_range=self.force_range,
            torque_range=self.torque_range,
            duration_s=self.duration_s,
            cooldown_s=self.cooldown_s,
            disturbed_fraction=self.disturbed_fraction,
        )


@dataclass(frozen=True)
class DomainRandomization(Addon):
    """Startup randomization of foot friction, encoder zero offset and base COM."""

    order: ClassVar[int] = 60
    friction_range: tuple[float, float] = (0.3, 1.2)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)  # rad
    base_com_ranges: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {
            0: (-0.025, 0.025),
            1: (-0.025, 0.025),
            2: (-0.03, 0.03),
        }
    )  # m, along x, y, z of the base body

    def apply(self, cfg, play):
        add_domain_randomization(
            cfg,
            friction_range=self.friction_range,
            encoder_bias_range=self.encoder_bias_range,
            base_com_ranges=self.base_com_ranges,
        )


# Presets.
DISTURBED = (BaseDisturbance(), DomainRandomization())
"""Random pushes and startup randomization, the setup of the ``*-Disturbed-*`` tasks."""

DEPLOYMENT = (
    BlindActor(),
    ObservationNoise(),
    ActionDelay(),
    MassRandomization(),
    GainRandomization(),
    BaseDisturbance(),
    DomainRandomization(),
)
"""Everything that narrows the sim-to-real gap: a blind noisy actor, control latency and
full randomization.  The actor input shrinks, so such a task cannot warm-start from a
checkpoint of the same experiment without it."""
