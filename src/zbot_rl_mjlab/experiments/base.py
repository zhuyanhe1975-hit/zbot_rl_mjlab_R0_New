"""Base class of every Zbot training experiment.

An experiment is a class.  The shared part (the Zbot environment and PPO configuration from
``env_cfg.py``) lives here; a subclass states only what it changes relative to its parent:

* ``reward_edit()``: the rewards it removes, re-weights, re-parameterizes, adds or replaces
  (see :class:`RewardEdit`);
* ``edit(cfg, play)``: anything else in the environment (commands, observations, events,
  curriculum, actions...);
* class attributes ``task_id``, ``experiment_name``, ``train_envs``, ``step_frequency``,
  ``max_iterations``;
* ``addons``: optional add-ons from ``addons.py`` (disturbance, domain randomization,
  observation noise, action delay, ...), which work on any experiment.

Building a configuration walks the inheritance chain from the most basic class to the most
derived one and applies each class's own ``reward_edit`` then ``edit``, so a subclass sees
exactly what its parents produced.  The add-ons of the whole chain are applied last, in a
fixed order.  A subclass never repeats its parents' edits.

    class VelocityEnergyTracking(VelocityTracking):
        task_id = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"
        experiment_name = "zbot_velocity_energy_tracking"

        def reward_edit(self):
            return RewardEdit(
                remove=("frequency_tracking",),
                weights={"lateral_velocity": -0.5},
                terms={"joint_energy": RewardTermCfg(...)},
            )
"""

from dataclasses import dataclass, field

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from ..env_cfg import env_cfg, ppo_cfg


@dataclass
class RewardEdit:
    """Changes one experiment makes to the rewards it inherits, applied in this order."""

    remove: tuple[str, ...] = ()
    """Reward names to delete."""
    weights: dict[str, float] = field(default_factory=dict)
    """New weight for an inherited reward."""
    params: dict[str, dict] = field(default_factory=dict)
    """Keyword arguments to merge into an inherited reward's parameters."""
    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Rewards to add; a name that already exists is replaced in its place."""

    def apply(self, rewards):
        for name in self.remove:
            self._require(rewards, name, "remove")
            del rewards[name]
        for name, weight in self.weights.items():
            self._require(rewards, name, "re-weight")
            rewards[name].weight = weight
        for name, params in self.params.items():
            self._require(rewards, name, "re-parameterize")
            rewards[name].params.update(params)
        rewards.update(self.terms)

    @staticmethod
    def _require(rewards, name, action):
        if name not in rewards:
            raise KeyError(
                f"cannot {action} reward {name!r}: not inherited (have {sorted(rewards)})"
            )


class Experiment:
    """One registered training task: the shared Zbot setup plus a subclass's edits."""

    task_id: str
    experiment_name: str
    train_envs: int | None = None
    """Parallel environments for training (playback always uses 1); None keeps env_cfg's."""
    step_frequency: tuple[float, float] | None = None
    """Commanded stepping frequency range in Hz; None keeps env_cfg's."""
    max_iterations: int | None = None
    """PPO iterations; None keeps the shared default."""
    addons: tuple = ()
    """Optional add-ons (``addons.py``) this class adds on top of its parents' ones."""

    def collected_addons(self):
        """Add-ons of this class and its parents, one per type, in their fixed order.

        A class listing an add-on of a type a parent already has replaces the parent's.
        """
        chosen = {}
        for cls in reversed(type(self).__mro__):
            for addon in cls.__dict__.get("addons", ()):
                chosen[type(addon)] = addon
        return sorted(chosen.values(), key=lambda addon: addon.order)

    def build_env(self, play=False):
        cfg = env_cfg(play=play)
        if self.train_envs is not None and not play:
            cfg.scene.num_envs = self.train_envs
        if self.step_frequency is not None:
            cfg.step_frequency_min, cfg.step_frequency_max = self.step_frequency
        # Base first: each class applies only its own hooks (not inherited ones).
        for cls in reversed(type(self).__mro__):
            reward_edit = cls.__dict__.get("reward_edit")
            if reward_edit is not None:
                reward_edit(self).apply(cfg.rewards)
            edit = cls.__dict__.get("edit")
            if edit is not None:
                edit(self, cfg, play)
        # Add-ons last, so they act on the finished experiment whatever it is.
        for addon in self.collected_addons():
            addon.apply(cfg, play)
        return cfg

    def build_ppo(self):
        cfg = ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def register(self):
        register_mjlab_task(
            task_id=self.task_id,
            env_cfg=self.build_env(),
            play_env_cfg=self.build_env(play=True),
            rl_cfg=self.build_ppo(),
        )


def variant(base, task_id, experiment_name, *addons, **attributes):
    """A new experiment: ``base`` plus add-ons, under its own task id and experiment name.

    Any experiment can become a disturbed or deployment-oriented one without a new class
    or file; ``attributes`` overrides class attributes such as ``max_iterations``.

        variant(OmniVelocityEnergy, "Mjlab-Zbot-6dof-Omni-Disturbed-Walking",
                "zbot_omni_disturbed", *DISTURBED)
    """
    parts = "+".join(type(addon).__name__ for addon in addons)
    return type(
        f"{base.__name__}+{parts}" if parts else base.__name__,
        (base,),
        {
            "task_id": task_id,
            "experiment_name": experiment_name,
            "addons": tuple(addons),
            **attributes,
        },
    )
