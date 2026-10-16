"""Every checkpoint archived in ``bests/`` must still be usable with the current code.

Cheap, CPU-only checks, run on every archived model against the task it is replayed with
(``bests/test.sh`` names that task).  They catch the changes that make an old checkpoint
unreplayable: a different set or order of actor observation terms, a different observation
or action size, another control rate or action scale, or another actuator.  They do not run
the simulator; ``python fall_check.py <TASK> <checkpoint>`` does, and should be run after any
change to physics.
"""

import fnmatch
import re
from pathlib import Path

import pytest
import torch
import yaml
from mjlab.tasks.registry import list_tasks, load_env_cfg

import zbot_rl_mjlab  # noqa: F401  (registers the tasks)
from zbot_rl_mjlab.robot import ARMATURE, DAMPING, SIM_EFFORT_LIMIT, STIFFNESS

BESTS = Path(__file__).resolve().parents[1] / "bests"
ENTRIES = sorted(p for p in BESTS.iterdir() if p.is_dir())

# Size of each actor observation term.
TERM_DIMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "yaw_error": 1,
    "joint_pos": 6,
    "joint_vel": 6,
    "actions": 6,
    "phase": 2,
    "frequency": 1,
    "target_speed": 1,
    "velocity_command": 3,
}
ACTION_DIM = 6


class _Loose(yaml.SafeLoader):
    """Reads the saved env.yaml without importing the python objects it names."""


def _sequence(loader, node):
    return loader.construct_sequence(node, deep=True)


def _python_object(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return suffix
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)


_Loose.add_constructor("tag:yaml.org,2002:python/tuple", _sequence)
_Loose.add_multi_constructor("tag:yaml.org,2002:python/", _python_object)


def archived_env(entry):
    return yaml.load((entry / "env.yaml").read_text(), Loader=_Loose)


def task_of(entry):
    """The task bests/test.sh replays this directory with (first matching case wins)."""
    cases = re.findall(
        r'^[ \t]*([^)\s][^)\n]*)\)[ \t]*echo[ \t]*"(Mjlab-[^"]+)"',
        (BESTS / "test.sh").read_text(),
        re.MULTILINE,
    )
    for patterns, task in cases:
        if any(fnmatch.fnmatchcase(entry.name, p.strip()) for p in patterns.split("|")):
            return task
    raise AssertionError(f"bests/test.sh has no task for {entry.name}")


def checkpoint(entry):
    models = sorted(entry.glob("model_*.pt"))
    assert len(models) == 1, f"{entry.name} should hold exactly one model_*.pt"
    return torch.load(models[0], map_location="cpu", weights_only=False)


def live_actor_terms(task):
    return list(load_env_cfg(task, play=True).observations["actor"].terms)


def test_every_actor_term_of_every_task_has_a_known_size():
    # A new observation term must be given a size here, or the checks below are blind to it.
    for task in (t for t in list_tasks() if t.startswith("Mjlab-Zbot")):
        unknown = set(live_actor_terms(task)) - set(TERM_DIMS)
        assert not unknown, f"{task}: add {sorted(unknown)} to TERM_DIMS"


def test_there_is_something_to_check():
    assert ENTRIES, "bests/ is empty"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_model_matches_the_current_task(entry):
    task = task_of(entry)
    live = load_env_cfg(task, play=True)
    saved = archived_env(entry)

    saved_terms = list(saved["observations"]["actor"]["terms"])
    assert saved_terms == live_actor_terms(task), (
        f"{entry.name}: the actor's observation terms changed since it was trained"
    )

    state = checkpoint(entry)["actor_state_dict"]
    obs_dim = sum(TERM_DIMS[name] for name in saved_terms)
    assert state["obs_normalizer._mean"].shape[-1] == obs_dim
    last = max(int(k.split(".")[1]) for k in state if k.startswith("mlp."))
    assert state[f"mlp.{last}.weight"].shape[0] == ACTION_DIM
    assert state["mlp.0.weight"].shape[1] == obs_dim

    if "frequency" in saved_terms:
        # The frequency observation is normalized by this range, so a different range
        # feeds the policy a different input even though every shape still matches.
        assert (saved["step_frequency_min"], saved["step_frequency_max"]) == (
            live.step_frequency_min,
            live.step_frequency_max,
        ), f"{entry.name}: the commanded stepping-frequency range changed"
    assert saved["decimation"] == live.decimation
    assert saved["sim"]["mujoco"]["timestep"] == live.sim.mujoco.timestep
    action = saved["actions"]["joint_pos"]
    assert action["scale"] == live.actions["joint_pos"].scale
    assert action["use_default_offset"] == live.actions["joint_pos"].use_default_offset


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_model_was_trained_with_the_current_actuator(entry):
    (actuator,) = archived_env(entry)["scene"]["entities"]["robot"]["articulation"][
        "actuators"
    ]
    assert actuator["stiffness"] == STIFFNESS
    assert actuator["damping"] == DAMPING
    assert actuator["effort_limit"] == SIM_EFFORT_LIMIT
    assert actuator["armature"] == ARMATURE
