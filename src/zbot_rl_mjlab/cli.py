"""Command-line switches that add the optional add-ons of ``experiments/addons.py`` to a task.

    python train.py Mjlab-Zbot-6dof-Deploy-Walking --disturbance --domain-randomization

``apply_cli(argv)`` removes these switches from ``argv`` and, for each task they name, replaces
the registered task with the same experiment plus the chosen add-ons.  The remaining
arguments (``--env.scene.num-envs 4096``, ``--agent.max-iterations 500`` ...) are returned
unchanged for mjlab's own parser.  Both ``train.py`` and ``play.py`` use it, so the same
switches build the same environment in training and in playback.

Switches (each adds one add-on with its default parameters):

    --disturbance             base force / torque pushes
    --domain-randomization    foot friction, encoder offset, base COM
    --mass-randomization      link mass and inertia
    --gain-randomization      PD gains
    --observation-noise       actor observation noise and bias
    --action-delay [N]        position targets arrive 0..N steps late (default 2; training only)
    --blind-actor             hide base_lin_vel and yaw_error from the actor
    --deployment              all of the above together

Tuning values with the form ``--option=value`` (an equals sign is required):

    --disturbance-fraction=0.5   --disturbance-force=8   --disturbance-torque=0.8
    --friction-range=0.4,1.0     --gain-scale=0.8,1.2    --mass-alpha=0.05
    --noise-scale=2.0 (multiplies every noise std and bias)

A switch that changes the actor's input (``--blind-actor``, ``--deployment``) makes a
checkpoint trained without it unusable; the task name is not changed, so a run trained with
switches must be replayed with the same switches.
"""

import sys

from mjlab.tasks import registry as mjlab_registry

from .experiments import addons as ad
from .experiments.base import variant

SWITCHES = {
    "--disturbance",
    "--domain-randomization",
    "--mass-randomization",
    "--gain-randomization",
    "--observation-noise",
    "--action-delay",
    "--blind-actor",
    "--deployment",
}
OPTIONS = {
    "--disturbance-fraction",
    "--disturbance-force",
    "--disturbance-torque",
    "--friction-range",
    "--gain-scale",
    "--mass-alpha",
    "--noise-scale",
}


class CliError(ValueError):
    pass


def _pair(text, name):
    try:
        low, high = (float(x) for x in text.split(","))
    except ValueError:
        raise CliError(f"{name} needs two numbers like 0.4,1.0, got {text!r}") from None
    if low >= high:
        raise CliError(f"{name}: the first number must be smaller, got {text!r}")
    return low, high


def _number(text, name, minimum=0.0):
    try:
        value = float(text)
    except ValueError:
        raise CliError(f"{name} needs a number, got {text!r}") from None
    if value < minimum:
        raise CliError(f"{name} must be at least {minimum}, got {value}")
    return value


def split_argv(argv):
    """Separate the add-on switches from the arguments meant for mjlab.

    Returns ``(rest, switches, options)``; ``switches`` maps a switch to its value (``True``
    or, for ``--action-delay``, an integer), ``options`` maps ``--name`` to its text.
    """
    rest, switches, options = [], {}, {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, equals, value = arg.partition("=")
        if arg in SWITCHES:
            if arg == "--action-delay" and i + 1 < len(argv) and argv[i + 1].isdigit():
                switches[arg] = int(argv[i + 1])
                i += 1
            else:
                switches[arg] = True
        elif name in OPTIONS:
            if not equals:
                raise CliError(f"{name} needs a value: write {name}=VALUE")
            options[name] = value
        elif arg.startswith(
            ("--disturbance", "--domain", "--mass-", "--gain-", "--noise")
        ):
            known = sorted(SWITCHES | OPTIONS)
            raise CliError(f"unknown option {arg!r}; the add-on options are {known}")
        else:
            rest.append(arg)
        i += 1
    return rest, switches, options


def build_addons(switches, options):
    """Add-on objects for the given switches and tuning options, in no particular order."""
    selected = set(switches)
    if "--deployment" in selected:
        selected |= SWITCHES - {"--deployment"}
    scale = _number(options.get("--noise-scale", "1"), "--noise-scale")
    result = []
    if "--disturbance" in selected:
        fraction = _number(
            options.get("--disturbance-fraction", "0.25"), "--disturbance-fraction"
        )
        if fraction > 1.0:
            raise CliError(f"--disturbance-fraction must be at most 1, got {fraction}")
        force = _number(options.get("--disturbance-force", "5"), "--disturbance-force")
        torque = _number(
            options.get("--disturbance-torque", "0.5"), "--disturbance-torque"
        )
        result.append(
            ad.BaseDisturbance(
                force_range=(-force, force),
                torque_range=(-torque, torque),
                disturbed_fraction=fraction,
            )
        )
    if "--domain-randomization" in selected:
        friction = _pair(options.get("--friction-range", "0.3,1.2"), "--friction-range")
        result.append(ad.DomainRandomization(friction_range=friction))
    if "--mass-randomization" in selected:
        alpha = _number(options.get("--mass-alpha", "0.07"), "--mass-alpha")
        result.append(ad.MassRandomization(alpha_range=(-alpha, alpha)))
    if "--gain-randomization" in selected:
        result.append(
            ad.GainRandomization(
                scale=_pair(options.get("--gain-scale", "0.7,1.3"), "--gain-scale")
            )
        )
    if "--observation-noise" in selected:
        result.append(
            ad.ObservationNoise(
                std={k: v * scale for k, v in ad.NOISE_STD.items()},
                bias={k: v * scale for k, v in ad.NOISE_BIAS.items()},
            )
        )
    if "--action-delay" in selected:
        lag = switches.get("--action-delay", ad.MAX_ACTION_LAG)
        result.append(ad.ActionDelay(max_lag=ad.MAX_ACTION_LAG if lag is True else lag))
    if "--blind-actor" in selected:
        result.append(ad.BlindActor())
    return result


def extended_task_ids():
    """Task ids of the Zbot experiments, so a switch is only applied to one of those."""
    return {cls.task_id: cls for cls in _experiments()}


def _experiments():
    from . import EXPERIMENTS

    return EXPERIMENTS


def apply_cli(argv=None):
    """Apply the add-on switches in ``argv`` and return ``argv`` without them.

    The first argument that names a registered Zbot task decides which task is rebuilt.
    With no switches nothing is touched, so the task is exactly the registered one.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    rest, switches, options = split_argv(argv)
    if not switches and not options:
        return rest
    if options and not switches:
        raise CliError(f"{sorted(options)} only tune a switch; add the switch too")
    tasks = extended_task_ids()
    chosen = next((a for a in rest if a in tasks), None)
    if chosen is None:
        raise CliError(
            "add-on switches need a Zbot task name as an argument, e.g. "
            "Mjlab-Zbot-6dof-Deploy-Walking"
        )
    base = tasks[chosen]
    new = variant(base, chosen, base.experiment_name, *build_addons(switches, options))
    # Replace the registered task with the extended one under the same task id.
    mjlab_registry._REGISTRY.pop(chosen, None)
    new().register()
    print(
        f"[INFO] {chosen}: add-ons {[type(a).__name__ for a in new().collected_addons()]}"
    )
    return rest
