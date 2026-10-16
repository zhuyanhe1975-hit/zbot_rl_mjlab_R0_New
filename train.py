"""Mjlab training CLI plus the Zbot add-on switches (see ``zbot_rl_mjlab/cli.py``).

    python train.py Mjlab-Zbot-6dof-Deploy-Walking --disturbance --domain-randomization \\
        --env.scene.num-envs 4096 --agent.max-iterations 500

Every other argument is mjlab's own and is passed through unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    import zbot_rl_mjlab  # noqa: F401, I001
    from zbot_rl_mjlab.cli import CliError, apply_cli
    from mjlab.scripts.train import main

    try:
        sys.argv[1:] = apply_cli()
    except CliError as error:
        sys.exit(f"error: {error}")
    main()
