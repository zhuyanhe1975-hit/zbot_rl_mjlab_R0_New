"""Register local Zbot tasks and invoke the MJLab training CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    import zbot_rl_mjlab  # noqa: F401, I001
    from mjlab.scripts.train import main

    main()
