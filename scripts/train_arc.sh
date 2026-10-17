#!/usr/bin/env bash
# Same-branch shallow arc; defaults come from SnakeToBipedArc (8192 envs / 4000 iterations).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/python.sh" train.py \
  Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc "$@"
