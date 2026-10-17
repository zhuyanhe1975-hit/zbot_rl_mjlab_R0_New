#!/usr/bin/env bash
# Randomly initialized actor/critic; no pretrained checkpoint is loaded by default.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/python.sh" train.py \
  Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust "$@"
