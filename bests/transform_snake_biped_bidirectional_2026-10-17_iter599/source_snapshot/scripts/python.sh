#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_ROOT="${MJLAB_ROOT:-/home/yhzhu/AI/mjlab}"
PYTHON_BIN="${ZBOT_PYTHON:-$MJLAB_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/zbot-cache}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$@"
