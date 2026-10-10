#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${ZBOT_LOG_ROOT:-$PROJECT_DIR/logs/rsl_rl}"
TASK_ID="${ZBOT_TASK_ID:-Mjlab-Zbot-6dof-Periodic-Stepping}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
NUM_ENVS="${ZBOT_NUM_ENVS:-1}"
VIEWER="${VIEWER:-native}"
if [[ "${1:-}" == "--task" ]]; then
  case "${2:-}" in
    stepping) TASK_ID="Mjlab-Zbot-6dof-Periodic-Stepping" ;;
    walking) TASK_ID="Mjlab-Zbot-6dof-Walking" ;;
    "Mjlab-Zbot-6dof-Periodic-Stepping"|"Mjlab-Zbot-6dof-Walking") TASK_ID="$2" ;;
    *) echo "Unknown task '$2' (choose stepping or walking)" >&2; exit 2 ;;
  esac
  shift 2
fi
if [[ "$TASK_ID" == "Mjlab-Zbot-6dof-Walking" ]]; then
  EXPERIMENT="zbot_walking"
else
  EXPERIMENT="zbot_periodic_stepping"
fi
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_ENVS="$1"
  shift
elif [[ "${1:-}" == "--num-envs" ]]; then
  NUM_ENVS="${2:-}"
  shift 2
fi
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "num envs must be a positive integer" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: ./preview_train.sh [--task stepping|walking] [NUM_ENVS]
       ./preview_train.sh [--task TASK] --num-envs NUM_ENVS

Opens a Viewer for the newest checkpoint and refreshes it as training saves
new checkpoints. Each environment samples its own stepping frequency.
HELP
  exit 0
fi
mkdir -p "$PROJECT_DIR/outputs"
last_checkpoint=""
viewer_pid=""
seen_training=0
training_running() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill -0 "$TRAIN_PID" 2>/dev/null
  else
    pgrep -f 'mjlab\.scripts\.train|zbot_rl_mjlab.*train' >/dev/null 2>&1
  fi
}
latest_checkpoint() {
  find "$LOG_ROOT/$EXPERIMENT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2- || true
}
launch_viewer() {
  local checkpoint="$1"
  if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
    echo "Refreshing viewer: $checkpoint"
    kill "$viewer_pid" 2>/dev/null || true
    wait "$viewer_pid" 2>/dev/null || true
  else
    echo "Opening viewer: $checkpoint"
  fi
  ZBOT_TASK_ID="$TASK_ID" ZBOT_LOG_ROOT="$LOG_ROOT" \
    "$PROJECT_DIR/scripts/python.sh" "$PROJECT_DIR/play.py" "$TASK_ID" --env.scene.num-envs "$NUM_ENVS" --checkpoint-file "$checkpoint" --viewer "$VIEWER" \
    >"$PROJECT_DIR/outputs/preview-viewer.log" 2>&1 &
  viewer_pid=$!
}
echo "Watching $LOG_ROOT/$EXPERIMENT for $TASK_ID checkpoints"
while true; do
  running=0
  if training_running; then
    seen_training=1
    running=1
  fi
  checkpoint="$(latest_checkpoint)"
  if [[ -n "$checkpoint" && "$checkpoint" != "$last_checkpoint" && -s "$checkpoint" ]]; then
    before="$(stat -c '%s' "$checkpoint")"
    sleep 1
    after="$(stat -c '%s' "$checkpoint")"
    if [[ "$before" == "$after" ]]; then
      launch_viewer "$checkpoint"
      last_checkpoint="$checkpoint"
    fi
  fi
  if (( seen_training )) && (( ! running )) && [[ -z "$checkpoint" || "$checkpoint" == "$last_checkpoint" ]]; then
    echo "Training ended; latest checkpoint has been handed to the viewer."
    if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
      wait "$viewer_pid" 2>/dev/null || true
    fi
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done
