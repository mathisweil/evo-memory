#!/bin/bash
# Background launcher: starts the dispatcher fully detached and prints PID.
# Usage: bash scripts/_claude_launch.sh <idx> <label>
set -u
IDX="${1:?usage: $0 <idx> <label>}"
LABEL="${2:?usage: $0 <idx> <label>}"
PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
HOST=$(hostname -s)
LAUNCH_LOG="$PROJ/logs/launch_${LABEL}_${HOST}.out"
mkdir -p "$PROJ/logs"
nohup bash "$PROJ/scripts/_claude_dispatch_eval.sh" "$IDX" \
    > "$LAUNCH_LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "PID=$PID"
echo "LAUNCH_LOG=$LAUNCH_LOG"
