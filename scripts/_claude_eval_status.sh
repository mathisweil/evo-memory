#!/bin/bash
# Check liveness + GPU usage of the 8 dispatched evals.
# Usage: bash scripts/_claude_eval_status.sh
#
# Reads logs/_claude_dispatch_status.txt to find host/pid pairs.
# For each, ssh's in and checks whether the dispatcher pid is alive
# and prints a one-line GPU summary.

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
STATUS=$PROJ/logs/_claude_dispatch_status.txt

if [ ! -f "$STATUS" ]; then
    echo "no status file at $STATUS"
    exit 1
fi

printf '%-14s %-22s %-9s %-7s %s\n' HOST LABEL PID STATE GPU
printf '%-14s %-22s %-9s %-7s %s\n' ---- ----- --- ----- ---

# host  idx  label  pid  launcher_log  err
grep -v '^#' "$STATUS" | while IFS=$'\t' read -r host idx label pid rest; do
    [ -z "$host" ] && continue
    info=$(ssh -n -o BatchMode=yes -o ConnectTimeout=4 "$host" \
        "kill -0 $pid 2>/dev/null && echo ALIVE || echo DEAD; \
         nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
        2>/dev/null)
    state=$(echo "$info" | sed -n '1p')
    gpu=$(echo "$info" | sed -n '2p')
    [ -z "$state" ] && state=SSH_ERR
    printf '%-14s %-22s %-9s %-7s %s\n' "$host" "$label" "$pid" "$state" "$gpu"
done

echo
echo "Logs (NFS-shared, tail any of them):"
ls -1t $PROJ/logs/eval_*.log 2>/dev/null | head -8
