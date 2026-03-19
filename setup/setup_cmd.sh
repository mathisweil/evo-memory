#!/usr/bin/env bash
# =============================================================================
# GPU VM Bootstrap
#
# Clones the repo then runs setup.sh — use this for first-time machine setup.
#
# Download and run:
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/main/setup/setup_cmd.sh \
#       -o /tmp/setup_cmd.sh
#   bash /tmp/setup_cmd.sh
#   bash /tmp/setup_cmd.sh --dir ~/my-workspace          # custom directory
#   bash /tmp/setup_cmd.sh --gpu 2                       # pin GPU
#   bash /tmp/setup_cmd.sh --branch my-feature           # alternate branch
#   bash /tmp/setup_cmd.sh --noclaude                    # skip Claude Code
#
# UCL GPU machines (dsml MSc):
#   bash /tmp/setup_cmd.sh --dir /cs/student/project_msc/2025/dsml/$(whoami)
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/mathisweil/evo-memory.git"
DEFAULT_BRANCH="main"

# ---------------------------------------------------------------------------
# Parse arguments — extract --dir and --branch before forwarding rest
# ---------------------------------------------------------------------------
CUSTOM_DIR=""
BRANCH="${DEFAULT_BRANCH}"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)         CUSTOM_DIR="$2"; shift 2 ;;
        --dir=*)       CUSTOM_DIR="${1#*=}"; shift ;;
        --branch)      BRANCH="$2"; shift 2 ;;
        --branch=*)    BRANCH="${1#*=}"; shift ;;
        *)             FORWARD_ARGS+=("$1"); shift ;;
    esac
done

WORK_DIR="${CUSTOM_DIR:-${HOME}/evo-memory-workspace}"
REPO_DIR="${WORK_DIR}/evo-memory"

echo '============================================================'
echo ' evo-memory — GPU Bootstrap'
echo '============================================================'
echo "Workspace: ${WORK_DIR}"
echo "Branch:    ${BRANCH}"
echo ''

# ---------------------------------------------------------------------------
# Clone or update the repo
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"

if [ -d "${REPO_DIR}" ]; then
    echo 'Repo already exists — pulling latest...'
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    echo 'Cloning repo...'
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

echo ''

# ---------------------------------------------------------------------------
# Delegate to setup.sh (all remaining args forwarded)
# ---------------------------------------------------------------------------
exec bash "${REPO_DIR}/setup/setup.sh" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
