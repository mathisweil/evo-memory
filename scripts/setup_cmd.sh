#!/usr/bin/env bash
# Run this on the GPU VM to set everything up.
#
# The one-liner clones the repo first, then runs setup.sh from the local
# copy — so you always execute the latest version (no stale curl cache).
#
# Works from any shell (bash, zsh, csh, tcsh — doesn't matter).
#
# Auto-detect first GPU:
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/scripts/setup_cmd.sh -o /tmp/setup_cmd.sh && bash /tmp/setup_cmd.sh
#
# Pin to a specific GPU (e.g. GPU 2 on a multi-GPU machine):
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/scripts/setup_cmd.sh -o /tmp/setup_cmd.sh && bash /tmp/setup_cmd.sh --gpu 2
#
# Or if you've already cloned the repo:
#   bash /cs/student/project_msc/2025/csml/sruppage/SNLP/FT-NAMM/evo-memory/scripts/setup.sh
#   bash /cs/student/project_msc/2025/csml/sruppage/SNLP/FT-NAMM/evo-memory/scripts/setup.sh --gpu 2

set -euo pipefail

WORK_DIR="/cs/student/project_msc/2025/csml/sruppage/SNLP/FT-NAMM"
REPO_DIR="${WORK_DIR}/evo-memory"
BRANCH="es-fine-tuning"
REPO_URL="https://github.com/mathisweil/evo-memory.git"

mkdir -p "${WORK_DIR}"

# Clone or update the repo so we have the latest setup.sh locally
if [ -d "${REPO_DIR}" ]; then
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

# Now run the real setup.sh from the local clone (never stale)
exec bash "${REPO_DIR}/scripts/setup.sh" "$@"
