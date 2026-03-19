#!/usr/bin/env bash
# =============================================================================
# migrate.sh — evo-memory directory restructuring
#
# Performs the structural file moves that cannot be expressed as code edits:
#   1. cfgs/ → config/          (Hydra configs; replaces any stale partial copy)
#   2. namm/evaluator.py        → namm/evaluation/evaluator.py
#   3. utils/longbench.py       → (removed; replaced by namm/evaluation/longbench.py)
#   4. utils/longbench_metrics.py → (removed; replaced by namm/evaluation/metrics.py)
#   5. Delete all "* 2.*" backup files and stale root-level module dirs
#
# All Python import paths were already updated by the code-edit pass.
# Running this script finalises the layout so the file locations match
# the new import paths.
#
# Usage:
#   bash migrate.sh             # dry-run: prints commands, makes no changes
#   bash migrate.sh --apply     # execute for real
#
# Safe to re-run: every operation is guarded by an existence check.
# POSIX-compatible: works with bash 3 (macOS default).
# =============================================================================

set -euo pipefail

APPLY=false
if [[ "${1:-}" == "--apply" ]]; then
    APPLY=true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

run() {
    if $APPLY; then
        echo "  + $*"
        "$@"
    else
        echo "  [dry-run] $*"
    fi
}

# ── 1. Rename cfgs/ → config/ ────────────────────────────────────────────────
echo ""
echo "==> Step 1: Rename cfgs/ → config/"
if [ ! -d "cfgs" ]; then
    echo "  Already done (cfgs/ not present)."
elif [ -f "config/config.yaml" ]; then
    echo "  Already done (config/config.yaml exists)."
else
    # Remove any partial/stale config/ directory before the git mv
    if [ -d "config" ]; then
        echo "  Removing stale partial config/ before rename..."
        run rm -rf config
    fi
    run git mv cfgs config
fi

# ── 2. Move namm/evaluator.py → namm/evaluation/evaluator.py ─────────────────
echo ""
echo "==> Step 2: Move namm/evaluator.py → namm/evaluation/evaluator.py"
if [ ! -f "namm/evaluator.py" ]; then
    echo "  Already done (namm/evaluator.py not present)."
else
    # Remove the temporary re-export shim so git mv can place the real file there
    if [ -f "namm/evaluation/evaluator.py" ]; then
        # Use git rm if tracked, plain rm if untracked
        if git ls-files --error-unmatch "namm/evaluation/evaluator.py" >/dev/null 2>&1; then
            run git rm namm/evaluation/evaluator.py
        else
            run rm -f "namm/evaluation/evaluator.py"
        fi
    fi
    run git mv namm/evaluator.py namm/evaluation/evaluator.py
    # Fix the one relative import inside the moved file:
    #   from namm.evaluation.longbench import build_chat
    # → from .longbench import build_chat
    # (was already updated in the code-edit pass to use the full module path,
    # which works both before and after the move — no further change needed)
    echo "  Import path already updated in code-edit pass."
fi

# ── 3. Remove utils/longbench.py ─────────────────────────────────────────────
echo ""
echo "==> Step 3: Remove utils/longbench.py (replaced by namm/evaluation/longbench.py)"
if [ -f "utils/longbench.py" ]; then
    run git rm utils/longbench.py
else
    echo "  Already done."
fi

# ── 4. Remove utils/longbench_metrics.py ─────────────────────────────────────
echo ""
echo "==> Step 4: Remove utils/longbench_metrics.py (replaced by namm/evaluation/metrics.py)"
if [ -f "utils/longbench_metrics.py" ]; then
    run git rm utils/longbench_metrics.py
else
    echo "  Already done."
fi

# ── 5. Delete "* 2.*" backup files and stale root-level dirs ─────────────────
echo ""
echo "==> Step 5: Delete '* 2.*' backup files and stale root-level dirs"

# Root-level stale module dirs (old layout, superseded by namm/)
for d in policy models; do
    if [ -d "${d}" ]; then
        # Check if tracked by git
        if git ls-files --error-unmatch "${d}/" >/dev/null 2>&1; then
            run git rm -r "${d}"
        else
            run rm -rf "${d}"
        fi
    fi
done

# All untracked "* 2.*" files (backup copies with spaces in names)
FOUND=0
while IFS= read -r -d '' f; do
    run rm -f "${f}"
    FOUND=$((FOUND + 1))
done < <(find . -not -path './.git/*' -name "* 2.*" -type f -print0 2>/dev/null | sort -z)

if [ "${FOUND}" -eq 0 ]; then
    echo "  No '* 2.*' backup files found."
else
    echo "  Found ${FOUND} backup file(s)."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if $APPLY; then
    echo "Migration complete. Suggested next step:"
    echo "  git add -u && git add namm/evaluation/ namm/run_utils.py scripts/experiment_utils.py"
    echo "  git commit -m 'refactor: restructure — cfgs→config, namm/evaluation/, namm/run_utils'"
else
    echo "Dry-run complete. To apply: bash migrate.sh --apply"
fi
