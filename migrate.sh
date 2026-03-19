#!/usr/bin/env bash
# =============================================================================
# migrate.sh — restructure evo-memory into a modular layout
#
# Proposed new structure
# ----------------------
#   config/               cfgs/              (Hydra configs)
#   data/longbench/       LongBench/         (dataset config JSONs)
#   data/choubun/         ChouBun/
#   models/               memory_llms/       (LLM wrappers)
#   evolution/            memory_evolution/  (CMA-ES optimizer)
#   ops/                  stateless_parallel_modules/
#   policy/               memory_policy/     (eviction policy)
#   policy/embedding/     memory_policy/deep_embedding_*.py
#   policy/scoring/       memory_policy/deep_scoring*.py
#   training/             memory_trainer.py + lora_grad_trainer.py
#   training/datasets/    lora_*_dataset.py
#   evaluation/           memory_evaluator.py + task_sampler.py
#   metrics/              longbench_metrics.py + choubun_metrics.py
#   utils/                utils*.py
#   scripts/              *.sh
#
# Usage
# -----
#   bash migrate.sh           # dry-run: preview all changes, touch nothing
#   bash migrate.sh --apply   # execute every rename, move, and import rewrite
# =============================================================================

set -euo pipefail

DRY_RUN=true
[[ "${1:-}" == "--apply" ]] && DRY_RUN=false

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

# Guard: verify we are in the right directory
[[ -f main.py && -d cfgs && -d memory_policy ]] \
    || { echo "ERROR: must be run from the evo-memory repo root"; exit 1; }

RUN() {
    if $DRY_RUN; then
        printf "  [dry] %s\n" "$*"
    else
        eval "$@"
    fi
}

WRITE_FILE() {         # WRITE_FILE <path> <content>
    local path="$1"
    local content="$2"
    if $DRY_RUN; then
        printf "  [dry] write %s\n" "$path"
    else
        printf '%s\n' "$content" > "$path"
        git add "$path"
        echo "  wrote $path"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — create directories that have no direct git mv source
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: scaffold new directories ==="
RUN "mkdir -p data training/datasets evaluation metrics utils scripts"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — whole-directory renames (git tracks each file)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: rename top-level packages ==="
RUN "git mv cfgs                      config"
RUN "git mv LongBench                 data/longbench"
RUN "git mv ChouBun                   data/choubun"
RUN "git mv memory_llms               models"
RUN "git mv memory_evolution          evolution"
RUN "git mv stateless_parallel_modules ops"
RUN "git mv memory_policy             policy"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — restructure policy/ into embedding/ and scoring/ sub-packages
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 3: split policy/ into sub-packages ==="
RUN "mkdir -p policy/embedding policy/scoring"

# embedding sub-package (four files, each renamed to a cleaner stem)
RUN "git mv policy/deep_embedding.py            policy/embedding/base.py"
RUN "git mv policy/deep_embedding_shared.py     policy/embedding/shared.py"
RUN "git mv policy/deep_embedding_spectogram.py policy/embedding/spectogram.py"
RUN "git mv policy/deep_embedding_wrappers.py   policy/embedding/wrappers.py"

# scoring sub-package
RUN "git mv policy/deep_scoring.py              policy/scoring/base.py"
RUN "git mv policy/deep_scoring_bam.py          policy/scoring/bam.py"

# flat renames that stay in policy/
RUN "git mv policy/base_deep_components.py      policy/components.py"
RUN "git mv policy/deep_selection.py            policy/selection.py"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — move root-level modules into new packages
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 4: move root-level modules into packages ==="

# training/
RUN "git mv memory_trainer.py    training/namm_trainer.py"
RUN "git mv lora_grad_trainer.py training/lora_trainer.py"
RUN "git mv lora_sft_dataset.py  training/datasets/sft.py"
RUN "git mv lora_ntp_dataset.py  training/datasets/ntp.py"

# evaluation/
RUN "git mv memory_evaluator.py  evaluation/evaluator.py"
RUN "git mv task_sampler.py      evaluation/task_sampler.py"

# metrics/
RUN "git mv longbench_metrics.py metrics/longbench.py"
RUN "git mv choubun_metrics.py   metrics/choubun.py"

# utils/
RUN "git mv utils.py             utils/core.py"
RUN "git mv utils_hydra.py       utils/hydra.py"
RUN "git mv utils_log.py         utils/log.py"
RUN "git mv utils_longbench.py   utils/longbench.py"

# scripts/
RUN "git mv run_namm_instruct_eval.sh  scripts/run_namm_eval.sh"
RUN "git mv run_m1sft_evals.sh         scripts/run_m1sft_evals.sh"
RUN "git mv run_32k_evals.sh           scripts/run_32k_evals.sh"
RUN "git mv run_all_evals_then_namm.sh scripts/run_all_evals.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — write new __init__.py files
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 5: write __init__.py files ==="

# Stub packages (empty __init__.py)
for pkg in training training/datasets evaluation metrics utils \
           policy/embedding policy/scoring; do
    WRITE_FILE "${pkg}/__init__.py" ""
done

# policy/__init__.py — re-export everything the old memory_policy/__init__.py did,
# but now pointing at the restructured sub-module paths.
WRITE_FILE "policy/__init__.py" \
"from .base import (
    MemoryPolicy, ParamMemoryPolicy, Recency, AttnRequiringRecency,
)
from .base_dynamic import (
    DynamicMemoryPolicy, DynamicParamMemoryPolicy,
    RecencyParams, AttentionParams,
)
from .auxiliary_losses import (
    MemoryPolicyAuxiliaryLoss, SparsityAuxiliaryLoss, L2NormAuxiliaryLoss,
)
from .deep import DeepMP
from .embedding.spectogram import (
    STFTParams, AttentionSpectrogram, fft_avg_mask, fft_ema_mask,
)
from .embedding.base import RecencyExponents, NormalizedRecencyExponents
from .scoring.base import (
    MLPScoring, GeneralizedScoring, make_scaled_one_hot_init, TCNScoring,
)
from .selection import DynamicSelection, TopKSelection, BinarySelection
from .components import (
    EMAParams, ComponentOutputParams, wrap_torch_initializer,
    DeepMemoryPolicyComponent, TokenEmbedding, JointEmbeddings,
    ScoringNetwork, SelectionNetwork,
)
from .shared import SynchronizableBufferStorage, RegistrationCompatible
from .embedding.shared import PositionalEmbedding, Embedding
from .embedding.wrappers import RecencyEmbeddingWrapper"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — rewrite Python imports across all .py files
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 6: rewrite Python imports ==="

if $DRY_RUN; then
    echo "  [dry] would run Python import rewriter on all .py files"
else
python3 - << 'PYEOF'
from pathlib import Path

# Order is intentional: longer / more specific patterns before their prefixes
# so that e.g. deep_embedding_spectogram is caught before deep_embedding.
REPLACEMENTS = [
    # ── External package renames ──────────────────────────────────────────
    ("from memory_llms",                 "from models"),
    ("import memory_llms",               "import models"),
    ("from memory_policy",               "from policy"),
    ("import memory_policy",             "import policy"),
    ("from memory_evolution",            "from evolution"),
    ("import memory_evolution",          "import evolution"),
    ("from stateless_parallel_modules",  "from ops"),
    ("import stateless_parallel_modules","import ops"),
    # root-level modules that moved into packages
    ("from memory_trainer import",       "from training.namm_trainer import"),
    ("from memory_evaluator import",     "from evaluation.evaluator import"),
    ("from lora_grad_trainer import",    "from training.lora_trainer import"),
    ("from lora_sft_dataset import",     "from training.datasets.sft import"),
    ("from lora_ntp_dataset import",     "from training.datasets.ntp import"),
    ("from task_sampler import",         "from evaluation.task_sampler import"),
    ("from utils import",                "from utils.core import"),
    ("from utils_hydra import",          "from utils.hydra import"),
    ("from utils_log import",            "from utils.log import"),
    ("from utils_longbench import",      "from utils.longbench import"),
    ("from longbench_metrics import",    "from metrics.longbench import"),
    ("from choubun_metrics import",      "from metrics.choubun import"),
    # ── policy sub-module renames — absolute (run after policy→policy) ────
    # specific before generic to prevent partial matches
    ("from policy.deep_embedding_spectogram", "from policy.embedding.spectogram"),
    ("from policy.deep_embedding_shared",     "from policy.embedding.shared"),
    ("from policy.deep_embedding_wrappers",   "from policy.embedding.wrappers"),
    ("from policy.deep_embedding",            "from policy.embedding.base"),
    ("from policy.deep_scoring_bam",          "from policy.scoring.bam"),
    ("from policy.deep_scoring",              "from policy.scoring.base"),
    ("from policy.deep_selection",            "from policy.selection"),
    ("from policy.base_deep_components",      "from policy.components"),
    # ── policy sub-module renames — relative (within policy/ package) ─────
    ("from .base_deep_components",       "from .components"),
    ("from .deep_embedding_spectogram",  "from .embedding.spectogram"),
    ("from .deep_embedding_shared",      "from .embedding.shared"),
    ("from .deep_embedding_wrappers",    "from .embedding.wrappers"),
    ("from .deep_embedding",             "from .embedding.base"),
    ("from .deep_scoring_bam",           "from .scoring.bam"),
    ("from .deep_scoring",               "from .scoring.base"),
    ("from .deep_selection",             "from .selection"),
    # ── Data config path references ───────────────────────────────────────
    ("LongBench/config",                 "data/longbench/config"),
    ("ChouBun/config",                   "data/choubun/config"),
    # ── Hydra entry-point config_path ─────────────────────────────────────
    ("config_path='cfgs'",               "config_path='config'"),
]

SKIP = {".git", ".planning", "__pycache__", ".mypy_cache", "migrate.sh"}

changed = []
for path in sorted(Path(".").rglob("*.py")):
    if any(part in SKIP for part in path.parts):
        continue
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        changed.append(str(path))

for f in changed:
    print(f"  updated {f}")
print(f"\n  {len(changed)} file(s) modified")
PYEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — rewrite Hydra _target_: fields in all YAML configs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 7: rewrite Hydra _target_: fields in config/**/*.yaml ==="

if $DRY_RUN; then
    echo "  [dry] would update _target_: fields in config/**/*.yaml"
else
python3 - << 'PYEOF'
from pathlib import Path

# Ordered: more specific before their prefixes
YAML_REPLACEMENTS = [
    ("_target_: memory_evaluator.",            "_target_: evaluation.evaluator."),
    ("_target_: memory_trainer.",              "_target_: training.namm_trainer."),
    ("_target_: memory_policy.",               "_target_: policy."),
    ("_target_: memory_evolution.",            "_target_: evolution."),
    ("_target_: memory_llms.",                 "_target_: models."),
    ("_target_: stateless_parallel_modules.",  "_target_: ops."),
    ("_target_: task_sampler.",                "_target_: evaluation.task_sampler."),
    ("_target_: utils_hydra.",                 "_target_: utils.hydra."),
]

changed = []
for path in sorted(Path("config").rglob("*.yaml")):
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in YAML_REPLACEMENTS:
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        changed.append(str(path))

for f in changed:
    print(f"  updated {f}")
print(f"\n  {len(changed)} file(s) modified")
PYEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if $DRY_RUN; then
    echo "=== Dry-run complete — nothing was changed. ==="
    echo "    Run  bash migrate.sh --apply  to execute."
else
    echo "=== Migration complete. ==="
    echo ""
    echo "Suggested next steps:"
    echo "  1. Smoke-test imports:"
    echo "       python -c 'import models, policy, evolution, ops, training, evaluation, metrics, utils'"
    echo "  2. Verify Hydra config resolves:"
    echo "       python main.py +run=rh_instruct_recency --cfg job"
    echo "  3. Run tests:"
    echo "       pytest tests/ -x"
    echo "  4. Commit:"
    echo "       git add -A && git commit -m 'refactor: modular directory restructure'"
fi
