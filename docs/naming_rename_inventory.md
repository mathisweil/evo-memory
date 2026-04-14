# Naming Rename Inventory (pre-rename snapshot)

Captured before the cleanup described in `claude_code_naming_cleanup.md`. This is
the safety net: every file:line reference to an old name as it stood at the time
of the rename. Filesystem paths under `results/` and historical references in
WandB/GCS are intentionally NOT included here — they are not renamed.

## Old name → file:line references (source only)

### `lora_rh_m1_instruct_5t.yaml` / `lora_rh_m1_instruct_5t`
- `README.md:38` — project tree entry
- `README.md:151` — experiments table
- `README.md:257` — example command
- `experiment_specification.md:102,112,120,128` — M1 commands and config row
- `experiment_specification.md:486` — eval example
- `.claude/rules/training.md:11,35` — config list and method table
- `.claude/skills/research-context/SKILL.md:65,74` — condition table
- `scripts/run_lora.py:9,16,55` — module docstring + argparse help
- `scripts/run_all_experiments.sh:138,311` — bash invocations
- `scripts/configs/lora_rh_m1_instruct_5t.yaml:5,11` — header comments
- `scripts/configs/deprecated/lora_default.yaml:3` — deprecated header
- `scripts/configs/deprecated/lora_m1_only.yaml:3` — deprecated header
- `scripts/configs/deprecated/lora_rh_m1_instruct.yaml:2` — deprecated header
- `grad_lora_finetuning/trainer.py:75` — docstring
- `docs/config_code_audit.md:11`
- `docs/config_key_trace.md:16,31,79,80`
- `docs/config_comparison_matrix.md:13`
- `docs/experiment_critical_review.md:80`
- `docs/experiment_recommendations.md:20,124,129`
- `docs/m1_recency_investigation.md` (none for filename, see method-name section)
- `docs/namm_memory_investigation.md:7,85`
- `docs/naming_mapping.md:16`

### `lora_rh_m4_instruct_5t.yaml` / `lora_rh_m4_instruct_5t`
- `README.md:39,152,261`
- `experiment_specification.md:210,217`
- `claude_code_namm_oom.md:143,183,184`
- `.claude/rules/training.md:12,36`
- `.claude/skills/research-context/SKILL.md:67,74`
- `scripts/run_lora.py:12`
- `scripts/run_all_experiments.sh:226,238,274`
- `scripts/configs/lora_rh_m1_instruct_5t.yaml:5` — cross-reference comment
- `scripts/configs/lora_rh_m4_instruct_5t.yaml:15` — header comment
- `scripts/configs/deprecated/lora_default.yaml:4`
- `scripts/configs/deprecated/lora_rh_m4_instruct.yaml:2`
- `docs/config_code_audit.md:67`
- `docs/config_key_trace.md:17,31,81,82`
- `docs/config_comparison_matrix.md:13`
- `docs/experiment_critical_review.md:80`
- `docs/experiment_recommendations.md:39,103`
- `docs/m3_rerun_plan.md:8,14,38`
- `docs/namm_memory_investigation.md:9,87`
- `docs/naming_mapping.md:18,43`

### `joint_lora_m4_5t.yaml`
- `README.md:40,153,266`
- `experiment_specification.md:250,256`
- `claude_code_namm_oom.md:155,185`
- `.claude/rules/training.md:37`
- `.claude/skills/research-context/SKILL.md:68`
- `scripts/run_all_experiments.sh:151,338,348`
- `scripts/configs/joint_default.yaml:5`
- `scripts/configs/joint_lora_m4_5t.yaml:9` — header self-reference
- `scripts/configs/lora_rh_m1_instruct_5t.yaml:6` — cross-ref
- `grad_lora_finetuning/trainer.py:75`
- `docs/compute_budget_validation.md:57`
- `docs/config_code_audit.md:89,195,197,199`
- `docs/config_key_trace.md:18,19`
- `docs/config_comparison_matrix.md:13`
- `docs/experiment_critical_review.md:197,199,241`
- `docs/experiment_recommendations.md:64,70,80,85,163,165`
- `docs/m4_joint_training_analysis.md:77,152,156`
- `docs/m4_readiness_review.md:117`
- `docs/namm_memory_investigation.md:11,82`
- `docs/naming_mapping.md:19,49`

### `lora_rh_m1_instruct.yaml` / `lora_rh_m4_instruct.yaml` (deprecated, non-5t)
- `scripts/configs/deprecated/lora_rh_m1_instruct.yaml:2,6`
- `scripts/configs/deprecated/lora_rh_m4_instruct.yaml:2,6,12`
- `docs/config_code_audit.md:176,177,187,188`
- `docs/experiment_critical_review.md:98`
- `docs/naming_mapping.md:43`

### Method values (`rh_m1_lora_instruct_5t`, `rh_m4_frozen_5t`, `rh_m4_frozen`)
- `experiment_specification.md:129,218,350,423,441,443`
- `.claude/rules/training.md:35,36,39,45`
- `.claude/skills/research-context/SKILL.md:74`
- `scripts/configs/lora_rh_m1_instruct_5t.yaml:21`
- `scripts/configs/lora_rh_m4_instruct_5t.yaml:4,5,22`
- `scripts/configs/deprecated/lora_rh_m4_instruct.yaml:12`
- `grad_lora_finetuning/trainer.py:79`
- `scripts/generate_paper_figures.py:103` — glob pattern over results dir
- `docs/naming_mapping.md:33,56,67`

### `lora_m1_only.yaml`
- `docs/config_code_audit.md:173`
- `scripts/configs/deprecated/lora_m1_only.yaml:3,6`

## What is intentionally NOT being renamed (per spec, kept verbatim)

- WandB run names — external/permanent.
- GCS checkpoint paths (`gs://.../lora-m4-frozen-5t-...`) — external/permanent.
- Filesystem paths under `results/` (e.g. `results/rh_m1_lora_instruct_5t/42/best_ckpt.pt`,
  `results/rh_m4_frozen_5t/42/...`, the `results/main_table_5t/M4/` directory).
- Hydra config names (`namm_bam_i1_llama32_1b_5t`, `rh_multi_qa_5t`).
- All `results/main_table_5t/**/command.sh` and `README.md` artefacts (run logs,
  not source).
- `_claude_*.sh` helper scripts that hardcode the historical `results/rh_*` paths
  to existing checkpoint files on disk.
- `scripts/organize_eval_results.py` command-string templates that embed the
  historical `results/rh_*` checkpoint paths.
- `scripts/generate_paper_figures.py:103` glob over `**/rh_m4_frozen/...` — this
  globs the existing on-disk results directory layout.
