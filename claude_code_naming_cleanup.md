# Claude Code Task: Clean Up Naming Conventions Across the Project

## Problem

The current file and method naming is confusing:

1. **`rh_` prefix is meaningless.** It originally separated multi-task instruct experiments from single-task exploratory runs. That distinction no longer matters — all FAIR-01 experiments use the multi-task instruct setup. `rh` adds noise to every filename, method name, and WandB reference.

2. **`lora_rh_m4_instruct_5t.yaml` is the M3 config.** The file has "m4" in its name but it's the M3 experiment (LoRA + frozen NAMM). This is the single biggest source of confusion in the project — the experiment spec has a WARNING block about it, the analysis spec has a naming note, and multiple docs reference the mislabelling.

3. **The `method` field inside configs** (`rh_m1_lora_instruct_5t`, `rh_m4_frozen_5t`) flows into WandB run names, GCS paths, and results.json. New runs should use clean method names.

4. **Inconsistent suffixes.** Some configs have `_5t`, some have `_instruct`, some have both, some have neither.

## Target Naming Scheme

### Config files (`scripts/configs/`)

| Current name | New name | Why |
|-------------|----------|-----|
| `lora_rh_m1_instruct_5t.yaml` | `m1_lora_5t.yaml` | Drop `rh`, drop `instruct` (all configs use instruct) |
| `lora_rh_m4_instruct_5t.yaml` | `m3_lora_frozen_namm_5t.yaml` | Fix the M3/M4 confusion, drop `rh` |
| `lora_rh_m1_instruct.yaml` | `m1_lora.yaml` | Old non-5t variant, drop `rh` |
| `lora_rh_m4_instruct.yaml` | `m3_lora_frozen_namm.yaml` | Old non-5t variant, fix M3/M4, drop `rh` |
| `lora_m1_only.yaml` | Keep or rename to `m1_lora_single_task.yaml` | This is the old single-task config |
| `joint_lora_m4_5t.yaml` | `m4_joint_lora_5t.yaml` | Prefix with condition ID for consistency |
| `joint_default.yaml` | Keep as is | Generic default, not condition-specific |
| `eval_main_table.yaml` | Keep as is | Descriptive |
| `eval_default.yaml` | Keep as is | Descriptive |
| `es_default.yaml` | Keep as is | Generic default |
| `es_m1_only.yaml` | Keep as is | Already clear |

The pattern becomes: `{condition}_{method}_{task_set}.yaml`
- `m1_lora_5t.yaml` — condition M1, method LoRA, 5-task subset
- `m3_lora_frozen_namm_5t.yaml` — condition M3, method LoRA+frozenNAMM, 5-task
- `m4_joint_lora_5t.yaml` — condition M4, method joint LoRA+NAMM, 5-task

### Method names (inside configs, flows to WandB)

| Current `method` value | New value |
|----------------------|-----------|
| `rh_m1_lora_instruct_5t` | `m1_lora_5t` |
| `rh_m4_frozen_5t` | `m3_lora_frozen_namm_5t` |

### Hydra task config

| Current | New |
|---------|-----|
| `rh_multi_qa_5t` | Keep as is |

The Hydra task config `rh_multi_qa_5t` lives in `config/task/` and is referenced by the Hydra run preset `namm_bam_i1_llama32_1b_5t`. Renaming it would require updating the Hydra config tree, which is risky and touches the NAMM training pipeline. **Leave it alone** — it's an internal Hydra identifier, not user-facing.

### Run config

| Current | Keep? |
|---------|-------|
| `namm_bam_i1_llama32_1b_5t` | Yes — Hydra preset name, not worth the risk |
| `full_cache_baseline_llama32_1b` | Yes |
| `recency_baseline_llama32_1b` | Yes |

---

## Step 1: Inventory Everything That References the Old Names

Before renaming anything, do a full grep of the repo. For EACH old name, find every reference:

```bash
# Run these for each old name
grep -rn "lora_rh_m1_instruct_5t" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "lora_rh_m4_instruct_5t" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "lora_rh_m1_instruct" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "lora_rh_m4_instruct" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "joint_lora_m4_5t" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "rh_m1_lora_instruct_5t" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "rh_m4_frozen_5t" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "rh_m4_frozen" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
grep -rn "lora_m1_only" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh"
```

Write the full inventory to `docs/naming_rename_inventory.md` — every file, line number, and the old string. This is the safety net.

---

## Step 2: Rename Config Files

```bash
# In scripts/configs/
mv lora_rh_m1_instruct_5t.yaml    m1_lora_5t.yaml
mv lora_rh_m4_instruct_5t.yaml    m3_lora_frozen_namm_5t.yaml
mv joint_lora_m4_5t.yaml          m4_joint_lora_5t.yaml

# Old non-5t variants (if they exist)
mv lora_rh_m1_instruct.yaml       m1_lora.yaml
mv lora_rh_m4_instruct.yaml       m3_lora_frozen_namm.yaml
```

---

## Step 3: Update `method` Field Inside Configs

In `m1_lora_5t.yaml`:
```yaml
method: m1_lora_5t              # was rh_m1_lora_instruct_5t
```

In `m3_lora_frozen_namm_5t.yaml`:
```yaml
method: m3_lora_frozen_namm_5t  # was rh_m4_frozen_5t
```

**Important:** This changes what appears in WandB for NEW runs. Old completed runs will still show `rh_m1_lora_instruct_5t` and `rh_m4_frozen_5t` in WandB — that's fine, they're historical. Add a comment in each config:
```yaml
# Historical WandB runs used method: rh_m1_lora_instruct_5t (pre-rename)
```

---

## Step 4: Update All References

For every file found in Step 1, update the reference to the new name. The main files to update:

### `README.md`
- Project structure tree: update config file names
- Experiments table: update Config column
- Example commands: update `--config` paths
- Any inline references to config files

### `experiment_specification.md`
- All `--config scripts/configs/...` commands
- Config path references in parameter tables
- Method name references
- The M3/M4 naming WARNING block — this can now be simplified since the file name matches the experiment number

### `analysis_specification.md`
- Any config file references

### `.claude/rules/training.md`
- Config references and method names

### `docs/*.md`
- `docs/m3_rerun_plan.md` — config references
- `docs/m4_readiness_review.md` — config references
- `docs/config_comparison_matrix.md` — config names in headers
- `docs/config_key_trace.md` — config file names
- `docs/namm_memory_investigation.md` — config references
- Any other docs that reference config files

### Python scripts
- Check if any scripts have hardcoded config paths (unlikely — they use `--config` from CLI)
- Check `scripts/organize_eval_results.py` (if it exists) — it may reference method names

### Other config files
- `joint_default.yaml` comments may reference old names
- `eval_main_table.yaml` usage examples reference old config names

---

## Step 5: Simplify the M3/M4 Naming Warning

The experiment spec currently has a large WARNING block about the M3/M4 naming confusion. Now that the config file is properly named `m3_lora_frozen_namm_5t.yaml`, simplify this to:

```markdown
> **Historical naming:** WandB runs and GCS paths use `rh_m4_5t_cs*` for M3 runs and 
> `rh_m4_frozen` as the method name. This predates the current M-numbering. Config files 
> have been renamed to match the experiment specification (M3 = LoRA + frozen NAMM). 
> The `results/main_table_5t/M4/` directory contains M3 results, not M4 joint results.
```

The key change: the warning is now about **historical artefacts** (WandB, GCS, results dirs), not about current config files.

---

## Step 6: Verify Nothing Broke

After all renames and reference updates:

1. **Smoke test M1:**
   ```bash
   python scripts/run_lora.py --config scripts/configs/m1_lora_5t.yaml \
       --run_name smoke_rename_m1 --num_epochs 1 --eval_interval 5 --no-gcs
   ```

2. **Smoke test M3** (will OOM at bs=2 unless Fix A was applied, use --batch_size 1):
   ```bash
   python scripts/run_lora.py --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
       --run_name smoke_rename_m3 --num_epochs 1 --eval_interval 5 --no-gcs \
       --namm_checkpoint <any-m2-checkpoint> --batch_size 1
   ```

3. **Smoke test M4:**
   ```bash
   python scripts/run_joint.py --config scripts/configs/m4_joint_lora_5t.yaml \
       --run_name smoke_rename_m4 --adapter_type lora \
       --num_outer_loops 1 --namm_iterations_per_stage 2 \
       --lora_epochs_per_stage 1 --population_size 2 --mini_batch_size 2
   ```

4. **Smoke test eval:**
   ```bash
   python scripts/run_eval.py --config scripts/configs/eval_main_table.yaml \
       --run_config full_cache_baseline_llama32_1b --num_samples 5
   ```

If any smoke test fails due to a missed reference, fix it.

---

## Step 7: Final Checklist

After everything is done, verify:

- [ ] No file in the repo contains `lora_rh_m1_instruct_5t` (except in historical notes/comments)
- [ ] No file contains `lora_rh_m4_instruct_5t` (except in historical notes/comments)
- [ ] No file contains `joint_lora_m4_5t` (except in historical notes/comments)
- [ ] The experiment spec's M3 section references `m3_lora_frozen_namm_5t.yaml`, not the old name
- [ ] The README experiments table references the new config names
- [ ] `method` fields in configs match the new naming scheme
- [ ] Old non-5t configs are renamed if they exist
- [ ] All docs reference the new names

Run the final grep to confirm:
```bash
grep -rn "lora_rh_" --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh" | grep -v "Historical\|pre-rename\|WandB runs"
```

This should return zero results (excluding historical/comment references).

---

## What NOT to Rename

- **WandB run names** — these are permanent and external. Old runs will always show `rh_m4_5t_cs1024`. Fine.
- **GCS checkpoint paths** — same. `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs1024-llama32-1b/` stays as is.
- **`results/main_table_5t/M4/` directory** — this contains M3 results under the old name. Don't rename filesystem results dirs mid-project. Document the mapping.
- **Hydra config names** — `namm_bam_i1_llama32_1b_5t`, `rh_multi_qa_5t`, etc. These are deep in the Hydra config tree and are referenced by the NAMM training pipeline. Not worth the risk.
- **The `run_config` key values** in YAML configs — these point to Hydra presets. Keep as is.

---

## Output Summary

```
scripts/configs/
├── m1_lora_5t.yaml                  # was lora_rh_m1_instruct_5t.yaml
├── m3_lora_frozen_namm_5t.yaml      # was lora_rh_m4_instruct_5t.yaml
├── m4_joint_lora_5t.yaml            # was joint_lora_m4_5t.yaml
├── m1_lora.yaml                     # was lora_rh_m1_instruct.yaml (if exists)
├── m3_lora_frozen_namm.yaml         # was lora_rh_m4_instruct.yaml (if exists)
├── joint_default.yaml               # unchanged (comments updated)
├── eval_default.yaml                # unchanged (comments updated)
├── eval_main_table.yaml             # unchanged (usage examples updated)

docs/
├── naming_rename_inventory.md       # full grep results before rename

# Updated references in:
README.md
experiment_specification.md
analysis_specification.md
.claude/rules/training.md
docs/*.md (all docs that reference configs)
```
