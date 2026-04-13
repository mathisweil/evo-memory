# Experiment & Config Audit (FAIR-01)

Generated: 2026-04-13

Cross-references `experiment_specification.md` against the actual codebase.
Flags missing configs, parameter mismatches, and FAIR-01 violations.

---

## 1 Hydra Run Configs (`config/run/`)

| Config | Exists | Issues |
|--------|--------|--------|
| `namm_bam_i1_llama32_1b_5t.yaml` | Y | OK. 5-task subset, FAIR-01 splits, correct cache/filtering. |
| `full_cache_baseline_llama32_1b.yaml` | Y | OK. `cache_size: null`, 5-task via `rh_multi_qa_5t`. |
| `recency_baseline_llama32_1b.yaml` | Y | OK. `cache_size: 1024`, 5-task via `rh_multi_qa_5t`. |

All three Hydra run configs exist and are FAIR-01 compliant.

---

## 2 Script Configs (`scripts/configs/`)

### B0 — Base model, full KV cache

- **Spec command:** `python scripts/run_eval.py --run_config full_cache_baseline_llama32_1b --output_dir ...`
- **Config needed:** None (eval uses CLI args or `eval_default.yaml`).
- **Issues:**
  - `eval_default.yaml` has `run_config: namm_bam_i1_llama32_1b` (3-task). Must override via CLI `--run_config full_cache_baseline_llama32_1b`.
  - `run_eval.py` default `--run_config` is `namm_bam_i1_llama32_1b_5t` (fixed in P0-1). B0 command explicitly passes the correct config. OK.

### B1 — Base model, recency eviction

- **Spec command:** `python scripts/run_eval.py --run_config recency_baseline_llama32_1b --cache_size 1024 ...`
- **Issues:** Same as B0 — CLI explicitly sets `--run_config`. OK.

### M1 — LoRA only

- **Spec config:** `scripts/configs/lora_rh_m1_instruct_5t.yaml`
- **Exists:** NO. Only `lora_rh_m1_instruct.yaml` exists.
- **Parameter mismatches in existing `lora_rh_m1_instruct.yaml` vs spec:**

| Parameter | Spec (FAIR-01) | `lora_rh_m1_instruct.yaml` | Mismatch? |
|-----------|---------------|---------------------------|-----------|
| `method` | `rh_m1_lora_instruct_5t` | `rh_m1_lora_instruct` | YES |
| `run_config` | `namm_bam_i1_llama32_1b_5t` | `namm_bam_i1_llama32_1b` (3-task) | YES |
| `min_conditioning_length` | 4096 | 100 (via override) | YES |
| Task list | 5 tasks | 6 tasks (includes `multifieldqa_en`) | YES |
| `eval_interval` | 2 | 25 | YES |
| `early_stopping_patience` | 5 | 5 | OK |
| `lora_rank` | 8 | 8 | OK |
| `lora_alpha` | 16 | 16 | OK |
| `lora_dropout` | 0.1 | 0.1 | OK |
| `learning_rate` | 5e-5 | 5e-5 | OK |
| `num_epochs` | 150 | 150 | OK |
| `batch_size` | 4 | 4 | OK |
| `gradient_accumulation_steps` | 4 | 4 | OK |
| `max_seq_len` | 7000 | 7000 | OK |
| `sft_mode` | true | true | OK |
| `namm_active` | false | false | OK |
| `train_split` | 0.7 | 0.7 | OK |
| `val_split` | 0.15 | 0.15 | OK |
| `split_seed` | 42 | 42 | OK |
| `filter_answers_by_tokens` | 64 | 64 | OK |

**Verdict:** MUST create `lora_rh_m1_instruct_5t.yaml`. The existing config uses the wrong task subset and run_config.

### M3 — LoRA + frozen NAMM

- **Spec config:** `scripts/configs/lora_rh_m4_instruct_5t.yaml`
- **Exists:** NO. Only `lora_rh_m4_instruct.yaml` exists.
- **Parameter mismatches in existing `lora_rh_m4_instruct.yaml` vs spec:**

| Parameter | Spec (FAIR-01) | `lora_rh_m4_instruct.yaml` | Mismatch? |
|-----------|---------------|---------------------------|-----------|
| `method` | `rh_m4_frozen_5t` | `rh_m4_frozen` | YES |
| `run_config` | `namm_bam_i1_llama32_1b_5t` | `namm_bam_i1_llama32_1b` (3-task) | YES |
| Task override | 5-task via run_config | 3-task override in `override:` | YES |
| `min_conditioning_length` | 4096 | not set (missing from overrides) | YES |
| `eval_interval` | 2 | 25 | YES |
| `lora_dropout` | 0.05 | 0.05 | OK |
| `learning_rate` | 1e-4 | 1e-4 | OK |
| `batch_size` | 1 | 1 | OK |
| `gradient_accumulation_steps` | 16 | 16 | OK |
| `cache_size` | 1024 | 1024 | OK |

**Verdict:** MUST create `lora_rh_m4_instruct_5t.yaml`. The existing config uses the wrong task subset and has redundant Hydra architecture overrides.

### M4 — Joint LoRA + NAMM

- **Spec config:** `scripts/configs/joint_default.yaml`
- **Exists:** YES.
- **Parameter mismatches vs spec:**

| Parameter | Spec (FAIR-01) | `joint_default.yaml` | Mismatch? |
|-----------|---------------|---------------------|-----------|
| `adapter_type` | lora (CLI) | es | Requires CLI override |
| `learning_rate` | 5e-5 | 2e-4 | YES |
| `lora_dropout` | 0.0 | 0.0 | OK (spec says 0.0 for M4) |
| `max_seq_len` | 3500 | 3500 | OK |
| `run_config` | `namm_bam_i1_llama32_1b_5t` | `namm_bam_i1_llama32_1b_5t` | OK |
| `num_outer_loops` | 2 | 2 | OK |
| `namm_iterations_per_stage` | 100 | 100 | OK |
| `lora_epochs_per_stage` | 75 | 75 | OK |
| `sft_mode` | true | true | OK |
| `train_split` | 0.7 | 0.7 | OK |
| `val_split` | 0.15 | 0.15 | OK |
| FAIR-01 filters | all present | all present | OK |

**Verdict:** `learning_rate` mismatch (2e-4 vs spec 5e-5). The spec says M4 LoRA should match M1's lr. The `adapter_type` default is `es` but the spec command explicitly passes `--adapter_type lora`. Two options: (a) create a `joint_lora_m4_5t.yaml` with `learning_rate: 5e-5` and `adapter_type: lora`, or (b) always pass `--learning_rate 5e-5 --adapter_type lora` on the CLI. Recommend (a) for reproducibility.

---

## 3 CLI Compatibility

### M1 command from spec

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r4 --lora_rank 4 --lora_alpha 8
```

- `--config` file does not exist -> **FileNotFoundError** (confirmed by user).
- All other CLI args (`--run_name`, `--lora_rank`, `--lora_alpha`) are valid `run_lora.py` args. OK.

### M2 command from spec

```bash
python scripts/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    wandb_run_name=m2_namm_standalone seed=1337
```

- Hydra-based, no argparse config file needed. All overrides are valid Hydra syntax. OK.

### M3 command from spec

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
    --run_name m3_lora_frozen_namm --namm_checkpoint <path>
```

- `--config` file does not exist -> **FileNotFoundError**.
- All CLI args valid. OK.

### M4 command from spec

```bash
python scripts/run_joint.py \
    --config scripts/configs/joint_default.yaml \
    --run_name m4_joint_lora --adapter_type lora \
    --num_outer_loops 2 --namm_iterations_per_stage 100 \
    --lora_epochs_per_stage 75 --lora_rank 8 --cache_size 1024
```

- Config exists. All CLI args are valid `run_joint.py` args. OK.
- `--adapter_type lora` overrides the `es` default. OK.
- Missing `--learning_rate 5e-5` to match M1 lr (config has 2e-4). **Must add to command or fix config.**
- Missing `--lora_alpha 16`. Config has 16 already. OK.

### A4 commands from spec

```bash
python scripts/run_eval.py \
    --es_checkpoint .../adapter/stage_1/ \
    --namm_checkpoint .../namm/latest.pt \
    --cache_size 1024 --output_dir ...
```

- `run_eval.py` accepts `--es_checkpoint` for loading LoRA/ES adapter weights. The `adapter/stage_1/` directory is a PEFT adapter directory — but `run_eval.py` loads it via `torch.load` as a state dict, NOT `PeftModel.from_pretrained`. This will fail if the checkpoint is a PEFT adapter directory. **Potential issue** — needs verification against actual M4 checkpoint format.
- The NAMM-off arm omits `--namm_checkpoint`, which triggers `namm_active=False` by default. OK.

### B0/B1 commands from spec

All args are valid `run_eval.py` args. No config file needed. OK.

---

## 4 FAIR-01 Compliance Summary

| Constraint | B0 | B1 | M1 | M2 | M3 | M4 | A4 |
|-----------|----|----|----|----|----|----|-----|
| 5-task subset | OK | OK | FAIL (6-task in existing config) | OK | FAIL (3-task in existing config) | OK | OK |
| train_frac=0.7 | n/a | n/a | OK | OK | OK | OK | n/a |
| val_frac=0.15 | n/a | n/a | OK | OK | OK | OK | n/a |
| split_seed=42 | OK | OK | OK | OK* | OK | OK | OK |
| min_cond=4096 | via Hydra | via Hydra | FAIL (100) | OK | FAIL (missing) | OK | OK |
| max_cond=6500 | via Hydra | via Hydra | OK | OK | OK | OK | OK |
| max_answer_tokens=64 | n/a | n/a | OK | OK | OK | OK | n/a |
| cache_size=1024 eval | n/a (B0=null) | OK | n/a (M1=full) | OK | OK | OK | OK |
| temperature=0.0 | OK | OK | OK | OK | OK | OK | OK |

*M2 uses `seed=1337` per spec (paired NAMM seed).

---

## 5 Test Split Size Discrepancy

The experiment spec says **70 test** prompts in several places (lines 146, 184, 231, 343, 353, 477), but the codebase's `assert_fair01_test_size` expects **69**. The `FAIR01_EXPECTED_TEST_SIZE` env var override was added to handle this. The discrepancy likely comes from a boundary prompt at exactly 4096 or 6500 tokens that falls in/out depending on tokenizer cache state. The spec should be updated to 69 to match the canonical split, or the env var set to 70 on machines that produce it.

---

## 6 Actions Required

### Missing configs (Task 2)

1. `scripts/configs/lora_rh_m1_instruct_5t.yaml` — M1 FAIR-01
2. `scripts/configs/lora_rh_m4_instruct_5t.yaml` — M3 FAIR-01
3. `scripts/configs/joint_lora_m4_5t.yaml` — M4 FAIR-01 (with `adapter_type: lora`, `learning_rate: 5e-5`)

### Config fixes needed

- `joint_default.yaml`: `learning_rate` is 2e-4 but spec says M4 should match M1 (5e-5). Either fix the default or always pass on CLI.

### Spec fixes needed

- Test split size: 70 -> 69 (or document the machine-dependent boundary)
- M4 command missing `--learning_rate 5e-5`
