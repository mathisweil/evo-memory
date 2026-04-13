# Naming Mapping: M-numbering vs Historical Names

Generated: 2026-04-13

The codebase predates the M-numbering in the experiment spec. This document is the
definitive mapping between all naming systems.

---

## Definitive mapping

| Spec | Description | WandB method | Config file | Results dir | Script |
|------|-------------|-------------|-------------|-------------|--------|
| B0 | Base, full cache | — | `full_cache_baseline_llama32_1b` (Hydra) | `B0/` | `run_eval.py` |
| B1 | Base + recency | — | `recency_baseline_llama32_1b` (Hydra) | `B1/cs{N}/` | `run_eval.py` |
| M1 | LoRA only | `rh_m1_lora_instruct_5t` | `lora_rh_m1_instruct_5t.yaml` | `M1/` | `run_lora.py` |
| M2 | Standalone NAMM | (auto from Hydra) | `namm_bam_i1_llama32_1b_5t` (Hydra) | `M2/cs{N}/` | `run_namm.py` |
| M3 | LoRA + frozen NAMM | `rh_m4_frozen_5t` | `lora_rh_m4_instruct_5t.yaml` | `M4/cs{N}/` | `run_lora.py` |
| M4 | Joint LoRA + NAMM | `joint_lora` | `joint_lora_m4_5t.yaml` | `joint_lora/` | `run_joint.py` |
| A4 | M4 NAMM on/off | — | `eval_main_table.yaml` | `A4/cs{N}_no_namm/` | `run_eval.py` |

---

## The M3/M4 confusion

This is the critical ambiguity. In the spec:

- **M3** = LoRA fine-tuned with a frozen NAMM active during training
- **M4** = Joint alternating NAMM + LoRA co-training

In the codebase and WandB:

- **`rh_m4_frozen`** / **`rh_m4_frozen_5t`** = what the spec calls **M3** (frozen NAMM)
- **`joint_lora`** = what the spec calls **M4** (joint)
- **`results/main_table_5t/M4/`** = contains **M3** (frozen NAMM) results, NOT M4 (joint)

### Where "M4" means M3

1. `scripts/organize_eval_results.py` — `JOBS` entries `lora_m4_cs{1024,2048}_5t` map to `dst: "M4/cs{N}"`. These are M3 results.
2. `results/main_table_5t/M4/` directory — M3 eval results.
3. `results/main_table_5t/A4/` directory — A4 ablation was run on M3 checkpoints, not M4 (joint has not been run).
4. WandB runs named `rh_m4_5t_cs*` — all M3.
5. `scripts/configs/lora_rh_m4_instruct.yaml` and `lora_rh_m4_instruct_5t.yaml` — M3 configs.
6. GCS paths `lora-m4-frozen-5t-cs*` — M3 checkpoints.

### Where "M4" means M4

1. `scripts/run_joint.py` — `method: joint_lora` or `joint_es`. This is the actual M4.
2. `scripts/configs/joint_default.yaml` and `joint_lora_m4_5t.yaml` — M4 configs.
3. `experiment_specification.md` §3 M4-LoRA — correct usage.

---

## Recommendation

**Do not rename.** The `rh_m4_frozen` string appears in:

- WandB run history (immutable)
- GCS checkpoint paths (referenced by M3 eval scripts)
- `experiments/experiment_N/` directory structures
- `organize_eval_results.py` job definitions
- Analysis notebooks and report scripts

Renaming would break reproducibility of existing results. Instead:

1. All new code and configs use the spec naming (M1/M2/M3/M4).
2. Comments in configs explicitly note the mapping (done in `lora_rh_m4_instruct_5t.yaml`).
3. This document serves as the canonical reference.
4. The `experiment_specification.md` warning in §6 stays as-is.
