# Naming Mapping: Current vs Historical Names

Generated: 2026-04-13. Updated 2026-04-14 after the source-side rename.

Source-side configs and method names now match the M-numbering in the experiment
spec. External artefacts (WandB, GCS, on-disk results dirs) still carry the old
strings — they are immutable and remain documented here.

---

## Current source-side names

| Spec | Description | `method` (in YAML / new WandB runs) | Config file | Script |
|------|-------------|-------------------------------------|-------------|--------|
| B0 | Base, full cache | — | `full_cache_baseline_llama32_1b` (Hydra) | `run_eval.py` |
| B1 | Base + recency | — | `recency_baseline_llama32_1b` (Hydra) | `run_eval.py` |
| M1 | LoRA only | `m1_lora_5t` | `scripts/configs/m1_lora_5t.yaml` | `run_lora.py` |
| M2 | Standalone NAMM | (auto from Hydra) | `namm_bam_i1_llama32_1b_5t` (Hydra) | `run_namm.py` |
| M3 | LoRA + frozen NAMM | `m3_lora_frozen_namm_5t` | `scripts/configs/m3_lora_frozen_namm_5t.yaml` | `run_lora.py` |
| M4 | Joint LoRA + NAMM | `joint_lora` | `scripts/configs/m4_joint_lora_5t.yaml` | `run_joint.py` |
| A4 | M4 NAMM on/off | — | `scripts/configs/eval_main_table.yaml` | `run_eval.py` |

---

## Historical artefacts (do not rewrite)

These strings live outside source and are intentionally NOT updated:

| External system | Old string | What it actually is |
|---|---|---|
| WandB run names | `rh_m1_5t_v2`, completed M1 runs | M1 |
| WandB run names | `rh_m4_5t_cs{1024,2048,3072}` | **M3** (frozen NAMM, despite the `m4` in the name) |
| WandB `method` field on completed runs | `rh_m1_lora_instruct_5t` | M1 |
| WandB `method` field on completed runs | `rh_m4_frozen_5t`, `rh_m4_frozen` | M3 |
| GCS paths | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/` | M1 |
| GCS paths | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs*-llama32-1b/` | **M3** |
| Results dir | `results/main_table_5t/M1/` | M1 |
| Results dir | `results/main_table_5t/M4/cs{N}/` | **M3** (frozen NAMM eval) |
| Results dir | `results/main_table_5t/A4/cs1024_no_namm/` | A4 ablation on M3 checkpoints |
| Local checkpoint paths | `results/rh_m1_lora_instruct_5t/42/best_ckpt.pt` | M1 best checkpoint |
| Local checkpoint paths | `results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt` | M3 best checkpoint |
| Hydra task config | `rh_multi_qa_5t` | 5-task QA subset (current) |
| Hydra run preset | `namm_bam_i1_llama32_1b_5t` | M2/M3/M4 NAMM preset (current) |

The `m4` in the historical strings is a leftover from before the M-numbering
existed; M4 (joint) had not been run when those strings were coined.

---

## Where the legacy strings still appear in source

The following source files reference legacy strings on purpose, because they
point at on-disk checkpoints whose paths have not been moved:

- `scripts/organize_eval_results.py` — `JOBS` entries embed `results/rh_*` paths
  for command-string templates.
- `scripts/_claude_*.sh` helper scripts — hardcoded checkpoint paths under
  `results/rh_*`.
- `scripts/generate_paper_figures.py:103` — globs `**/rh_m4_frozen/...` against
  the on-disk results layout.
- `results/main_table_5t/**/{command,README}.{sh,md}` — eval-run logs captured
  at the time of the run; not source.

If a future cleanup re-organises `results/`, those references should be updated
in lockstep.
