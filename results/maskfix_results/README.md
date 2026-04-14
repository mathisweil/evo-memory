# Maskfix Results — Deterministic Evaluation

All results use:
- **batch_size=1** (deterministic, no OOM retry artifacts)
- **Maskfix NAMM** (correct attention, causal mask fix applied)
- **cs=1024** cache budget where eviction is active
- **5-task QA** subset: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e
- **Greedy decoding** (temperature=0)
- **Chat template** applied to all prompts

## Results

| Condition | KV cache | test micro F1 | ext micro F1 |
|-----------|------:|---:|---:|
| B0 plain | full | 22.41 | 22.42 |
| Trunc plain 1024 | 1024 | 17.89 | 17.88 |
| M2 NAMM cs1024 | 1024 | 19.27 | 18.70 |
| M1 LoRA (full cache) | full | 27.97 | 25.75 |
| Trunc LoRA 1024 | 1024 | 25.08 | 23.71 |
| M1 under NAMM cs1024 | 1024 | 27.29 | 21.76 |
| M4 LoRA +NAMM cs1024 | 1024 | 33.51 | 25.84 |
| A4 LoRA (no NAMM) | full | 36.07 | 24.91 |

## Plots

- `plots/mean_f1_test.png` — Mean F1 bar chart, test split
- `plots/mean_f1_extended_test.png` — Mean F1 bar chart, extended test
- `plots/per_task_test.png` — Per-task grouped bars, test
- `plots/per_task_extended_test.png` — Per-task grouped bars, extended test

## Checkpoints used

- NAMM: `eval_results/namm_cs1024_maskfix/ckpt.pt` (iter 135, GCS backup available)
- M4 LoRA: `checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt`
- M1 LoRA: `results/rh_m1_lora_instruct_5t/42/best_ckpt.pt`
- GCS: `gs://statistical-nlp/evo-memory/checkpoints_backup_20260414/`
