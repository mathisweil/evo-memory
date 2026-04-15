# checkpoints_backup/

Archived training checkpoints for the 5-task LongBench QA experiments. Each subfolder contains a `best_ckpt.pt` (LoRA) or `ckpt.pt` (NAMM), plus the training config and validation metrics where available.

## LoRA checkpoints

| Folder | Condition | best step | val avg F1 | lr | dropout | NAMM active | cache_size | Notes |
|---|---|---|---|---|---|---|---|---|
| `lora_m1_original` | M1 (LoRA only) | 336 | 45.48 | 5e-5 | 0.1 | No | full | Original M1 run, old hyperparams |
| `lora_m1_lr1e4_matched` | M1 (LoRA only) | 64 | 38.35 | **1e-4** | **0.05** | No | full | M1 with hyperparams matched to M4 cs1024 maskfix. Early-stopped at step 104 (patience=20). |
| `lora_m4_cs1024_original` | M4 (LoRA + NAMM cs1024) | 340 | 45.59 | 1e-4 | 0.05 | Yes | 1024 | Original M4 cs1024 run, pre-maskfix NAMM |
| `lora_m4_cs1024_maskfix` | M4 (LoRA + NAMM cs1024) | 222 / 260 | 47.17 / 52.06 | 1e-4 | 0.05 | Yes | 1024 | M4 cs1024 with maskfix NAMM. Two checkpoints saved at different steps; run crashed. |
| `lora_m4_cs2048_step244` | M4 (LoRA + NAMM cs2048) | 244 | 44.86 | 1e-4 | 0.05 | Yes | 2048 | M4 cs2048, stopped early — needs resumed training |
| `lora_m3_cs2048_maskfix` | M3 (LoRA + NAMM cs2048) | 74 | 25.82 | 5e-5 | ? | Yes | 2048 | M3 cs2048 with maskfix NAMM, lr=5e-5 (not matched to M4) |

## NAMM checkpoints

| Folder | cache_size | best iter | val tasks_agg | val mean F1 | Notes |
|---|---|---|---|---|---|
| `namm_cs1024_original` | 1024 | 105 | 0.002790 | 27.90 | Pre-maskfix, used for original M4 cs1024 LoRA training |
| `namm_cs1024_maskfix` | 1024 | 135 | 0.002034 | 20.34 | Post-maskfix |
| `namm_cs2048_original` | 2048 | 120 | 0.003040 | 30.40 | Trained on powerglide, used for M4 cs2048 LoRA training |
| `namm_cs2048_maskfix` | 2048 | 75 (iter_75.pt) | 0.002198 | 21.98 | Post-maskfix, includes eval results in `bs1_*` subdirs |

## Naming conventions

- **`original`**: checkpoint from before the attention mask fix ("maskfix"). These are the checkpoints used in the `results/main_table_5t/` eval matrix.
- **`maskfix`**: checkpoint from after the attention mask fix. Represents a corrected NAMM training that should be used going forward.
- **`lr1e4_matched`**: M1 run with `lr=1e-4, dropout=0.05` to match the M4 cs1024 hyperparams for a controlled comparison.
- **`step244`**: run that stopped early and may need to be resumed.

## Val metric format

`val_metrics.csv` columns: `step, train_lb_{task}, ..., train_lb_avg_f1, val_lb_{task}, ..., val_lb_avg_f1`. The `avg_f1` is micro (prompt-count-weighted), matching the LoRA trainer's `_evaluate_f1` method.
