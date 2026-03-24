# Claude Code Prompt — Paper Figures & Tables for `evo-memory`

Generate all graphs, tables, and result assets needed for the research paper.

---

## Context

This project fine-tunes Llama-3.2-1B-Instruct on Qasper (token-level F1) via LoRA or Evolutionary Strategies (ES), co-trained with a NAMM (Neural Attention Memory Model) KV-cache eviction policy. All experiment results live under `experiments/experiment_N/` and follow the schema in `experiment_specification.md`.

**Your task:** create `scripts/generate_paper_figures.py` — a self-contained script that reads all experiment results and produces every table and figure needed for the paper.

---

## 1 · Result Collection

Write a helper that walks the experiments directory and loads every `results.json` it finds. The standard schema is:

```json
{
  "f1": 0.0,
  "exact_match": 0.0,
  "num_samples": 200,
  "cache_size": 1024,
  "method": "m1_lora_only"
}
```

Joint runs produce a **list** of entries (one per outer loop).

Map each found result to one of the canonical run names below by matching the directory path:

| ID | Path pattern |
|----|-------------|
| B0 | `experiments/*/baseline/results.json` |
| B1 | `experiments/*/es_recency/b1_recency/results.json` |
| M1-r4 | `experiments/*/m1_lora_only/m1_r4/results.json` |
| M1-r8 | `experiments/*/m1_lora_only/m1_r8/results.json` |
| M1-r16 | `experiments/*/m1_lora_only/m1_r16/results.json` |
| M1-ES | `experiments/*/es_only/m1_es/results.json` |
| M2 | `outputs/**/results.json` — most recent with `wandb_run_name=m2_namm_standalone` |
| M3-LoRA | `outputs/**/results.json` — `wandb_run_name=m3_namm_on_lora` |
| M3-ES | `outputs/**/results.json` — `wandb_run_name=m3_namm_on_es` |
| M4-LoRA | `experiments/*/joint_lora/m4_joint_lora/results.json` |
| M4-ES | `experiments/*/joint_es/m4_joint_es/results.json` |
| A2-512 | `experiments/ablations/a2_cache/m2_cache512/results.json` |
| A2-2048 | `experiments/ablations/a2_cache/m2_cache2048/results.json` |
| A4-on | `experiments/ablations/a4_modularity/m4_namm_on/results.json` |
| A4-off | `experiments/ablations/a4_modularity/m4_namm_off/results.json` |
| A5-LoRA | `experiments/*/m1_lora_only/a5_lora_frozen_namm/results.json` |
| A5-ES | `experiments/*/es_namm/a5_es_frozen_namm/results.json` |

For any result that is **missing**, substitute `None` and emit a visible `WARNING` — never crash, always produce partial output.

Also check `outputs/` for NAMM training logs: look for `outputs/**/training_log.json`, `outputs/**/fitness_history.json`, or any JSON/CSV containing per-generation fitness/F1 values for M2 and M4 NAMM stages.

---

## 2 · Summary CSV

Before generating any figure, write `paper_figures/all_results.csv` with columns:

```
condition_id, condition_name, optimizer, namm, cache_size, f1, exact_match, num_samples
```

All figures and tables should be generated from this file — it is the single source of truth.

---

## 3 · Figures

Save all figures to `paper_figures/`. Use **matplotlib** with a clean academic style (`seaborn-whitegrid`, 10 pt font, `tight_layout`). Export as both `.pdf` (for LaTeX) and `.png` (for review). Use a **consistent colour palette** across every plot:

| Group | Colour |
|-------|--------|
| B0 / B1 baselines | grey |
| M1-LoRA family | blue shades |
| M1-ES family | orange shades |
| M2 — NAMM only | green |
| M3 sequential | purple |
| M4 joint | red |
| A5 frozen-NAMM | teal |

---

### Figure 1 — Main results bar chart (gradient conditions, FAIR-01)

Grouped horizontal bar chart. Conditions in order: B0, B1, M1-LoRA-r8, M2, M3-LoRA, M4-LoRA. X-axis: Qasper token-level F1. Add a vertical dashed line at B0 (full-cache upper reference). Label each bar with its F1 value.

Output: `paper_figures/fig1_main_results.pdf`

---

### Figure 2 — ES variants bar chart

Same style as Figure 1. Conditions: B0, B1, M1-ES, M1-ES+M2-NAMM (M1-ES weights evaluated with frozen M2 NAMM), M2, M3-ES, M4-ES. Add a caption note that ES is not compute-equivalent to gradient methods.

Output: `paper_figures/fig2_es_results.pdf`

---

### Figure 3 — LoRA rank ablation (A1)

Bar chart. X-axis: rank ∈ {4, 8, 16}. Y-axis: Qasper F1. Add error bars if multiple seeds are available. Annotate the chosen rank r=8 with an asterisk.

Output: `paper_figures/fig3_rank_ablation.pdf`

---

### Figure 4 — Cache size sweep / accuracy–memory Pareto (A2)

Line plot. X-axis: cache size in tokens (512, 1024, 2048). Y-axis: Qasper F1. Plot the M2 curve. Add a secondary x-axis showing compression ratio (full\_cache / cache\_size, assuming full\_cache = 4086). Add horizontal dashed reference lines for B0 and B1.

Output: `paper_figures/fig4_cache_sweep.pdf`

---

### Figure 5 — Modularity test (A4)

Three-bar chart: M1-LoRA-r8 · M4-LoRA with NAMM disabled at eval · M4-LoRA with NAMM enabled at eval. Illustrates whether M4 depends on NAMM at inference time.

Output: `paper_figures/fig5_modularity.pdf`

---

### Figure 6 — Frozen NAMM ablation (A5)

Side-by-side bars: M1-LoRA vs A5-LoRA (LoRA trained under frozen NAMM), and M1-ES vs A5-ES. Isolates whether training under compressed context helps independently of co-optimisation.

Output: `paper_figures/fig6_frozen_namm.pdf`

---

### Figure 7 — NAMM training curve (M2)

Line plot. X-axis: CMA-ES generation (0–300). Y-axis: best-of-generation fitness (F1). If population mean + std are available, add a shaded confidence band. Load from M2 training logs under `outputs/`. If no per-generation log exists, skip with a warning and produce a placeholder.

Output: `paper_figures/fig7_namm_training_curve.pdf`

---

### Figure 8 — Joint training learning curve (M4-LoRA)

Line plot. X-axis: outer loop index (1, 2, …). Two lines: NAMM F1 after each NAMM stage, LoRA F1 after each LoRA stage. Load from the list entries in `M4-LoRA/results.json`.

Output: `paper_figures/fig8_joint_learning_curve.pdf`

---

## 4 · Tables

Render all tables as **images** using `matplotlib` (`matplotlib.table` or `pandas` + `df.style`). Export each as both `.pdf` and `.png`, consistent with the figures. Save to `paper_figures/tables/`. Use the same 10 pt font and clean academic style as the figures. Bold the best F1 value in each table.

---

### Table 1 — Main results (gradient conditions)

Columns: Condition · Fine-tuning · Memory Policy · Cache · Qasper F1

Rows: B0, B1, M1-LoRA-r8, M2, M3-LoRA, M4-LoRA. Add a horizontal rule between baselines and trained conditions.

Output: `paper_figures/tables/tab1_main_results.pdf`

---

### Table 2 — ES variants

Same structure as Table 1. Add a footnote below the table: *"† ES is not compute-equivalent to gradient methods (≈38,400 vs ≈1,600 forward passes); direct comparison is invalid."*

Output: `paper_figures/tables/tab2_es_results.pdf`

---

### Table 3 — LoRA rank ablation (A1)

Columns: Rank r · # Trainable Params · Qasper F1

Compute trainable parameter counts analytically for Llama-3.2-1B (hidden\_size = 2048, num\_layers = 16, targeting q\_proj and v\_proj):

```
params = 2 × num_layers × 2 × hidden_size × r
```

Annotate the r=8 row as the main condition.

Output: `paper_figures/tables/tab3_rank_ablation.pdf`

---

### Table 4 — Full results

Consolidates all conditions. Columns: ID · Condition · Optimizer · NAMM · Cache · F1

Include all 17 conditions: B0, B1, M1-r4, M1-r8, M1-r16, M1-ES, M2, M3-LoRA, M3-ES, M4-LoRA, M4-ES, A2-512, A2-2048, A4-on, A4-off, A5-LoRA, A5-ES.

Output: `paper_figures/tables/tab4_full_results.pdf`

---

## 5 · CLI

Add a CLI at the bottom of `scripts/generate_paper_figures.py`:

```bash
python scripts/generate_paper_figures.py \
    --experiment_dir experiments/experiment_N \
    --outputs_dir outputs \
    --paper_figures_dir paper_figures
```

At the end of every run, print a summary listing:
- Each figure/table generated successfully with its output path
- Each figure/table skipped due to missing data, with a note on which `results.json` was not found
