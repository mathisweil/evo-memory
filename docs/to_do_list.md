# To-Do List

Ordered steps to get the full NAMM + ES fine-tuning pipeline running and produce results.

Smoke tests are done. Environment works. All three pipelines (NAMM eval, NAMM training, ES fine-tuning) have been verified on a single GPU.

---

## Phase 0: Validate Baselines

- [x] **0.1** ~~Train NAMM scoring network~~ — using pretrained checkpoint (`namm_pretrained_romain_v2.pt`)

- [x] **0.2** Evaluate trained NAMM checkpoint across cache sizes — done at c1024/c3072/c5120 via experiment_2 baselines

- [x] **0.3** Run full-cache baseline evaluation — es_only baseline F1=14.08

- [x] **0.4** Run recency baseline evaluation — obtained from es_recency training baselines

- [x] **0.5** Record base model row of the results grid

| Model | Eviction | cache | qasper |
| ----- | -------- | ----- | ------ |
| base  | full     | full  | 14.08  |
| base  | NAMM     | 1024  | 11.65  |
| base  | NAMM     | 3072  | 13.12  |
| base  | NAMM     | 5120  | 14.43  |
| base  | recency  | 1024  | 1.28   |
| base  | recency  | 3072  | 5.05   |
| base  | recency  | 5120  | 10.50  |

---

## Phase 1: ES Fine-Tune Without NAMM (Control)

- [x] **1.1** Run ES fine-tuning with full cache (no NAMM checkpoint) — es_only_mb16, 50 iter, F1: 14.08→30.78

- [x] **1.2** Evaluate ES-fine-tuned (no NAMM) model under all eviction policies

| Policy        | F1                           |
| ------------- | ---------------------------- |
| Full cache    | 30.78 (= es_only final eval) |
| NAMM c1024    | 21.06 (post-hoc)             |
| NAMM c3072    | 28.82 (post-hoc)             |
| NAMM c5120    | 29.33 (post-hoc)             |
| Recency c1024 | 1.20                         |
| Recency c3072 | 8.69                         |
| Recency c5120 | 20.59                        |

- [x] **1.3** Record ES-FT (no NAMM) row of the results grid

---

## Phase 2: ES Fine-Tune With NAMM (Core Experiment)

- [x] **2.1** Run ES fine-tuning with frozen NAMM eviction active — 3 runs completed:

| Run           | Baseline F1 | Final F1 |
| ------------- | ----------- | -------- |
| es_namm c1024 | 11.65       | 22.53    |
| es_namm c3072 | 13.12       | 27.46    |
| es_namm c5120 | 14.43       | 31.85    |

- [ ] **2.2** Re-run ES+NAMM training to convergence (~1000 iterations)
  Experiment_2 ran only 50 iterations (matching the ES paper's setup for small tests), but the paper uses 500–1000 for full convergence. Resume from experiment_2 checkpoints or start fresh:
  ```bash
  for CS in 1024 3072 5120; do
      python scripts/run_es.py \
          --run_name cache${CS}_i1000 \
          --namm_checkpoint latest \
          --num_iterations 1000 \
          --cache_size $CS \
          --save_every 100
  done
  ```

- [x] ~~**2.3** Cross-eval ES+NAMM under other policies~~ — dropped (not needed for main story)

- [x] **2.4** Record ES-FT (with NAMM) row of the results grid (50-iter preliminary)

---

## NOTE: Recency Baseline Gap — RESOLVED

All recency baselines and ES+recency training runs are now complete in experiment_2:

- Base model + recency: from es_recency training baselines (c1024=1.28, c3072=5.05, c5120=10.50)
- ES-only + recency eval: eval_recency runs (c1024=1.20, c3072=8.69, c5120=20.59)
- ES+recency training: es_recency runs (c1024=2.56, c3072=8.99, c5120=20.35)

NAMM clearly outperforms recency at all cache sizes, both for the base model and after ES fine-tuning.

---

## Phase 3: Compare and Analyse

- [ ] **3.1** Assemble the full results grid and compare
  - Does ES-FT (with NAMM) beat base model under NAMM? -> cooperation hypothesis
  - Does ES-FT (with NAMM) also improve under full cache? -> generalisation
  - Does ES-FT (with NAMM) beat ES-FT (no NAMM) under NAMM? -> specialisation
  - Does NAMM beat recency baseline? -> learned eviction value

- [ ] **3.2** Compare results from ES with NAMM vs ES without NAMM
  ```bash
  cat experiments/experiment_N/es_namm/run_name/results.json
  cat experiments/experiment_N/es_only/run_name/results.json
  ```
  Key: convergence speed, final reward, variance.

- [ ] **3.3** Generate a comparison report
  ```bash
  python scripts/generate_report.py
  ```

---

## Phase 4: Hyperparameter Sweeps (If Time Permits)

- [ ] **4.1** Sigma sweep with NAMM active
  ```bash
  for SIGMA in 0.0005 0.001 0.005; do
      python scripts/run_es.py \
          --run_name sigma_sweep_${SIGMA} \
          --namm_checkpoint latest \
          --num_iterations 50 \
          --sigma $SIGMA
  done
  ```

- [ ] **4.2** Population size sweep
  ```bash
  for POP in 4 8 16; do
      python scripts/run_es.py \
          --run_name pop_sweep_${POP} \
          --namm_checkpoint latest \
          --num_iterations 50 \
          --population_size $POP
  done
  ```

- [ ] **4.3** Noise mode comparison (correlated vs iid)
  ```bash
  for MODE in correlated iid; do
      python scripts/run_es.py \
          --run_name noise_${MODE} \
          --namm_checkpoint latest \
          --num_iterations 50 \
          --noise_mode $MODE
  done
  ```

---

## Phase 5: Alternating Optimisation (If Staleness Is a Problem)

- [ ] **5.1** Design alternating schedule (e.g. 50 ES iters -> 50 CMA-ES iters -> repeat)
- [ ] **5.2** Implement outer script that alternates between `scripts/run_es.py` and NAMM re-training via `scripts/run_namm.py`
- [ ] **5.3** Run alternating optimisation and compare with frozen-policy results
- [ ] **5.4** Ablate alternation frequency (every 25/50/100 ES iterations)

---

## Status

Phases 0–1 are **complete**. Phase 2 has preliminary results at 50 iterations but likely needs ~1000 for convergence (step 2.2).

**Next:**
- **2.2**: Re-run ES+NAMM to 1000 iterations (main compute blocker)
- **3.1**: Assemble results grid (can do preliminary analysis with 50-iter data now)

Phases 4–5 are stretch goals if time permits.
