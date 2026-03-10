# To-Do List

Ordered steps to get the full NAMM + ES fine-tuning pipeline running and produce results.

Smoke tests are done. Environment works. All three pipelines (NAMM eval, NAMM training, ES fine-tuning) have been verified on a single GPU.

---

## Phase 0: Validate Baselines

- [x] **0.1** ~~Train NAMM scoring network~~ — using pretrained checkpoint (`namm_pretrained_romain_v2.pt`)

- [x] **0.2** Evaluate trained NAMM checkpoint across cache sizes — done at c1024/c3072/c5120 via experiment_2 baselines

- [x] **0.3** Run full-cache baseline evaluation — es_only baseline F1=14.08

- [ ] **0.4** Run recency baseline evaluation
  ```bash
  python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml'
  ```

- [ ] **0.5** Record base model row of the results grid

  | Model | Eviction | cache | qasper |
  |---|---|---|---|
  | base | full | full | 14.08 |
  | base | NAMM | 1024 | 11.65 |
  | base | NAMM | 3072 | 13.12 |
  | base | NAMM | 5120 | 14.43 |
  | base | recency | 1024 | **TODO** |

---

## Phase 1: ES Fine-Tune Without NAMM (Control)

- [x] **1.1** Run ES fine-tuning with full cache (no NAMM checkpoint) — es_only_mb16, 50 iter, F1: 14.08→30.78

- [ ] **1.2** Evaluate ES-fine-tuned (no NAMM) model under all three eviction policies

  | Policy | Done? | F1 |
  |---|---|---|
  | Full cache | **DONE** | 30.78 (= es_only final eval) |
  | NAMM c1024 | **DONE** | 21.06 (post-hoc) |
  | NAMM c3072 | **DONE** | 28.82 (post-hoc) |
  | NAMM c5120 | **DONE** | 29.33 (post-hoc) |
  | Recency c1024 | **TODO** | — |

  Recency eval still needed:
  ```bash
  CKPT=experiments/experiment_2/es_only/es_only_mb16/checkpoints/es_checkpoint_final.pt
  python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
  ```

- [ ] **1.3** Record ES-FT (no NAMM) row of the results grid — blocked on recency eval

---

## Phase 2: ES Fine-Tune With NAMM (Core Experiment)

- [x] **2.1** Run ES fine-tuning with frozen NAMM eviction active — 3 runs completed:

  | Run | Baseline F1 | Final F1 |
  |---|---|---|
  | es_namm c1024 | 11.65 | 22.53 |
  | es_namm c3072 | 13.12 | 27.46 |
  | es_namm c5120 | 14.43 | 31.85 |

- [ ] **2.2** Evaluate ES-fine-tuned (with NAMM) model under all three eviction policies — **NOT DONE**
  Need: full-cache eval and recency eval for each of the 3 ES+NAMM checkpoints.

- [ ] **2.3** Record ES-FT (with NAMM) row of the results grid — blocked on 2.2

---

## NOTE: Recency Baseline Gap

Experiment 2 evaluated NAMM eviction and full-cache, but **no recency (sliding window) baseline** was run. This is needed to show that NAMM's learned eviction actually outperforms the simplest eviction strategy. Without it, a reviewer could argue that any fixed-window policy achieves similar results.

**Still needed:**
- Recency baseline on base model (step 0.4)
- Recency eval on ES-FT (no NAMM) weights (step 1.2)
- Recency eval on ES-FT (with NAMM) weights (step 2.2)
- Full-cache eval on ES-FT (with NAMM) weights (step 2.2)
- Optionally: ES fine-tune with recency eviction active (analogous to ES+NAMM but with a non-learned policy) — this would show whether the LLM can cooperate with *any* eviction policy or specifically benefits from NAMM's learned one

Use `run@_global_=recency_baseline_llama32_1b.yaml` config.

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
          --namm_checkpoint /path/to/ckpt.pt \
          --num_iterations 50 \
          --sigma $SIGMA
  done
  ```

- [ ] **4.2** Population size sweep
  ```bash
  for POP in 4 8 16; do
      python scripts/run_es.py \
          --run_name pop_sweep_${POP} \
          --namm_checkpoint /path/to/ckpt.pt \
          --num_iterations 50 \
          --population_size $POP
  done
  ```

- [ ] **4.3** Noise mode comparison (correlated vs iid)
  ```bash
  for MODE in correlated iid; do
      python scripts/run_es.py \
          --run_name noise_${MODE} \
          --namm_checkpoint /path/to/ckpt.pt \
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

## Critical Path

The minimum to produce a meaningful result:

```
0.1 Train NAMM (~44h)
 -> 0.2-0.4 Baselines (~1h each)
     -> 1.1 ES fine-tune no NAMM
     |   -> 1.2 Evaluate (~1h)
     -> 2.1 ES fine-tune with NAMM  [can run in parallel with 1.1 if two GPUs]
         -> 2.2 Evaluate (~1h)
             -> 3.1 Compare results
```

**Total estimated wall time (single GPU, sequential):** ~65h (~3 days)
**With two GPUs (Phases 1 and 2 in parallel):** ~55h (~2.5 days)
