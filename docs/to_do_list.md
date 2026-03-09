# To-Do List

Ordered steps to get the full NAMM + ES fine-tuning pipeline running and produce results.

Smoke tests are done. Environment works. All three pipelines (NAMM eval, NAMM training, ES fine-tuning) have been verified on a single GPU.

---

## Phase 0: Validate Baselines

- [ ] **0.1** Train NAMM scoring network for real (200 iterations, ~44h)
  ```bash
  torchrun --standalone --nproc_per_node=1 scripts/run_namm.py \
      run@_global_=namm_bam_i1_llama32_1b.yaml
  ```
  Record: total wall time, final fitness, checkpoint path.

- [ ] **0.2** Evaluate trained NAMM checkpoint across cache sizes
  ```bash
  for CACHE in 128 256 512 1024; do
      python scripts/run_namm.py \
          'run@_global_=namm_bam_eval_llama32_1b.yaml' \
          init_from=/path/to/ckpt.pt \
          cache_size=$CACHE
  done
  ```
  Confirm results match or improve on the README table.

- [ ] **0.3** Run full-cache baseline evaluation
  ```bash
  python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml'
  ```

- [ ] **0.4** Run recency baseline evaluation
  ```bash
  python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml'
  ```

- [ ] **0.5** Record base model row of the results grid

  | Model | Eviction | cache | qasper | passage_ret | narrativeqa |
  |---|---|---|---|---|---|
  | base | full | 4096 | | | |
  | base | NAMM | 1024 | | | |
  | base | recency | 1024 | | | |

---

## Phase 1: ES Fine-Tune Without NAMM (Control)

- [ ] **1.1** Run ES fine-tuning with full cache (no NAMM checkpoint)
  ```bash
  python scripts/run_es.py \
      --run_name no_namm_full \
      --num_iterations 150 \
      --population_size 8 \
      --mini_batch_size 16 \
      --sigma 0.001 \
      --alpha 0.0005
  ```
  Monitor reward curve. Results saved to `experiments/experiment_N/es_only/no_namm_full/`.

- [ ] **1.2** Evaluate ES-fine-tuned (no NAMM) model under all three eviction policies
  ```bash
  CKPT=experiments/experiment_N/es_only/no_namm_full/checkpoints/es_checkpoint_final.pt

  python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT
  python scripts/run_namm.py 'run@_global_=namm_bam_eval_llama32_1b.yaml' init_from=$CKPT cache_size=1024
  python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
  ```

- [ ] **1.3** Record ES-FT (no NAMM) row of the results grid

  | Model | Eviction | cache | qasper | passage_ret | narrativeqa |
  |---|---|---|---|---|---|
  | ES-FT (no NAMM) | full | 4096 | | | |
  | ES-FT (no NAMM) | NAMM | 1024 | | | |
  | ES-FT (no NAMM) | recency | 1024 | | | |

---

## Phase 2: ES Fine-Tune With NAMM (Core Experiment)

- [ ] **2.1** Run ES fine-tuning with frozen NAMM eviction active
  ```bash
  python scripts/run_es.py \
      --run_name with_namm_full \
      --namm_checkpoint /path/to/ckpt.pt \
      --num_iterations 150 \
      --population_size 8 \
      --mini_batch_size 16 \
      --sigma 0.001 \
      --alpha 0.0005
  ```
  Monitor reward curve. Results saved to `experiments/experiment_N/es_namm/with_namm_full/`.

- [ ] **2.2** Evaluate ES-fine-tuned (with NAMM) model under all three eviction policies
  ```bash
  CKPT=experiments/experiment_N/es_namm/with_namm_full/checkpoints/es_checkpoint_final.pt

  python scripts/run_namm.py 'run@_global_=namm_bam_eval_llama32_1b.yaml' init_from=$CKPT cache_size=1024
  python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT
  python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
  ```

- [ ] **2.3** Record ES-FT (with NAMM) row of the results grid

  | Model | Eviction | cache | qasper | passage_ret | narrativeqa |
  |---|---|---|---|---|---|
  | ES-FT (with NAMM) | NAMM | 1024 | | | |
  | ES-FT (with NAMM) | full | 4096 | | | |
  | ES-FT (with NAMM) | recency | 1024 | | | |

---

## Phase 3: Compare and Analyse

- [ ] **3.1** Assemble the full 3x3 results grid and compare
  - Does ES-FT (with NAMM) beat base model under NAMM? -> cooperation hypothesis
  - Does ES-FT (with NAMM) also improve under full cache? -> generalisation
  - Does ES-FT (with NAMM) beat ES-FT (no NAMM) under NAMM? -> specialisation

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
