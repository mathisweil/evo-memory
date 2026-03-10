# Combined ES Fine-Tuning + NAMM

Fine-tuning LLaMA 3.2-1B-Instruct weights via evolutionary strategies while NAMM's trained eviction policy manages the KV cache. This is the core experiment: can the base model learn to cooperate with its eviction policy?

This combines two existing works:
- **ES fine-tuning**: [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper) (Qiu et al.)
- **NAMM**: Neural Attention Memory Model for KV cache eviction (Weil et al.)

---

## How Both Systems Interact

NAMM and ES fine-tuning operate on completely different parameters at different levels:

| System | What it optimises | Optimiser | During combined run |
|---|---|---|---|
| NAMM | Scoring network (~hundreds of params) | CMA-ES | **Frozen** — pre-trained checkpoint loaded, weights fixed |
| ES fine-tuning | Base LLM weights (~1.24B params) | NES | **Active** — perturbs and updates all base LLM parameters |

The NAMM scoring network is trained first (Stage 1), then frozen. ES fine-tuning (Stage 2) modifies the base LLM weights while the frozen NAMM policy continues to run inside every attention layer, evicting tokens from the KV cache.

### Two-level optimisation view

```
Outer loop (ES):  optimise LLM weights θ to maximise E[reward(θ)]
                  where reward(θ) = F1 on Qasper with NAMM eviction active

Inner loop (fixed): for each forward pass, NAMM scoring network φ* (frozen)
                    scores and evicts tokens from the KV cache
                    φ* was pre-trained on the base model's attention patterns
```

The reward signal for ES passes through the entire pipeline: perturbed LLM weights -> attention patterns -> NAMM scoring -> eviction decisions -> generation quality -> F1 score. ES treats this whole chain as a black box.

### The cooperation hypothesis

If ES successfully fine-tunes the LLM under NAMM eviction, the model may learn to:
- Concentrate important information in tokens that NAMM scores highly
- Produce attention patterns that are more "compressible" under eviction
- Become more robust to cache size constraints

Conversely, if the frozen NAMM policy becomes stale as weights drift, we'd expect reward to plateau or degrade — motivating the alternating optimisation approach (Phase 4 in the research plan).

---

## Parameters (Combined System)

### ES hyperparameters (argparse)

| Parameter | Default | Meaning |
|---|---|---|
| `--run_name` | (required) | Name for this run (e.g. `cache1024_i50`) |
| `--experiment` | auto | Existing experiment ID to add to, or creates new |
| `--method` | auto | `es_namm` or `es_only` (auto-detected from `--namm_checkpoint`) |
| `--sigma` | `0.001` | Noise scale for LLM weight perturbations |
| `--alpha` | `0.0005` | ES learning rate |
| `--population_size` | `8` | Perturbed models per ES iteration |
| `--num_iterations` | `50` | Total ES iterations |
| `--noise_mode` | `correlated` | Noise correlation mode |
| `--initial_seed` | `33` | NumPy random seed for reproducibility |
| `--mini_batch_size` | `16` | Qasper samples per evaluation |
| `--batch_size` | config value | GPU inference batch size for the evaluator |
| `--filter_by_length` | `None` (config default: 6500) | Override the Hydra `filter_by_length` value. Omit to use config default (6500) |
| `--train_samples` | `150` | Qasper samples in training pool |
| `--n_examples` | `10` | Number of Q/A examples to capture during final eval |
| `--resume_checkpoint` | `None` | Path to checkpoint to resume from |
| `--gcs` | `True` | Enable GCS experiment management and checkpointing |
| `--checkpoint_every` | `10` | Save checkpoint to GCS every N iterations |

### NAMM-specific

| Parameter | Value | Meaning |
|---|---|---|
| `--namm_checkpoint` | (required) | Path to pre-trained NAMM `ckpt.pt` |
| `--run_config` | `namm_bam_i1_llama32_1b` | Hydra config (defines cache_size, eviction delay, etc.) |
| `--cache_size` | `None` | Override cache size for NAMM eviction |
| `cache_size` | 1024 | KV-cache budget (from Hydra config) |
| `memory_policy_fixed_delay` | 256 | Tokens between eviction calls (from Hydra config) |

### Model/task config (from Hydra)

| Parameter | Default | Meaning |
|---|---|---|
| `max_new_tokens` | 64 | Max generated tokens per sample. Also filters out samples whose shortest answer exceeds this |
| `max_position_id` | 6500 (= `filter_by_length`) | Max conditioning window; samples longer than this are dropped |

### What the NAMM checkpoint contains

The checkpoint (`ckpt.pt`) stores:
- `evo_state['best_member']` — flat parameter vector for the scoring network (loaded as NAMM weights)
- EMA normalisation buffers — running statistics from the embedding module
- CMA-ES state (mean, sigma, covariance) — not used during ES fine-tuning

When loaded, the scoring network parameters are set via `memory_policy.set_params_batch_idxs(np.zeros([1]))` — fixing the policy to use a single parameter set (not a population) for all evaluations.

---

## Experiment Hierarchy

Results are organised under:
```
experiments/experiment_N/{es_namm,es_only}/run_name/
    config.json      # full configuration snapshot
    results.json     # final eval scores
    examples.json    # captured Q/A examples from final eval
    checkpoints/     # final checkpoint only
```

A `manifest.json` in `experiments/` tracks all experiments and their runs.

---

## Combined Workflow

```
# Stage 1: Train NAMM (done separately, checkpoint already exists)
# φ* = CMA-ES result from training on base LLaMA

# Stage 2: ES fine-tuning with frozen NAMM
load_model()                                     # LLaMA 3.2-1B-Instruct + NAMM hooks in every attention layer
load_namm_checkpoint(namm_checkpoint)             # load φ* into scoring network, freeze it
base_params = get_base_llm_param_names(model)     # 147 tensors, excludes memory_policy.*

for iteration in range(num_iterations):           # 50 iterations
    seeds = random_integers(population_size)
    rewards = []

    for seed in seeds:                            # population_size = 8
        perturb_weights(model, seed, sigma,       # perturb ONLY base LLM params
                        base_params, noise_mode)

        # This evaluation runs the FULL pipeline:
        #   1. Tokenise Qasper input
        #   2. Run LLaMA forward pass (perturbed weights)
        #   3. At every 256-token boundary, NAMM scores and evicts tokens (frozen φ*)
        #   4. Generate up to 64 tokens
        #   5. Compute F1 vs ground truth
        reward = evaluate(model, mini_batch)
        rewards.append(reward)

        restore_weights(model, seed, sigma,
                        base_params, noise_mode)

    normalized = (rewards - mean) / (std + 1e-8)
    apply_es_update(model, seeds, normalized,     # update ONLY base LLM params
                    sigma, alpha, base_params,
                    population_size, noise_mode)

# After training: baseline eval + final full eval + Q/A example capture
# Only the final checkpoint is saved (no intermediate checkpoints)
```

### What happens during each forward pass

```
Input tokens: [t1, t2, ..., tN]     (N ~ 2000-4000 for Qasper)

For each chunk of 256 tokens:
  1. LLaMA processes chunk -> attention KV pairs added to cache
  2. If |cache| > cache_size:
     a. NAMM embedding: STFT spectrogram of KV vectors + recency encoding
     b. NAMM scoring: BAM network scores each token (using frozen φ*)
     c. Selection: keep top-cache_size tokens, evict the rest
  3. Continue to next chunk

Generate up to 64 tokens using the evicted cache
Compute F1 against ground truth answer
```

---

## Forward Pass Calculation

The combined system has the same forward pass count as standalone ES fine-tuning, but each forward pass is slightly more expensive due to NAMM eviction overhead:

```
Forward passes per ES iteration:
  = population_size x mini_batch_size
  = 8 x 16
  = 128 forward passes (each with NAMM eviction)

Total forward passes:
  = num_iterations x population_size x mini_batch_size
  = 50 x 8 x 16
  = 6,400 forward passes
```

### Cost comparison

| Pipeline | Fwd passes (50 iter) | Cost per fwd pass | Relative cost |
|---|---|---|---|
| ES only (full cache) | 6,400 | Full LLaMA inference, no eviction | 1.0x |
| ES + NAMM | 6,400 | Full LLaMA inference + NAMM scoring/eviction | ~1.05-1.1x [TODO: measure] |
| NAMM only (200 iter, pop=8) | 25,600 | Full LLaMA inference + NAMM scoring/eviction | -- |

The NAMM scoring overhead is small (tiny MLP+attention on a few hundred tokens), so combined cost ~ ES-only cost. The dominant cost is always the LLaMA forward pass.

### Total wall time estimate

| Config | Est. time/iter | Total estimate |
|---|---|---|
| pop=8, mini_batch=16, bs=8, NAMM active | ~12-15 min | ~10-13h (50 iter) |
| pop=8, mini_batch=4, bs=8, NAMM active | ~3-4 min | ~2.5-3.3h (50 iter) |
| [TODO: measure actual overhead of NAMM vs no-NAMM] | | |

---

## Comparison: Standalone vs Combined

| Aspect | NAMM only | ES only | ES + NAMM |
|---|---|---|---|
| What's optimised | Scoring network φ | LLM weights θ | LLM weights θ (φ frozen) |
| Optimiser | CMA-ES | NES | NES |
| Parameter count | ~hundreds | ~1.24B | ~1.24B (+ frozen φ) |
| Forward passes/iter | pop x samples = 128 | pop x mini_batch = 128 | pop x mini_batch = 128 |
| Typical runtime | ~44h (200 iter) | ~10h (50 iter) | ~10-13h (50 iter) |
| Eviction during eval | Yes (optimised) | Optional (if checkpoint loaded) | Yes (frozen) |
| Reward signal | F1 reflects eviction quality | F1 reflects LLM quality | F1 reflects LLM+eviction interaction |

---

## Reflections and Open Questions

### Policy staleness
The NAMM scoring network was trained on the base model's attention patterns. As ES modifies the LLM weights, the attention patterns change. Does the frozen NAMM policy remain effective? Signs of staleness:
- Reward plateaus before matching full-cache fine-tuning performance
- NAMM starts evicting "wrong" tokens (important tokens scored low)
- Validation reward declines after initial improvement

If staleness is a problem -> Phase 4 (alternating optimisation) is needed.

### Is the reward signal rich enough?
The F1 score on 16 Qasper samples (default mini_batch_size) captures the combined effect of LLM quality AND eviction quality. ES can't distinguish between "better LLM weights" and "weights that happen to work better with this particular frozen NAMM policy". Is this conflation a problem, or does it naturally steer the model toward cooperation?

### Sigma in the combined setting
With NAMM eviction active, the loss landscape may be different than without eviction. Eviction introduces a discrete, non-smooth operation (top-k selection). Small weight perturbations might cause different tokens to be evicted, creating discontinuities in the reward. Does this affect the optimal `sigma`?

### LoRA integration path
Instead of perturbing all 1.24B parameters, ES could perturb only LoRA adapters (~10M parameters). This would:
- Reduce the effective dimensionality of the search
- Require much smaller population for reliable gradient estimates
- Allow direct comparison with gradient-based LoRA fine-tuning
- Keep the base model frozen (only adapters change), which might preserve NAMM policy validity longer

Implementation: add LoRA adapters via PEFT, then modify `get_base_llm_param_names()` to return only LoRA parameters.

### Joint optimisation (future)
Rather than freezing NAMM and only optimising LLM weights, could we optimise both simultaneously? Two approaches:
1. **Alternating**: 50 ES iterations on LLM weights, then 50 CMA-ES iterations on NAMM, repeat
2. **Bilevel**: Use ES for LLM weights with NAMM as an inner optimisation (re-train NAMM from scratch for each LLM perturbation — prohibitively expensive)
3. **Single ES**: Concatenate NAMM params + LLM params into one vector and run ES on the combined space — but the scale mismatch (~hundreds vs ~1.24B) would make this poorly conditioned

---

## Tests to Run

### 1. Measure NAMM overhead in ES fine-tuning
```bash
# Compare ES fine-tuning with and without NAMM
python scripts/run_es.py \
    --run_name timing_no_namm \
    --num_iterations 10 \
    --population_size 8 \
    --mini_batch_size 16

python scripts/run_es.py \
    --run_name timing_with_namm \
    --namm_checkpoint /path/to/ckpt.pt \
    --num_iterations 10 \
    --population_size 8 \
    --mini_batch_size 16
```

### 2. Full combined run
```bash
python scripts/run_es.py \
    --run_name full_es_namm \
    --namm_checkpoint /path/to/ckpt.pt \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16 \
    --sigma 0.001 \
    --alpha 0.0005
```
Record: reward curve, total wall time, final checkpoint.

### 3. Compare ES-only vs ES+NAMM results
```bash
# Compare results.json from both runs
cat experiments/experiment_N/es_only/run_name/results.json
cat experiments/experiment_N/es_namm/run_name/results.json
```
Key question: does the NAMM run converge to a similar reward as the no-NAMM run? Higher? Lower? Slower?

### 4. Evaluate the combined model under all eviction policies
```bash
CKPT=experiments/experiment_N/es_namm/run_name/checkpoints/es_checkpoint_final.pt

# Under NAMM (same policy used during training)
python scripts/run_namm.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=$CKPT \
    cache_size=1024

# Under full cache (upper bound — does fine-tuning under NAMM hurt full-cache perf?)
python scripts/run_namm.py \
    'run@_global_=full_cache_baseline_llama32_1b.yaml' \
    init_from=$CKPT

# Under recency (does it generalise to a different eviction policy?)
python scripts/run_namm.py \
    'run@_global_=recency_baseline_llama32_1b.yaml' \
    init_from=$CKPT \
    cache_size=1024
```

### 5. Build the 2x3 comparison grid
Run tests 4 above for both ES-only and ES+NAMM checkpoints. Fill in:

| Model | Eviction | cache | qasper | passage_ret | narrativeqa |
|---|---|---|---|---|---|
| base | full | 4096 | 8.30 | 3.59 | 7.32 |
| base | NAMM | 1024 | 7.00 | 3.58 | 6.91 |
| base | recency | 1024 | 1.76 | 0.78 | 0.80 |
| ES-FT (no NAMM) | full | 4096 | [TODO] | [TODO] | [TODO] |
| ES-FT (no NAMM) | NAMM | 1024 | [TODO] | [TODO] | [TODO] |
| ES-FT (no NAMM) | recency | 1024 | [TODO] | [TODO] | [TODO] |
| ES-FT (with NAMM) | NAMM | 1024 | [TODO] | [TODO] | [TODO] |
| ES-FT (with NAMM) | full | 4096 | [TODO] | [TODO] | [TODO] |
| ES-FT (with NAMM) | recency | 1024 | [TODO] | [TODO] | [TODO] |

### 6. Sigma sweep with NAMM active
```bash
for SIGMA in 0.0005 0.001 0.005; do
    python scripts/run_es.py \
        --run_name sigma_namm_${SIGMA} \
        --namm_checkpoint /path/to/ckpt.pt \
        --num_iterations 50 \
        --sigma $SIGMA
done
```
Does the optimal sigma differ with/without NAMM eviction active?

---

## Commands Reference

**Train NAMM first (Stage 1):**
```bash
torchrun --standalone --nproc_per_node=1 scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```

**Then ES fine-tune with NAMM (Stage 2):**
```bash
python scripts/run_es.py \
    --run_name my_run \
    --namm_checkpoint experiments/.../ckpt.pt \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16
```

**Evaluate:**
```bash
python scripts/run_namm.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=/path/to/es_checkpoint_final.pt \
    cache_size=1024
```

**Key files:**
- `scripts/run_es.py` — combined pipeline entry point
- `namm/trainer.py` — NAMM CMA-ES training (Stage 1)
- `es_finetuning/trainer.py` — ESTrainer (Stage 2)
- `cfgs/run/namm_bam_i1_llama32_1b.yaml` — Hydra config for model/task/eviction

---

NAMM paper: [An Evolved Universal Transformer Memory](https://arxiv.org/abs/2410.13166) — Cetin et al., SakanaAI 2024
ES paper: [Scalable Gradient-Free Fine-Tuning of Language Models](https://arxiv.org/abs/2509.24372) — Qiu et al. 2025
