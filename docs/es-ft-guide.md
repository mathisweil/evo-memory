# ES Fine-Tuning of Base LLM Weights

Gradient-free weight optimisation of LLaMA 3.2-1B-Instruct using evolutionary strategies, with optional NAMM eviction active during evaluation.

---

## What is ES Fine-Tuning?

Evolutionary Strategy (ES) fine-tuning optimises the base LLM weights without computing gradients. Instead of backpropagation, it:

1. **Perturbs** the model weights with random noise
2. **Evaluates** each perturbed model on a task
3. **Estimates** the gradient direction from the reward-noise correlation
4. **Updates** the weights in the direction that correlates with higher reward

This is the OpenAI-ES / Natural Evolution Strategy (NES) approach. It scales to large models because it never needs to store activations or compute a backward pass — only forward passes.

The key advantage for our setting: ES can optimise through any evaluation pipeline, including non-differentiable components like NAMM's KV-cache eviction and text generation.

---

## Parameters

### ES hyperparameters (argparse)

| Parameter | Default | Meaning |
|---|---|---|
| `--run_name` | (required) | Name for this run (e.g. `cache1024_i50`) |
| `--experiment` | auto | Existing experiment ID to add to, or creates new |
| `--method` | auto | `es_namm` or `es_only` (auto-detected from `--namm_checkpoint`) |
| `--sigma` | `0.001` | Noise scale for weight perturbations |
| `--alpha` | `0.0005` | ES learning rate (step size for weight update) |
| `--population_size` | `8` | Number of perturbed models evaluated per iteration |
| `--num_iterations` | `150` | Total ES iterations |
| `--noise_mode` | `correlated` | Noise correlation: `correlated` or `iid` |
| `--initial_seed` | `33` | NumPy random seed for reproducibility |
| `--mini_batch_size` | `16` | Qasper samples per population member evaluation |
| `--namm_checkpoint` | `None` | Path to pre-trained NAMM `ckpt.pt` (optional) |
| `--run_config` | `namm_bam_i1_llama32_1b` | Hydra config name for model/task setup |
| `--batch_size` | config value | GPU inference batch size for the evaluator |
| `--filter_by_length` | `None` (config default: 6500) | Override the Hydra `filter_by_length` value. Omit to use config default (6500) |
| `--cache_size` | `None` | Override cache size for NAMM eviction |
| `--train_samples` | `150` | Qasper samples in training pool |
| `--val_samples` | `50` | Qasper samples for validation |
| `--n_examples` | `10` | Number of Q/A examples to capture during final eval |
| `--resume_checkpoint` | `None` | Path to checkpoint to resume from |

### Model/task config (from Hydra)

| Parameter | Default | Meaning |
|---|---|---|
| `cache_size` | 1024 | KV-cache budget (tokens kept after eviction) |
| `memory_policy_fixed_delay` | 256 | Tokens between NAMM eviction calls |
| `max_new_tokens` | 64 | Max generated tokens per sample. Also filters out samples whose shortest answer exceeds this |
| `max_position_id` | 6500 (= `filter_by_length`) | Max conditioning window; samples longer than this are dropped |

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

## ES Training Workflow

```
for iteration in range(num_iterations):          # 150 iterations
    seeds = random_integers(population_size)      # one seed per population member

    resample_batch(mini_batch_size)               # draw a SHARED batch for this iteration
                                                  # all pop members evaluated on same data

    rewards = []
    for seed in seeds:                            # population_size = 8
        perturb_weights(model, seed, sigma)       # add noise: p += sigma * noise(seed)
        reward = evaluate(model, mini_batch)      # run NAMM inference on mini_batch_size Qasper samples
        rewards.append(reward)
        restore_weights(model, seed, sigma)       # subtract same noise: p -= sigma * noise(seed)

    # Normalise rewards to zero mean, unit variance
    normalized = (rewards - mean(rewards)) / (std(rewards) + 1e-8)

    # Update weights: gradient estimate from reward-noise correlation
    for each parameter p:
        update = sum(normalized[k] * noise_k for k in population) / population_size
        p += alpha * update

# After training: baseline eval + final full eval + Q/A example capture
# Only the final checkpoint is saved (no intermediate checkpoints)
```

**Important:** All population members in a given iteration are evaluated on the same batch of samples (resampled once via `pre_step_fn`). This matches how NAMM's CMA-ES trainer works and ensures reward differences between members reflect weight quality, not sample variance.

**Note:** Validation during training has been removed. Only a baseline evaluation (before training) and a final full evaluation (after training) are performed. Q/A examples are captured during the final evaluation.

### Step-by-step breakdown

**Perturb** (`perturb_weights`):
```python
generator.manual_seed(seed)           # deterministic noise from seed
noise = torch.randn(p.shape)          # same shape as parameter
p.data += sigma * noise               # shift weights
```

**Restore** (`restore_weights`):
```python
generator.manual_seed(seed)           # SAME seed → SAME noise
noise = torch.randn(p.shape)
p.data -= sigma * noise               # undo the perturbation exactly
```

**Update** (`apply_es_update`):
```python
# For each population member k, replay its noise and weight by normalised reward
for k, seed in enumerate(seeds):
    generator.manual_seed(seed)
    noise = torch.randn(p.shape)
    p.data += (alpha / population_size) * normalized[k] * noise
```

The deterministic seed-based noise replay is the key trick: we never store the noise tensors, only the seeds. Memory cost is O(population_size) integers regardless of model size.

---

## Noise Injection Modes

### `correlated` (default)
Every parameter tensor in the model receives noise from the **same** random seed per population member. This means the noise patterns across layers are correlated — if layer 1's Q projection gets a positive perturbation in a certain direction, layer 2's Q projection gets a similarly-structured perturbation.

```python
# Same seed used for ALL parameter tensors for a given population member
gen.manual_seed(seed)
for p in model.parameters():
    noise = torch.randn(p.shape, generator=gen)
    p.data += sigma * noise
```

### `iid`
Each parameter tensor gets its own independent random seed. Perturbations across layers are statistically independent.

```python
# Different seed per parameter tensor
for i, p in enumerate(model.parameters()):
    gen.manual_seed(seed + i)
    noise = torch.randn(p.shape, generator=gen)
    p.data += sigma * noise
```

**Trade-off:** Correlated noise means fewer effective degrees of freedom per perturbation, which can make the gradient estimate more biased but lower variance. IID noise gives an unbiased estimate but may need more population members to reduce variance. The paper uses correlated as default.

---

## Forward Pass Calculation

Each ES iteration evaluates `population_size` perturbed models, each on `mini_batch_size` samples:

```
Forward passes per ES iteration:
  = population_size x mini_batch_size
  = 8 x 16
  = 128 forward passes

Total forward passes for full training:
  = num_iterations x population_size x mini_batch_size
  = 150 x 8 x 16
  = 19,200 forward passes
```

Each forward pass runs the full LLaMA inference pipeline (with NAMM eviction if a checkpoint is loaded). The cost per forward pass is the same as NAMM evaluation — dominated by the LLaMA attention layers, not the tiny NAMM scoring network.

---

## Which Tensors Are Optimised

ES perturbs **all base LLM parameters** — everything except `memory_policy.*`. For LLaMA 3.2-1B-Instruct, this includes:

| Tensor group | Count | Shape | Total params |
|---|---|---|---|
| `model.embed_tokens.weight` | 1 | (128256, 2048) | 262M |
| `self_attn.{q,k,v,o}_proj.weight` | 4 x 16 layers | (2048, 2048) or GQA variants | ~268M |
| `mlp.{gate,up,down}_proj.weight` | 3 x 16 layers | (2048, 5632) and transpose | ~541M |
| `{input,post_attention}_layernorm.weight` | 2 x 16 layers | (2048,) | 65K |
| `model.norm.weight` | 1 | (2048,) | 2K |
| `lm_head.weight` | 1 | (128256, 2048) | 262M |
| **Total** | **147 tensors** | | **~1.24B parameters** |

Every one of these tensors gets perturbed by `sigma * noise` and then restored after evaluation. The perturbation is applied in the model's native dtype (bfloat16).

---

## Timing

**Measured (smoke test, single Quadro RTX 6000 24 GB):**

| Config | Time/iter | GPU mem |
|---|---|---|
| pop=2, mini_batch=2, bs=8 | ~22s | 11 GB |

**Extrapolated to full runs:**

| Config | Est. time/iter | Total estimate |
|---|---|---|
| pop=8, mini_batch=16, bs=8 | ~12 min | ~30h (150 iter) |
| pop=8, mini_batch=4, bs=8 | ~3 min | ~7.5h (150 iter) |
| pop=4, mini_batch=4, bs=8 | ~1.5 min | ~3.8h (150 iter) |
| pop=8, mini_batch=8, bs=8 | [TODO: measure] | [TODO] |

**Logging:** Results are written to `experiments/experiment_N/{es_namm,es_only}/run_name/`:
- `config.json` — full run configuration
- `results.json` — final evaluation scores
- `examples.json` — captured Q/A examples

**Checkpoint format:** `experiments/experiment_N/.../run_name/checkpoints/es_checkpoint_final.pt` — dict of `{param_name: tensor}` for optimised base LLM parameters only. Only the final checkpoint is saved.

---

## Reflections and Open Questions

### Parameter count
ES is perturbing all 1.24 billion parameters simultaneously. The gradient estimate from a population of 8 is extremely low-rank — we're estimating a billion-dimensional gradient from 8 samples. Classical ES theory says you need `O(n)` population members for reliable estimates, which is obviously intractable.

Why does it work at all? Likely because:
- The loss landscape has low effective dimensionality (not all 1.24B dimensions matter equally)
- Reward normalisation focuses the update on the relative ordering of perturbations
- Correlated noise reduces the effective search space further

But is pop=8 enough? Would pop=16 or pop=32 converge faster per iteration (despite taking longer per iteration)?

### Compute cost vs gradient-based fine-tuning
For reference, LoRA fine-tuning of LLaMA 3.2-1B-Instruct with rank 16 would update ~10M parameters using standard backprop. ES updates 1.24B parameters using 128 forward passes per iteration (with mini_batch_size=16). The convergence rate is likely much slower for ES.

Is the gradient-free property worth the cost? For this project, yes — because we need to optimise through the non-differentiable NAMM eviction pipeline. But exploring LoRA + straight-through estimators could be a faster alternative.

### Sigma sensitivity
`sigma=0.001` is a tiny perturbation relative to the weight magnitudes (typically O(0.01-1.0) in bfloat16). Too small -> negligible reward differences -> noisy gradient. Too large -> perturbations break the model -> all candidates score ~0. The sweet spot depends on the loss landscape curvature. Has sigma=0.001 been validated, or should we sweep?

### Convergence
150 iterations x 8 population = 1,200 total perturbation-evaluation pairs. For a 1.24B parameter model, this is minuscule. Are we seeing meaningful convergence, or just noise? The final eval results should show a clear improvement over baseline if learning is happening.

### Mini-batch size
Each perturbed model is evaluated on 16 Qasper samples (default). The F1 reward on 16 samples is moderately noisy. If two perturbations get different random samples, the reward difference may reflect sample variance rather than weight quality.

---

## Tests to Run

### 1. Timing at different configs
```bash
# Sweep population_size and mini_batch_size, 5 iterations each
for POP in 4 8 16; do
    for MB in 2 4 8 16; do
        python scripts/run_es.py \
            --run_name timing_pop${POP}_mb${MB} \
            --num_iterations 5 \
            --population_size $POP \
            --mini_batch_size $MB
    done
done
```

### 2. Sigma sweep
```bash
for SIGMA in 0.0001 0.0005 0.001 0.005 0.01; do
    python scripts/run_es.py \
        --run_name sigma_${SIGMA} \
        --num_iterations 50 \
        --sigma $SIGMA
done
```
Compare: reward convergence curves across sigma values.

### 3. Noise mode comparison
```bash
for MODE in correlated iid; do
    python scripts/run_es.py \
        --run_name noise_${MODE} \
        --num_iterations 50 \
        --noise_mode $MODE
done
```
Compare: convergence speed and final reward.

### 4. Full ES fine-tuning run (no NAMM)
```bash
python scripts/run_es.py \
    --run_name full_no_namm \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 16 \
    --sigma 0.001 \
    --alpha 0.0005
```
Record: total wall time, final training and validation reward, checkpoint path.

### 5. Full ES fine-tuning run (with NAMM)
```bash
python scripts/run_es.py \
    --run_name full_with_namm \
    --namm_checkpoint /path/to/ckpt.pt \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 16 \
    --sigma 0.001 \
    --alpha 0.0005
```
Record: total wall time, reward curve, does it converge differently with NAMM active?

### 6. Evaluate ES-fine-tuned model
```bash
# After training, evaluate under all three eviction policies
for CONFIG in full_cache_baseline_llama32_1b namm_bam_eval_llama32_1b recency_baseline_llama32_1b; do
    python scripts/run_namm.py \
        "run@_global_=${CONFIG}.yaml" \
        init_from=/path/to/es_checkpoint_final.pt
done
```

---

## Commands Reference

**Train (no NAMM):**
```bash
python scripts/run_es.py \
    --run_name my_run \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 16
```

**Train (with NAMM):**
```bash
python scripts/run_es.py \
    --run_name my_run \
    --namm_checkpoint /path/to/ckpt.pt \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 16
```

**Monitor:**
```bash
# Results in experiments/experiment_N/{es_namm,es_only}/run_name/results.json
```

**Key files:**
- `scripts/run_es.py` — entry point and argument parser
- `es_finetuning/trainer.py` — ESTrainer loop
- `es_finetuning/noise.py` — perturb/restore/update functions
- `es_finetuning/config.py` — ESConfig dataclass

---

ES method reference: [Scalable Gradient-Free Fine-Tuning of Language Models](https://arxiv.org/abs/2509.24372) — Qiu et al. 2025
