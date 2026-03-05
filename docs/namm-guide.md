# NAMM Memory Policy Training via CMA-ES

Training a neural KV-cache eviction policy for LLaMA 3.2-1B using Covariance Matrix Adaptation Evolution Strategy.

---

## What is NAMM?

NAMM (Neural Attentive Memory Management) is a learned KV-cache eviction policy. Instead of keeping all tokens in the cache (expensive) or using a simple heuristic like recency (poor quality), NAMM trains a small scoring network that decides which tokens to keep.

The scoring network runs inside every attention layer. Every `memory_policy_fixed_delay` tokens (default 256), it scores all cached tokens and keeps only the top-`cache_size` by score.

### 3-stage pipeline

1. **Embedding** — Each cached token's KV vectors are transformed into a fixed-size feature via STFT spectrogram + recency positional encoding, reduced by exponential moving average.
2. **Scoring** — A small MLP+attention network (BAM: Block Attention Memory) reads the per-token embeddings and outputs one scalar score per token per head.
3. **Selection** — Tokens are ranked by score; the top-`cache_size` tokens are kept, the rest are evicted from the KV cache.

The scoring network is tiny (hundreds of parameters). The LLaMA model weights are completely frozen during NAMM training — only the scoring network is optimised.

---

## Parameters

### CMA-ES and training parameters

| Parameter | Paper (Mistral-7B) | Our setup (LLaMA 3.2-1B) | Meaning |
|---|---|---|---|
| `max_iters` | 200 | 200 | CMA-ES generations |
| `pop_size` | 32 | 8 | Population size (candidates per generation) |
| `samples_batch_size` | 16 | 16 | Qasper samples evaluated per CMA-ES step |
| `batch_size` | — | 1 | GPU inference batch size (1 for 16 GB, 4–8 for 24 GB) |
| `cache_size` | 1024 | 1024 | KV-cache budget (tokens kept after eviction) |
| `memory_policy_fixed_delay` | 256 | 256 | Tokens between eviction calls |
| `max_new_tokens` | 64 | 64 | Max generated tokens per evaluation sample |
| `elite_ratio` | 0.5 | 0.5 | Top fraction used in CMA mean update |
| `init_sigma` | 0.065 | 0.065 | Initial CMA-ES step size |
| `c_m` | 1.0 | 1.0 | Mean update learning rate |
| `prefer_mean_to_best` | true | true | Checkpoint the CMA mean, not the best member |
| `scoring_initializer` | 0 | 0 | Scoring network initialised to all zeros |
| `per_head` | false | false | Shared scoring params across attention heads |
| `per_layer` | false | false | Shared scoring params across transformer layers |

### BAM scoring network architecture

| Parameter | Value | Meaning |
|---|---|---|
| `scoring_attn_hidden_dim` | 32 | Hidden dimension of the attention scoring head |
| `scoring_attn_num_heads` | 1 | Number of attention heads in scoring network |
| `scoring_attn_bias` | true | Use bias in scoring attention |
| `scoring_attn_masking_strategy` | `backward` | Causal masking |
| `embedding_reduction_mode` | `ema` | Exponential moving average on token embeddings |
| `embedding_ema_coeff` | 0.99 | EMA decay coefficient |

### Token embedding (STFT spectrogram)

| Parameter | Value |
|---|---|
| `n_fft` | 32 |
| `hop_length` | 16 |
| `window_fn` | Hann, length 32 |
| `recency_embed_dim` | 8 |
| `recency_max_freq` | 50000 |
| `joining_strategy` | `append` |

---

## CMA-ES Training Workflow

CMA-ES is a derivative-free optimiser that maintains a multivariate Gaussian over the parameter space. Each generation:

```
for iteration in range(max_iters):        # 200 generations
    candidates = cma.ask()                 # sample pop_size parameter vectors from N(mean, sigma^2 * C)

    for candidate in candidates:           # pop_size = 8
        policy.set_params(candidate)       # load scoring network weights
        fitness = evaluate(policy, task)   # run LLaMA with this eviction policy on samples_batch_size Qasper samples

    cma.tell(fitnesses)                    # update mean, covariance C, step size sigma

    if fitness improved:
        save_checkpoint()                  # saves mean (not best member) to ckpt.pt
```

### What `ask()` does
- Draw `z ~ N(0, I)` for each population member
- Transform: `x = mean + sigma * B * D * z` where `C = B * D^2 * B^T` is the eigendecomposition of the covariance
- Return `pop_size` candidate parameter vectors

### What `tell()` does
- Sort candidates by fitness (descending — higher F1 is better)
- Update `mean` via weighted average of top-`elite_ratio` fraction
- Update evolution paths `p_sigma` and `p_c`
- Update covariance `C` via rank-1 + rank-mu updates
- Update step size `sigma` via cumulative step-size adaptation (CSA)

### CMA-ES internal hyperparameters (auto-computed from `pop_size` and `param_size`)
| Symbol | Formula | Role |
|---|---|---|
| `c_sigma` | `(mu_eff + 2) / (n + mu_eff + 5)` | Step-size adaptation rate |
| `d_sigma` | `1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + c_sigma` | Step-size damping |
| `c_c` | `(4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)` | Rank-1 path decay |
| `c_1` | `alpha_cov / ((n+1.3)^2 + mu_eff)` | Rank-1 learning rate |
| `c_mu` | `min(1-c_1, alpha_cov * (mu_eff - 2 + 1/mu_eff) / ((n+2)^2 + alpha_cov*mu_eff/2))` | Rank-mu learning rate |

Where `n = min(param_size, 40000)` for numerical stability, `alpha_cov = 2`.

---

## Forward Pass Calculation

Each CMA-ES iteration requires `pop_size` evaluations, each evaluating `samples_batch_size` Qasper samples:

```
Forward passes per CMA-ES iteration:
  = pop_size x samples_batch_size
  = 8 x 16
  = 128 forward passes

Total forward passes for full training:
  = max_iters x pop_size x samples_batch_size
  = 200 x 8 x 16
  = 25,600 forward passes
```

Each forward pass processes a long Qasper document (typically 2000–4000 tokens input, 64 tokens generated), with the NAMM scoring network running at every `memory_policy_fixed_delay=256` token boundary.

---

## Timing

**Measured (smoke test, single Quadro RTX 6000 24 GB):**

| Config | Time/iter | GPU mem |
|---|---|---|
| pop=2, samples=2, new_tok=16, bs=8 | ~50s | 11 GB |

**Extrapolated to full run:**

| Config | Est. time/iter | Total estimate |
|---|---|---|
| pop=8, samples=16, new_tok=64, bs=8 | ~13 min | ~44h (200 iter) |
| pop=8, samples=16, new_tok=64, bs=4 | [TODO: measure] | [TODO] |
| pop=8, samples=16, new_tok=64, bs=1 | [TODO: measure] | [TODO] |

**Notes:**
- batch_size=8 uses ~11 GB on a 24 GB GPU. This is the sweet spot.
- batch_size=16 uses ~22 GB and is sluggish due to memory pressure.
- Always override `eval_samples_batch_size` — the default (128) silently makes eval steps very slow.

---

## Reflections and Open Questions

### Benchmark choice
NAMM trains on QASPER (scientific paper QA). The paper shows it generalises zero-shot to passage retrieval and NarrativeQA. But QASPER F1 scores are low even for the full-cache baseline (8.30). Is there enough signal in the reward for CMA-ES to learn effectively? Would training on a task with higher absolute scores give a stronger gradient signal?

### Metric noise
CMA-ES uses F1 score on a random 16-sample batch as fitness. F1 on 16 samples is noisy — variance comes from both sample selection and the stochastic generation. Does this noise slow down CMA-ES convergence? Would larger `samples_batch_size` (e.g. 32) help, or does CMA-ES handle noisy fitness naturally through its population-based averaging?

### Parameter count
The scoring network is tiny — shared across all 16 layers and all 8 KV heads. The total parameter vector is a few hundred floats. CMA-ES maintains a full covariance matrix over this space (`O(n^2)` memory). If `per_head=true` or `per_layer=true` were enabled, the parameter count would grow 8x or 16x respectively. Is the shared parameterisation a meaningful bottleneck?

### Population size
We use pop_size=8 vs the paper's 32. Smaller population means faster iterations but worse CMA-ES covariance estimation and more generations to converge. Is 8 sufficient for this parameter space, or are we paying for it in final quality?

### Initialisation
All scoring parameters start at zero (`scoring_initializer=0`), meaning the initial policy assigns equal scores to all tokens — effectively random eviction. CMA-ES must discover meaningful scoring from this blank slate. Would a warm-start (e.g. initialising towards recency-like scoring) speed convergence?

---

## Tests to Run

These experiments fill in the `[TODO]` timing blanks and answer open questions above.

### 1. Timing at different batch sizes
```bash
# Measure time per CMA-ES iteration at different batch_size values
for BS in 1 2 4 8; do
    torchrun --standalone --nproc_per_node=1 run_namm_training.py \
        run@_global_=namm_bam_i1_llama32_1b.yaml \
        max_iters=3 \
        batch_size=$BS \
        wandb_log=false
done
```
Record: wall time per iteration, GPU memory at each batch_size.

### 2. Effect of population size on convergence
```bash
# Compare pop=4 vs pop=8 vs pop=16 on short runs (50 iters)
for POP in 4 8 16; do
    torchrun --standalone --nproc_per_node=1 run_namm_training.py \
        run@_global_=namm_bam_i1_llama32_1b.yaml \
        max_iters=50 \
        pop_size=$POP \
        wandb_project=namm_pop_sweep
done
```
Compare: best fitness at iteration 50, time per iteration.

### 3. Effect of samples_batch_size on fitness noise
```bash
# Compare samples=8 vs samples=16 vs samples=32
for SAMPLES in 8 16 32; do
    torchrun --standalone --nproc_per_node=1 run_namm_training.py \
        run@_global_=namm_bam_i1_llama32_1b.yaml \
        max_iters=50 \
        samples_batch_size=$SAMPLES \
        wandb_project=namm_samples_sweep
done
```
Compare: fitness variance per iteration, convergence speed.

### 4. Full 200-iteration training run
```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```
Record: total wall time, final fitness, checkpoint path. This is the checkpoint needed for ES fine-tuning (Stage 2).

### 5. Evaluate trained NAMM at different cache sizes
```bash
for CACHE in 128 256 512 1024; do
    python run_namm_training.py \
        'run@_global_=namm_bam_eval_llama32_1b.yaml' \
        init_from=/path/to/ckpt.pt \
        cache_size=$CACHE
done
```
Record: qasper F1, passage_retrieval accuracy, narrativeqa F1 at each cache size.

---

## Commands Reference

**Train:**
```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```

**Evaluate:**
```bash
python run_namm_training.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=/path/to/ckpt.pt \
    cache_size=1024
```

**Key config file:** `cfgs/run/namm_bam_i1_llama32_1b.yaml`

**Checkpoint location:** `experiments/.../ckpt.pt` (saved when validation F1 improves)

---

Original paper: [An Evolved Universal Transformer Memory](https://arxiv.org/abs/2410.13166) — Cetin et al., SakanaAI 2024
CMA-ES reference: [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) — Hansen 2016
