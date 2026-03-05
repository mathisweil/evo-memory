# NAMM on LLaMA 3.2-1B

Reproduction of [An Evolved Universal Transformer Memory](https://arxiv.org/abs/2410.13166) (SakanaAI)
ported to LLaMA 3.2-1B, with fixes for PyTorch 2.7 compatibility and batch parallelism in the scoring network.

---

## Dependencies — read this first

This is where most setup time goes. Pin these versions exactly.

**System requirement: GLIBC ≥ 2.28**
Check with `ldd --version`. RHEL 7 / CentOS 7 (GLIBC 2.17) will not work with
PyTorch 2.3+. You need RHEL 8/9, Rocky 8/9, Ubuntu 20.04+, or equivalent.

**Critical version pins:**
```
torch==2.3.1          (cu121 build)
transformers==4.41.2  (4.45+ breaks DynamicCache API — hard failure at runtime)
peft==0.11.1          (newer versions depend on transformers 4.45+)
numpy<2               (numpy 2.x breaks many downstream packages)
```

**Install:**
```bash
source scripts/activate.sh
```

**HuggingFace access** (LLaMA is a gated model):
```bash
huggingface-cli login   # paste your token from huggingface.co/settings/tokens
```
Then request access to `meta-llama/Llama-3.2-1B` on HuggingFace if you haven't already.

**HF cache** (avoid filling your home quota):
```bash
export HF_HOME=/path/to/large/storage/.hf_cache
```

**wandb** (optional but recommended):
```bash
wandb login   # paste your API key from wandb.ai/authorize
```

---

## Training NAMM (stage 1)

Trains the NAMM scoring network on QASPER with CMA-ES. LLaMA weights stay frozen.

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```

Key config values (`cfgs/run/namm_bam_i1_llama32_1b.yaml`):
- `max_iters: 200` — CMA-ES iterations
- `pop_size: 8` — population size
- `cache_size: 1024` — KV-cache budget during training
- `batch_size: 4` — prompts per forward pass (sweet spot for 16GB GPU)

The best checkpoint is saved to `experiments/.../ckpt.pt` when validation F1 improves.

---

## Evaluation

Run all three methods (NAMM, recency, full-cache) across cache sizes:

```bash
bash scripts/run_eval_comparison.sh
```

Edit the `NAMM_CKPT` path at the top of the script to point to your checkpoint.
Results are logged to wandb under `project=memory_evolution_hf`, `group=Llama-3.2-1B/eval-comparison`.

Or run a single NAMM eval manually:

```bash
python run_namm_training.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=/path/to/ckpt.pt \
    cache_size=512
```

---

## Results (LLaMA 3.2-1B, 3-task LongBench subset)

| Method      | cache_size | qasper | passage_ret | narrativeqa |
|-------------|------------|--------|-------------|-------------|
| Full-cache  | 4096       | 8.30   | 3.59        | 7.32        |
| NAMM        | 1024       | 7.00   | 3.58        | 6.91        |
| NAMM        | 512        | 7.07   | 3.52        | 7.03        |
| NAMM        | 256        | 7.02   | 3.57        | 6.34        |
| NAMM        | 128        | 6.46   | 2.92        | 5.67        |
| Recency     | 1024       | 1.76   | 0.78        | 0.80        |
| Recency     | 512        | 1.12   | 0.00        | 1.01        |
| Recency     | 256        | 0.76   | 0.00        | 1.03        |
| Recency     | 128        | 0.44   | 0.00        | 0.60        |

NAMM trained only on QASPER generalises to passage retrieval and NarrativeQA zero-shot.

---

## Codebase changes vs original SakanaAI repo

- `memory_trainer.py` — `torch.load(..., weights_only=False)` for PyTorch 2.7 compat
- `stateless_parallel_modules/attention.py` — attn_mask expansion fix for `pop_size > 1` with `batch_size > 1` in both `StatelessAttention` and `MonoHeadStatelessAttention`
- `cfgs/model/wrapped_llm/llama32-1b.yaml` — LLaMA 3.2-1B model config (`max_position_id=4096`)
- `cfgs/run/*_llama32_1b.yaml` — training and eval configs for LLaMA 3.2-1B
- `cfgs/task/lb_3subset_eval.yaml` — 3-task eval subset (qasper, passage_ret, narrativeqa)
- `scripts/run_eval_comparison.sh` — script to sweep all methods and cache sizes

---

Original paper: [arxiv 2410.13166](https://arxiv.org/abs/2410.13166) — Cetin et al., SakanaAI 2024
