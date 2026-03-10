# TPU Port Plan

Port ES fine-tuning (+ NAMM) from 4x RTX 6000 (UCL cluster) to Google Cloud TPU v4-32 (on-demand, us-central2-b) via TRC allocation.

**Goal**: 10h/experiment → ~20 min/experiment by parallelising population members across TPU chips.

---

## Phase 0: TPU VM Setup & Environment

**Create `setup/setup_tpu.sh`** — a new setup script specifically for the TPU VM environment.

Must handle:
- Python 3.10+ venv creation
- `torch` + `torch_xla` installation (matching versions, e.g. torch 2.3 + torch_xla 2.3)
- HuggingFace `transformers`, `accelerate`, `datasets`
- All other deps from requirements.txt **except** `bitsandbytes` (GPU-only, not needed)
- `HF_HOME` set to a persistent location on the VM
- Download Llama 3.2 1B weights on first run
- Clone repo or rsync from UCL cluster
- No CUDA env vars (`CUDA_VISIBLE_DEVICES` etc.)

```bash
# Rough outline for setup_tpu.sh
python3 -m venv venv
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install transformers accelerate datasets hydra-core numpy'<2'
pip install rouge jieba fuzzywuzzy einops scipy sentencepiece tensorboard
# ... remaining deps from requirements.txt minus bitsandbytes
```

**Provision the TPU VM:**
```bash
gcloud compute tpus tpu-vm create es-finetune \
    --zone=us-central2-b \
    --accelerator-type=v4-32 \
    --version=tpu-vm-v4-pt-2.3  # or latest pytorch image
```

---

## Phase 1: Device Abstraction

Replace all hardcoded CUDA references with a device variable. These changes should be backwards-compatible (still works on GPU).

### Files to change:

| File | What to change |
|---|---|
| `scripts/run_es.py` | Replace `memory_model.cuda()` with `memory_model.to(device)`. Replace `.to('cuda')` (2x) with `.to(device)`. Wrap `torch.backends.cuda/cudnn` settings in `if torch.cuda.is_available()` |
| `scripts/run_eval.py` | Same pattern as run_es.py |
| `namm/evaluator.py` | Replace `self.memory_policy.cuda()` with `.to(device)` |
| `cfgs/config.yaml` | Change `device: 'cuda'` to be overridable (or auto-detect) |
| `es_finetuning/noise.py` | `torch.Generator(device=p.device)` — already device-agnostic, but verify on XLA. Wrap `torch.cuda.synchronize/empty_cache` in conditional |
| `es_finetuning/trainer.py` | Wrap `torch.cuda.is_available()` memory tracking in conditional |
| `es_finetuning/utils.py` | Wrap `torch.cuda.*` calls in conditional |
| `utils/helpers.py` | Replace `.cuda()` calls in `pack_attn_mxs()` with `.to(device)` |

### Device detection helper:

```python
# es_finetuning/device.py (new file)
import torch

def get_device():
    """Return the best available device."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def sync_device(device):
    """Synchronise the device (barrier)."""
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def empty_cache(device):
    """Free device memory cache if applicable."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # XLA manages memory automatically — no-op on TPU
```

---

## Phase 2: Fix CUDA-Only APIs

### 2a. `@torch.cuda.amp.custom_fwd` decorator

**File:** `namm/policy/base.py` (lines 122, 269)

```python
# Before (CUDA-only)
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)

# After (generic, works on all devices)
@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
# OR simply remove if not needed (we're not using AMP on TPU)
```

Note: `torch.amp.custom_fwd` (without `.cuda`) is available since PyTorch 2.0. May need to conditionalise the `device_type` argument, or just remove the decorators since we run in bf16 natively on TPU.

### 2b. `torch.Generator(device=...)` in noise.py

Currently uses `p.device` which is correct. But `torch.Generator` may not support XLA devices directly. If it fails:

```python
# Fallback: generate noise on CPU, then move to device
gen = torch.Generator(device="cpu")
gen.manual_seed(effective_seed)
noise = torch.randn(p.shape, dtype=p.dtype, device="cpu", generator=gen)
noise = noise.to(p.device)
p.data.add_(sigma * noise)
```

This preserves deterministic seeding (the important part) while being device-agnostic. Slight overhead from CPU→TPU transfer but noise generation is negligible vs inference.

### 2c. `torch.cuda.synchronize()` / `empty_cache()`

**File:** `es_finetuning/noise.py` (3 locations), `es_finetuning/utils.py`

Replace with `sync_device()` / `empty_cache()` from the new device helper (Phase 1).

### 2d. TF32 / cuDNN settings

**Files:** `scripts/run_es.py`, `scripts/run_eval.py`

Wrap in conditional:
```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

---

## Phase 3: Multi-Chip Population Parallelism

This is the key speedup. Instead of evaluating 8 population members sequentially on one device, distribute them across TPU chips.

### Architecture:

```
TPU v4-32 = 4 TPU hosts × 8 chips = 32 chips
We need 8 chips (1 per population member)

Chip 0: perturb(seed[0]) → evaluate → reward[0]
Chip 1: perturb(seed[1]) → evaluate → reward[1]
...
Chip 7: perturb(seed[7]) → evaluate → reward[7]

All chips: gather rewards → normalize → apply ES update on chip 0 → broadcast weights
```

### Implementation options:

**Option A: torch_xla SPMD (simplest)**
- Use `torch_xla.distributed` to spawn 8 workers
- Each worker loads the model, receives its seed, evaluates, returns reward
- Master process aggregates and applies update

**Option B: Manual multiprocessing**
- Similar to multi-GPU DataParallel but with XLA devices
- `xmp.spawn(fn, args=(...,), nprocs=8)`

**Option C: Start with single-chip first**
- Get the code working on 1 TPU chip
- Benchmark single-chip vs single-GPU speedup
- Then add multi-chip parallelism

**Recommended: Option C first, then A.**

---

## Phase 4: Testing & Validation

### 4a. Smoke test (single chip, no NAMM)

```bash
python scripts/run_es.py \
    --run_name tpu_smoke \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

Verify: rewards are computed, weights update, checkpoint saves.

### 4b. Determinism check

Run the same config (same seed) on GPU and TPU. Compare:
- Seeds generated per iteration (should be identical — NumPy, not device-dependent)
- Rewards per population member (may differ slightly due to bf16 vs fp16 numerics)
- Final checkpoint weights (should be close but not bit-identical)

### 4c. NAMM on TPU

```bash
python scripts/run_es.py \
    --run_name tpu_namm_smoke \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain_v2.pt \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

If this fails, the likely culprit is:
1. `@torch.cuda.amp.custom_fwd` decorator → fix per Phase 2a
2. `torch.Generator` on XLA device → fix per Phase 2b
3. Dynamic KV cache shapes causing XLA recompilation → profile and add padding if needed

### 4d. Benchmark

| Config | GPU (1x RTX 6000) | TPU (1x v4 chip) | TPU (8x v4 chips) |
|---|---|---|---|
| pop=2, mb=2, iter=2 | measure | measure | measure |
| pop=8, mb=16, iter=5 | measure | measure | measure |

---

## Phase 5: Production Runs

Once validated, run the actual experiments:

```bash
# ES-only on TPU
python scripts/run_es.py \
    --run_name es_only_tpu \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16

# ES + NAMM at different cache sizes
for cache in 1024 3072 5120; do
    python scripts/run_es.py \
        --run_name es_namm_c${cache}_tpu \
        --namm_checkpoint exp_local/pretrained/namm_pretrained_romain_v2.pt \
        --cache_size $cache \
        --num_iterations 50 \
        --population_size 8 \
        --mini_batch_size 16
done
```

With 8-chip parallelism, each run should take ~15-20 minutes. All 4 runs in ~1 hour.

Can also run longer experiments (200+ iterations) to check convergence, which would have been impractical on GPU.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `torch.Generator` doesn't work on XLA device | Medium | Low | Generate noise on CPU, transfer to TPU (Phase 2b) |
| NAMM attention decorator breaks on TPU | High | Low | Remove `@torch.cuda.amp.custom_fwd` or replace with generic (Phase 2a) |
| KV cache dynamic shapes cause XLA retracing | Medium | Medium | Pad cache to fixed size, or accept first-sample compilation overhead |
| Spot instance preemption (if using spot) | N/A | N/A | Using on-demand v4-32, not spot |
| Numerical differences GPU vs TPU | Low | Low | Accept small differences; validate final F1 is in same range |
| torch_xla version incompatibility | Medium | Medium | Pin versions; use Google's prebuilt TPU VM images |

---

## File Checklist

New files to create:
- [ ] `setup/setup_tpu.sh` — TPU-specific environment setup
- [ ] `es_finetuning/device.py` — device detection and sync helpers

Files to modify:
- [ ] `scripts/run_es.py` — device abstraction, remove hardcoded CUDA
- [ ] `scripts/run_eval.py` — device abstraction, remove hardcoded CUDA
- [ ] `es_finetuning/noise.py` — conditional sync, device-agnostic generator
- [ ] `es_finetuning/trainer.py` — conditional CUDA memory tracking
- [ ] `es_finetuning/utils.py` — conditional CUDA calls
- [ ] `namm/evaluator.py` — `.cuda()` → `.to(device)`
- [ ] `namm/policy/base.py` — replace `@torch.cuda.amp.custom_fwd`
- [ ] `utils/helpers.py` — `.cuda()` → `.to(device)`
- [ ] `cfgs/config.yaml` — parameterise device

Files NOT to modify (NAMM upstream, out of scope):
- `namm/trainer.py` — CMA-ES training, not used for ES fine-tuning on TPU
- `scripts/run_namm.py` — NAMM training with DDP, separate concern

---

## Estimated Timeline

| Phase | Time | Notes |
|---|---|---|
| Phase 0: VM setup + setup_tpu.sh | 2h | Provision VM, install deps, clone repo, download model |
| Phase 1: Device abstraction | 1-2h | Mechanical find-and-replace, backwards-compatible |
| Phase 2: Fix CUDA-only APIs | 1h | Decorators, generators, sync calls |
| Phase 3: Multi-chip parallelism | 2-4h | Main engineering effort |
| Phase 4: Testing | 1-2h | Smoke tests, benchmarks, NAMM validation |
| **Total** | **~1 day** | |
