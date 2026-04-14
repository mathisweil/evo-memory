#!/usr/bin/env python3
"""Report 5 -- Attention Entropy Under Eviction.

Measures attention entropy in each model's **actual operating regime**:
  - M1: full-context forward (no NAMM, no eviction)
  - M2: forward with apply_memory_policy=True (NAMM eviction, no LoRA)
  - M3: forward with apply_memory_policy=True (NAMM eviction + LoRA)

All use long (>4096 token) prompts so NAMM actually evicts for M2/M3.

Entropy is computed at the last query position.  Since evicted tokens
have zero attention by construction (they are physically absent from
the KV cache), H(evicted model) = H(retained tokens only).  This is
mathematically equivalent to computing entropy over the full prompt
with evicted positions set to zero, because 0*log(0) = 0.

Saves to: analysis/report_5/data/entropy_data.npz

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/report_5/scripts/generate_data.py
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HOME", ".hf_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "entropy_data.npz"

# Checkpoint paths
ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M2_NAMM_MASKFIX_CKPT = ARTIFACTS / "M2_cs1024_maskfix" / "ckpt.pt"
M3_MASKFIX_LORA_CKPT = ARTIFACTS / "M3_cs1024_maskfix" / "best_ckpt.pt"

# NAMM config
RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
MIN_CONDITIONING_LENGTH = 4096

NUM_LAYERS = 16
NUM_HEADS = 32
SAMPLES_PER_TASK = 73  # balanced across 5 tasks (min available)


# ---------------------------------------------------------------------------
# NAMM model setup
# ---------------------------------------------------------------------------

def build_model_and_data(device: str = "cuda"):
    import torch
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )
    with torch.no_grad():
        memory_policy, memory_model, memory_evaluator, _, _ = make_eval_model(
            cfg=cfg)
    memory_model.to(device)
    memory_model.eval()
    tokenizer = memory_evaluator.tokenizer
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    task_sampler.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT, val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=MIN_CONDITIONING_LENGTH,
        tokenizer=tokenizer,
    )
    return cfg, memory_policy, memory_model, memory_evaluator, task_sampler, tokenizer


def load_namm_weights(
    memory_model: Any, memory_policy: Any,
    namm_ckpt_path: Path, device: str = "cuda",
) -> None:
    import torch
    logger.info("Loading NAMM checkpoint: %s", namm_ckpt_path)
    ckpt = torch.load(str(namm_ckpt_path), map_location="cpu", weights_only=False)
    evo_state = ckpt["evolution_state"]
    params_vec = evo_state.get("mean", evo_state["best_member"])
    params = params_vec.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)
    buffers_prefix = "stored_buffers_to_save."
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    logger.info("  NAMM loaded (%d params)", params_vec.shape[0])


def load_lora_weights(
    memory_model: Any, lora_ckpt_path: Path, device: str = "cuda",
) -> None:
    import torch
    logger.info("Loading LoRA checkpoint: %s", lora_ckpt_path)
    ckpt = torch.load(str(lora_ckpt_path), map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    lora_sd = ckpt["lora_state_dict"]
    if not memory_model.has_lora_adapters():
        rank = lora_cfg.get("rank", 8)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        memory_model.apply_lora_adapters(rank=rank, target_modules=target_modules)
    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_sd:
            p.data.copy_(lora_sd[n].to(p.device))
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No LoRA weights loaded from {lora_ckpt_path}")
    logger.info("  LoRA loaded (%d tensors)", loaded)


def reset_lora_weights(memory_model: Any) -> None:
    for _n, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()
    logger.info("  Reset LoRA weights to zero")


def reset_policy_state(memory_model: Any) -> None:
    if hasattr(memory_model.memory_policy, "initialize_buffers"):
        memory_model.memory_policy.initialize_buffers()
    elif hasattr(memory_model.memory_policy, "reset"):
        memory_model.memory_policy.reset()


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

def get_prompts(
    task_sampler: Any, tokenizer: Any, per_task: int = SAMPLES_PER_TASK,
) -> list[dict]:
    """Get balanced prompts from all splits (train+val+test)."""
    import torch
    prompts: list[dict] = []
    for task_name in sorted(task_sampler.lb_prompts_per_task.keys()):
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        # Gather indices from all splits
        all_idxs: list[int] = []
        for split_attr in ("_train_idxs_per_task", "_val_idxs_per_task",
                           "_test_idxs_per_task"):
            split_dict = getattr(task_sampler, split_attr, None)
            if split_dict and task_name in split_dict:
                all_idxs.extend(split_dict[task_name])
        for idx in all_idxs[:per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=FILTER_BY_TOKENS)
            seq_len = ids["input_ids"].shape[1]
            prompts.append({
                "input_ids": ids["input_ids"],
                "task": task_name,
                "idx": int(idx),
                "seq_len": seq_len,
            })
    logger.info("Prepared %d prompts (%d per task, seq_len range: %d-%d)",
                len(prompts), per_task,
                min(p["seq_len"] for p in prompts) if prompts else 0,
                max(p["seq_len"] for p in prompts) if prompts else 0)
    return prompts


# ---------------------------------------------------------------------------
# Entropy extraction: full context (no eviction)
# ---------------------------------------------------------------------------

def extract_entropy_full_context(
    memory_model: Any, prompts: list[dict], device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per_head_entropy, union_entropy).

    per_head_entropy: (NUM_LAYERS, NUM_HEADS) mean per-head entropy
    union_entropy:    (NUM_LAYERS,) entropy of head-averaged distribution
    """
    import torch

    entropy_sum = torch.zeros(NUM_LAYERS, NUM_HEADS)
    union_entropy_sum = torch.zeros(NUM_LAYERS)
    count = 0

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        reset_policy_state(memory_model)
        with torch.no_grad():
            outputs = memory_model(
                input_ids=input_ids, use_cache=True,
                output_attentions=True, return_dict=True,
                apply_memory_policy=False,
            )
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            attn = layer_attn[0].float()
            attn_last = attn[:, -1, :]  # (n_heads, seq)
            # Per-head entropy
            attn_clamped = attn_last.clamp(min=1e-12)
            h = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
            entropy_sum[layer_idx] += h.cpu()
            # Union: average across heads, then entropy
            union_dist = attn_last.mean(dim=0).clamp(min=1e-12)  # (seq,)
            h_union = -(union_dist * union_dist.log()).sum()
            union_entropy_sum[layer_idx] += h_union.cpu()
        count += 1
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Full-ctx sample %d/%d done (seq_len=%d)",
                         i + 1, len(prompts), p["seq_len"])
        del outputs
        torch.cuda.empty_cache()

    return (entropy_sum / count).numpy(), (union_entropy_sum / count).numpy()


# ---------------------------------------------------------------------------
# Entropy extraction: with NAMM eviction
# ---------------------------------------------------------------------------

def extract_entropy_with_eviction(
    memory_model: Any, prompts: list[dict], device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (per_head_entropy, union_entropy, sample_info)."""
    import torch

    entropy_sum = torch.zeros(NUM_LAYERS, NUM_HEADS)
    union_entropy_sum = torch.zeros(NUM_LAYERS)
    sample_info: list[dict] = []
    count = 0

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        reset_policy_state(memory_model)
        with torch.no_grad():
            outputs = memory_model(
                input_ids=input_ids, use_cache=True,
                output_attentions=True, return_dict=True,
                apply_memory_policy=True,
            )
        past_kv = outputs.past_key_values
        if past_kv is not None:
            if isinstance(past_kv, tuple):
                cache_len = past_kv[0][0].shape[-2]
            elif hasattr(past_kv, "key_cache"):
                cache_len = past_kv.key_cache[0].shape[-2]
            else:
                cache_len = p["seq_len"]
        else:
            cache_len = p["seq_len"]

        for layer_idx, layer_attn in enumerate(outputs.attentions):
            attn = layer_attn[0].float()
            attn_last = attn[:, -1, :]  # (n_heads, cache_len)
            attn_clamped = attn_last.clamp(min=1e-12)
            h = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
            entropy_sum[layer_idx] += h.cpu()
            union_dist = attn_last.mean(dim=0).clamp(min=1e-12)
            h_union = -(union_dist * union_dist.log()).sum()
            union_entropy_sum[layer_idx] += h_union.cpu()

        count += 1
        retention = cache_len / p["seq_len"]
        sample_info.append({
            "seq_len": p["seq_len"],
            "cache_len": cache_len,
            "retention_ratio": retention,
            "task": p["task"],
        })
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Evicted sample %d/%d done (seq_len=%d, cache=%d, ret=%.2f)",
                         i + 1, len(prompts), p["seq_len"], cache_len, retention)
        del outputs
        torch.cuda.empty_cache()

    return ((entropy_sum / count).numpy(),
            (union_entropy_sum / count).numpy(),
            sample_info)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute() -> None:
    import torch

    if OUT_PATH.exists():
        logger.info("Data already exists: %s — skipping", OUT_PATH)
        return

    for path, label in [
        (M1_LORA_CKPT, "M1 LoRA"),
        (M2_NAMM_MASKFIX_CKPT, "M2 NAMM maskfix"),
        (M3_MASKFIX_LORA_CKPT, "M3 LoRA maskfix"),
    ]:
        if not path.exists():
            logger.error("Checkpoint not found: %s (%s)", path, label)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    logger.info("=== Building model infrastructure ===")
    cfg, memory_policy, memory_model, memory_evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    prompts = get_prompts(task_sampler, tokenizer)

    # ── M1: full context, no eviction ─────────────────────────────────
    logger.info("=== M1: Loading LoRA weights ===")
    load_namm_weights(memory_model, memory_policy, M2_NAMM_MASKFIX_CKPT, device)
    load_lora_weights(memory_model, M1_LORA_CKPT, device)

    logger.info("=== M1: Extracting entropy (full context) ===")
    m1_entropy, m1_union = extract_entropy_full_context(
        memory_model, prompts, device)

    # ── M3: with NAMM eviction ────────────────────────────────────────
    logger.info("=== M3: Loading LoRA weights ===")
    load_lora_weights(memory_model, M3_MASKFIX_LORA_CKPT, device)

    logger.info("=== M3: Extracting entropy (with eviction) ===")
    m3_entropy, m3_union, m3_sample_info = extract_entropy_with_eviction(
        memory_model, prompts, device)

    # ── M2: NAMM eviction, no LoRA ───────────────────────────────────
    logger.info("=== M2: Zeroing LoRA weights ===")
    reset_lora_weights(memory_model)

    logger.info("=== M2: Extracting entropy (with eviction, no LoRA) ===")
    m2_entropy, m2_union, m2_sample_info = extract_entropy_with_eviction(
        memory_model, prompts, device)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    m3_cache_lens = np.array([s["cache_len"] for s in m3_sample_info])
    m3_seq_lens = np.array([s["seq_len"] for s in m3_sample_info])
    m3_retention = np.array([s["retention_ratio"] for s in m3_sample_info])
    m2_cache_lens = np.array([s["cache_len"] for s in m2_sample_info])
    m2_retention = np.array([s["retention_ratio"] for s in m2_sample_info])

    np.savez(
        str(OUT_PATH),
        m1_entropy=m1_entropy,
        m2_entropy=m2_entropy,
        m3_entropy=m3_entropy,
        m1_union_entropy=m1_union,
        m2_union_entropy=m2_union,
        m3_union_entropy=m3_union,
        m2_cache_lens=m2_cache_lens,
        m2_retention=m2_retention,
        m3_cache_lens=m3_cache_lens,
        m3_seq_lens=m3_seq_lens,
        m3_retention=m3_retention,
    )
    logger.info("Saved %s", OUT_PATH)

    del memory_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    compute()
