#!/usr/bin/env python3
"""Diagnose whether NAMM split processing causes attention entropy collapse.

Processes a single prompt through the NAMM pipeline, hooking into each
split-processing chunk to record:
  - attention_mask shape vs actual KV cache size
  - per-head attention entropy at each chunk
  - per-head max attention weight (sharp = high, uniform = 1/N)

Compares against the same prompt processed WITHOUT NAMM (full context,
no split processing) as a control.

Usage:
    source activate.sh
    PYTHONPATH=. .venv/bin/python scripts/diagnose_attention_mask.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("attn_diag")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── Config ──────────────────────────────────────────────────────────────────
ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M2_NAMM_CKPT = ARTIFACTS / "M2_cs1024" / "ckpt.pt"
M3_LORA_CKPT = ARTIFACTS / "M3_cs1024" / "best_ckpt.pt"

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
NUM_LAYERS = 16
NUM_HEADS = 32  # query heads (8 KV heads expanded via GQA)


def load_model_and_prompt():
    """Load model infrastructure and get one test prompt."""
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    with torch.no_grad():
        memory_policy, memory_model, memory_evaluator, _, _ = make_eval_model(
            cfg=cfg
        )

    # Load NAMM weights
    logger.info("Loading NAMM checkpoint...")
    ckpt = torch.load(M2_NAMM_CKPT, map_location="cpu", weights_only=False)
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

    logger.info("  NAMM loaded (%d params)", params_vec.shape[0])

    memory_model.to(device)
    memory_policy.to(device)
    memory_model.eval()

    # Set batch indices AFTER moving to GPU so they land on the right device
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # Get one test prompt
    tokenizer = memory_evaluator.tokenizer
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=0.7, split_seed=42
    )
    task_sampler.filter_by_token_count(tokenizer, 6500)
    task_sampler.filter_answers_by_token_count(tokenizer, 64)
    task_sampler.apply_train_val_test_split(
        train_frac=0.7, val_frac=0.15,
        max_conditioning_length=6500,
        min_conditioning_length=4096,
        tokenizer=tokenizer,
    )

    # Pick one test prompt from qasper (a long one)
    test_idxs = task_sampler._test_idxs_per_task
    for task_name in ["lb/qasper", "lb/hotpotqa_e", "lb/2wikimqa"]:
        if task_name in test_idxs and len(test_idxs[task_name]) > 0:
            idx = test_idxs[task_name][0]
            prompt = task_sampler.lb_prompts_per_task[task_name][idx]
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=6500
            )["input_ids"]
            logger.info(
                "Selected prompt: task=%s idx=%d seq_len=%d",
                task_name, idx, input_ids.shape[1],
            )
            return memory_model, memory_policy, tokenizer, input_ids.to(device), device

    raise RuntimeError("No test prompts found")


def compute_entropy(attn_weights):
    """Compute Shannon entropy per head from attention weights.

    Args:
        attn_weights: (n_heads, q_len, kv_len) float tensor

    Returns:
        (n_heads,) entropy values, (n_heads,) max attention weight
    """
    # Use last query position as representative
    attn = attn_weights[:, -1, :].float().clamp(min=1e-12)
    entropy = -(attn * torch.log(attn)).sum(dim=-1)  # (n_heads,)
    max_weight = attn.max(dim=-1).values  # (n_heads,)
    return entropy.cpu().numpy(), max_weight.cpu().numpy()


def diagnose_with_namm(memory_model, memory_policy, input_ids, device):
    """Run the prompt through NAMM split processing, hooking into each chunk.

    Instead of calling memory_model() directly (which does split processing
    internally), we replicate the split loop manually to inspect each chunk.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 1: NAMM split processing (cache_size=%d)", CACHE_SIZE)
    logger.info("=" * 70)

    seq_len = input_ids.shape[1]
    chunk_size = memory_model.memory_policy_fixed_delay or 256
    logger.info(
        "Input: %d tokens, chunk_size=%d, expected %d chunks",
        seq_len, chunk_size, (seq_len + chunk_size - 1) // chunk_size,
    )

    # Check attn_implementation
    attn_impl = getattr(memory_model.config, "_attn_implementation", "unknown")
    logger.info("Attention implementation: %s", attn_impl)

    # Build position_ids and attention_mask as the split loop does
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # Split into chunks
    split_points = list(range(chunk_size, seq_len, chunk_size))
    if split_points and split_points[-1] != seq_len:
        split_points.append(seq_len)
    elif not split_points:
        split_points = [seq_len]

    chunk_boundaries = [0] + split_points
    n_chunks = len(split_points)
    logger.info("Chunk boundaries: %s", chunk_boundaries)

    past_key_values = None
    chunk_diagnostics = []

    memory_policy.initialize_buffers()

    for chunk_idx in range(n_chunks):
        start = chunk_boundaries[chunk_idx]
        end = chunk_boundaries[chunk_idx + 1]
        chunk_ids = input_ids[:, start:end]
        chunk_pos = position_ids[:, start:end]

        # This is exactly what the split loop does (line 444-445 of llama.py)
        curr_attn_mask = attention_mask[:, :end]

        # Get actual KV cache size before this chunk
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                kv_len = past_key_values[0][0].shape[-2]
            else:
                kv_len = past_key_values.key_cache[0].shape[-2]
        else:
            kv_len = 0

        is_last = chunk_idx == n_chunks - 1

        with torch.no_grad():
            outputs = memory_model(
                input_ids=chunk_ids,
                attention_mask=curr_attn_mask,
                position_ids=chunk_pos,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                apply_memory_policy=not is_last,  # evict on all but last
                limit_new_tokens=None,
                output_hidden_states=False,
                skip_lm_head=True,
            )

        past_key_values = outputs.past_key_values

        # Get KV size after this chunk (including eviction)
        if isinstance(past_key_values, tuple):
            kv_len_after = past_key_values[0][0].shape[-2]
        else:
            kv_len_after = past_key_values.key_cache[0].shape[-2]

        # Compute attention entropy from the returned attention weights
        if outputs.attentions is not None:
            # Pick layer 0 and layer 8 as representative
            entropies = {}
            max_weights = {}
            for layer_idx in [0, 7, 15]:
                if layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[layer_idx][0]  # (n_heads, q, kv)
                    ent, maxw = compute_entropy(attn)
                    entropies[layer_idx] = ent
                    max_weights[layer_idx] = maxw

            # Theoretical uniform entropy
            uniform_entropy = np.log(kv_len_after) if kv_len_after > 0 else 0

            diag = {
                "chunk_idx": chunk_idx,
                "token_range": f"{start}-{end}",
                "chunk_tokens": end - start,
                "attn_mask_len": curr_attn_mask.shape[-1],
                "kv_before": kv_len,
                "kv_after": kv_len_after,
                "mask_kv_mismatch": curr_attn_mask.shape[-1] - kv_len,
                "uniform_entropy": uniform_entropy,
                "entropies": entropies,
                "max_weights": max_weights,
            }
            chunk_diagnostics.append(diag)

            # Log summary
            for layer_idx in sorted(entropies.keys()):
                mean_ent = entropies[layer_idx].mean()
                std_ent = entropies[layer_idx].std()
                mean_maxw = max_weights[layer_idx].mean()
                near_uniform = abs(mean_ent - uniform_entropy) < 0.1

                logger.info(
                    "  Chunk %2d [%4d-%4d] mask=%4d kv_before=%4d kv_after=%4d | "
                    "Layer %2d: entropy=%.3f±%.3f (uniform=%.3f) maxw=%.3f %s",
                    chunk_idx, start, end, curr_attn_mask.shape[-1],
                    kv_len, kv_len_after,
                    layer_idx, mean_ent, std_ent, uniform_entropy,
                    mean_maxw,
                    "*** COLLAPSED ***" if near_uniform else "",
                )
        else:
            logger.info(
                "  Chunk %2d [%4d-%4d] mask=%4d kv=%4d->%4d (no attentions)",
                chunk_idx, start, end, curr_attn_mask.shape[-1],
                kv_len, kv_len_after,
            )

        del outputs
        torch.cuda.empty_cache()

    return chunk_diagnostics


def diagnose_without_namm(memory_model, input_ids, device):
    """Run the same prompt WITHOUT NAMM (full context, no eviction) as control."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 2: Full context, no NAMM (control)")
    logger.info("=" * 70)

    from namm.policy.base import Recency

    # Swap to passthrough policy
    original_policy = memory_model.memory_policy
    memory_model.swap_memory_policy(Recency(cache_size=None))

    seq_len = input_ids.shape[1]
    logger.info("Input: %d tokens, no split processing", seq_len)

    with torch.no_grad():
        outputs = memory_model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=True,
            apply_memory_policy=False,
            output_hidden_states=False,
            skip_lm_head=True,
        )

    if outputs.attentions is not None:
        for layer_idx in [0, 7, 15]:
            if layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx][0]
                ent, maxw = compute_entropy(attn)
                kv_len = attn.shape[-1]
                uniform_ent = np.log(kv_len)
                logger.info(
                    "  Layer %2d: entropy=%.3f±%.3f (uniform=%.3f) maxw=%.3f kv=%d",
                    layer_idx, ent.mean(), ent.std(), uniform_ent,
                    maxw.mean(), kv_len,
                )

    # Restore original policy
    memory_model.swap_memory_policy(original_policy)

    del outputs
    torch.cuda.empty_cache()


def diagnose_truncated(memory_model, input_ids, device):
    """Run with input truncated to last 1024 tokens (no NAMM) as another control."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 3: Truncated to last %d tokens (no NAMM)", CACHE_SIZE)
    logger.info("=" * 70)

    from namm.policy.base import Recency

    original_policy = memory_model.memory_policy
    memory_model.swap_memory_policy(Recency(cache_size=None))

    trunc_ids = input_ids[:, -CACHE_SIZE:]
    logger.info("Input: %d tokens (truncated from %d)", trunc_ids.shape[1], input_ids.shape[1])

    with torch.no_grad():
        outputs = memory_model(
            input_ids=trunc_ids,
            use_cache=True,
            output_attentions=True,
            apply_memory_policy=False,
            output_hidden_states=False,
            skip_lm_head=True,
        )

    if outputs.attentions is not None:
        for layer_idx in [0, 7, 15]:
            if layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx][0]
                ent, maxw = compute_entropy(attn)
                kv_len = attn.shape[-1]
                uniform_ent = np.log(kv_len)
                logger.info(
                    "  Layer %2d: entropy=%.3f±%.3f (uniform=%.3f) maxw=%.3f kv=%d",
                    layer_idx, ent.mean(), ent.std(), uniform_ent,
                    maxw.mean(), kv_len,
                )

    memory_model.swap_memory_policy(original_policy)
    del outputs
    torch.cuda.empty_cache()


def main():
    logger.info("=" * 70)
    logger.info("Attention Mask Diagnostic: NAMM Split Processing")
    logger.info("=" * 70)

    memory_model, memory_policy, tokenizer, input_ids, device = (
        load_model_and_prompt()
    )

    # Check attn_implementation
    attn_impl = getattr(memory_model.config, "_attn_implementation", "unknown")
    logger.info("Model attn_implementation: %s", attn_impl)

    # Test 1: NAMM with split processing
    namm_diags = diagnose_with_namm(memory_model, memory_policy, input_ids, device)

    # Test 2: Full context, no NAMM
    diagnose_without_namm(memory_model, input_ids, device)

    # Test 3: Truncated, no NAMM
    diagnose_truncated(memory_model, input_ids, device)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if namm_diags:
        logger.info("Mask-KV mismatches by chunk:")
        for d in namm_diags:
            mismatch = d["mask_kv_mismatch"]
            collapsed_layers = []
            for layer_idx, ent in d["entropies"].items():
                if abs(ent.mean() - d["uniform_entropy"]) < 0.1:
                    collapsed_layers.append(layer_idx)
            logger.info(
                "  Chunk %2d: mask=%4d kv=%4d mismatch=%+5d collapsed_layers=%s",
                d["chunk_idx"], d["attn_mask_len"], d["kv_after"],
                mismatch, collapsed_layers or "none",
            )

        any_collapsed = any(
            abs(d["entropies"].get(l, np.array([0])).mean() - d["uniform_entropy"]) < 0.1
            for d in namm_diags
            for l in d["entropies"]
        )
        if any_collapsed:
            logger.info("")
            logger.info(
                "*** ATTENTION COLLAPSE DETECTED ***"
                " Some chunks show entropy near log(N), confirming the bug."
            )
        else:
            logger.info("")
            logger.info(
                "No attention collapse detected. Entropy is below log(N) "
                "for all chunks. The mask mismatch exists but does not "
                "cause uniform attention in this configuration."
            )


if __name__ == "__main__":
    main()
