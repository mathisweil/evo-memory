"""Analyze how LoRA adaptation changes attention patterns under NAMM eviction.

Uses forward hooks on attention layers to capture attention weights from
EVERY chunk during split-processing — not just the final chunk. This gives
us the attention patterns at each eviction step, showing how the model
attends to the cache as it grows and gets evicted.

Three model variants on the SAME prompts with the SAME NAMM checkpoint:
  (a) Plain model + NAMM          — baseline under eviction
  (b) M1 LoRA + NAMM              — LoRA NOT trained with eviction
  (c) M4 LoRA + NAMM              — LoRA trained WITH eviction

Per chunk per layer per head, captures:
  - Attention entropy (sharp vs diffuse)
  - Fraction of attention on tokens from the first/middle/last third of
    the original prompt
  - Mean attention distance (how far back each head reaches)
  - Whether the chunk is pre-eviction, post-eviction, or the final chunk

Also captures per-prompt:
  - Final hidden state of the last token (generation-critical)
  - Hidden state norms per layer

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/eviction_representation_analysis.py \
        --namm_checkpoint <path> --cache_size 1024 \
        --variant plain|m1|m4 \
        --splits test extended_test \
        --output_dir analysis_out/eviction_repr
"""

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device


class AttentionCapture:
    """Forward hook that captures attention weights from every layer at
    every forward pass (including intermediate split-processing chunks)."""

    def __init__(self, model):
        self.captured_chunks = []  # list of dicts, one per forward call
        self._current_chunk = {}
        self._hooks = []

        # Register hooks on all attention layers
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            hook = attn.register_forward_hook(
                self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights, past_key_value[, query_states])
            # attn_weights is (bs, n_heads, q_len, kv_len) or None
            attn_weights = output[1]
            if attn_weights is not None:
                # Store as float16 on CPU to save memory
                self._current_chunk[layer_idx] = (
                    attn_weights[0].detach().cpu().half())  # (n_heads, q_len, kv_len)
        return hook_fn

    def start_prompt(self):
        """Call before each prompt's forward pass."""
        self.captured_chunks = []
        self._current_chunk = {}

    def end_chunk(self):
        """Call after each chunk's forward pass to flush captured attention."""
        if self._current_chunk:
            self.captured_chunks.append(dict(self._current_chunk))
            self._current_chunk = {}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def compute_chunk_metrics(attn_weights, n_total_tokens, kv_len_at_chunk):
    """Compute metrics from one layer's attention at one chunk.

    Args:
        attn_weights: (n_heads, q_len, kv_len) float16
        n_total_tokens: total prompt length (for third-based position binning)
        kv_len_at_chunk: how many tokens are in the cache at this point

    Returns: dict with per-head metrics
    """
    n_heads, q_len, kv_len = attn_weights.shape
    attn = attn_weights.float()

    third = n_total_tokens / 3.0
    # The KV positions in the cache after eviction are non-contiguous.
    # We can't know exact original positions without the eviction trace.
    # But we CAN compute entropy and relative attention patterns.

    metrics = {
        "entropy_per_head": [],
        "max_attn_per_head": [],
        "kv_len": int(kv_len),
        "q_len": int(q_len),
    }

    for h in range(n_heads):
        # Use attention from the LAST query token in this chunk
        attn_row = attn[h, -1, :]  # (kv_len,)
        attn_np = attn_row.numpy()

        # Entropy
        attn_clipped = np.clip(attn_np, 1e-10, 1.0)
        entropy = float(-np.sum(attn_clipped * np.log(attn_clipped)))
        metrics["entropy_per_head"].append(entropy)

        # Max attention weight (how "peaked" is the attention)
        metrics["max_attn_per_head"].append(float(attn_np.max()))

    # Attention concentration: what fraction of mass is in top-k positions
    # (averaged over heads)
    for topk in [10, 50]:
        fracs = []
        for h in range(n_heads):
            attn_row = attn[h, -1, :].numpy()
            topk_actual = min(topk, len(attn_row))
            topk_mass = float(np.sort(attn_row)[-topk_actual:].sum())
            fracs.append(topk_mass)
        metrics[f"attn_top{topk}_frac"] = fracs

    # Attention to first/last quarter of the KV cache
    # (after eviction, the first quarter of the cache contains the
    # oldest surviving tokens — likely early-document tokens)
    quarter = max(1, kv_len // 4)
    first_q_fracs = []
    last_q_fracs = []
    for h in range(n_heads):
        attn_row = attn[h, -1, :].numpy()
        first_q_fracs.append(float(attn_row[:quarter].sum()))
        last_q_fracs.append(float(attn_row[-quarter:].sum()))
    metrics["attn_first_quarter"] = first_q_fracs
    metrics["attn_last_quarter"] = last_q_fracs

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--cache_size", type=int, required=True)
    p.add_argument("--variant", type=str, required=True,
                   choices=["plain", "m1", "m4"])
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--run_config", type=str,
                   default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "val", "test", "extended_test"])
    p.add_argument("--extended_max_conditioning_length", type=int, default=8192)
    p.add_argument("--max_prompts_per_task", type=int, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.variant in ("m1", "m4") and not args.lora_checkpoint:
        raise ValueError(f"--lora_checkpoint required for variant={args.variant}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false", "wandb_project=Experiments",
        f"filter_by_length={args.filter_by_length}",
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
        "+protected_tail_n=5",
    ]

    with initialize(version_base=None, config_path="../config",
                    job_name="eviction_repr_analysis"):
        cfg = compose(config_name="config", overrides=overrides)

    print(f"Building model (variant={args.variant})...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)

    # Load NAMM
    print(f"Loading NAMM: {args.namm_checkpoint}")
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                      weights_only=False)
    evo_state = ckpt['evolution_state']
    prefer_mean = cfg.get('prefer_mean_to_best', True)
    params_vec = (evo_state['mean'] if (prefer_mean and 'mean' in evo_state)
                  else evo_state['best_member'])
    memory_model.set_memory_params(params_vec.unsqueeze(0).to(device))
    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {k[len(buffers_prefix):]: v.to(device)
                    for k, v in evo_state.items()
                    if k.startswith(buffers_prefix)}
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    memory_policy.record_eval_stats = True
    memory_policy.initialize_stat_objects()

    # Load LoRA
    if args.lora_checkpoint:
        print(f"Loading LoRA: {args.lora_checkpoint}")
        lora_ckpt = torch.load(args.lora_checkpoint, map_location="cpu",
                               weights_only=False)
        lora_cfg = lora_ckpt.get('lora_config', {})
        memory_model.apply_lora_adapters(
            rank=lora_cfg.get('rank', 8),
            target_modules=lora_cfg.get('target_modules', ['q_proj', 'v_proj']))
        memory_model.to(device)
        lora_sd = lora_ckpt['lora_state_dict']
        loaded = 0
        for n, p in memory_model.model.named_parameters():
            if n in lora_sd:
                p.data.copy_(lora_sd[n].to(p.device, dtype=p.dtype))
                loaded += 1
        print(f"  Loaded {loaded} LoRA tensors")

    # Register attention capture hooks
    capture = AttentionCapture(memory_model)

    # We need to intercept the split-processing loop to call end_chunk()
    # after each intermediate forward pass. Monkey-patch the model's forward.
    original_forward = memory_model.forward

    def patched_forward(*a, **kw):
        result = original_forward(*a, **kw)
        capture.end_chunk()
        return result

    memory_model.forward = patched_forward

    # Task sampler
    print("Building task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get('train_frac', 0.7),
        val_frac=cfg.get('val_frac', 0.15),
        max_conditioning_length=cfg.get('split_max_conditioning_length',
                                        cfg.get('max_conditioning_length', 6500)),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        tokenizer=tokenizer,
        extended_max_conditioning_length=(
            args.extended_max_conditioning_length
            if "extended_test" in args.splits else None),
    )
    task_sampler.apply_chat_template_to_prompts(tokenizer)
    raw_prompts = task_sampler.lb_prompts_per_task

    out_path = os.path.join(
        args.output_dir,
        f"repr_{args.variant}_cs{args.cache_size}_{timestamp}.jsonl")
    print(f"Output: {out_path}")
    n_total = 0

    with open(out_path, "w") as f_out:
        header = {
            "_header": True,
            "variant": args.variant,
            "cache_size": args.cache_size,
            "namm_checkpoint": os.path.abspath(args.namm_checkpoint),
            "lora_checkpoint": (os.path.abspath(args.lora_checkpoint)
                                if args.lora_checkpoint else None),
            "timestamp": timestamp,
        }
        f_out.write(json.dumps(header) + "\n")

        for split_name in args.splits:
            split_idxs = task_sampler.get_split_indices(split_name)
            for task_name in sorted(split_idxs.keys()):
                task_indices = list(split_idxs[task_name])
                if args.max_prompts_per_task:
                    task_indices = task_indices[:args.max_prompts_per_task]
                print(f"\n  {split_name}/{task_name}: {len(task_indices)} prompts")

                for i, orig_idx in enumerate(task_indices):
                    prompt = raw_prompts[task_name][int(orig_idx)]
                    enc = tokenizer(prompt, add_special_tokens=True,
                                   return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    n_tok = int(input_ids.shape[-1])

                    if n_tok <= args.cache_size:
                        continue

                    # Reset state
                    memory_policy.initialize_stat_objects()
                    capture.start_prompt()

                    # Forward pass — the patched forward calls end_chunk()
                    # after each split-processing chunk automatically
                    with torch.no_grad():
                        outputs = memory_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            output_attentions=True,
                            use_cache=True,
                            apply_memory_policy=True,
                        )
                    # Flush the final chunk
                    capture.end_chunk()

                    # Process captured chunks
                    n_chunks = len(capture.captured_chunks)
                    chunks_data = []
                    for chunk_idx, chunk_attn in enumerate(capture.captured_chunks):
                        chunk_layers = {}
                        for layer_idx, attn_weights in chunk_attn.items():
                            kv_len = attn_weights.shape[-1]
                            metrics = compute_chunk_metrics(
                                attn_weights, n_tok, kv_len)
                            metrics["layer"] = layer_idx
                            chunk_layers[layer_idx] = metrics
                        chunks_data.append({
                            "chunk_idx": chunk_idx,
                            "is_final": chunk_idx == n_chunks - 1,
                            "layers": chunk_layers,
                        })

                    # Final hidden states
                    hidden_per_layer = outputs.hidden_states
                    hidden_norms = []
                    for layer_idx in range(1, len(hidden_per_layer)):
                        h = hidden_per_layer[layer_idx][0]
                        hidden_norms.append({
                            "layer": layer_idx - 1,
                            "last_token_norm": float(torch.norm(h[-1]).item()),
                            "mean_norm": float(torch.norm(h, dim=-1).mean().item()),
                        })

                    record = {
                        "task": task_name,
                        "split": split_name,
                        "original_idx": int(orig_idx),
                        "n_tokens": n_tok,
                        "n_chunks": n_chunks,
                        "variant": args.variant,
                        "cache_size": args.cache_size,
                        "chunks": chunks_data,
                        "hidden_norms": hidden_norms,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    n_total += 1

                    if (i + 1) % 10 == 0:
                        print(f"    [{i+1}/{len(task_indices)}] "
                              f"{n_chunks} chunks captured")

                    del outputs, hidden_per_layer
                    torch.cuda.empty_cache()

    capture.remove_hooks()
    print(f"\nDone: {n_total} prompts → {out_path}")

    # Summary
    records = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("_header"):
                records.append(r)

    if records:
        print(f"\n{'='*70}")
        print(f"  Summary: {args.variant} under NAMM cs{args.cache_size}")
        print(f"  {n_total} prompts")
        print(f"{'='*70}")

        # Compare early chunks (before eviction bites hard) vs late chunks
        early_entropy = defaultdict(list)  # layer → list of entropy values
        late_entropy = defaultdict(list)
        early_top10 = defaultdict(list)
        late_top10 = defaultdict(list)

        for r in records:
            n_chunks = r["n_chunks"]
            for chunk in r["chunks"]:
                cidx = chunk["chunk_idx"]
                is_early = cidx < n_chunks // 3
                is_late = cidx >= 2 * n_chunks // 3
                for layer_str, metrics in chunk["layers"].items():
                    layer = int(layer_str)
                    ent = np.mean(metrics["entropy_per_head"])
                    t10 = np.mean(metrics["attn_top10_frac"])
                    if is_early:
                        early_entropy[layer].append(ent)
                        early_top10[layer].append(t10)
                    elif is_late:
                        late_entropy[layer].append(ent)
                        late_top10[layer].append(t10)

        layers = sorted(early_entropy.keys())
        print(f"\n{'Layer':>6s} {'Early Ent':>10s} {'Late Ent':>10s} {'Δ':>8s} "
              f"{'Early Top10':>12s} {'Late Top10':>12s}")
        print("-" * 62)
        for l in layers:
            ee = np.mean(early_entropy[l])
            le = np.mean(late_entropy[l])
            et = np.mean(early_top10[l])
            lt = np.mean(late_top10[l])
            print(f"{l:6d} {ee:10.3f} {le:10.3f} {le-ee:+8.3f} "
                  f"{et:12.4f} {lt:12.4f}")


if __name__ == "__main__":
    main()
