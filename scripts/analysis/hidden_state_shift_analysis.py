"""Measure the hidden-state distributional shift caused by KV cache eviction.

For each prompt, runs three forward passes:
  (a) Full context (no eviction, no truncation) — reference
  (b) NAMM-evicted context (with a trained NAMM checkpoint)
  (c) Truncated input (last N tokens, clean forward pass)

Then computes per-layer metrics comparing (b) vs (a) and (c) vs (a):
  - L2 distance of the LAST token's hidden state (the generation-critical one)
  - Cosine similarity of the LAST token's hidden state
  - Mean L2 / cosine over ALL shared token positions

The hypothesis: if NAMM eviction causes a larger distributional shift than
truncation, that explains why truncation sometimes produces better generations
despite having less context. The LoRA trained under eviction (M4) should show
a smaller shift than the LoRA trained without (M1).

Outputs a JSONL file with per-prompt, per-layer metrics + a summary CSV.

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/hidden_state_shift_analysis.py \
        --namm_checkpoint <path> \
        --cache_size 1024 \
        --lora_checkpoint <path>  # optional \
        --splits test extended_test \
        --output_dir analysis_out/hidden_state_shift
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure hidden-state shift from NAMM eviction vs truncation")
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--cache_size", type=int, required=True)
    p.add_argument("--run_config", type=str,
                   default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "val", "test", "extended_test"])
    p.add_argument("--extended_max_conditioning_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Must be 1 for hidden state comparison (prompts have "
                        "different lengths)")
    p.add_argument("--max_prompts_per_task", type=int, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def compute_shift_metrics(hidden_a, hidden_b, mask_a=None, mask_b=None):
    """Compare hidden states from two forward passes.

    Args:
        hidden_a: tuple of (num_layers+1,) tensors, each (1, seq_len_a, dim)
        hidden_b: tuple of (num_layers+1,) tensors, each (1, seq_len_b, dim)

    Returns per-layer dict with L2/cosine for last token and mean over all.
    """
    n_layers = len(hidden_a) - 1  # exclude embedding layer
    metrics = []

    for layer_idx in range(1, len(hidden_a)):  # skip embedding (idx 0)
        ha = hidden_a[layer_idx][0]  # (seq_len_a, dim)
        hb = hidden_b[layer_idx][0]  # (seq_len_b, dim)

        # Last token comparison (the generation-critical position)
        last_a = ha[-1]  # (dim,)
        last_b = hb[-1]  # (dim,)

        l2_last = torch.norm(last_a - last_b).item()
        cos_last = F.cosine_similarity(
            last_a.unsqueeze(0), last_b.unsqueeze(0)).item()

        # Norm of reference for relative L2
        norm_a = torch.norm(last_a).item()

        metrics.append({
            "layer": layer_idx - 1,  # 0-indexed layer
            "l2_last_token": l2_last,
            "l2_last_token_relative": l2_last / (norm_a + 1e-8),
            "cosine_last_token": cos_last,
            "norm_ref_last": norm_a,
        })

    return metrics


def forward_full_context(model, input_ids, attention_mask, device):
    """Full forward pass, no eviction. Returns hidden states tuple."""
    # Disable the memory policy by setting apply_memory_policy=False
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=True,
            apply_memory_policy=False,
        )
    return outputs.hidden_states


def forward_with_namm(model, input_ids, attention_mask, device):
    """Forward pass WITH the active memory policy (NAMM eviction)."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=True,
            apply_memory_policy=True,
        )
    return outputs.hidden_states


def forward_truncated(model, input_ids, attention_mask, n_trunc, device):
    """Forward pass on truncated input (last n_trunc tokens). No eviction."""
    trunc_ids = input_ids[..., -n_trunc:]
    trunc_mask = attention_mask[..., -n_trunc:]
    with torch.no_grad():
        outputs = model(
            input_ids=trunc_ids.to(device),
            attention_mask=trunc_mask.to(device),
            output_hidden_states=True,
            use_cache=True,
            apply_memory_policy=False,
        )
    return outputs.hidden_states


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    # Build config
    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false",
        "wandb_project=Experiments",
        f"filter_by_length={args.filter_by_length}",
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
        "+protected_tail_n=5",
    ]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
        overrides.append(f"eval_max_batch_size={args.batch_size}")

    with initialize(version_base=None, config_path="../config",
                    job_name="hidden_state_shift"):
        cfg = compose(config_name="config", overrides=overrides)

    # Build model
    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)
    memory_evaluator.device = device

    # Load NAMM checkpoint
    print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                      weights_only=False)
    evo_state = ckpt['evolution_state']
    prefer_mean = cfg.get('prefer_mean_to_best', True)
    if prefer_mean and 'mean' in evo_state:
        params_vec = evo_state['mean']
    else:
        params_vec = evo_state['best_member']
    params = params_vec.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)
    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    memory_policy.record_eval_stats = True
    memory_policy.initialize_stat_objects()

    # Load LoRA if provided
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
        for n, p in memory_model.model.named_parameters():
            if n in lora_sd:
                p.data.copy_(lora_sd[n].to(p.device, dtype=p.dtype))

    # Build task sampler
    print("Building task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    train_frac = cfg.get('train_frac', 0.7)
    val_frac = cfg.get('val_frac', 0.15)
    max_cond = cfg.get('split_max_conditioning_length',
                       cfg.get('max_conditioning_length', 6500))
    min_cond = cfg.get('min_conditioning_length', None)
    ext_max_cond = (args.extended_max_conditioning_length
                    if "extended_test" in args.splits else None)
    task_sampler.apply_train_val_test_split(
        train_frac=train_frac, val_frac=val_frac,
        max_conditioning_length=max_cond,
        min_conditioning_length=min_cond,
        tokenizer=tokenizer,
        extended_max_conditioning_length=ext_max_cond,
    )
    # Apply chat template
    task_sampler.apply_chat_template_to_prompts(tokenizer)

    raw_prompts = task_sampler.lb_prompts_per_task
    bos = getattr(tokenizer, 'bos_token', '') or ''

    lora_label = "with_lora" if args.lora_checkpoint else "no_lora"
    out_path = os.path.join(
        args.output_dir,
        f"shift_cs{args.cache_size}_{lora_label}_{timestamp}.jsonl")

    print(f"Output: {out_path}")
    n_total = 0

    with open(out_path, "w") as f_out:
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
                    input_ids = enc["input_ids"]
                    attention_mask = enc["attention_mask"]
                    n_tok = input_ids.shape[-1]

                    # Skip very short prompts
                    if n_tok <= args.cache_size:
                        continue

                    # (a) Full context, no eviction
                    memory_policy.initialize_stat_objects()
                    hidden_full = forward_full_context(
                        memory_model, input_ids, attention_mask, device)

                    # (b) With NAMM eviction
                    memory_policy.initialize_stat_objects()
                    hidden_namm = forward_with_namm(
                        memory_model, input_ids, attention_mask, device)

                    # (c) Truncated input
                    hidden_trunc = forward_truncated(
                        memory_model, input_ids, attention_mask,
                        args.cache_size, device)

                    # Compute shift metrics
                    shift_namm = compute_shift_metrics(hidden_full, hidden_namm)
                    shift_trunc = compute_shift_metrics(hidden_full, hidden_trunc)

                    record = {
                        "task": task_name,
                        "split": split_name,
                        "original_idx": int(orig_idx),
                        "n_tokens": n_tok,
                        "cache_size": args.cache_size,
                        "lora": lora_label,
                        "namm_vs_full": shift_namm,
                        "trunc_vs_full": shift_trunc,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    n_total += 1

                    if (i + 1) % 10 == 0:
                        # Quick summary for this prompt
                        avg_cos_namm = np.mean(
                            [m["cosine_last_token"] for m in shift_namm])
                        avg_cos_trunc = np.mean(
                            [m["cosine_last_token"] for m in shift_trunc])
                        print(f"    [{i+1}/{len(task_indices)}] "
                              f"cos_namm={avg_cos_namm:.4f} "
                              f"cos_trunc={avg_cos_trunc:.4f}")

                    # Free GPU memory
                    del hidden_full, hidden_namm, hidden_trunc
                    torch.cuda.empty_cache()

    print(f"\nDone: {n_total} prompts written to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Mean cosine similarity of last token (higher = less shift)")
    print("=" * 70)

    records = []
    with open(out_path) as f:
        for line in f:
            records.append(json.loads(line))

    if records:
        # Aggregate per layer
        n_layers = len(records[0]["namm_vs_full"])
        print(f"\n{'Layer':>6s} {'NAMM cos':>10s} {'Trunc cos':>10s} "
              f"{'NAMM L2rel':>10s} {'Trunc L2rel':>10s}")
        print("-" * 50)
        for l in range(n_layers):
            cos_n = np.mean([r["namm_vs_full"][l]["cosine_last_token"]
                            for r in records])
            cos_t = np.mean([r["trunc_vs_full"][l]["cosine_last_token"]
                            for r in records])
            l2_n = np.mean([r["namm_vs_full"][l]["l2_last_token_relative"]
                           for r in records])
            l2_t = np.mean([r["trunc_vs_full"][l]["l2_last_token_relative"]
                           for r in records])
            print(f"{l:6d} {cos_n:10.4f} {cos_t:10.4f} "
                  f"{l2_n:10.4f} {l2_t:10.4f}")

        # Overall
        cos_n_all = np.mean([[r["namm_vs_full"][l]["cosine_last_token"]
                             for l in range(n_layers)] for r in records])
        cos_t_all = np.mean([[r["trunc_vs_full"][l]["cosine_last_token"]
                             for l in range(n_layers)] for r in records])
        print(f"\n  Overall mean cosine: NAMM={cos_n_all:.4f}  "
              f"Trunc={cos_t_all:.4f}")
        print(f"  Interpretation: {'NAMM causes MORE shift' if cos_n_all < cos_t_all else 'Truncation causes MORE shift'}")


if __name__ == "__main__":
    main()
