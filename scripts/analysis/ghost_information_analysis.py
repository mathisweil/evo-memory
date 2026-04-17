"""Quantify ghost information in KV cache after NAMM eviction.

For tokens that survive in the last `cache_size` positions, compare their
key/value vectors between:
  (a) NAMM eviction: full prompt processed, then evicted to cache_size entries.
      Surviving tokens' KVs were computed attending to tokens that were later evicted.
  (b) Truncation: only the last cache_size tokens fed to the model.
      KVs computed attending only to tokens that are actually present.

The L2 distance between (a) and (b) for the same token IS the ghost
information — the measurable imprint left by evicted tokens on surviving
entries' representations.

Outputs per-prompt, per-layer:
  - ghost_l2_mean: mean L2 distance across all shared tokens
  - ghost_l2_last: L2 distance for the last token (generation-critical)
  - ghost_cosine_mean: mean cosine similarity (1 = identical, 0 = orthogonal)
  - ghost_cosine_last: cosine for last token
  - ghost_relative_l2: L2 / norm of the truncation KV (relative magnitude)
  - n_shared_tokens: how many tokens are compared

Also computes ghost magnitude vs position (are early-surviving tokens
more ghost-contaminated than recent ones?).

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/ghost_information_analysis.py \
        --namm_checkpoint <path> --cache_size 1024 \
        --splits test extended_test \
        --output_dir analysis_out/ghost_info
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
from namm.policy.base import Recency
from es_finetuning.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--cache_size", type=int, required=True)
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--splits", nargs="+", default=["test", "extended_test"],
                   choices=["train", "val", "test", "extended_test"])
    p.add_argument("--extended_max_conditioning_length", type=int, default=8192)
    p.add_argument("--max_prompts_per_task", type=int, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def extract_kv_cache(past_key_values):
    """Extract KV cache as list of (key, value) tuples per layer.
    Each key/value is (n_heads, seq_len, head_dim)."""
    kvs = []
    for layer_kv in past_key_values:
        k, v = layer_kv  # each is (batch=1, n_heads, seq_len, head_dim)
        kvs.append((k[0].cpu(), v[0].cpu()))  # remove batch dim
    return kvs


def compute_ghost_metrics(kv_namm, kv_trunc, n_layers):
    """Compare KV caches from NAMM vs truncation.

    Both caches have the same number of layers. NAMM cache may be longer
    (cache_size entries from scattered positions) while trunc cache has
    exactly cache_size entries from contiguous tail positions.

    We compare the LAST min(namm_len, trunc_len) positions — these are
    the tokens that exist in both caches (the tail of the prompt).
    """
    metrics_per_layer = []

    for layer_idx in range(n_layers):
        k_namm, v_namm = kv_namm[layer_idx]
        k_trunc, v_trunc = kv_trunc[layer_idx]
        # shapes: (n_heads, seq_len_x, head_dim)

        namm_len = k_namm.shape[1]
        trunc_len = k_trunc.shape[1]
        shared_len = min(namm_len, trunc_len)

        # Align from the END (both should end at the same token)
        k_n = k_namm[:, -shared_len:, :].float()
        k_t = k_trunc[:, -shared_len:, :].float()
        v_n = v_namm[:, -shared_len:, :].float()
        v_t = v_trunc[:, -shared_len:, :].float()

        # Per-position L2 distance (averaged over heads)
        # k shape: (n_heads, shared_len, head_dim)
        k_l2 = torch.norm(k_n - k_t, dim=-1).mean(dim=0)  # (shared_len,)
        v_l2 = torch.norm(v_n - v_t, dim=-1).mean(dim=0)  # (shared_len,)

        # Per-position cosine similarity (averaged over heads)
        k_cos = F.cosine_similarity(
            k_n.reshape(-1, k_n.shape[-1]),
            k_t.reshape(-1, k_t.shape[-1])).reshape(
                k_n.shape[0], shared_len).mean(dim=0)  # (shared_len,)
        v_cos = F.cosine_similarity(
            v_n.reshape(-1, v_n.shape[-1]),
            v_t.reshape(-1, v_t.shape[-1])).reshape(
                v_n.shape[0], shared_len).mean(dim=0)  # (shared_len,)

        # Norms of truncation KVs (for relative L2)
        k_norm = torch.norm(k_t, dim=-1).mean(dim=0)  # (shared_len,)
        v_norm = torch.norm(v_t, dim=-1).mean(dim=0)

        # Aggregate metrics
        metrics = {
            "layer": layer_idx,
            "n_shared": int(shared_len),
            "namm_cache_len": int(namm_len),
            "trunc_cache_len": int(trunc_len),
            # Keys
            "key_l2_mean": float(k_l2.mean()),
            "key_l2_last": float(k_l2[-1]),
            "key_cosine_mean": float(k_cos.mean()),
            "key_cosine_last": float(k_cos[-1]),
            "key_relative_l2_mean": float((k_l2 / (k_norm + 1e-8)).mean()),
            # Values
            "val_l2_mean": float(v_l2.mean()),
            "val_l2_last": float(v_l2[-1]),
            "val_cosine_mean": float(v_cos.mean()),
            "val_cosine_last": float(v_cos[-1]),
            "val_relative_l2_mean": float((v_l2 / (v_norm + 1e-8)).mean()),
            # Ghost magnitude by position quartile
            # (first quarter of shared = oldest surviving, last = most recent)
            "key_l2_q1": float(k_l2[:shared_len//4].mean()),
            "key_l2_q4": float(k_l2[-shared_len//4:].mean()),
            "val_l2_q1": float(v_l2[:shared_len//4].mean()),
            "val_l2_q4": float(v_l2[-shared_len//4:].mean()),
        }
        metrics_per_layer.append(metrics)

    return metrics_per_layer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
                    job_name="ghost_info"):
        cfg = compose(config_name="config", overrides=overrides)

    # Build model
    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)

    # Load NAMM
    print(f"Loading NAMM: {args.namm_checkpoint}")
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']
    prefer_mean = cfg.get('prefer_mean_to_best', True)
    params_vec = (evo_state['mean'] if (prefer_mean and 'mean' in evo_state)
                  else evo_state['best_member'])
    memory_model.set_memory_params(params_vec.unsqueeze(0).to(device))
    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {k[len(buffers_prefix):]: v.to(device)
                    for k, v in evo_state.items() if k.startswith(buffers_prefix)}
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    memory_policy.record_eval_stats = True
    memory_policy.initialize_stat_objects()

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

    n_layers = len(memory_model.model.layers)
    out_path = os.path.join(
        args.output_dir,
        f"ghost_cs{args.cache_size}_{timestamp}.jsonl")
    print(f"Output: {out_path}")
    n_total = 0

    with open(out_path, "w") as f_out:
        header = {
            "_header": True,
            "cache_size": args.cache_size,
            "namm_checkpoint": os.path.abspath(args.namm_checkpoint),
            "n_layers": n_layers,
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

                    # (a) NAMM eviction: full prompt, NAMM active
                    memory_policy.initialize_stat_objects()
                    with torch.no_grad():
                        outputs_namm = memory_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            use_cache=True,
                            apply_memory_policy=True,
                        )
                    kv_namm = extract_kv_cache(outputs_namm.past_key_values)

                    # (b) Full context: all tokens, NO eviction
                    # Input > cache_size but apply_memory_policy=False so
                    # no eviction happens — the full KV cache is retained.
                    memory_policy.initialize_stat_objects()
                    with torch.no_grad():
                        outputs_full = memory_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            use_cache=True,
                            apply_memory_policy=False,
                        )
                    kv_full = extract_kv_cache(outputs_full.past_key_values)

                    # (c) Truncation: last cache_size tokens, no eviction
                    trunc_ids = input_ids[..., -args.cache_size:]
                    trunc_mask = attention_mask[..., -args.cache_size:]
                    memory_policy.initialize_stat_objects()
                    with torch.no_grad():
                        outputs_trunc = memory_model(
                            input_ids=trunc_ids,
                            attention_mask=trunc_mask,
                            output_hidden_states=False,
                            use_cache=True,
                            apply_memory_policy=True,
                        )
                    kv_trunc = extract_kv_cache(outputs_trunc.past_key_values)

                    # Compute ghost metrics for all three pairs:
                    # namm vs full:  ghost info (same RoPE, isolates eviction effect)
                    # namm vs trunc: ghost + RoPE confound (what we had before)
                    # full vs trunc: pure context difference (no ghost, just less context)
                    layer_metrics_namm_full = compute_ghost_metrics(
                        kv_namm, kv_full, n_layers)
                    layer_metrics_namm_trunc = compute_ghost_metrics(
                        kv_namm, kv_trunc, n_layers)
                    layer_metrics_full_trunc = compute_ghost_metrics(
                        kv_full, kv_trunc, n_layers)

                    record = {
                        "task": task_name,
                        "split": split_name,
                        "original_idx": int(orig_idx),
                        "n_tokens": n_tok,
                        "cache_size": args.cache_size,
                        "namm_vs_full": layer_metrics_namm_full,
                        "namm_vs_trunc": layer_metrics_namm_trunc,
                        "full_vs_trunc": layer_metrics_full_trunc,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    n_total += 1

                    if (i + 1) % 10 == 0:
                        m_nf = layer_metrics_namm_full[-1]
                        m_nt = layer_metrics_namm_trunc[-1]
                        print(f"    [{i+1}/{len(task_indices)}] "
                              f"namm_vs_full key_cos={m_nf['key_cosine_mean']:.4f} "
                              f"namm_vs_trunc key_cos={m_nt['key_cosine_mean']:.4f}")

                    del outputs_namm, outputs_full, outputs_trunc
                    del kv_namm, kv_full, kv_trunc
                    torch.cuda.empty_cache()

    print(f"\nDone: {n_total} prompts → {out_path}")

    # Summary
    records = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("_header"):
                records.append(r)

    if records:
        for pair_key, pair_label in [
            ("namm_vs_full",  "NAMM vs Full context (pure ghost, same RoPE)"),
            ("namm_vs_trunc", "NAMM vs Truncation (ghost + RoPE confound)"),
            ("full_vs_trunc", "Full vs Truncation (context loss, no ghost)"),
        ]:
            print(f"\n{'='*80}")
            print(f"  {pair_label} (cs={args.cache_size}, {n_total} prompts)")
            print(f"{'='*80}")
            print(f"{'Layer':>6s} {'Key cos':>8s} {'Val cos':>8s} "
                  f"{'Key L2rel':>10s} {'Val L2rel':>10s} "
                  f"{'Key L2 q1':>10s} {'Key L2 q4':>10s}")
            print("-" * 70)
            for l in range(n_layers):
                kc = np.mean([r[pair_key][l]["key_cosine_mean"] for r in records])
                vc = np.mean([r[pair_key][l]["val_cosine_mean"] for r in records])
                kr = np.mean([r[pair_key][l]["key_relative_l2_mean"] for r in records])
                vr = np.mean([r[pair_key][l]["val_relative_l2_mean"] for r in records])
                kq1 = np.mean([r[pair_key][l]["key_l2_q1"] for r in records])
                kq4 = np.mean([r[pair_key][l]["key_l2_q4"] for r in records])
                print(f"{l:6d} {kc:8.4f} {vc:8.4f} {kr:10.4f} {vr:10.4f} "
                      f"{kq1:10.4f} {kq4:10.4f}")

            # By prompt length
            print(f"\n  Key cosine (last layer) by prompt length:")
            for lo, hi, label in [(4096,5500,"in-dist"),(5500,6500,"in-dist"),
                                   (6500,7500,"near-OOD"),(7500,8200,"far-OOD")]:
                vals = [r[pair_key][-1]["key_cosine_mean"]
                        for r in records if lo <= r["n_tokens"] < hi]
                if vals:
                    print(f"    {lo}-{hi} ({label}): cos={np.mean(vals):.4f} n={len(vals)}")


if __name__ == "__main__":
    main()
