"""Deep analysis of NAMM scoring, eviction decisions, and attention patterns
between M1 (post-hoc NAMM) and M4 (NAMM in-loop) LoRA checkpoints.

Addresses:
  1. Kept-set decomposition: tokens kept by both, M1-only, M4-only, both-evicted
  2. Score distributions conditioned on kept/evicted status per model
  3. Score concentration: does M4 produce narrower scores around kept tokens?
  4. Score gap: top-k vs non-kept score separation per model
  5. Per-token attention difference on shared kept tokens
  6. Positional analysis: where in the prompt are model-specific kept tokens?
  7. Score rank correlation between M1 and M4

Requires GPU. Outputs JSON + plots.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device
from analyze_retained_tokens import reset_memory_policy_state

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────
# Score capture (same as v1)
# ────────────────────────────────────────────────────────────────────────────

class ScoreCapture:
    def __init__(self, memory_policy, n_layers):
        self.memory_policy = memory_policy
        self.n_layers = n_layers
        self._original_fn = memory_policy.record_deep_stats
        self.reset()

    def reset(self):
        self.token_scores = [None] * self.n_layers
        self.retained_idxs = [None] * self.n_layers

    def install(self):
        capture = self
        def patched(layer_id, **kwargs):
            if "token_scores" in kwargs:
                capture.token_scores[layer_id] = kwargs["token_scores"].detach().cpu()
            if "retained_idxs" in kwargs:
                capture.retained_idxs[layer_id] = kwargs["retained_idxs"].detach().cpu()
            capture._original_fn(layer_id=layer_id, **kwargs)
        self.memory_policy.record_deep_stats = patched

    def uninstall(self):
        self.memory_policy.record_deep_stats = self._original_fn


# ────────────────────────────────────────────────────────────────────────────
# Per-prompt deep analysis
# ────────────────────────────────────────────────────────────────────────────

def get_position_sets(retained_idxs, cache_position_ids, n_heads):
    """Convert retained indices to sets of original positions per head,
    then compute union across heads."""
    head_sets = []
    for h in range(n_heads):
        if cache_position_ids is not None and cache_position_ids.dim() == 3:
            positions = cache_position_ids[0, h, :].tolist()
        elif cache_position_ids is not None and cache_position_ids.dim() == 2:
            positions = cache_position_ids[0, :].tolist()
        else:
            positions = retained_idxs[0, h, :].tolist()
        head_sets.append(set(int(p) for p in positions))
    union = set().union(*head_sets) if head_sets else set()
    return head_sets, union


def analyze_prompt_pair(
    m1_data: Dict,
    m4_data: Dict,
    n_layers: int,
    n_input: int,
) -> Dict:
    """Full analysis for one prompt, comparing M1 and M4."""
    layers = []

    for layer_id in range(n_layers):
        ts_m1 = m1_data["token_scores"][layer_id]
        ts_m4 = m4_data["token_scores"][layer_id]
        ri_m1 = m1_data["retained_idxs"][layer_id]
        ri_m4 = m4_data["retained_idxs"][layer_id]
        cp_m1 = m1_data["cache_position_ids"][layer_id]
        cp_m4 = m4_data["cache_position_ids"][layer_id]

        if any(x is None for x in [ts_m1, ts_m4, ri_m1, ri_m4]):
            layers.append(None)
            continue

        n_heads = ts_m1.shape[1]
        seq_len = ts_m1.shape[2]

        # ── A. Kept-set decomposition (union over heads, ABSOLUTE positions) ─
        _, m1_union = get_position_sets(ri_m1, cp_m1, n_heads)
        _, m4_union = get_position_sets(ri_m4, cp_m4, n_heads)

        both_kept = m1_union & m4_union
        m1_only = m1_union - m4_union
        m4_only = m4_union - m1_union
        all_abs_positions = set(range(n_input))
        both_evicted_approx = all_abs_positions - (m1_union | m4_union)

        # ── B. Per-model score analysis using CACHE-LOCAL indices ──────────
        # Score tensors are indexed by cache slot, not absolute position.
        # Use retained_idxs (cache-local) for kept/evicted split within
        # each model independently.
        scores_m1_avg = ts_m1[0].mean(dim=0).numpy()  # (m1_cache_len,)
        scores_m4_avg = ts_m4[0].mean(dim=0).numpy()  # (m4_cache_len,)
        seq_len_m1 = ts_m1.shape[2]
        seq_len_m4 = ts_m4.shape[2]

        def score_stats_from_array(vals):
            if len(vals) == 0:
                return {"mean": None, "std": None, "min": None, "max": None,
                        "median": None, "iqr": None, "n": 0}
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "n": len(vals),
            }

        # Per-model kept/evicted using cache-local retained_idxs (union over heads)
        m1_kept_local = set()
        m4_kept_local = set()
        for h in range(n_heads):
            m1_kept_local.update(ri_m1[0, h, :].tolist())
            m4_kept_local.update(ri_m4[0, h, :].tolist())
        m1_evicted_local = sorted(set(range(seq_len_m1)) - m1_kept_local)
        m4_evicted_local = sorted(set(range(seq_len_m4)) - m4_kept_local)
        m1_kept_local = sorted(m1_kept_local)
        m4_kept_local = sorted(m4_kept_local)

        m1_kept_scores = scores_m1_avg[m1_kept_local] if m1_kept_local else np.array([])
        m1_evicted_scores = scores_m1_avg[m1_evicted_local] if m1_evicted_local else np.array([])
        m4_kept_scores = scores_m4_avg[m4_kept_local] if m4_kept_local else np.array([])
        m4_evicted_scores = scores_m4_avg[m4_evicted_local] if m4_evicted_local else np.array([])

        # Cross-model score comparison on SHARED absolute positions.
        # Build position→score maps for each model.
        # For model M, cache slot i has absolute position from the
        # pre-eviction position tracking. We approximate this: after
        # eviction, cache_position_ids gives abs positions of survivors.
        # For evicted tokens we don't have the abs→score mapping (they
        # were removed). So cross-model score comparison is limited to
        # the union of survived tokens.
        m1_scores_both_kept = score_stats_from_array(m1_kept_scores)
        m1_scores_m1_only = {"n": len(m1_only)}  # can't index M1 scores by M4's positions
        m4_scores_both_kept = score_stats_from_array(m4_kept_scores)
        m4_scores_m4_only = {"n": len(m4_only)}

        # ── C. Score concentration ────────────────────────────────────────
        m1_all_std = float(np.std(scores_m1_avg))
        m4_all_std = float(np.std(scores_m4_avg))
        m1_kept_std = float(np.std(m1_kept_scores)) if len(m1_kept_scores) > 0 else None
        m4_kept_std = float(np.std(m4_kept_scores)) if len(m4_kept_scores) > 0 else None

        # ── D. Score gap: mean(kept) - mean(evicted) ──────────────────────
        m1_score_gap = (float(np.mean(m1_kept_scores) - np.mean(m1_evicted_scores))
                        if len(m1_kept_scores) > 0 and len(m1_evicted_scores) > 0 else None)
        m4_score_gap = (float(np.mean(m4_kept_scores) - np.mean(m4_evicted_scores))
                        if len(m4_kept_scores) > 0 and len(m4_evicted_scores) > 0 else None)

        # Top-k vs bottom scores (cache-local, per model)
        n_kept_m1 = len(m1_kept_local)
        n_kept_m4 = len(m4_kept_local)
        m1_sorted = np.sort(scores_m1_avg)
        m4_sorted = np.sort(scores_m4_avg)
        m1_topk_mean = float(np.mean(m1_sorted[-n_kept_m1:])) if n_kept_m1 > 0 else None
        m1_bottomk_mean = float(np.mean(m1_sorted[:-n_kept_m1])) if n_kept_m1 < seq_len_m1 else None
        m4_topk_mean = float(np.mean(m4_sorted[-n_kept_m4:])) if n_kept_m4 > 0 else None
        m4_bottomk_mean = float(np.mean(m4_sorted[:-n_kept_m4])) if n_kept_m4 < seq_len_m4 else None

        # ── E. Score rank correlation ─────────────────────────────────────
        # Only meaningful if both score tensors cover the same tokens.
        # At layer 0, both caches start from the same prompt so seq_len
        # matches. At later layers, caches diverge. Compute correlation
        # only when seq_lens match.
        if seq_len_m1 == seq_len_m4 and seq_len_m1 > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(scores_m1_avg, scores_m4_avg)
        else:
            spearman_r, spearman_p = None, None

        # ── F. Positional analysis of model-specific tokens ───────────────
        third = n_input / 3.0
        def positional_breakdown(positions):
            if not positions:
                return {"first_third": 0, "middle_third": 0, "last_third": 0,
                        "mean_pos": None, "std_pos": None}
            arr = np.array(sorted(positions))
            return {
                "first_third": int(np.sum(arr < third)),
                "middle_third": int(np.sum((arr >= third) & (arr < 2*third))),
                "last_third": int(np.sum(arr >= 2*third)),
                "mean_pos": float(np.mean(arr)),
                "std_pos": float(np.std(arr)),
                "mean_pos_frac": float(np.mean(arr) / max(n_input, 1)),
            }

        # ── G. Score histograms for this layer ────────────────────────────
        all_scores_combined = np.concatenate([scores_m1_avg, scores_m4_avg])
        lo = float(np.percentile(all_scores_combined, 1))
        hi = float(np.percentile(all_scores_combined, 99))
        if lo >= hi:
            lo, hi = float(all_scores_combined.min()) - 0.1, float(all_scores_combined.max()) + 0.1
        hist_bins = np.linspace(lo, hi, 50)
        m1_hist_kept = np.histogram(m1_kept_scores, bins=hist_bins)[0].tolist() if len(m1_kept_scores) else [0]*49
        m1_hist_evicted = np.histogram(m1_evicted_scores, bins=hist_bins)[0].tolist() if len(m1_evicted_scores) else [0]*49
        m4_hist_kept = np.histogram(m4_kept_scores, bins=hist_bins)[0].tolist() if len(m4_kept_scores) else [0]*49
        m4_hist_evicted = np.histogram(m4_evicted_scores, bins=hist_bins)[0].tolist() if len(m4_evicted_scores) else [0]*49

        layer_result = {
            # A: Kept-set decomposition
            "n_both_kept": len(both_kept),
            "n_m1_only": len(m1_only),
            "n_m4_only": len(m4_only),
            "n_both_evicted": len(both_evicted_approx),
            "n_m1_total_kept": len(m1_union),
            "n_m4_total_kept": len(m4_union),
            "jaccard": len(both_kept) / max(len(m1_union | m4_union), 1),

            # B: Scores per category (cache-local, per model)
            "m1_scores_both_kept": m1_scores_both_kept,
            "m1_scores_m1_only": m1_scores_m1_only,
            "m4_scores_both_kept": m4_scores_both_kept,
            "m4_scores_m4_only": m4_scores_m4_only,

            # C: Score concentration
            "m1_all_score_std": m1_all_std,
            "m4_all_score_std": m4_all_std,
            "m1_kept_score_std": m1_kept_std,
            "m4_kept_score_std": m4_kept_std,

            # D: Score gap
            "m1_score_gap": m1_score_gap,
            "m4_score_gap": m4_score_gap,
            "m1_topk_mean": m1_topk_mean,
            "m1_bottomk_mean": m1_bottomk_mean,
            "m4_topk_mean": m4_topk_mean,
            "m4_bottomk_mean": m4_bottomk_mean,

            # E: Rank correlation
            "spearman_r": float(spearman_r) if spearman_r is not None else None,
            "spearman_p": float(spearman_p) if spearman_p is not None else None,

            # F: Positional
            "m1_only_position": positional_breakdown(m1_only),
            "m4_only_position": positional_breakdown(m4_only),
            "both_kept_position": positional_breakdown(both_kept),

            # G: Histograms
            "hist_bins": hist_bins.tolist(),
            "m1_hist_kept": m1_hist_kept,
            "m1_hist_evicted": m1_hist_evicted,
            "m4_hist_kept": m4_hist_kept,
            "m4_hist_evicted": m4_hist_evicted,
        }

        # ── H. Per-token attention on shared kept tokens ──────────────────
        attn_m1 = m1_data["attention_weights"]
        attn_m4 = m4_data["attention_weights"]
        if (layer_id < len(attn_m1) and layer_id < len(attn_m4)
                and cp_m1 is not None and cp_m4 is not None):
            a_m1 = attn_m1[layer_id][0]  # (n_attn_heads, n_q, n_kv_m1)
            a_m4 = attn_m4[layer_id][0]  # (n_attn_heads, n_q, n_kv_m4)
            n_attn_heads = a_m1.shape[0]

            # Build position→kv_index maps for each model
            if cp_m1.dim() == 3:
                n_kv_heads = cp_m1.shape[1]
            else:
                n_kv_heads = 1
            gqa_ratio = n_attn_heads // n_kv_heads

            shared_attn_diffs = []
            shared_attn_corrs = []

            for ah in range(n_attn_heads):
                kv_h = ah // gqa_ratio
                if cp_m1.dim() == 3:
                    pos_m1 = cp_m1[0, kv_h, :].tolist()
                else:
                    pos_m1 = cp_m1[0, :].tolist()
                if cp_m4.dim() == 3:
                    pos_m4 = cp_m4[0, kv_h, :].tolist()
                else:
                    pos_m4 = cp_m4[0, :].tolist()

                pos2idx_m1 = {int(p): i for i, p in enumerate(pos_m1)}
                pos2idx_m4 = {int(p): i for i, p in enumerate(pos_m4)}
                shared = sorted(set(pos2idx_m1.keys()) & set(pos2idx_m4.keys()))

                if len(shared) < 5:
                    continue

                # Attention on last query token over shared positions
                a1 = a_m1[ah, -1, :]  # (n_kv_m1,)
                a4 = a_m4[ah, -1, :]  # (n_kv_m4,)

                a1_shared = torch.stack([a1[pos2idx_m1[p]] for p in shared]).float()
                a4_shared = torch.stack([a4[pos2idx_m4[p]] for p in shared]).float()

                # Normalize over shared set
                a1_norm = a1_shared / (a1_shared.sum() + 1e-10)
                a4_norm = a4_shared / (a4_shared.sum() + 1e-10)

                # Mean absolute difference
                mad = float((a1_norm - a4_norm).abs().mean())
                shared_attn_diffs.append(mad)

                # Correlation
                if a1_norm.std() > 1e-8 and a4_norm.std() > 1e-8:
                    corr = float(torch.corrcoef(
                        torch.stack([a1_norm, a4_norm]))[0, 1])
                    if not np.isnan(corr):
                        shared_attn_corrs.append(corr)

            layer_result["shared_attn_mad_mean"] = (
                float(np.mean(shared_attn_diffs)) if shared_attn_diffs else None)
            layer_result["shared_attn_corr_mean"] = (
                float(np.mean(shared_attn_corrs)) if shared_attn_corrs else None)
        else:
            layer_result["shared_attn_mad_mean"] = None
            layer_result["shared_attn_corr_mean"] = None

        layers.append(layer_result)

    return {"n_input_tokens": n_input, "layers": layers}


# ────────────────────────────────────────────────────────────────────────────
# Model run (same as v1)
# ────────────────────────────────────────────────────────────────────────────

def run_prompts(model, policy, tokenizer, prompts, device, capture, n_layers):
    bos = getattr(tokenizer, "bos_token", None) or ""
    results = []
    for p_idx, raw in enumerate(prompts):
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            add_generation_prompt=True, tokenize=False)
        if bos and templated.startswith(bos):
            templated = templated[len(bos):]
        enc = tokenizer(templated, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        n_input = int(input_ids.shape[1])

        reset_memory_policy_state(policy)
        policy.initialize_stat_objects()
        capture.reset()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True, apply_memory_policy=True,
                output_attentions=True)

        cache_pos = []
        for l in range(n_layers):
            cp = policy.cache_position_ids[l]
            cache_pos.append(cp.detach().cpu() if cp is not None else None)

        attn = [a.detach().cpu() for a in outputs.attentions] if outputs.attentions else []

        results.append({
            "n_input_tokens": n_input,
            "token_scores": [s.clone() if s is not None else None for s in capture.token_scores],
            "retained_idxs": [r.clone() if r is not None else None for r in capture.retained_idxs],
            "cache_position_ids": cache_pos,
            "attention_weights": attn,
        })
        if (p_idx + 1) % 10 == 0:
            print(f"    {p_idx+1}/{len(prompts)}")
    return results


def swap_lora(model, path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["lora_state_dict"].items()}
    model.model.load_state_dict(sd, strict=False)
    model.to(device)
    print(f"  Loaded: step={ckpt.get('best_step','?')} val={ckpt.get('best_val_score','?')}")


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def aggregate_per_layer(all_analyses, key, n_layers):
    """Collect a scalar metric across prompts, return per-layer mean."""
    out = []
    for l in range(n_layers):
        vals = []
        for a in all_analyses:
            if a["layers"][l] is not None:
                v = a["layers"][l].get(key)
                if v is not None:
                    vals.append(v)
        out.append(float(np.mean(vals)) if vals else None)
    return out


def aggregate_nested(all_analyses, layer_key, nested_key, n_layers):
    out = []
    for l in range(n_layers):
        vals = []
        for a in all_analyses:
            if a["layers"][l] is not None:
                d = a["layers"][l].get(layer_key)
                if d and d.get(nested_key) is not None:
                    vals.append(d[nested_key])
        out.append(float(np.mean(vals)) if vals else None)
    return out


def safe_plot(x, y, *args, **kwargs):
    """Plot ignoring None values."""
    mask = [v is not None for v in y]
    xf = [x[i] for i in range(len(x)) if mask[i]]
    yf = [y[i] for i in range(len(y)) if mask[i]]
    plt.plot(xf, yf, *args, **kwargs)


def make_plots(all_analyses, n_layers, out_dir):
    layers = list(range(n_layers))

    # ── Plot 1: Kept-set decomposition ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    both = aggregate_per_layer(all_analyses, "n_both_kept", n_layers)
    m1o = aggregate_per_layer(all_analyses, "n_m1_only", n_layers)
    m4o = aggregate_per_layer(all_analyses, "n_m4_only", n_layers)
    ax.bar(layers, both, label="Both kept", color="#4caf50")
    ax.bar(layers, m1o, bottom=both, label="M1 only", color="#1f77b4")
    b2 = [a+b for a,b in zip(both, m1o)]
    ax.bar(layers, m4o, bottom=b2, label="M4 only", color="#e377c2")
    ax.set_xlabel("Layer"); ax.set_ylabel("Token count")
    ax.set_title("Kept-Token Decomposition Per Layer")
    ax.legend(fontsize=9)

    ax = axes[1]
    jacc = aggregate_per_layer(all_analyses, "jaccard", n_layers)
    ax.plot(layers, jacc, "o-", color="#2e7d32", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Jaccard")
    ax.set_title("Kept-Token Jaccard (M1 vs M4)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kept_decomposition.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Score gap and concentration ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    m1_gap = aggregate_per_layer(all_analyses, "m1_score_gap", n_layers)
    m4_gap = aggregate_per_layer(all_analyses, "m4_score_gap", n_layers)
    safe_plot(layers, m1_gap, "o-", color="#1f77b4", label="M1", linewidth=2)
    safe_plot(layers, m4_gap, "s-", color="#e377c2", label="M4", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("mean(kept) - mean(evicted)")
    ax.set_title("Score Gap (kept vs evicted)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    m1_kstd = aggregate_per_layer(all_analyses, "m1_kept_score_std", n_layers)
    m4_kstd = aggregate_per_layer(all_analyses, "m4_kept_score_std", n_layers)
    m1_astd = aggregate_per_layer(all_analyses, "m1_all_score_std", n_layers)
    m4_astd = aggregate_per_layer(all_analyses, "m4_all_score_std", n_layers)
    safe_plot(layers, m1_kstd, "o-", color="#1f77b4", label="M1 kept std")
    safe_plot(layers, m4_kstd, "s-", color="#e377c2", label="M4 kept std")
    safe_plot(layers, m1_astd, "o--", color="#1f77b4", alpha=0.4, label="M1 all std")
    safe_plot(layers, m4_astd, "s--", color="#e377c2", alpha=0.4, label="M4 all std")
    ax.set_xlabel("Layer"); ax.set_ylabel("Score std")
    ax.set_title("Score Concentration\n(lower kept_std = more concentrated)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    m1_tk = aggregate_per_layer(all_analyses, "m1_topk_mean", n_layers)
    m1_bk = aggregate_per_layer(all_analyses, "m1_bottomk_mean", n_layers)
    m4_tk = aggregate_per_layer(all_analyses, "m4_topk_mean", n_layers)
    m4_bk = aggregate_per_layer(all_analyses, "m4_bottomk_mean", n_layers)
    safe_plot(layers, m1_tk, "o-", color="#1f77b4", label="M1 top-k mean")
    safe_plot(layers, m1_bk, "o--", color="#1f77b4", alpha=0.4, label="M1 bottom mean")
    safe_plot(layers, m4_tk, "s-", color="#e377c2", label="M4 top-k mean")
    safe_plot(layers, m4_bk, "s--", color="#e377c2", alpha=0.4, label="M4 bottom mean")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean NAMM score")
    ax.set_title("Top-k vs Bottom Scores")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_gap_concentration.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: Rank correlation + attention on shared tokens ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    sr = aggregate_per_layer(all_analyses, "spearman_r", n_layers)
    safe_plot(layers, sr, "o-", color="#2e7d32", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Spearman rho")
    ax.set_title("Score Rank Correlation (M1 vs M4)")
    ax.set_ylim(-0.1, 1.05); ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    ax = axes[1]
    mad = aggregate_per_layer(all_analyses, "shared_attn_mad_mean", n_layers)
    safe_plot(layers, mad, "o-", color="#d32f2f", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean abs difference")
    ax.set_title("Per-Token Attention Difference\n(on shared kept tokens, normalized)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    corr = aggregate_per_layer(all_analyses, "shared_attn_corr_mean", n_layers)
    safe_plot(layers, corr, "o-", color="#1565c0", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Pearson correlation")
    ax.set_title("Attention Correlation on Shared Tokens")
    ax.set_ylim(-0.1, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rank_corr_attention.png"), dpi=150)
    plt.close(fig)

    # ── Plot 4: Positional analysis of model-specific tokens ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    m1_first = aggregate_nested(all_analyses, "m1_only_position", "first_third", n_layers)
    m1_mid = aggregate_nested(all_analyses, "m1_only_position", "middle_third", n_layers)
    m1_last = aggregate_nested(all_analyses, "m1_only_position", "last_third", n_layers)
    w = 0.25
    x = np.arange(n_layers)
    ax.bar(x - w, m1_first, w, label="First 1/3", color="#ef5350")
    ax.bar(x, m1_mid, w, label="Middle 1/3", color="#ffca28")
    ax.bar(x + w, m1_last, w, label="Last 1/3", color="#66bb6a")
    ax.set_xlabel("Layer"); ax.set_ylabel("Count")
    ax.set_title("M1-only tokens: where in the prompt?")
    ax.legend(fontsize=9)

    ax = axes[1]
    m4_first = aggregate_nested(all_analyses, "m4_only_position", "first_third", n_layers)
    m4_mid = aggregate_nested(all_analyses, "m4_only_position", "middle_third", n_layers)
    m4_last = aggregate_nested(all_analyses, "m4_only_position", "last_third", n_layers)
    ax.bar(x - w, m4_first, w, label="First 1/3", color="#ef5350")
    ax.bar(x, m4_mid, w, label="Middle 1/3", color="#ffca28")
    ax.bar(x + w, m4_last, w, label="Last 1/3", color="#66bb6a")
    ax.set_xlabel("Layer"); ax.set_ylabel("Count")
    ax.set_title("M4-only tokens: where in the prompt?")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "positional_analysis.png"), dpi=150)
    plt.close(fig)

    # ── Plot 5: Score histograms (selected layers) ────────────────────────
    show_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    fig, axes = plt.subplots(len(show_layers), 2, figsize=(14, 3*len(show_layers)))

    for row, lid in enumerate(show_layers):
        for col, (model_label, hist_kept_key, hist_evicted_key, color) in enumerate([
            ("M1", "m1_hist_kept", "m1_hist_evicted", "#1f77b4"),
            ("M4", "m4_hist_kept", "m4_hist_evicted", "#e377c2"),
        ]):
            ax = axes[row][col]
            # Average histograms across prompts
            kept_hists = []
            evicted_hists = []
            bins_list = []
            for a in all_analyses:
                if a["layers"][lid] is not None:
                    kept_hists.append(a["layers"][lid][hist_kept_key])
                    evicted_hists.append(a["layers"][lid][hist_evicted_key])
                    bins_list.append(a["layers"][lid]["hist_bins"])
            if not kept_hists:
                continue
            kept_avg = np.mean(kept_hists, axis=0)
            evicted_avg = np.mean(evicted_hists, axis=0)
            bins = np.array(bins_list[0])
            centers = 0.5 * (bins[:-1] + bins[1:])
            w = (bins[1] - bins[0]) * 0.4
            ax.bar(centers - w/2, kept_avg, w, color="#4caf50", alpha=0.7, label="Kept")
            ax.bar(centers + w/2, evicted_avg, w, color="#f44336", alpha=0.7, label="Evicted")
            ax.set_title(f"{model_label} — Layer {lid}", fontsize=10)
            ax.set_xlabel("NAMM score"); ax.set_ylabel("Count")
            if row == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_histograms_v2.png"), dpi=150)
    plt.close(fig)

    # ── Plot 6: Kept vs evicted score means per model ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (model_label, kept_key, evicted_gap_key, color) in enumerate([
        ("M1", "m1_scores_both_kept", "m1_score_gap", "#1f77b4"),
        ("M4", "m4_scores_both_kept", "m4_score_gap", "#e377c2"),
    ]):
        ax = axes[col]
        kept_mean = aggregate_nested(all_analyses, kept_key, "mean", n_layers)
        gap = aggregate_per_layer(all_analyses, evicted_gap_key, n_layers)
        # evicted_mean ≈ kept_mean - gap
        evicted_mean = [
            (k - g if k is not None and g is not None else None)
            for k, g in zip(kept_mean, gap)]

        safe_plot(layers, kept_mean, "o-", color="#4caf50", label="Kept mean", linewidth=2)
        safe_plot(layers, evicted_mean, "x--", color="#f44336", label="Evicted mean", linewidth=2)
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean NAMM score")
        ax.set_title(f"{model_label} — Kept vs Evicted Scores")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scores_by_category.png"), dpi=150)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_lora_checkpoint", type=str, required=True)
    p.add_argument("--m4_lora_checkpoint", type=str, required=True)
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    overrides = [
        f"run@_global_={args.run_config}", "wandb_log=false",
        "wandb_project=Experiments",
        f"cache_size={args.cache_size}", f"max_memory_length={args.cache_size}",
    ]
    with initialize(version_base=None, config_path="../config",
                    job_name="analyze_v2"):
        cfg = compose(config_name="config", overrides=overrides)

    print("Building model...")
    with torch.no_grad():
        (policy, model, evaluator, _, _) = make_eval_model(cfg=cfg)
    model.to(device)

    # Load NAMM
    print(f"Loading NAMM: {args.namm_checkpoint}")
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu", weights_only=False)
    evo = ckpt["evolution_state"]
    pv = evo["mean"] if cfg.get("prefer_mean_to_best", True) and "mean" in evo else evo["best_member"]
    model.set_memory_params(pv.unsqueeze(0).to(device))
    bp = "stored_buffers_to_save."
    bd = {k[len(bp):]: v.to(device) for k, v in evo.items() if k.startswith(bp)}
    if bd:
        model.load_buffers_dict(buffers_dict=bd)
    policy.set_params_batch_idxs(np.zeros([1]))
    policy.record_eval_stats = True
    policy.initialize_stat_objects()
    n_layers = policy.num_memory_layers

    model.apply_lora_adapters(rank=8, target_modules=["q_proj", "v_proj"])

    # Task sampler
    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    task_sampler.filter_answers_by_token_count(
        tokenizer, cfg.get("max_answer_tokens", cfg.get("max_new_tokens", 64)))
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get("train_frac", 0.7),
        val_frac=cfg.get("val_frac", 0.15),
        max_conditioning_length=cfg.get("split_max_conditioning_length",
                                         cfg.get("max_conditioning_length", 6500)),
        min_conditioning_length=cfg.get("min_conditioning_length", None),
        tokenizer=tokenizer)
    split_idxs = task_sampler.get_split_indices(args.split)
    prompts = []
    for task in sorted(split_idxs.keys()):
        for oi in split_idxs[task]:
            prompts.append(task_sampler.lb_prompts_per_task[task][int(oi)])
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  {len(prompts)} prompts")

    # Run both models
    capture = ScoreCapture(policy, n_layers)
    capture.install()

    model_results = {}
    for label, path in [("m1", args.m1_lora_checkpoint), ("m4", args.m4_lora_checkpoint)]:
        print(f"\n{'='*60}\nRunning {label.upper()}\n{'='*60}")
        swap_lora(model, path, device)
        model_results[label] = run_prompts(model, policy, tokenizer, prompts, device, capture, n_layers)

    capture.uninstall()

    # Analyze
    print(f"\n{'='*60}\nAnalyzing...\n{'='*60}")
    all_analyses = []
    for p_idx in range(len(prompts)):
        a = analyze_prompt_pair(
            model_results["m1"][p_idx], model_results["m4"][p_idx],
            n_layers, model_results["m1"][p_idx]["n_input_tokens"])
        all_analyses.append(a)

    # Summary
    print(f"\nPer-layer summary (averaged across {len(prompts)} prompts):")
    print(f"{'L':>3} | {'Jaccard':>7} | {'M1gap':>7} | {'M4gap':>7} | "
          f"{'M1kStd':>7} | {'M4kStd':>7} | {'Spearman':>8} | "
          f"{'AttnMAD':>7} | {'AttnCorr':>8} | {'M1only':>6} | {'M4only':>6}")
    print("-" * 105)
    for l in range(n_layers):
        j = aggregate_per_layer(all_analyses, "jaccard", n_layers)[l]
        m1g = aggregate_per_layer(all_analyses, "m1_score_gap", n_layers)[l]
        m4g = aggregate_per_layer(all_analyses, "m4_score_gap", n_layers)[l]
        m1ks = aggregate_per_layer(all_analyses, "m1_kept_score_std", n_layers)[l]
        m4ks = aggregate_per_layer(all_analyses, "m4_kept_score_std", n_layers)[l]
        sr = aggregate_per_layer(all_analyses, "spearman_r", n_layers)[l]
        mad = aggregate_per_layer(all_analyses, "shared_attn_mad_mean", n_layers)[l]
        ac = aggregate_per_layer(all_analyses, "shared_attn_corr_mean", n_layers)[l]
        m1o = aggregate_per_layer(all_analyses, "n_m1_only", n_layers)[l]
        m4o = aggregate_per_layer(all_analyses, "n_m4_only", n_layers)[l]
        def f(v, w=7): return f"{v:{w}.4f}" if v is not None else f"{'N/A':>{w}}"
        print(f"L{l:2d} | {f(j)} | {f(m1g)} | {f(m4g)} | {f(m1ks)} | {f(m4ks)} | "
              f"{f(sr,8)} | {f(mad)} | {f(ac,8)} | {f(m1o,6)} | {f(m4o,6)}")

    # Save
    # Strip tensors from analyses for JSON serialization
    json_path = os.path.join(args.output_dir, "deep_analysis.json")
    with open(json_path, "w") as fp:
        json.dump({"n_prompts": len(prompts), "n_layers": n_layers,
                    "analyses": all_analyses}, fp, default=str)
    print(f"\nSaved: {json_path}")

    make_plots(all_analyses, n_layers, args.output_dir)
    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
