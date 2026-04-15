"""Section C: Eviction mask drift under LoRA fine-tuning.

Since NAMM's eviction decisions depend on the LLM's attention activations,
fine-tuning the LLM changes which tokens get evicted even with frozen NAMM
weights. This script quantifies how much the eviction mask changes between
the base model (no LoRA) and LoRA-finetuned models (M1, M4).

For the same prompts under the same frozen NAMM policy:
  1. Base model (no LoRA) → eviction decisions
  2. M1 LoRA → eviction decisions
  3. M4 LoRA → eviction decisions

Computes per-layer:
  - Jaccard(base, M1) and Jaccard(base, M4): how much did LoRA change eviction?
  - Jaccard(M1, M4): how differently do the two LoRA variants affect eviction?
  - Positional analysis: do the drifted tokens come from specific prompt regions?
  - Drift magnitude: fraction of tokens that changed eviction status

If drift is small (Jaccard > 0.9), LoRA+NAMM training is stable — the eviction
pattern is robust to parameter updates. If drift is large, the coupled
optimization concern from the paper is empirically validated.

Requires GPU.

Usage:
    python scripts/analyze_eviction_mask_drift.py \\
        --m1_lora_checkpoint checkpoints_backup/lora_m1_original/best_ckpt.pt \\
        --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \\
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \\
        --cache_size 1024 \\
        --output_dir analysis_out/eviction_mask_drift
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

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
from analyze_retained_tokens import (
    reset_memory_policy_state, run_prompt_forward, extract_kept_positions)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def swap_lora(model, path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["lora_state_dict"].items()}
    model.model.load_state_dict(sd, strict=False)
    model.to(device)
    print(f"  Loaded: step={ckpt.get('best_step','?')} val={ckpt.get('best_val_score','?')}")


def zero_lora(model):
    """Zero out all LoRA parameters to get base model behavior."""
    for name, param in model.model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data.zero_()
    print("  LoRA weights zeroed (base model behavior)")


def run_all_prompts(model, policy, tokenizer, prompts, device, n_layers):
    """Run all prompts, return list of kept_per_layer (union over heads)."""
    bos = getattr(tokenizer, "bos_token", None) or ""
    all_kept = []
    for p_idx, raw in enumerate(prompts):
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            add_generation_prompt=True, tokenize=False)
        if bos and templated.startswith(bos):
            templated = templated[len(bos):]
        enc = tokenizer(templated, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        n_input = int(input_ids.shape[1])

        run_prompt_forward(model, policy, input_ids, device)
        kept, _ = extract_kept_positions(policy, n_input, aggregation="union")
        all_kept.append({"kept_per_layer": kept, "n_input": n_input})

        if (p_idx + 1) % 10 == 0:
            print(f"    {p_idx+1}/{len(prompts)}")
    return all_kept


def compare_kept_sets(kept_a, kept_b, n_layers, n_input):
    """Compare two models' kept sets per layer."""
    layers = []
    for l in range(n_layers):
        sa = set(kept_a[l])
        sb = set(kept_b[l])
        union = sa | sb
        jaccard = len(sa & sb) / max(len(union), 1)
        # Tokens that changed status
        a_only = sa - sb
        b_only = sb - sa
        drift_frac = (len(a_only) + len(b_only)) / max(len(union), 1)

        # Positional breakdown of drifted tokens
        third = n_input / 3.0
        def pos_breakdown(positions):
            if not positions:
                return {"first": 0, "middle": 0, "last": 0}
            arr = np.array(sorted(positions))
            return {
                "first": int(np.sum(arr < third)),
                "middle": int(np.sum((arr >= third) & (arr < 2*third))),
                "last": int(np.sum(arr >= 2*third)),
            }

        layers.append({
            "jaccard": jaccard,
            "n_shared": len(sa & sb),
            "n_a_only": len(a_only),
            "n_b_only": len(b_only),
            "drift_frac": drift_frac,
            "a_only_pos": pos_breakdown(a_only),
            "b_only_pos": pos_breakdown(b_only),
        })
    return layers


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
                    job_name="mask_drift"):
        cfg = compose(config_name="config", overrides=overrides)

    print("Building model...")
    with torch.no_grad():
        (policy, model, evaluator, _, _) = make_eval_model(cfg=cfg)
    model.to(device)

    # Load NAMM
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu", weights_only=False)
    evo = ckpt["evolution_state"]
    pv = evo["mean"] if cfg.get("prefer_mean_to_best", True) and "mean" in evo else evo["best_member"]
    model.set_memory_params(pv.unsqueeze(0).to(device))
    bp = "stored_buffers_to_save."
    bd = {k[len(bp):]: v.to(device) for k, v in evo.items() if k.startswith(bp)}
    if bd:
        model.load_buffers_dict(buffers_dict=bd)
    policy.set_params_batch_idxs(np.zeros([1]))
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

    # ── Run three conditions ──────────────────────────────────────────────
    conditions = {}

    # 1. Base model (zero LoRA)
    print(f"\n{'='*60}\nBASE (LoRA zeroed)\n{'='*60}")
    zero_lora(model)
    conditions["base"] = run_all_prompts(model, policy, tokenizer, prompts, device, n_layers)

    # 2. M1 LoRA
    print(f"\n{'='*60}\nM1 LoRA\n{'='*60}")
    swap_lora(model, args.m1_lora_checkpoint, device)
    conditions["m1"] = run_all_prompts(model, policy, tokenizer, prompts, device, n_layers)

    # 3. M4 LoRA
    print(f"\n{'='*60}\nM4 LoRA\n{'='*60}")
    swap_lora(model, args.m4_lora_checkpoint, device)
    conditions["m4"] = run_all_prompts(model, policy, tokenizer, prompts, device, n_layers)

    # ── Compare ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nCOMPARING\n{'='*60}")
    pairs = [("base", "m1"), ("base", "m4"), ("m1", "m4")]
    pair_results = {}

    for a_label, b_label in pairs:
        per_prompt = []
        for p_idx in range(len(prompts)):
            n_input = conditions[a_label][p_idx]["n_input"]
            comp = compare_kept_sets(
                conditions[a_label][p_idx]["kept_per_layer"],
                conditions[b_label][p_idx]["kept_per_layer"],
                n_layers, n_input)
            per_prompt.append(comp)
        pair_results[f"{a_label}_vs_{b_label}"] = per_prompt

    # ── Aggregate and print ───────────────────────────────────────────────
    agg = {}
    for pair_name, per_prompt in pair_results.items():
        jacc_per_layer = []
        drift_per_layer = []
        for l in range(n_layers):
            j_vals = [pp[l]["jaccard"] for pp in per_prompt]
            d_vals = [pp[l]["drift_frac"] for pp in per_prompt]
            jacc_per_layer.append(float(np.mean(j_vals)))
            drift_per_layer.append(float(np.mean(d_vals)))
        agg[pair_name] = {"jaccard": jacc_per_layer, "drift_frac": drift_per_layer}

    print(f"\n{'Layer':>5} | {'Base→M1':>8} | {'Base→M4':>8} | {'M1→M4':>8}")
    print(f"{'':>5} | {'Jaccard':>8} | {'Jaccard':>8} | {'Jaccard':>8}")
    print("-" * 45)
    for l in range(n_layers):
        bm1 = agg["base_vs_m1"]["jaccard"][l]
        bm4 = agg["base_vs_m4"]["jaccard"][l]
        mm = agg["m1_vs_m4"]["jaccard"][l]
        print(f"  L{l:2d} | {bm1:8.3f} | {bm4:8.3f} | {mm:8.3f}")

    print(f"\n  Mean Jaccard:")
    for pair_name in agg:
        mean_j = np.mean(agg[pair_name]["jaccard"])
        mean_d = np.mean(agg[pair_name]["drift_frac"])
        label = pair_name.replace("_vs_", " → ").replace("base", "Base")
        print(f"    {label}: Jaccard={mean_j:.3f}, drift={mean_d:.1%}")

    mean_bm1 = np.mean(agg["base_vs_m1"]["jaccard"])
    mean_bm4 = np.mean(agg["base_vs_m4"]["jaccard"])
    if mean_bm1 > 0.9 and mean_bm4 > 0.9:
        print("\n  → Eviction mask drift is SMALL. LoRA+NAMM training is stable:")
        print("    the eviction pattern is robust to parameter updates.")
    elif mean_bm4 > mean_bm1:
        print("\n  → M4 drifts MORE from base than M1 — training under eviction")
        print("    actively reshapes the attention patterns NAMM uses for scoring.")
    else:
        print("\n  → M1 drifts MORE — even full-cache LoRA changes eviction patterns")
        print("    substantially. The concern about coupled optimization is empirically valid.")

    # ── Save ──────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "eviction_mask_drift.json")
    with open(json_path, "w") as f:
        json.dump({"n_prompts": len(prompts), "n_layers": n_layers,
                    "aggregate": agg}, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for pair_name, color, marker in [
        ("base_vs_m1", "#1f77b4", "o"),
        ("base_vs_m4", "#e377c2", "s"),
        ("m1_vs_m4", "#2e7d32", "D"),
    ]:
        label = pair_name.replace("_vs_", " → ").replace("base", "Base")
        ax.plot(layers, agg[pair_name]["jaccard"], f"{marker}-",
                color=color, label=label, linewidth=2, markersize=5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Jaccard similarity")
    ax.set_title("Eviction Mask Stability Under LoRA")
    ax.set_ylim(0.4, 1.05); ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.5, label="stability threshold")

    ax = axes[1]
    for pair_name, color, marker in [
        ("base_vs_m1", "#1f77b4", "o"),
        ("base_vs_m4", "#e377c2", "s"),
        ("m1_vs_m4", "#2e7d32", "D"),
    ]:
        label = pair_name.replace("_vs_", " → ").replace("base", "Base")
        drift = [d * 100 for d in agg[pair_name]["drift_frac"]]
        ax.plot(layers, drift, f"{marker}-", color=color, label=label,
                linewidth=2, markersize=5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Token drift (%)")
    ax.set_title("Fraction of Tokens Changing Eviction Status")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Does LoRA Fine-Tuning Change NAMM's Eviction Decisions?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "eviction_mask_drift.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
