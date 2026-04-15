"""Section B: Compare M1, M1+NAMM, and A4 (=M4 weights) under different eval conditions.

Three conditions using the SAME prompts:
  1. M1-matched, full cache (apply_memory_policy=False)
  2. M1-matched, NAMM active (apply_memory_policy=True, cs1024)
  3. A4 = M4 weights, full cache (apply_memory_policy=False)

Per prompt captures:
  - Last-token hidden states at every layer
  - Attention weights (last query → all KV positions)
  - Per-layer attention entropy

Analyses:
  - Hidden state L2/cosine between all 3 pairs
  - Attention entropy per layer per condition
  - Attention pattern correlation (M1 full vs A4 full — do they focus differently?)
  - Positional attention mass: where in the prompt does each condition attend?

Requires GPU.

Usage:
    python scripts/analyze_full_cache_comparison.py \\
        --m1_lora_checkpoint checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \\
        --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \\
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \\
        --cache_size 1024 \\
        --output_dir analysis_out/latest_analysis_matched/07_full_cache_comparison
"""

import argparse
import json
import os
import sys
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
from analyze_retained_tokens import reset_memory_policy_state

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def swap_lora(model, path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["lora_state_dict"].items()}
    model.model.load_state_dict(sd, strict=False)
    model.to(device)
    print(f"  Loaded: step={ckpt.get('best_step','?')} val={ckpt.get('best_val_score','?')}")


def run_prompt(model, policy, input_ids, device, apply_memory_policy):
    """Forward pass, return hidden states + attention weights."""
    reset_memory_policy_state(policy)
    policy.initialize_stat_objects()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            apply_memory_policy=apply_memory_policy,
            output_hidden_states=True,
            output_attentions=not apply_memory_policy,  # skip for NAMM (OOM)
        )

    hidden = [h[0, -1, :].detach().cpu().float() for h in outputs.hidden_states]

    # Attention: tuple of (batch, n_heads, seq_q, seq_kv) per layer
    # Only for full-cache runs (NAMM run skips to avoid OOM)
    attn = []
    if outputs.attentions is not None:
        for a in outputs.attentions:
            # Last query token's attention over all KV
            attn.append(a[0, :, -1, :].detach().cpu().float())  # (n_heads, n_kv)
    del outputs
    torch.cuda.empty_cache()

    return hidden, attn


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
                    job_name="full_cache_cmp"):
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
    policy.record_eval_stats = False

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

    bos = getattr(tokenizer, "bos_token", None) or ""

    # ── Three conditions ──────────────────────────────────────────────────
    conditions = [
        ("M1_full",  args.m1_lora_checkpoint, False, "M1-matched, full cache"),
        ("M1_namm",  args.m1_lora_checkpoint, True,  "M1-matched, NAMM cs1024"),
        ("A4_full",  args.m4_lora_checkpoint, False, "A4 (M4 weights), full cache"),
    ]

    all_data = {}  # cond_label -> list of {hidden, attn, n_input}
    for cond_label, ckpt_path, apply_mp, desc in conditions:
        print(f"\n{'='*60}\n{desc} ({cond_label})\n{'='*60}")
        swap_lora(model, ckpt_path, device)
        cond_data = []
        for p_idx, raw in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            hidden, attn = run_prompt(model, policy, input_ids, device, apply_mp)
            cond_data.append({
                "hidden": hidden,
                "attn": attn,
                "n_input": int(input_ids.shape[1]),
            })
            if (p_idx + 1) % 10 == 0:
                print(f"    {p_idx+1}/{len(prompts)}")
        all_data[cond_label] = cond_data

    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nAnalyzing...\n{'='*60}")
    n_prompts = len(prompts)
    n_layers = len(all_data["M1_full"][0]["hidden"])

    # Pairs to compare
    pairs = [
        ("M1_full", "M1_namm",  "M1 full → M1+NAMM"),
        ("M1_full", "A4_full",  "M1 full → A4 full"),
        ("M1_namm", "A4_full",  "M1+NAMM → A4 full"),
    ]

    # Per-layer hidden state comparison
    pair_l2 = {p[2]: np.zeros((n_prompts, n_layers)) for p in pairs}
    pair_cos = {p[2]: np.zeros((n_prompts, n_layers)) for p in pairs}

    for p_idx in range(n_prompts):
        for a_key, b_key, label in pairs:
            for l in range(n_layers):
                ha = all_data[a_key][p_idx]["hidden"][l]
                hb = all_data[b_key][p_idx]["hidden"][l]
                pair_l2[label][p_idx, l] = float((ha - hb).norm())
                pair_cos[label][p_idx, l] = float(
                    torch.nn.functional.cosine_similarity(
                        ha.unsqueeze(0), hb.unsqueeze(0)))

    # Per-layer attention entropy (full-cache conditions only)
    attn_entropy = {}
    for cond_label in ["M1_full", "A4_full"]:
        ent = np.zeros((n_prompts, n_layers - 1))  # n_layers-1 attention layers
        for p_idx in range(n_prompts):
            attn_list = all_data[cond_label][p_idx]["attn"]
            for l in range(min(len(attn_list), n_layers - 1)):
                # attn_list[l] shape: (n_heads, n_kv)
                a = attn_list[l].float()
                eps = 1e-10
                h = -(a * (a + eps).log()).sum(dim=-1).mean()  # mean over heads
                ent[p_idx, l] = float(h)
        attn_entropy[cond_label] = ent

    # Attention pattern correlation between M1_full and A4_full
    attn_corr = np.zeros((n_prompts, n_layers - 1))
    attn_mad = np.zeros((n_prompts, n_layers - 1))
    for p_idx in range(n_prompts):
        a_m1 = all_data["M1_full"][p_idx]["attn"]
        a_a4 = all_data["A4_full"][p_idx]["attn"]
        for l in range(min(len(a_m1), len(a_a4), n_layers - 1)):
            # Both are (n_heads, n_kv) — same size since both full cache
            m1 = a_m1[l].float()  # (n_heads, n_kv)
            a4 = a_a4[l].float()
            # Per-head correlation, mean
            corrs = []
            for h in range(m1.shape[0]):
                if m1[h].std() > 1e-8 and a4[h].std() > 1e-8:
                    c = float(torch.corrcoef(torch.stack([m1[h], a4[h]]))[0, 1])
                    if not np.isnan(c):
                        corrs.append(c)
            attn_corr[p_idx, l] = float(np.mean(corrs)) if corrs else 0
            attn_mad[p_idx, l] = float((m1 - a4).abs().mean())

    # Positional attention mass (thirds of prompt)
    pos_mass = {}
    for cond_label in ["M1_full", "A4_full"]:
        first = np.zeros((n_prompts, n_layers - 1))
        middle = np.zeros((n_prompts, n_layers - 1))
        last = np.zeros((n_prompts, n_layers - 1))
        for p_idx in range(n_prompts):
            n_input = all_data[cond_label][p_idx]["n_input"]
            third = n_input // 3
            attn_list = all_data[cond_label][p_idx]["attn"]
            for l in range(min(len(attn_list), n_layers - 1)):
                a = attn_list[l].float().mean(dim=0)  # (n_kv,) avg over heads
                first[p_idx, l] = float(a[:third].sum())
                middle[p_idx, l] = float(a[third:2*third].sum())
                last[p_idx, l] = float(a[2*third:].sum())
        pos_mass[cond_label] = {"first": first, "middle": middle, "last": last}

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")

    print("\nHidden state L2 distance (mean across prompts):")
    print(f"{'Layer':>5} | {'M1→M1+NAMM':>12} | {'M1→A4':>12} | {'M1+NAMM→A4':>12}")
    print("-" * 50)
    for l in range(n_layers):
        vals = [f"{pair_l2[p[2]][:, l].mean():12.4f}" for p in pairs]
        print(f"  L{l:2d} | {'|'.join(vals)}")
    print(f"\n  Means:")
    for _, _, label in pairs:
        print(f"    {label}: {pair_l2[label].mean():.4f}")

    print("\nAttention entropy (mean across prompts, full-cache conditions):")
    for cond in ["M1_full", "A4_full"]:
        print(f"  {cond}: {attn_entropy[cond].mean():.4f}")

    print("\nAttention correlation M1_full vs A4_full:")
    print(f"  Mean: {attn_corr.mean():.4f}")

    print("\nPositional attention mass (M1_full vs A4_full):")
    for cond in ["M1_full", "A4_full"]:
        pm = pos_mass[cond]
        print(f"  {cond}: first={pm['first'].mean():.3f} "
              f"middle={pm['middle'].mean():.3f} last={pm['last'].mean():.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_data = {
        "n_prompts": n_prompts, "n_layers": n_layers,
        "hidden_l2": {k: v.mean(axis=0).tolist() for k, v in pair_l2.items()},
        "hidden_cos": {k: v.mean(axis=0).tolist() for k, v in pair_cos.items()},
        "attn_entropy": {k: v.mean(axis=0).tolist() for k, v in attn_entropy.items()},
        "attn_corr": attn_corr.mean(axis=0).tolist(),
        "attn_mad": attn_mad.mean(axis=0).tolist(),
        "pos_mass": {cond: {r: v.mean(axis=0).tolist()
                            for r, v in regions.items()}
                     for cond, regions in pos_mass.items()},
    }
    with open(os.path.join(args.output_dir, "full_cache_comparison.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────
    layers = np.arange(n_layers)
    attn_layers = np.arange(n_layers - 1)
    pair_colors = {"M1 full → M1+NAMM": "#f44336",
                   "M1 full → A4 full": "#1565c0",
                   "M1+NAMM → A4 full": "#4caf50"}

    # Plot 1: Hidden state L2 + cosine
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for _, _, label in pairs:
        ax.plot(layers, pair_l2[label].mean(axis=0), "o-",
                color=pair_colors[label], label=label, linewidth=2, markersize=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 distance")
    ax.set_title("Hidden State Distance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for _, _, label in pairs:
        ax.plot(layers, pair_cos[label].mean(axis=0), "o-",
                color=pair_colors[label], label=label, linewidth=2, markersize=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Hidden State Direction"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.01)

    plt.suptitle("M1 full vs M1+NAMM vs A4 full: Hidden States",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "hidden_states.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Attention entropy + correlation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for cond, color, label in [("M1_full", "#1f77b4", "M1 (full cache)"),
                                ("A4_full", "#bcbd22", "A4 (full cache)")]:
        ax.plot(attn_layers, attn_entropy[cond].mean(axis=0), "o-",
                color=color, label=label, linewidth=2, markersize=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("Entropy (nats)")
    ax.set_title("Attention Entropy\n(lower = more focused)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(attn_layers, attn_corr.mean(axis=0), "o-", color="#2e7d32", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Pearson correlation")
    ax.set_title("Attention Pattern Correlation\n(M1 full vs A4 full)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(attn_layers, attn_mad.mean(axis=0), "o-", color="#d32f2f", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean abs difference")
    ax.set_title("Attention MAD\n(M1 full vs A4 full)"); ax.grid(alpha=0.3)

    plt.suptitle("M1 vs A4 Under Full Cache: Attention Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "attention_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Positional attention mass
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (cond, label) in enumerate([
        ("M1_full", "M1 (full cache)"), ("A4_full", "A4 (full cache)")
    ]):
        ax = axes[ax_idx]
        pm = pos_mass[cond]
        ax.plot(attn_layers, pm["first"].mean(axis=0), "o-",
                color="#ef5350", label="First 1/3", linewidth=2, markersize=4)
        ax.plot(attn_layers, pm["middle"].mean(axis=0), "s-",
                color="#ffca28", label="Middle 1/3", linewidth=2, markersize=4)
        ax.plot(attn_layers, pm["last"].mean(axis=0), "^-",
                color="#66bb6a", label="Last 1/3", linewidth=2, markersize=4)
        ax.set_xlabel("Layer"); ax.set_ylabel("Attention mass")
        ax.set_title(f"{label}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Where in the Prompt Does Each Model Attend?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "positional_attention.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: L2 ratio — which comparison is bigger?
    fig, ax = plt.subplots(figsize=(10, 5))
    m1_namm_l2 = pair_l2["M1 full → M1+NAMM"].mean(axis=0)
    m1_a4_l2 = pair_l2["M1 full → A4 full"].mean(axis=0)
    ratio = m1_a4_l2 / np.maximum(m1_namm_l2, 1e-10)
    colors = ["#1565c0" if r > 1 else "#f44336" for r in ratio]
    ax.bar(layers, ratio, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("L2(M1→A4) / L2(M1→M1+NAMM)")
    ax.set_title("Which Changes Hidden States More: LoRA Adaptation or NAMM Eviction?\n"
                 "(blue >1 = LoRA difference larger, red <1 = NAMM eviction larger)")
    for i, r in enumerate(ratio):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "l2_ratio.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
