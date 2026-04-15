"""Section B Part B: Hidden state drift under eviction.

For each model (M1, M4), run the SAME prompts twice:
  1. Full cache (apply_memory_policy=False)
  2. NAMM active (apply_memory_policy=True)

Capture the last token's hidden state at every layer. Compute per-layer
L2 distance between the two conditions. If M4 shows smaller drift than
M1, training under eviction made the model's representations more
invariant to pruning.

Also computes:
  - Cosine similarity of hidden states (full vs NAMM) per layer
  - Relative drift: ||h_full - h_namm|| / ||h_full|| per layer
  - Layer-wise drift profile: does drift accumulate or stay constant?

Requires GPU.

Usage:
    python scripts/analyze_hidden_state_drift.py \\
        --m1_lora_checkpoint checkpoints_backup/lora_m1_original/best_ckpt.pt \\
        --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \\
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \\
        --cache_size 1024 \\
        --output_dir analysis_out/hidden_state_drift
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


def get_last_token_hidden_states(
    model, policy, input_ids, device, apply_memory_policy: bool,
) -> List[torch.Tensor]:
    """Run forward pass and return last token's hidden state at each layer.

    Returns list of (hidden_dim,) tensors, one per layer + embedding.
    """
    reset_memory_policy_state(policy)
    policy.initialize_stat_objects()

    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            apply_memory_policy=apply_memory_policy,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (batch, seq_len_of_last_chunk, hidden_dim)
    # One entry per layer + 1 for embedding output = n_layers + 1 total
    # Take last token position from each
    hidden_states = []
    if outputs.hidden_states is not None:
        for hs in outputs.hidden_states:
            hidden_states.append(hs[0, -1, :].detach().cpu().float())
    return hidden_states


def analyze_drift(
    h_full: List[torch.Tensor],
    h_namm: List[torch.Tensor],
) -> Dict:
    """Compute per-layer drift metrics between full-cache and NAMM hidden states."""
    n_layers = len(h_full)
    l2_dists = []
    cosine_sims = []
    relative_drifts = []

    for l in range(n_layers):
        diff = h_full[l] - h_namm[l]
        l2 = float(diff.norm())
        norm_full = float(h_full[l].norm())
        cos = float(torch.nn.functional.cosine_similarity(
            h_full[l].unsqueeze(0), h_namm[l].unsqueeze(0)))

        l2_dists.append(l2)
        cosine_sims.append(cos)
        relative_drifts.append(l2 / max(norm_full, 1e-10))

    return {
        "l2_dist": l2_dists,
        "cosine_sim": cosine_sims,
        "relative_drift": relative_drifts,
        "n_layers": n_layers,
    }


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
                    job_name="hidden_drift"):
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
    prompt_meta = []
    for task in sorted(split_idxs.keys()):
        for oi in split_idxs[task]:
            prompts.append(task_sampler.lb_prompts_per_task[task][int(oi)])
            prompt_meta.append({"task": task, "orig_idx": int(oi)})
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        prompt_meta = prompt_meta[:args.max_prompts]
    print(f"  {len(prompts)} prompts")

    bos = getattr(tokenizer, "bos_token", None) or ""

    # ── Run each model under both conditions ──────────────────────────────
    all_drifts = {}  # model_label -> list of per-prompt drift dicts

    for model_label, ckpt_path in [
        ("m1", args.m1_lora_checkpoint),
        ("m4", args.m4_lora_checkpoint),
    ]:
        print(f"\n{'='*60}\n{model_label.upper()}: {ckpt_path}\n{'='*60}")
        swap_lora(model, ckpt_path, device)

        drifts = []
        for p_idx, raw in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            # Full cache forward
            h_full = get_last_token_hidden_states(
                model, policy, input_ids, device, apply_memory_policy=False)

            # NAMM forward
            h_namm = get_last_token_hidden_states(
                model, policy, input_ids, device, apply_memory_policy=True)

            drift = analyze_drift(h_full, h_namm)
            drift["n_input_tokens"] = int(input_ids.shape[1])
            drift["task"] = prompt_meta[p_idx]["task"]
            drifts.append(drift)

            if (p_idx + 1) % 10 == 0:
                print(f"    {p_idx+1}/{len(prompts)}")

        all_drifts[model_label] = drifts

    # ── Aggregate ─────────────────────────────────────────────────────────
    n_layers = all_drifts["m1"][0]["n_layers"]
    agg = {}
    for model_label in ["m1", "m4"]:
        drifts = all_drifts[model_label]
        l2_per_layer = np.array([d["l2_dist"] for d in drifts])  # (n_prompts, n_layers)
        cos_per_layer = np.array([d["cosine_sim"] for d in drifts])
        rel_per_layer = np.array([d["relative_drift"] for d in drifts])
        agg[model_label] = {
            "l2_mean": l2_per_layer.mean(axis=0).tolist(),
            "l2_std": l2_per_layer.std(axis=0).tolist(),
            "cosine_mean": cos_per_layer.mean(axis=0).tolist(),
            "cosine_std": cos_per_layer.std(axis=0).tolist(),
            "rel_drift_mean": rel_per_layer.mean(axis=0).tolist(),
            "rel_drift_std": rel_per_layer.std(axis=0).tolist(),
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nHIDDEN STATE DRIFT SUMMARY\n{'='*60}")
    print(f"\n{'Layer':>6} | {'M1 L2':>10} | {'M4 L2':>10} | {'M1 cos':>8} | "
          f"{'M4 cos':>8} | {'M1 rel%':>8} | {'M4 rel%':>8}")
    print("-" * 75)
    for l in range(n_layers):
        m1_l2 = agg["m1"]["l2_mean"][l]
        m4_l2 = agg["m4"]["l2_mean"][l]
        m1_cos = agg["m1"]["cosine_mean"][l]
        m4_cos = agg["m4"]["cosine_mean"][l]
        m1_rel = agg["m1"]["rel_drift_mean"][l] * 100
        m4_rel = agg["m4"]["rel_drift_mean"][l] * 100
        winner = "←" if m4_l2 < m1_l2 else ""
        print(f"  L{l:2d}  | {m1_l2:10.4f} | {m4_l2:10.4f} {winner:>1}| "
              f"{m1_cos:8.5f} | {m4_cos:8.5f} | {m1_rel:7.2f}% | {m4_rel:7.2f}%")

    avg_m1 = np.mean(agg["m1"]["l2_mean"])
    avg_m4 = np.mean(agg["m4"]["l2_mean"])
    print(f"\n  Mean L2 drift: M1={avg_m1:.4f}  M4={avg_m4:.4f}")
    if avg_m4 < avg_m1:
        pct = (1 - avg_m4/avg_m1) * 100
        print(f"  → M4 has {pct:.1f}% less hidden state drift under eviction")
        print(f"    Training under eviction makes representations more invariant to pruning")
    else:
        pct = (avg_m4/avg_m1 - 1) * 100
        print(f"  → M4 has {pct:.1f}% MORE drift — specialization, not robustness")

    # ── Save ──────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "hidden_drift.json")
    with open(json_path, "w") as f:
        json.dump({"n_prompts": len(prompts), "n_layers": n_layers,
                    "aggregate": agg,
                    "per_prompt": {k: v for k, v in all_drifts.items()}},
                  f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    layers = np.arange(n_layers)

    # Plot 1: L2 drift comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    m1_l2 = np.array(agg["m1"]["l2_mean"])
    m4_l2 = np.array(agg["m4"]["l2_mean"])
    m1_l2_std = np.array(agg["m1"]["l2_std"])
    m4_l2_std = np.array(agg["m4"]["l2_std"])
    ax.plot(layers, m1_l2, "o-", color="#1f77b4", label="M1 (post-hoc)", linewidth=2)
    ax.fill_between(layers, m1_l2 - m1_l2_std, m1_l2 + m1_l2_std,
                     color="#1f77b4", alpha=0.15)
    ax.plot(layers, m4_l2, "s-", color="#e377c2", label="M4 (in-loop)", linewidth=2)
    ax.fill_between(layers, m4_l2 - m4_l2_std, m4_l2 + m4_l2_std,
                     color="#e377c2", alpha=0.15)
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 distance")
    ax.set_title("Hidden State Drift: Full Cache → NAMM\n(lower = more robust to eviction)")
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 2: Cosine similarity
    ax = axes[1]
    m1_cos = np.array(agg["m1"]["cosine_mean"])
    m4_cos = np.array(agg["m4"]["cosine_mean"])
    ax.plot(layers, m1_cos, "o-", color="#1f77b4", label="M1", linewidth=2)
    ax.plot(layers, m4_cos, "s-", color="#e377c2", label="M4", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Hidden State Direction Similarity\n(full cache vs NAMM)")
    ax.set_ylim(0.5, 1.01); ax.legend(); ax.grid(alpha=0.3)

    # Plot 3: Relative drift
    ax = axes[2]
    m1_rel = np.array(agg["m1"]["rel_drift_mean"]) * 100
    m4_rel = np.array(agg["m4"]["rel_drift_mean"]) * 100
    ax.plot(layers, m1_rel, "o-", color="#1f77b4", label="M1", linewidth=2)
    ax.plot(layers, m4_rel, "s-", color="#e377c2", label="M4", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Relative drift (%)")
    ax.set_title("Relative Hidden State Drift\n(||h_full - h_namm|| / ||h_full||)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Does Training Under Eviction Reduce Hidden State Drift?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "hidden_drift.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Drift ratio M4/M1
    fig, ax = plt.subplots(figsize=(10, 5))
    ratio = m4_l2 / np.maximum(m1_l2, 1e-10)
    colors = ["#4caf50" if r < 1 else "#f44336" for r in ratio]
    ax.bar(layers, ratio, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("M4 drift / M1 drift")
    ax.set_title("Drift Ratio Per Layer\n(green < 1 = M4 more robust, red > 1 = M1 more robust)")
    for i, r in enumerate(ratio):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "drift_ratio.png"), dpi=150)
    plt.close(fig)

    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
