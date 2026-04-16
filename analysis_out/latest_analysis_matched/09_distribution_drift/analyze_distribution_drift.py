"""Distribution drift analysis: how much does adding/removing NAMM
change p(x_{t+1} | x_{1:t}) during generation?

Compares two pairs:
  Pair 1: M1 full cache  vs  M1 + post-hoc NAMM   (adding NAMM hurts M1)
  Pair 2: M4 + NAMM      vs  A4 full cache         (removing NAMM from M4)

Per generated token, captures:
  - JS divergence of the full output distribution
  - Top-k overlap (k=5, k=10) of the highest-probability tokens
  - Per-prompt F1 under each condition (from pre-computed eval JSONs)

Outputs:
  - drift_results.json: per-prompt, per-step divergence data
  - js_divergence_by_pair.png: mean JS divergence per generation step
  - topk_overlap_by_pair.png: mean top-k overlap per generation step
  - aggregate_summary.png: mean drift vs F1 delta scatter + bar summaries
  - per_prompt_drift_vs_f1.png: scatter of per-prompt drift vs F1 change

Requires GPU (~40 min on 3090 Ti, 4 conditions × 70 prompts).
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def reset_memory_policy_state(policy: Any) -> None:
    if hasattr(policy, "initialize_buffers"):
        policy.initialize_buffers()
    elif hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "initialize_stat_objects"):
        policy.initialize_stat_objects()


def swap_lora(model: Any, path: str, device: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["lora_state_dict"].items()}
    model.model.load_state_dict(sd, strict=False)
    model.to(device)
    logger.info("  Loaded LoRA: step=%s val=%s",
                ckpt.get("best_step", "?"), ckpt.get("best_val_score", "?"))


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two logit vectors (base e)."""
    p = F.softmax(logits_a.float(), dim=-1)
    q = F.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="sum", log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction="sum", log_target=False)
    return float(0.5 * (kl_pm + kl_qm))


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    """Fraction of top-k token IDs shared between two distributions."""
    top_a = set(logits_a.topk(k).indices.tolist())
    top_b = set(logits_b.topk(k).indices.tolist())
    return len(top_a & top_b) / k


# ────────────────────────────────────────────────────────────────────────────
# Generation with logit capture
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_logits(
    model: Any,
    policy: Any,
    input_ids: torch.Tensor,
    device: str,
    apply_memory_policy: bool,
    max_gen_tokens: int = 128,
) -> Dict[str, Any]:
    """Greedy-decode, capturing full logit vectors at each step.

    Returns dict with:
      - logits: list of (vocab_size,) tensors (CPU, float32)
      - token_ids: list of greedy token IDs
    """
    reset_memory_policy_state(policy)

    # Prefill
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        apply_memory_policy=apply_memory_policy,
        return_dict=True,
    )
    past_kv = outputs.past_key_values
    next_logits = outputs.logits[0, -1, :]  # (vocab,)

    all_logits: List[torch.Tensor] = []
    token_ids: List[int] = []

    for step in range(max_gen_tokens):
        all_logits.append(next_logits.cpu().float())

        next_token_id = next_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token_id.item()))

        eos_id = getattr(getattr(model, "config", None), "eos_token_id", 2)
        if isinstance(eos_id, list):
            eos_ids = set(eos_id)
        else:
            eos_ids = {eos_id}
        if next_token_id.item() in eos_ids:
            break

        next_input = next_token_id.unsqueeze(0)
        outputs = model(
            input_ids=next_input,
            past_key_values=past_kv,
            use_cache=True,
            apply_memory_policy=False,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[0, -1, :]

    del outputs, past_kv
    torch.cuda.empty_cache()

    return {"logits": all_logits, "token_ids": token_ids}


# ────────────────────────────────────────────────────────────────────────────
# Per-prompt comparison
# ────────────────────────────────────────────────────────────────────────────

def compare_logit_sequences(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two generation runs token-by-token.

    Uses the shorter sequence length (both are greedy, may differ due to
    different EOS timing under different cache states).
    """
    logits_a = result_a["logits"]
    logits_b = result_b["logits"]
    n = min(len(logits_a), len(logits_b))

    js_divs = []
    top5_overlaps = []
    top10_overlaps = []

    for t in range(n):
        js_divs.append(js_divergence(logits_a[t], logits_b[t]))
        top5_overlaps.append(topk_overlap(logits_a[t], logits_b[t], k=5))
        top10_overlaps.append(topk_overlap(logits_a[t], logits_b[t], k=10))

    return {
        "n_steps_compared": n,
        "n_steps_a": len(logits_a),
        "n_steps_b": len(logits_b),
        "tokens_a": result_a["token_ids"][:n],
        "tokens_b": result_b["token_ids"][:n],
        "token_match_rate": sum(
            1 for i in range(n)
            if result_a["token_ids"][i] == result_b["token_ids"][i]
        ) / max(n, 1),
        "js_divergence": js_divs,
        "top5_overlap": top5_overlaps,
        "top10_overlap": top10_overlaps,
        "mean_js": float(np.mean(js_divs)) if js_divs else None,
        "mean_top5": float(np.mean(top5_overlaps)) if top5_overlaps else None,
        "mean_top10": float(np.mean(top10_overlaps)) if top10_overlaps else None,
    }


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def pad_to_length(seqs: List[List[float]], length: int) -> np.ndarray:
    """Pad variable-length sequences with NaN, return (n_prompts, length)."""
    out = np.full((len(seqs), length), np.nan)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def plot_per_step_metric(
    pair1_seqs: List[List[float]],
    pair2_seqs: List[List[float]],
    ylabel: str,
    title: str,
    output_path: str,
    pair1_label: str = "M1: full vs +NAMM",
    pair2_label: str = "M4: +NAMM vs full",
    invert: bool = False,
):
    max_len = max(
        max((len(s) for s in pair1_seqs), default=0),
        max((len(s) for s in pair2_seqs), default=0),
    )
    if max_len == 0:
        return

    p1 = pad_to_length(pair1_seqs, max_len)
    p2 = pad_to_length(pair2_seqs, max_len)

    p1_mean = np.nanmean(p1, axis=0)
    p2_mean = np.nanmean(p2, axis=0)
    p1_se = np.nanstd(p1, axis=0) / np.sqrt(np.sum(~np.isnan(p1), axis=0).clip(1))
    p2_se = np.nanstd(p2, axis=0) / np.sqrt(np.sum(~np.isnan(p2), axis=0).clip(1))

    steps = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, p1_mean, color="#1f77b4", linewidth=1.5, label=pair1_label)
    ax.fill_between(steps, p1_mean - p1_se, p1_mean + p1_se,
                    color="#1f77b4", alpha=0.2)
    ax.plot(steps, p2_mean, color="#C44E52", linewidth=1.5, label=pair2_label)
    ax.fill_between(steps, p2_mean - p2_se, p2_mean + p2_se,
                    color="#C44E52", alpha=0.2)

    ax.set_xlabel("Generation step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_aggregate_bars(all_results: Dict, output_path: str):
    """Bar chart: mean JS divergence and token match rate for both pairs."""
    pair_labels = ["M1: full vs +NAMM", "M4: +NAMM vs full"]

    p1 = all_results["pair1_comparisons"]
    p2 = all_results["pair2_comparisons"]

    js_means = [
        np.mean([c["mean_js"] for c in p1 if c["mean_js"] is not None]),
        np.mean([c["mean_js"] for c in p2 if c["mean_js"] is not None]),
    ]
    match_rates = [
        np.mean([c["token_match_rate"] for c in p1]),
        np.mean([c["token_match_rate"] for c in p2]),
    ]
    top5_means = [
        np.mean([c["mean_top5"] for c in p1 if c["mean_top5"] is not None]),
        np.mean([c["mean_top5"] for c in p2 if c["mean_top5"] is not None]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#1f77b4", "#C44E52"]

    for ax, vals, ylabel, title in [
        (axes[0], js_means, "JS Divergence", "Mean JS Divergence"),
        (axes[1], match_rates, "Token Match Rate", "Greedy Token Match Rate"),
        (axes[2], top5_means, "Top-5 Overlap", "Mean Top-5 Overlap"),
    ]:
        bars = ax.bar(pair_labels, vals, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_drift_vs_f1(all_results: Dict, output_path: str):
    """Scatter: per-prompt mean JS divergence vs F1 change."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pair_key, f1_key_a, f1_key_b, title, color in [
        (axes[0], "pair1_comparisons", "m1_full_f1", "m1_namm_f1",
         "M1: full vs +NAMM", "#1f77b4"),
        (axes[1], "pair2_comparisons", "m4_namm_f1", "a4_full_f1",
         "M4: +NAMM vs full", "#C44E52"),
    ]:
        comps = all_results[pair_key]
        f1_a = all_results.get(f1_key_a, {})
        f1_b = all_results.get(f1_key_b, {})

        xs, ys = [], []
        for c in comps:
            if c["mean_js"] is None:
                continue
            prompt_key = str(c["original_idx"])
            if prompt_key in f1_a and prompt_key in f1_b:
                delta = f1_b[prompt_key] - f1_a[prompt_key]
                xs.append(c["mean_js"])
                ys.append(delta)

        if xs:
            ax.scatter(xs, ys, c=color, alpha=0.5, s=30, edgecolors="none")
            # Fit line
            if len(xs) > 2:
                from scipy import stats
                slope, intercept, r, p, se = stats.linregress(xs, ys)
                x_fit = np.linspace(min(xs), max(xs), 50)
                ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray",
                        linewidth=1)
                ax.text(0.05, 0.95,
                        f"r={r:.2f}, p={p:.3f}",
                        transform=ax.transAxes, fontsize=8, va="top")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Mean JS Divergence", fontsize=9)
        ax.set_ylabel("F1 change (pruned/full - baseline)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Distribution drift: JS divergence and top-k overlap "
                    "when adding/removing NAMM eviction.")
    p.add_argument("--m1_lora_checkpoint", type=str, required=True)
    p.add_argument("--m4_lora_checkpoint", type=str, required=True)
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--prompts_per_task", type=int, default=None,
                   help="Sample N prompts per task (deterministic). "
                        "Overrides --max_prompts.")
    p.add_argument("--max_gen_tokens", type=int, default=128)
    p.add_argument("--eval_json_dir", type=str, default=None,
                   help="Dir with per-prompt F1 JSONs (00_eval_results). "
                        "If provided, enables drift-vs-F1 scatter.")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def load_per_prompt_f1(eval_json_dir: str, split: str) -> Dict[str, Dict[str, float]]:
    """Load per-prompt F1 from eval JSONs for the drift-vs-F1 scatter.

    Returns dict mapping condition key -> {prompt_idx_str: f1}.
    """
    mapping = {
        "m1_full_f1": "m1_matched_full_cache.json",
        "m1_namm_f1": "m1_matched_under_namm_cs1024.json",
        "m4_namm_f1": "m4_lora_namm_cs1024.json",
        "a4_full_f1": "a4_m4_lora_no_namm.json",
    }
    out = {}
    for key, fname in mapping.items():
        path = os.path.join(eval_json_dir, fname)
        if not os.path.exists(path):
            logger.warning("F1 file not found: %s", path)
            continue
        with open(path) as f:
            d = json.load(f)
        split_data = d.get("scores_per_split", {}).get(split, {})
        per_prompt = split_data.get("per_prompt_f1", {})
        if per_prompt:
            # Flatten task -> {idx: f1} into {global_idx: f1}
            # We need to reconstruct the prompt ordering to match our iteration
            flat = {}
            for task_prompts in per_prompt.values():
                for idx, f1 in task_prompts.items():
                    flat[idx] = f1
            out[key] = flat
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    # ── Build model ──────────────────────────────────────────────────
    overrides = [
        f"run@_global_={args.run_config}", "wandb_log=false",
        "wandb_project=Experiments",
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
    ]
    with initialize(version_base=None, config_path="../../../config",
                    job_name="distribution_drift"):
        cfg = compose(config_name="config", overrides=overrides)

    logger.info("Building model...")
    with torch.no_grad():
        (policy, model, evaluator, _, _) = make_eval_model(cfg=cfg)
    model.to(device)

    # Load NAMM
    logger.info("Loading NAMM: %s", args.namm_checkpoint)
    namm_ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                           weights_only=False)
    evo = namm_ckpt["evolution_state"]
    pv = evo.get("mean", evo["best_member"])
    model.set_memory_params(pv.unsqueeze(0).to(device))
    bp = "stored_buffers_to_save."
    bd = {k[len(bp):]: v.to(device) for k, v in evo.items() if k.startswith(bp)}
    if bd:
        model.load_buffers_dict(buffers_dict=bd)
    policy.set_params_batch_idxs(np.zeros([1]))
    if hasattr(policy, "record_eval_stats"):
        policy.record_eval_stats = False

    model.apply_lora_adapters(rank=8, target_modules=["q_proj", "v_proj"])

    # ── Task sampler ─────────────────────────────────────────────────
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
    prompt_tasks = []
    prompt_original_idxs = []
    for task in sorted(split_idxs.keys()):
        task_ois = list(split_idxs[task])
        if args.prompts_per_task is not None:
            task_ois = task_ois[:args.prompts_per_task]
        for oi in task_ois:
            prompts.append(task_sampler.lb_prompts_per_task[task][int(oi)])
            prompt_tasks.append(task)
            prompt_original_idxs.append(int(oi))
    if args.max_prompts and args.prompts_per_task is None:
        prompts = prompts[:args.max_prompts]
        prompt_tasks = prompt_tasks[:args.max_prompts]
        prompt_original_idxs = prompt_original_idxs[:args.max_prompts]
    logger.info("  %d prompts from split '%s'", len(prompts), args.split)

    # ── Define conditions ────────────────────────────────────────────
    # (label, lora_ckpt_path, apply_memory_policy)
    conditions = [
        ("m1_full", args.m1_lora_checkpoint, False),
        ("m1_namm", args.m1_lora_checkpoint, True),
        ("m4_namm", args.m4_lora_checkpoint, True),
        ("a4_full", args.m4_lora_checkpoint, False),
    ]

    # ── Run all conditions ───────────────────────────────────────────
    bos = getattr(tokenizer, "bos_token", None) or ""
    results_by_condition: Dict[str, List[Dict]] = {}

    current_lora_path = None
    for cond_label, lora_path, apply_mp in conditions:
        logger.info("\n%s\nCondition: %s (NAMM=%s)\n%s",
                    "=" * 60, cond_label, apply_mp, "=" * 60)

        if lora_path != current_lora_path:
            swap_lora(model, lora_path, device)
            current_lora_path = lora_path

        cond_results = []
        for p_idx, raw_prompt in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            gen_result = generate_with_logits(
                model, policy, input_ids, device,
                apply_memory_policy=apply_mp,
                max_gen_tokens=args.max_gen_tokens,
            )
            cond_results.append(gen_result)

            if (p_idx + 1) % 10 == 0:
                logger.info("    %d/%d", p_idx + 1, len(prompts))

        results_by_condition[cond_label] = cond_results

    # ── Compare pairs ────────────────────────────────────────────────
    logger.info("\n%s\nComparing distributions...\n%s", "=" * 60, "=" * 60)

    pair1_comparisons = []  # M1 full vs M1+NAMM
    pair2_comparisons = []  # M4+NAMM vs A4 full

    for p_idx in range(len(prompts)):
        c1 = compare_logit_sequences(
            results_by_condition["m1_full"][p_idx],
            results_by_condition["m1_namm"][p_idx],
        )
        c1["task"] = prompt_tasks[p_idx]
        c1["original_idx"] = prompt_original_idxs[p_idx]
        pair1_comparisons.append(c1)

        c2 = compare_logit_sequences(
            results_by_condition["m4_namm"][p_idx],
            results_by_condition["a4_full"][p_idx],
        )
        c2["task"] = prompt_tasks[p_idx]
        c2["original_idx"] = prompt_original_idxs[p_idx]
        pair2_comparisons.append(c2)

    # Free logits from memory
    del results_by_condition
    torch.cuda.empty_cache()

    # ── Load F1 data if available ────────────────────────────────────
    f1_data = {}
    if args.eval_json_dir and os.path.isdir(args.eval_json_dir):
        f1_data = load_per_prompt_f1(args.eval_json_dir, args.split)
        logger.info("Loaded F1 data for %d conditions", len(f1_data))

    # ── Summary ──────────────────────────────────────────────────────
    def summarize_pair(comps, name):
        js_vals = [c["mean_js"] for c in comps if c["mean_js"] is not None]
        t5_vals = [c["mean_top5"] for c in comps if c["mean_top5"] is not None]
        t10_vals = [c["mean_top10"] for c in comps if c["mean_top10"] is not None]
        match_vals = [c["token_match_rate"] for c in comps]
        logger.info("\n%s:", name)
        logger.info("  Mean JS divergence:   %.4f (std %.4f)",
                     np.mean(js_vals), np.std(js_vals))
        logger.info("  Mean top-5 overlap:   %.4f (std %.4f)",
                     np.mean(t5_vals), np.std(t5_vals))
        logger.info("  Mean top-10 overlap:  %.4f (std %.4f)",
                     np.mean(t10_vals), np.std(t10_vals))
        logger.info("  Mean token match:     %.4f (std %.4f)",
                     np.mean(match_vals), np.std(match_vals))

    summarize_pair(pair1_comparisons, "Pair 1: M1 full vs M1+NAMM")
    summarize_pair(pair2_comparisons, "Pair 2: M4+NAMM vs A4 full")

    # ── Save JSON (strip logit sequences, keep per-step divergence) ──
    all_results = {
        "n_prompts": len(prompts),
        "max_gen_tokens": args.max_gen_tokens,
        "cache_size": args.cache_size,
        "pair1_comparisons": pair1_comparisons,
        "pair2_comparisons": pair2_comparisons,
    }
    all_results.update(f1_data)

    json_path = os.path.join(args.output_dir, "drift_results.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, default=str)
    logger.info("Saved %s", json_path)

    # ── Plots ────────────────────────────────────────────────────────
    plot_per_step_metric(
        [c["js_divergence"] for c in pair1_comparisons],
        [c["js_divergence"] for c in pair2_comparisons],
        ylabel="JS Divergence",
        title="Per-Step JS Divergence: Adding vs Removing NAMM",
        output_path=os.path.join(args.output_dir, "js_divergence_by_pair.png"),
    )

    plot_per_step_metric(
        [c["top5_overlap"] for c in pair1_comparisons],
        [c["top5_overlap"] for c in pair2_comparisons],
        ylabel="Top-5 Overlap",
        title="Per-Step Top-5 Overlap: Adding vs Removing NAMM",
        output_path=os.path.join(args.output_dir, "topk5_overlap_by_pair.png"),
    )

    plot_per_step_metric(
        [c["top10_overlap"] for c in pair1_comparisons],
        [c["top10_overlap"] for c in pair2_comparisons],
        ylabel="Top-10 Overlap",
        title="Per-Step Top-10 Overlap: Adding vs Removing NAMM",
        output_path=os.path.join(args.output_dir, "topk10_overlap_by_pair.png"),
    )

    plot_aggregate_bars(all_results, os.path.join(args.output_dir,
                                                   "aggregate_summary.png"))

    if f1_data:
        plot_drift_vs_f1(all_results, os.path.join(args.output_dir,
                                                     "drift_vs_f1.png"))

    logger.info("All outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
