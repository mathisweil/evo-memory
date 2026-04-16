"""Post-hoc filter: drop prompts where any condition generated >2x the gold
answer length. Those are repetition/hallucination loops, not real answers.

Recomputes aggregate metrics and regenerates plots on the filtered subset.
Operates on drift_results.json — no GPU re-run needed.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_task_sampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_gold_lengths(split: str, tokenizer) -> Dict[str, Dict[int, int]]:
    """Load gold answer lengths for every (task, original_idx) in split.

    Returns {task: {original_idx: shortest_gold_answer_len_in_tokens}}.
    Replicates the train/val/test split logic from run_utils so the
    indices align with what the drift analysis captured.
    """
    overrides = [
        "run@_global_=namm_bam_i1_llama32_1b_5t",
        "wandb_log=false",
        "wandb_project=Experiments",
        "cache_size=1024",
        "max_memory_length=1024",
    ]
    with initialize(version_base=None, config_path="../../../config",
                    job_name="gold_lengths"):
        cfg = compose(config_name="config", overrides=overrides)

    task_sampler = make_task_sampler(cfg=cfg)
    task_sampler.filter_answers_by_token_count(
        tokenizer, cfg.get("max_answer_tokens", cfg.get("max_new_tokens", 64)))
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get("train_frac", 0.7),
        val_frac=cfg.get("val_frac", 0.15),
        max_conditioning_length=cfg.get("split_max_conditioning_length",
                                         cfg.get("max_conditioning_length", 6500)),
        min_conditioning_length=cfg.get("min_conditioning_length", None),
        tokenizer=tokenizer)

    out: Dict[str, Dict[int, int]] = {}
    for task, jsons in task_sampler.lb_jsons_per_task.items():
        out[task] = {}
        for i, j in enumerate(jsons):
            answers = j.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            shortest = min(
                len(tokenizer.encode(a, add_special_tokens=False))
                for a in answers)
            out[task][i] = shortest
    return out


def pad_to_length(seqs: List[List[float]], length: int) -> np.ndarray:
    out = np.full((len(seqs), length), np.nan)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def plot_per_step_metric(pair1_seqs, pair2_seqs, ylabel, title, output_path,
                         pair1_label="M1: full vs +NAMM",
                         pair2_label="M4: +NAMM vs full"):
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


def plot_aggregate_bars(pair1, pair2, output_path):
    pair_labels = ["M1: full vs +NAMM", "M4: +NAMM vs full"]
    js_means = [
        np.mean([c["mean_js"] for c in pair1 if c["mean_js"] is not None]),
        np.mean([c["mean_js"] for c in pair2 if c["mean_js"] is not None]),
    ]
    match_rates = [
        np.mean([c["token_match_rate"] for c in pair1]),
        np.mean([c["token_match_rate"] for c in pair2]),
    ]
    top5_means = [
        np.mean([c["mean_top5"] for c in pair1 if c["mean_top5"] is not None]),
        np.mean([c["mean_top5"] for c in pair2 if c["mean_top5"] is not None]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#1f77b4", "#C44E52"]
    for ax, vals, ylabel, title in [
        (axes[0], js_means, "JS Divergence", "Mean JS Divergence"),
        (axes[1], match_rates, "Token Match Rate", "Greedy Token Match Rate"),
        (axes[2], top5_means, "Top-5 Overlap", "Mean Top-5 Overlap"),
    ]:
        bars = ax.bar(pair_labels, vals, color=colors, edgecolor="white",
                      width=0.5)
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


def plot_drift_vs_f1(pair1, pair2, f1_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, comps, f1_key_a, f1_key_b, title, color in [
        (axes[0], pair1, "m1_full_f1", "m1_namm_f1",
         "M1: full vs +NAMM", "#1f77b4"),
        (axes[1], pair2, "m4_namm_f1", "a4_full_f1",
         "M4: +NAMM vs full", "#C44E52"),
    ]:
        f1_a = f1_data.get(f1_key_a, {})
        f1_b = f1_data.get(f1_key_b, {})
        xs, ys = [], []
        for c in comps:
            if c["mean_js"] is None:
                continue
            prompt_key = str(c["original_idx"])
            if prompt_key in f1_a and prompt_key in f1_b:
                fa, fb = f1_a[prompt_key], f1_b[prompt_key]
                if fa == 0 and fb == 0:
                    continue
                delta = fb - fa
                xs.append(c["mean_js"])
                ys.append(delta)
        if xs:
            ax.scatter(xs, ys, c=color, alpha=0.5, s=30, edgecolors="none")
            if len(xs) > 2:
                from scipy import stats
                slope, intercept, r, p, se = stats.linregress(xs, ys)
                x_fit = np.linspace(min(xs), max(xs), 50)
                ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray",
                        linewidth=1)
                ax.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f} (n={len(xs)})",
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


def summarize(pair, name):
    js = [c["mean_js"] for c in pair if c["mean_js"] is not None]
    t5 = [c["mean_top5"] for c in pair if c["mean_top5"] is not None]
    t10 = [c["mean_top10"] for c in pair if c["mean_top10"] is not None]
    match = [c["token_match_rate"] for c in pair]
    logger.info("%s (n=%d):", name, len(pair))
    logger.info("  JS div:    %.4f (std %.4f)", np.mean(js), np.std(js))
    logger.info("  Top-5:     %.4f (std %.4f)", np.mean(t5), np.std(t5))
    logger.info("  Top-10:    %.4f (std %.4f)", np.mean(t10), np.std(t10))
    logger.info("  Match:     %.4f (std %.4f)", np.mean(match), np.std(match))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drift_json", type=str,
                   default=os.path.join(SCRIPT_DIR, "drift_results.json"))
    p.add_argument("--output_dir", type=str, default=SCRIPT_DIR)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--length_multiplier", type=float, default=2.0,
                   help="Drop prompts where any condition's generation > "
                        "multiplier * shortest_gold_answer_len tokens.")
    p.add_argument("--min_gold_tokens", type=int, default=1,
                   help="Minimum gold length to apply the multiplier against "
                        "(prevents dropping 1-token answers because they generated 3).")
    p.add_argument("--min_cap", type=int, default=5,
                   help="Allow generations up to max(min_cap, multiplier*gold_len) tokens.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load drift results
    with open(args.drift_json) as f:
        d = json.load(f)
    logger.info("Loaded %d prompts from %s", d["n_prompts"], args.drift_json)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Load gold answer lengths (matched to original_idx used in the drift run)
    logger.info("Computing gold answer lengths via task sampler...")
    gold_lengths = load_gold_lengths(args.split, tokenizer)

    # Filter: keep prompt if BOTH pairs' generations are <= length cap in BOTH conditions
    kept_indices = []
    dropped_reasons = []

    for i, (c1, c2) in enumerate(zip(d["pair1_comparisons"],
                                     d["pair2_comparisons"])):
        task = c1["task"]
        oi = c1["original_idx"]
        gold_len = gold_lengths.get(task, {}).get(oi)
        if gold_len is None or gold_len < args.min_gold_tokens:
            dropped_reasons.append((i, task, oi, "no_gold"))
            continue
        cap = max(args.min_cap, int(args.length_multiplier * gold_len))

        # Both conditions in both pairs must be under the cap
        lens = [c1["n_steps_a"], c1["n_steps_b"],
                c2["n_steps_a"], c2["n_steps_b"]]
        if max(lens) > cap:
            dropped_reasons.append((i, task, oi,
                                    f"too_long ({max(lens)}>{cap}, gold={gold_len})"))
            continue
        kept_indices.append(i)

    logger.info("Kept %d/%d prompts (length <= max(%d, %.1fx gold))",
                len(kept_indices), d["n_prompts"],
                args.min_cap, args.length_multiplier)
    logger.info("Dropped %d prompts", len(dropped_reasons))

    # Show dropped reasons summary
    from collections import Counter
    reason_summary = Counter(r[3].split(" ")[0] for r in dropped_reasons)
    for reason, count in reason_summary.most_common():
        logger.info("  %s: %d", reason, count)

    # Per-task retention
    per_task_kept = Counter()
    per_task_total = Counter()
    for i in range(d["n_prompts"]):
        task = d["pair1_comparisons"][i]["task"]
        per_task_total[task] += 1
        if i in kept_indices:
            per_task_kept[task] += 1
    logger.info("Per-task retention:")
    for task in sorted(per_task_total.keys()):
        logger.info("  %s: %d/%d", task, per_task_kept[task], per_task_total[task])

    # Filter comparisons
    pair1_filt = [d["pair1_comparisons"][i] for i in kept_indices]
    pair2_filt = [d["pair2_comparisons"][i] for i in kept_indices]

    logger.info("\n=== FILTERED RESULTS ===")
    summarize(pair1_filt, "Pair 1: M1 full vs M1+NAMM")
    summarize(pair2_filt, "Pair 2: M4+NAMM vs A4 full")

    logger.info("\n=== FOR REFERENCE: unfiltered ===")
    summarize(d["pair1_comparisons"], "Pair 1 unfiltered")
    summarize(d["pair2_comparisons"], "Pair 2 unfiltered")

    # F1 data for scatter: prefer values embedded in drift_results.json,
    # fall back to local eval JSONs for conditions missing from there.
    eval_dir = os.path.join(REPO_ROOT, "analysis_out",
                            "latest_analysis_matched", "00_eval_results")
    eval_file_mapping = {
        "m1_full_f1": "m1_matched_full_cache.json",
        "m1_namm_f1": "m1_matched_under_namm_cs1024.json",
        "m4_namm_f1": "m4_lora_namm_cs1024.json",
        "a4_full_f1": "a4_m4_lora_no_namm.json",
    }
    f1_data = {}
    for k, fname in eval_file_mapping.items():
        if k in d and d[k]:
            f1_data[k] = d[k]
            continue
        path = os.path.join(eval_dir, fname)
        if not os.path.exists(path):
            logger.warning("Eval JSON missing for %s: %s", k, path)
            continue
        with open(path) as ef:
            ed = json.load(ef)
        per_prompt = ed.get("scores_per_split", {}).get(
            args.split, {}).get("per_prompt_f1", {})
        if not per_prompt:
            logger.warning("%s has no per_prompt_f1 for split=%s",
                           fname, args.split)
            continue
        flat: Dict[str, float] = {}
        for task_prompts in per_prompt.values():
            for idx, f1 in task_prompts.items():
                flat[str(idx)] = f1
        f1_data[k] = flat
        logger.info("Loaded %s from %s (n=%d)", k, fname, len(flat))

    # Save filtered JSON
    filt_path = os.path.join(args.output_dir, "drift_results_filtered.json")
    with open(filt_path, "w") as f:
        json.dump({
            "n_prompts_kept": len(kept_indices),
            "n_prompts_total": d["n_prompts"],
            "kept_indices": kept_indices,
            "dropped_reasons": [
                {"idx": r[0], "task": r[1], "original_idx": r[2], "reason": r[3]}
                for r in dropped_reasons
            ],
            "length_multiplier": args.length_multiplier,
            "min_cap": args.min_cap,
            "pair1_comparisons": pair1_filt,
            "pair2_comparisons": pair2_filt,
            "per_task_kept": dict(per_task_kept),
            "per_task_total": dict(per_task_total),
            **f1_data,
        }, f, default=str)
    logger.info("Saved %s", filt_path)

    # Plots (filtered)
    suffix = "_filtered"
    plot_per_step_metric(
        [c["js_divergence"] for c in pair1_filt],
        [c["js_divergence"] for c in pair2_filt],
        ylabel="JS Divergence",
        title=f"Per-Step JS Divergence (filtered, n={len(kept_indices)})",
        output_path=os.path.join(args.output_dir,
                                 f"js_divergence_by_pair{suffix}.png"),
    )
    plot_per_step_metric(
        [c["top5_overlap"] for c in pair1_filt],
        [c["top5_overlap"] for c in pair2_filt],
        ylabel="Top-5 Overlap",
        title=f"Per-Step Top-5 Overlap (filtered, n={len(kept_indices)})",
        output_path=os.path.join(args.output_dir,
                                 f"topk5_overlap_by_pair{suffix}.png"),
    )
    plot_per_step_metric(
        [c["top10_overlap"] for c in pair1_filt],
        [c["top10_overlap"] for c in pair2_filt],
        ylabel="Top-10 Overlap",
        title=f"Per-Step Top-10 Overlap (filtered, n={len(kept_indices)})",
        output_path=os.path.join(args.output_dir,
                                 f"topk10_overlap_by_pair{suffix}.png"),
    )
    plot_aggregate_bars(pair1_filt, pair2_filt,
                        os.path.join(args.output_dir,
                                     f"aggregate_summary{suffix}.png"))
    plot_drift_vs_f1(pair1_filt, pair2_filt, f1_data,
                     os.path.join(args.output_dir, f"drift_vs_f1{suffix}.png"))


if __name__ == "__main__":
    main()
