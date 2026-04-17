"""Post-hoc filter degenerate generations (>2x gold answer length) and
produce the JS divergence bar plot comparing M1 toggle vs M4 toggle.

Operates on drift_results.json — no GPU re-run needed.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Dict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize
from namm.run_utils import make_task_sampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_gold_lengths(split: str, tokenizer) -> Dict[str, Dict[int, int]]:
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

    ts = make_task_sampler(cfg=cfg)
    ts.filter_answers_by_token_count(
        tokenizer, cfg.get("max_answer_tokens", cfg.get("max_new_tokens", 64)))
    ts.apply_train_val_test_split(
        train_frac=cfg.get("train_frac", 0.7),
        val_frac=cfg.get("val_frac", 0.15),
        max_conditioning_length=cfg.get("split_max_conditioning_length")
            or cfg.get("max_conditioning_length", 6500),
        min_conditioning_length=cfg.get("split_min_conditioning_length")
            or cfg.get("min_conditioning_length"),
        tokenizer=tokenizer)

    out: Dict[str, Dict[int, int]] = {}
    for task, jsons in ts.lb_jsons_per_task.items():
        out[task] = {}
        for i, j in enumerate(jsons):
            answers = j.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            out[task][i] = min(
                len(tokenizer.encode(a, add_special_tokens=False))
                for a in answers)
    return out


def plot_js_bar(pair1, pair2, output_path, n):
    js_means = [
        float(np.mean([c["mean_js"] for c in pair1 if c["mean_js"] is not None])),
        float(np.mean([c["mean_js"] for c in pair2 if c["mean_js"] is not None])),
    ]
    labels = ["M1: full ↔ +NAMM", "M4: +NAMM ↔ full"]
    colors = ["#1f77b4", "#C44E52"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, js_means, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, js_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Mean JS Divergence", fontsize=10)
    ax.set_title(f"Distributional shift under NAMM toggle (n={n})",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def summarize(pair, name):
    js = [c["mean_js"] for c in pair if c["mean_js"] is not None]
    logger.info("%s (n=%d):  JS=%.4f (std %.4f)",
                name, len(pair), np.mean(js), np.std(js))


def plot_stratified_with_baselines(pairs, labels, colors, output_path, n):
    """Bar plot of mean JS (± SE) for an arbitrary list of pairs."""
    means, sems = [], []
    for pair in pairs:
        js = [c["mean_js"] for c in pair if c["mean_js"] is not None]
        means.append(float(np.mean(js)))
        sems.append(float(np.std(js) / np.sqrt(len(js))) if js else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=sems, color=colors, edgecolor="black",
                  linewidth=1.2, width=0.6, capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.008,
                f"{m:.3f}\u00b1{s:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean JS Divergence", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.grid(axis="y", alpha=0.3)
    top = max(m + s for m, s in zip(means, sems))
    ax.set_ylim(0, top * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s  (n=%d)", output_path, n)
    logger.info("Saved %s", pdf_path)


def run_stratified_with_baselines(output_dir):
    """Unfiltered val+test, 3 bars: FTS-FC↔FTS-EC, FTE-FC↔FTE-EC, Base-FC↔Base-EC."""
    test_drift = json.load(open(os.path.join(SCRIPT_DIR, "drift_results.json")))
    test_base = json.load(open(os.path.join(SCRIPT_DIR,
                                            "drift_baseline_results.json")))
    val_drift = json.load(open(os.path.join(SCRIPT_DIR, "val",
                                            "drift_results.json")))
    val_base = json.load(open(os.path.join(SCRIPT_DIR, "val",
                                           "drift_baseline_results.json")))

    pair1 = test_drift["pair1_comparisons"] + val_drift["pair1_comparisons"]
    pair2 = test_drift["pair2_comparisons"] + val_drift["pair2_comparisons"]
    pair4 = test_base["pair4_comparisons"] + val_base["pair4_comparisons"]
    n = len(pair1)

    logger.info("=== unfiltered val+test (n=%d) ===", n)
    summarize(pair1, "FTS-FC \u2194 FTS-EC")
    summarize(pair2, "FTE-FC \u2194 FTE-EC")
    summarize(pair4, "Base-FC \u2194 Base-EC")

    labels = ["FTS-FC \u2194 FTS-EC",
              "FTE-FC \u2194 FTE-EC",
              "Base-FC \u2194 Base-EC"]
    colors = ["#1f77b4", "#C44E52", "#6BA368"]
    out_path = os.path.join(
        output_dir, "js_divergence_bar_stratified_with_baselines_unfiltered.png")
    plot_stratified_with_baselines([pair1, pair2, pair4], labels, colors,
                                   out_path, n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["filter", "stratified_with_baselines"],
                   default="filter")
    p.add_argument("--drift_json", type=str,
                   default=os.path.join(SCRIPT_DIR, "drift_results.json"))
    p.add_argument("--output_dir", type=str, default=SCRIPT_DIR)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--length_multiplier", type=float, default=2.0)
    p.add_argument("--min_gold_tokens", type=int, default=1)
    p.add_argument("--min_cap", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "stratified_with_baselines":
        run_stratified_with_baselines(args.output_dir)
        return

    with open(args.drift_json) as f:
        d = json.load(f)
    logger.info("Loaded %d prompts from %s", d["n_prompts"], args.drift_json)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    logger.info("Computing gold answer lengths via task sampler...")
    gold_lengths = load_gold_lengths(args.split, tokenizer)

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

    pair1_filt = [d["pair1_comparisons"][i] for i in kept_indices]
    pair2_filt = [d["pair2_comparisons"][i] for i in kept_indices]

    logger.info("\n=== FILTERED ===")
    summarize(pair1_filt, "Pair 1: M1 full ↔ +NAMM")
    summarize(pair2_filt, "Pair 2: M4 +NAMM ↔ full")
    logger.info("\n=== unfiltered (reference) ===")
    summarize(d["pair1_comparisons"], "Pair 1 unfiltered")
    summarize(d["pair2_comparisons"], "Pair 2 unfiltered")

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
        }, f, default=str)
    logger.info("Saved %s", filt_path)

    plot_js_bar(pair1_filt, pair2_filt,
                os.path.join(args.output_dir, "js_divergence_bar.png"),
                n=len(kept_indices))


if __name__ == "__main__":
    main()
