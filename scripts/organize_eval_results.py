"""Organize the test/extended_test eval results for the 5-task QA suite into a
clean directory structure mirroring the main-branch experiment specification.

Reads completed `ext_*` runs from `eval_results/<model>/ext_<ts>/results.json`
and copies them into `results/main_table_5t/<condition>/[<cs>/]/` with:
  - results.json           (raw)
  - eviction_traces.npz    (when present, NAMM/recency only)
  - command.sh             (exact command used to produce the run)
  - README.md              (one-paragraph description)

Also writes:
  - results/main_table_5t/README.md       (full layout, status, summary table)
  - results/main_table_5t/all_results.json  (aggregated B0/B1/M1/M2/M4/A4 → cs → task → F1)

Idempotent: re-run after more jobs finish to refresh.

Usage:
    python scripts/organize_eval_results.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

PROJ = Path("/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo")
SRC  = PROJ / "eval_results"
DST  = PROJ / "results" / "main_table_5t"

# (source dir under eval_results,  dest path under main_table_5t,  pretty name,  command snippet)
JOBS = [
    {
        "src": "plain_baseline_5t",
        "dst": "B0",
        "name": "B0 — plain Llama, full KV cache",
        "desc": "Base Llama-3.2-1B-Instruct with no eviction, no fine-tuning. "
                "Performance ceiling for the 5-task QA subset.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --plain --filter_by_length 8192 --batch_size 16 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/plain_baseline_5t"
        ),
    },
    {
        "src": "recency_cs1024_5t",
        "dst": "B1/cs1024",
        "name": "B1 — recency eviction, cache_size=1024",
        "desc": "Base model with a fixed recency policy (keep most recent, "
                "evict oldest). No learned policy, no training.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/recency_cs1024_5t"
        ),
    },
    {
        "src": "recency_cs2048_5t",
        "dst": "B1/cs2048",
        "name": "B1 — recency eviction, cache_size=2048",
        "desc": "Same as B1 cs1024 but with double the cache budget.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/recency_cs2048_5t"
        ),
    },
    {
        "src": "lora_m1_5t",
        "dst": "M1",
        "name": "M1 — LoRA only (no NAMM)",
        "desc": "LoRA SFT on the 5-task QA subset, full KV cache during training "
                "and eval (cache_size=8192). No eviction.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m1_5t"
        ),
    },
    {
        "src": "namm_cs1024_5t",
        "dst": "M2/cs1024",
        "name": "M2 — standalone NAMM, cache_size=1024",
        "desc": "Trained NAMM eviction policy on top of the frozen base model. "
                "No LoRA, no fine-tuning of the LM weights.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/"
            "rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/namm_cs1024_5t"
        ),
    },
    {
        "src": "namm_cs2048_5t",
        "dst": "M2/cs2048",
        "name": "M2 — standalone NAMM, cache_size=2048 (friend's checkpoint)",
        "desc": "Same as M2 cs1024 but using a NAMM checkpoint trained at "
                "cache_size=2048 by a collaborator.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/namm_cs2048_5t"
        ),
    },
    {
        "src": "lora_m4_cs1024_5t",
        "dst": "M4/cs1024",
        "name": "M4 — LoRA on frozen NAMM, cache_size=1024",
        "desc": "LoRA fine-tuned on top of a frozen NAMM (cs=1024). The LoRA "
                "and NAMM are evaluated together.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/"
            "rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \\\n"
            "    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m4_cs1024_5t"
        ),
    },
    {
        "src": "lora_m4_cs2048_5t",
        "dst": "M4/cs2048",
        "name": "M4 — LoRA on frozen NAMM, cache_size=2048",
        "desc": "Same as M4 cs1024 but with the cs=2048 NAMM and a LoRA "
                "trained against it.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \\\n"
            "    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m4_cs2048_5t"
        ),
    },
    {
        "src": "trunc_plain_1024_5t",
        "dst": "Trunc/plain_1024",
        "name": "Plain Llama, input truncated to last 1024 tokens",
        "desc": "Naive tail-only baseline: every prompt is decoded from its "
                "last 1024 token ids before the model sees it. No KV cache "
                "eviction, no policy hooks — the model simply runs on a "
                "shorter input. The cleanest 'StreamingLLM rolling-window' "
                "comparison: how much can we recover with no learned policy?",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --truncate_input_to 1024 \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label trunc \\\n"
            "    --output_dir eval_results/trunc_plain_1024_5t"
        ),
    },
    {
        "src": "trunc_plain_2048_5t",
        "dst": "Trunc/plain_2048",
        "name": "Plain Llama, input truncated to last 2048 tokens",
        "desc": "Same as Trunc/plain_1024 but with double the input budget.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --truncate_input_to 2048 \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label trunc \\\n"
            "    --output_dir eval_results/trunc_plain_2048_5t"
        ),
    },
    {
        "src": "trunc_lora_m1_1024_5t",
        "dst": "Trunc/lora_m1_1024",
        "name": "M1 LoRA, input truncated to last 1024 tokens",
        "desc": "M1 LoRA evaluated on its last-1024-token input. Pairs with "
                "Trunc/plain_1024 to isolate how much the LoRA recovers under "
                "naive truncation, and pairs with M2/M4 to see how learned "
                "eviction compares to the simplest possible baseline at the "
                "same budget.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --truncate_input_to 1024 \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label trunc \\\n"
            "    --output_dir eval_results/trunc_lora_m1_1024_5t"
        ),
    },
    {
        "src": "trunc_lora_m1_2048_5t",
        "dst": "Trunc/lora_m1_2048",
        "name": "M1 LoRA, input truncated to last 2048 tokens",
        "desc": "Same as Trunc/lora_m1_1024 but with double the input budget.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --truncate_input_to 2048 \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label trunc \\\n"
            "    --output_dir eval_results/trunc_lora_m1_2048_5t"
        ),
    },
    {
        "src": "lora_m1_recency_cs1024_5t",
        "dst": "M1_recency/cs1024",
        "name": "M1 LoRA + recency eviction, cache_size=1024",
        "desc": "M1 LoRA (trained with full cache) evaluated under a fixed "
                "recency policy at cs=1024. Tests how much of M1's gain "
                "survives aggressive cache compression with a naive eviction "
                "heuristic (no learned policy).",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m1_recency_cs1024_5t"
        ),
    },
    {
        "src": "lora_m1_recency_cs2048_5t",
        "dst": "M1_recency/cs2048",
        "name": "M1 LoRA + recency eviction, cache_size=2048",
        "desc": "Same as M1_recency cs1024 but with double the cache budget.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m1_recency_cs2048_5t"
        ),
    },
    {
        "src": "lora_m1_namm_cs1024_5t",
        "dst": "M1_under_NAMM/cs1024",
        "name": "M1 LoRA (no NAMM training) + NAMM eviction cs1024",
        "desc": "M1 LoRA evaluated under NAMM eviction it was NOT trained with. "
                "Measures the distribution shift penalty: the LoRA adapted to "
                "full-context attention patterns but now faces a post-eviction "
                "cache. Compare with M4/cs1024 (LoRA trained WITH NAMM) to "
                "quantify the value of training under eviction.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/"
            "rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m1_namm_cs1024_5t"
        ),
    },
    {
        "src": "lora_m1_namm_cs2048_5t",
        "dst": "M1_under_NAMM/cs2048",
        "name": "M1 LoRA (no NAMM training) + NAMM eviction cs2048",
        "desc": "Same as M1_under_NAMM/cs1024 but at cache_size=2048.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \\\n"
            "    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext \\\n"
            "    --output_dir eval_results/lora_m1_namm_cs2048_5t"
        ),
    },
    {
        "src": "lora_m4_cs1024_5t_ablation",
        "dst": "A4/cs1024_no_namm",
        "name": "A4 — M4 (cs1024) LoRA, NAMM disabled (full cache)",
        "desc": "Ablation: take the M4 cs=1024 LoRA but evaluate it WITHOUT "
                "its NAMM, with full cache (cs=8192). Measures how much the "
                "LoRA alone contributes vs. LoRA+NAMM together.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext_no_namm \\\n"
            "    --output_dir eval_results/lora_m4_cs1024_5t_ablation"
        ),
    },
    {
        "src": "lora_m4_cs2048_5t_ablation",
        "dst": "A4/cs2048_no_namm",
        "name": "A4 — M4 (cs2048) LoRA, NAMM disabled (full cache)",
        "desc": "Same ablation as A4 cs1024 but for the cs=2048 LoRA.",
        "cmd": (
            "env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \\\n"
            "    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \\\n"
            "    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \\\n"
            "    --splits test extended_test --run_label ext_no_namm \\\n"
            "    --output_dir eval_results/lora_m4_cs2048_5t_ablation"
        ),
    },
]

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]


def latest_ext_run(src_dir: Path) -> Optional[Path]:
    """Return the most recent eval-run subdir containing results.json, or None.

    Accepts any subdir whose name starts with one of the run-label prefixes
    used by eval_namm_splits.py: ext, trunc, classic, smoke. The latest by
    mtime wins, so re-running with a new label automatically supersedes the
    older run.
    """
    if not src_dir.exists():
        return None
    accepted_prefixes = ("ext", "trunc", "classic", "smoke")
    candidates = [
        d for d in src_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(accepted_prefixes)
        and (d / "results.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _compute_micro(scores: dict) -> tuple:
    """Compute (micro_mean_f1, n_prompts_total) from per_prompt_f1 if present.

    Falls back to (None, None) if per_prompt_f1 is not in `scores`.
    """
    ppf1 = scores.get("per_prompt_f1")
    if not ppf1:
        return None, None
    all_vals = []
    for task_dict in ppf1.values():
        all_vals.extend(task_dict.values())
    if not all_vals:
        return None, None
    return (sum(all_vals) / len(all_vals)) * 100.0, len(all_vals)


def extract_scores(results_json: dict) -> dict:
    """Pull per-task F1 for test and extended_test out of either schema.

    Returns:
        {"test":          {task: f1, ..., "mean": macro_f1, "micro": micro_f1, "n": int},
         "extended_test": {...}}

    `mean` is the unweighted mean of the 5 per-task F1s (macro). `micro` is
    the prompt-count-weighted mean computed from per_prompt_f1, comparable
    to the val_lb_avg_f1 metric reported during LoRA training.
    """
    out = {}

    def _enrich(split_scores: dict, fallback_macro: float) -> dict:
        d = {t: split_scores.get(t) for t in TASKS if t in split_scores}
        if not d:
            return None
        d["mean"] = split_scores.get("mean_f1", fallback_macro)
        # Prefer the value already in the JSON (may have been written by the
        # eval script or post-hoc patcher); recompute as a safety net.
        micro = split_scores.get("micro_mean_f1")
        n_total = split_scores.get("n_prompts_total")
        if micro is None or n_total is None:
            micro, n_total = _compute_micro(split_scores)
        if micro is not None:
            d["micro"] = micro
            d["n"] = n_total
        return d

    # Schema 1: plain baseline → results.<split>.<task>
    if "results" in results_json and isinstance(results_json["results"], dict):
        for split, scores in results_json["results"].items():
            if not isinstance(scores, dict):
                continue
            macro_fallback = (
                sum(scores[t] for t in TASKS if t in scores)
                / max(1, sum(1 for t in TASKS if t in scores))
            )
            d = _enrich(scores, macro_fallback)
            if d:
                out[split] = d
    # Schema 2: namm/lora → scores_per_split.<split>.<task>
    if "scores_per_split" in results_json and isinstance(results_json["scores_per_split"], dict):
        for split, scores in results_json["scores_per_split"].items():
            if not isinstance(scores, dict):
                continue
            macro_fallback = (
                sum(scores[t] for t in TASKS if t in scores)
                / max(1, sum(1 for t in TASKS if t in scores))
            )
            d = _enrich(scores, macro_fallback)
            if d:
                out[split] = d
    return out


def write_leaf(job: dict, run_dir: Optional[Path]) -> dict:
    """Copy artifacts and write README/command for one leaf. Returns scores dict."""
    leaf = DST / job["dst"]
    leaf.mkdir(parents=True, exist_ok=True)

    # command.sh always present
    (leaf / "command.sh").write_text("#!/bin/bash\nset -e\ncd " + str(PROJ) + "\n\n" + job["cmd"] + "\n")

    if run_dir is None:
        # Pending: write a placeholder README
        (leaf / "README.md").write_text(
            f"# {job['name']}\n\n"
            f"{job['desc']}\n\n"
            f"**Status:** ⏳ pending — no completed `ext_*` run found in "
            f"`eval_results/{job['src']}/`.\n\n"
            f"## Command\n\n```bash\n{job['cmd']}\n```\n"
        )
        return {}

    # Copy artifacts as REGULAR files (not symlinks) so the leaf is
    # self-contained and survives git checkouts on a fresh clone.
    src_results = run_dir / "results.json"
    shutil.copy2(src_results, leaf / "results.json")
    traces = run_dir / "eviction_traces.npz"
    if traces.exists():
        dst = leaf / "eviction_traces.npz"
        # Remove any prior symlink/file before copying so shutil.copy2 doesn't
        # follow a stale link to write into the source.
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(traces, dst)
    gens = run_dir / "generations.json"
    if gens.exists():
        dst = leaf / "generations.json"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(gens, dst)

    # Extract scores for the README
    with open(src_results) as f:
        rj = json.load(f)
    scores = extract_scores(rj)

    # Per-leaf README
    md = [f"# {job['name']}", "", job["desc"], ""]
    md.append(f"**Source run:** `eval_results/{job['src']}/{run_dir.name}/`")
    md.append("")
    md.append("## Results")
    md.append("")
    if scores:
        header = ["Task"] + list(scores.keys())
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for t in TASKS:
            row = [t]
            for split in scores:
                v = scores[split].get(t)
                row.append(f"{v:.2f}" if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        # mean row
        mean_row = ["**mean F1**"]
        for split in scores:
            v = scores[split].get("mean")
            mean_row.append(f"**{v:.2f}**" if v is not None else "—")
        md.append("| " + " | ".join(mean_row) + " |")
    else:
        md.append("_(no parseable scores)_")
    md.append("")
    md.append("## Command")
    md.append("")
    md.append("```bash")
    md.append(job["cmd"])
    md.append("```")
    md.append("")
    (leaf / "README.md").write_text("\n".join(md) + "\n")

    return scores


def main():
    DST.mkdir(parents=True, exist_ok=True)

    aggregate = {}   # {dst_path: {split: {task: f1}}}
    statuses  = {}   # {dst_path: "done" | "pending"}

    for job in JOBS:
        run_dir = latest_ext_run(SRC / job["src"])
        scores = write_leaf(job, run_dir)
        statuses[job["dst"]] = "done" if run_dir else "pending"
        if scores:
            aggregate[job["dst"]] = scores

    # Top-level all_results.json
    with open(DST / "all_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Top-level README
    md = [
        "# Main results table — 5-task QA eval (test + extended_test)",
        "",
        "Eval runs against the 5-task LongBench QA subset used throughout "
        "this fork (`qasper`, `2wikimqa`, `qasper_e`, `hotpotqa_e`, `2wikimqa_e`).",
        "",
        "Naming follows the milestones in `experiment_specification.md` "
        "(B0 baseline, B1 recency, M1 LoRA, M2 standalone NAMM, "
        "M4 LoRA + frozen NAMM, A4 ablation removing NAMM from M4).",
        "",
        "All runs use:",
        "",
        "- `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`",
        "- `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`",
        "- `extended_test`: filtered to length window `(6500, 8192]` (held-out long examples)",
        "- Greedy decoding (`do_sample=False`)",
        "",
        "Set sizes (post-filter): **70 test** + **224 extended_test** examples across 5 tasks.",
        "",
        "## Layout",
        "",
        "```",
        "results/main_table_5t/",
        "├── README.md                  ← this file",
        "├── all_results.json           ← aggregated F1 across all conditions",
        "│",
        "├── B0/                        ← plain Llama, full cache",
        "├── B1/cs{1024,2048}/          ← recency eviction baselines",
        "├── M1/                        ← LoRA only (no NAMM)",
        "├── M2/cs{1024,2048}/          ← standalone NAMM eviction",
        "├── M4/cs{1024,2048}/          ← LoRA on frozen NAMM",
        "└── A4/cs{1024,2048}_no_namm/  ← M4 LoRA, NAMM disabled (cs=8192)",
        "```",
        "",
        "Each leaf contains:",
        "- `results.json` — raw eval output (full per-prompt F1 + per-task aggregate)",
        "- `eviction_traces.npz` — symlink to per-prompt eviction signals (NAMM/recency only)",
        "- `command.sh` — exact command used to produce the run",
        "- `README.md` — condition description + parsed scores table",
        "",
        "## Status",
        "",
        "| Condition | Status |",
        "|-----------|--------|",
    ]
    for job in JOBS:
        s = statuses[job["dst"]]
        icon = "✅" if s == "done" else "⏳"
        md.append(f"| `{job['dst']}` — {job['name']} | {icon} {s} |")
    md.append("")
    md.append("Re-run `python scripts/organize_eval_results.py` to refresh after "
              "more jobs finish — the script is idempotent.")
    md.append("")

    # Summary table: condition → test/extended micro AND macro F1.
    # Micro = prompt-count-weighted (matches LoRA training-time val_lb_avg_f1).
    # Macro = unweighted mean over the 5 tasks (each task counts as 1/5).
    md.append("## Summary — micro and macro mean F1 across 5 tasks")
    md.append("")
    md.append("**Micro** = prompt-count-weighted mean (matches LoRA training "
              "`val_lb_avg_f1`). **Macro** = unweighted mean over the 5 tasks "
              "(each task = 1/5). Plots in `plots/` use the micro average as "
              "the headline metric.")
    md.append("")
    md.append("| Condition | test (micro) | test (macro) | extended_test (micro) | extended_test (macro) |")
    md.append("|-----------|-------------:|-------------:|----------------------:|----------------------:|")
    for job in JOBS:
        scores = aggregate.get(job["dst"], {})
        t = scores.get("test", {})
        e = scores.get("extended_test", {})
        def fmt(d, k):
            v = d.get(k)
            return f"{v:.2f}" if v is not None else "—"
        md.append(
            f"| `{job['dst']}` | {fmt(t, 'micro')} | {fmt(t, 'mean')} "
            f"| {fmt(e, 'micro')} | {fmt(e, 'mean')} |"
        )
    md.append("")

    # Per-task breakdown — both splits.
    # The 5 per-task columns are macro within their task (mean over the
    # task's prompts in the split). The "micro" column is the prompt-count-
    # weighted mean across all 5 tasks (= matches LoRA training val score).
    header = (["Condition"] + [t.replace("lb/", "") for t in TASKS]
              + ["macro", "micro"])

    def per_task_section(title: str, split: str):
        md.append(title)
        md.append("")
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for job in JOBS:
            scores = aggregate.get(job["dst"], {}).get(split, {})
            row = [f"`{job['dst']}`"]
            for t in TASKS:
                v = scores.get(t)
                row.append(f"{v:.2f}" if v is not None else "—")
            macro = scores.get("mean")
            micro = scores.get("micro")
            row.append(f"{macro:.2f}" if macro is not None else "—")
            row.append(f"**{micro:.2f}**" if micro is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    per_task_section("## Per-task F1 (test split)", "test")
    per_task_section("## Per-task F1 (extended_test split)", "extended_test")

    (DST / "README.md").write_text("\n".join(md) + "\n")

    print(f"Wrote {DST}")
    print(f"  done: {sum(1 for s in statuses.values() if s == 'done')}")
    print(f"  pending: {sum(1 for s in statuses.values() if s == 'pending')}")


if __name__ == "__main__":
    main()
