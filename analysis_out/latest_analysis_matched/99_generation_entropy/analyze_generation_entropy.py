"""Section D: Generation-Time Entropy — Does Confidence Degrade During Decoding?

Token eviction removes context that the model was pretrained to condition on.
During autoregressive generation, each token is conditioned on the (evicted)
KV cache plus all previously generated tokens.  Small perturbations in the
output distribution can compound over steps: an uncertain token shifts the
distribution for the next, and so on.

This analysis tracks the entropy of the output logits at each generation step
for three conditions:
  1. M1-matched, full cache   (no eviction — baseline)
  2. M1-matched, under NAMM   (eviction, post-hoc — no adaptation)
  3. A4 (M4 weights), under NAMM  (eviction, trained under eviction)

If eviction compounds errors during generation:
  - Condition 2 should show rising entropy relative to condition 1
  - Condition 3 (adapted) should stabilise this, staying closer to 1

Requires GPU.  Generates one token at a time (greedy) to collect per-step
logits without KV recomputation.

Checkpoints are downloaded automatically from GCS bucket ``statistical-nlp``
under ``NAMM_checkpoints/pretrained/final_cs1024/``.

Uses qasper / qasper_e by default, filtering for prompts whose gold answers
are >= 6 tokens so that generation is long enough to observe entropy dynamics.

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python \\
        analysis_out/latest_analysis_matched/08_generation_entropy/analyze_generation_entropy.py \\
        --output_dir analysis_out/latest_analysis_matched/08_generation_entropy
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(REPO_ROOT))

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# GCS checkpoint download
# ---------------------------------------------------------------------------

GCS_BUCKET = "statistical-nlp"
GCS_PROJECT = "statistical-nlp"
GCS_PREFIX = "NAMM_checkpoints/pretrained/final_cs1024/"
GCS_FILES = {
    "m1_lora": "m1_lora_matched.pt",
    "m4_lora": "m4_lora_namm.pt",
    "namm":    "namm_cs1024_maskfix.pt",
}

LOCAL_CKPT_DIR = os.path.join(REPO_ROOT, "experiment_artifacts", "gcs", "final_cs1024")


def ensure_gcs_checkpoints() -> Dict[str, str]:
    """Download checkpoints from GCS if not already cached locally.

    Returns dict mapping role -> local absolute path.
    """
    from google.cloud import storage

    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    client = storage.Client(project=GCS_PROJECT)
    bucket = client.bucket(GCS_BUCKET)

    paths: Dict[str, str] = {}
    for role, fname in GCS_FILES.items():
        local_path = os.path.join(LOCAL_CKPT_DIR, fname)
        if os.path.exists(local_path):
            logger.info("  %s: cached at %s", role, local_path)
        else:
            blob = bucket.blob(GCS_PREFIX + fname)
            size_mb = blob.size / 1024 / 1024 if blob.size else 0
            logger.info("  %s: downloading %s (%.1f MB)...", role, fname, size_mb)
            blob.download_to_filename(local_path)
        paths[role] = local_path
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Entropy of the softmax distribution over the vocab (nats)."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -float((probs * log_probs).sum(dim=-1))


def top1_prob_from_logits(logits: torch.Tensor) -> float:
    """Probability of the top-1 token."""
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs.max(dim=-1).values)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_entropy(
    model: Any,
    policy: Any,
    input_ids: torch.Tensor,
    device: str,
    apply_memory_policy: bool,
    max_gen_tokens: int = 128,
) -> Dict[str, List[float]]:
    """Greedy-decode one token at a time, recording per-step entropy.

    Returns dict with:
      - entropy: list of per-step output entropy (nats)
      - top1_prob: list of per-step top-1 probability
      - nll: list of per-step NLL of the chosen (greedy) token
      - token_ids: list of per-step greedy token IDs
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

    entropies: List[float] = []
    top1_probs: List[float] = []
    nlls: List[float] = []
    token_ids: List[int] = []

    for step in range(max_gen_tokens):
        ent = entropy_from_logits(next_logits)
        t1p = top1_prob_from_logits(next_logits)
        entropies.append(ent)
        top1_probs.append(t1p)

        # Greedy pick
        next_token_id = next_logits.argmax(dim=-1, keepdim=True)
        log_probs = torch.log_softmax(next_logits.float(), dim=-1)
        nll = -float(log_probs[next_token_id])
        nlls.append(nll)
        token_ids.append(int(next_token_id.item()))

        # Check for EOS
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", 2)
        if isinstance(eos_id, list):
            eos_ids = set(eos_id)
        else:
            eos_ids = {eos_id}
        if next_token_id.item() in eos_ids:
            break

        # Decode step — no eviction during generation
        next_input = next_token_id.unsqueeze(0)  # (1, 1)
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

    return {"entropy": entropies, "top1_prob": top1_probs, "nll": nlls,
            "token_ids": token_ids}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_spaghetti(results: Dict[str, Any], output_dir: str,
                   prompts: Optional[List[dict]] = None,
                   highlight_idx: int = 0) -> None:
    """Per-prompt spaghetti plot with entropy (top row) and NLL (bottom row).

    One column per condition.  A single prompt (highlight_idx) is drawn in
    bold, and its truncated prompt + generated text are shown in a text
    panel below the plots.
    """
    conds = [c for c in ["M1_full", "M1_namm", "A4_namm"] if c in results]
    n_conds = len(conds)

    colors = {"M1_full": "#2ca02c", "M1_namm": "#ff7f0e", "A4_namm": "#1f77b4"}

    # Compute mean generation length per condition
    mean_lens: Dict[str, float] = {}
    for cond in conds:
        runs = results[cond].get("raw_runs", [])
        mean_lens[cond] = float(np.mean([len(r["entropy"]) for r in runs])) if runs else 0

    titles = {
        "M1_full": f"M1-matched, full cache\nmean steps={mean_lens.get('M1_full', 0):.0f}",
        "M1_namm": f"M1-matched, under NAMM\n(post-hoc)  mean steps={mean_lens.get('M1_namm', 0):.0f}",
        "A4_namm": f"A4 (adapted), under NAMM\nmean steps={mean_lens.get('A4_namm', 0):.0f}",
    }

    fig, axes = plt.subplots(2, n_conds, figsize=(6 * n_conds, 10),
                             sharex=True)
    if n_conds == 1:
        axes = axes.reshape(2, 1)

    for col, cond in enumerate(conds):
        runs = results[cond].get("raw_runs", [])
        ax_ent = axes[0, col]
        ax_nll = axes[1, col]

        for i, run in enumerate(runs):
            ent = run["entropy"]
            nll = run["nll"]
            steps = range(len(ent))
            is_highlight = (i == highlight_idx)
            alpha = 0.12 if not is_highlight else 1.0
            lw = 0.6 if not is_highlight else 2.5
            zorder = 1 if not is_highlight else 10
            color = colors[cond] if not is_highlight else "black"

            ax_ent.plot(steps, ent, color=color, alpha=alpha,
                        linewidth=lw, zorder=zorder)
            ax_nll.plot(steps, nll, color=color, alpha=alpha,
                        linewidth=lw, zorder=zorder)

        ax_ent.set_title(titles[cond], fontsize=11, fontweight="bold")
        ax_ent.grid(True, alpha=0.3)
        ax_nll.set_xlabel("Decode Step", fontsize=10)
        ax_nll.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Output Entropy (nats)", fontsize=11)
    axes[1, 0].set_ylabel("NLL of Greedy Token (nats)", fontsize=11)

    fig.suptitle("Per-Prompt Generation Entropy and NLL",
                 fontsize=13, fontweight="bold")

    # Text panel: show the highlighted prompt and each condition's generation
    text_lines = []
    if prompts is not None and highlight_idx < len(prompts):
        prompt_text = prompts[highlight_idx]["text"]
        text_lines.append(f"Prompt (idx={highlight_idx}, "
                          f"task={prompts[highlight_idx]['task']}):")
        # Wrap prompt text manually at ~120 chars
        for i in range(0, len(prompt_text), 120):
            text_lines.append(f"  {prompt_text[i:i+120]}")
        text_lines.append("")
        answers = prompts[highlight_idx].get("answers", [])
        text_lines.append(f"Ground-truth answers: {answers}")
        text_lines.append("")
        for cond in conds:
            runs = results[cond].get("raw_runs", [])
            if highlight_idx < len(runs) and "text" in runs[highlight_idx]:
                gen_text = runs[highlight_idx]["text"]
                n_steps = len(runs[highlight_idx]["entropy"])
                text_lines.append(f"{cond} ({n_steps} steps): {gen_text}")

    if text_lines:
        text_block = "\n".join(text_lines)
        fig.text(0.02, -0.02, text_block, fontsize=8.5, fontfamily="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat",
                           alpha=0.5))

    fig.tight_layout(rect=[0, 0.22 if text_lines else 0, 1, 0.96])
    path = os.path.join(output_dir, "entropy_spaghetti.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_lora_checkpoint", type=str, default=None,
                   help="Override M1 LoRA path (default: download from GCS)")
    p.add_argument("--m4_lora_checkpoint", type=str, default=None,
                   help="Override M4 LoRA path (default: download from GCS)")
    p.add_argument("--namm_checkpoint", type=str, default=None,
                   help="Override NAMM policy path (default: download from GCS)")
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["lb/qasper", "lb/qasper_e"],
                   help="LongBench tasks to use")
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--max_gen_tokens", type=int, default=128)
    p.add_argument("--min_answer_tokens", type=int, default=6,
                   help="Only use prompts whose gold answer has >= this many tokens")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def aggregate_per_step(all_runs: List[Dict]) -> Dict[str, List[float]]:
    """Aggregate per-step metrics across prompts, padding shorter sequences."""
    max_len = max(len(r["entropy"]) for r in all_runs)

    ent_matrix = np.full((len(all_runs), max_len), np.nan)
    t1p_matrix = np.full((len(all_runs), max_len), np.nan)
    nll_matrix = np.full((len(all_runs), max_len), np.nan)

    for i, r in enumerate(all_runs):
        n = len(r["entropy"])
        ent_matrix[i, :n] = r["entropy"]
        t1p_matrix[i, :n] = r["top1_prob"]
        nll_matrix[i, :n] = r["nll"]

    return {
        "mean_entropy": np.nanmean(ent_matrix, axis=0).tolist(),
        "std_entropy": np.nanstd(ent_matrix, axis=0).tolist(),
        "mean_top1_prob": np.nanmean(t1p_matrix, axis=0).tolist(),
        "std_top1_prob": np.nanstd(t1p_matrix, axis=0).tolist(),
        "mean_nll": np.nanmean(nll_matrix, axis=0).tolist(),
        "std_nll": np.nanstd(nll_matrix, axis=0).tolist(),
        "n_active": np.sum(~np.isnan(ent_matrix), axis=0).tolist(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoints — download from GCS if not overridden
    logger.info("Resolving checkpoints...")
    gcs_paths = ensure_gcs_checkpoints()
    m1_ckpt = args.m1_lora_checkpoint or gcs_paths["m1_lora"]
    m4_ckpt = args.m4_lora_checkpoint or gcs_paths["m4_lora"]
    namm_ckpt = args.namm_checkpoint or gcs_paths["namm"]
    logger.info("  M1 LoRA:  %s", m1_ckpt)
    logger.info("  M4 LoRA:  %s", m4_ckpt)
    logger.info("  NAMM:     %s", namm_ckpt)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    # Override tasks to use the requested set (e.g. summarization)
    tasks_str = str(args.tasks)  # e.g. "['lb/gov_report', 'lb/multi_news']"
    overrides = [
        f"run@_global_={args.run_config}", "wandb_log=false",
        "wandb_project=Experiments",
        f"cache_size={args.cache_size}", f"max_memory_length={args.cache_size}",
        f"tasks={tasks_str}",
    ]
    with initialize(version_base=None, config_path="../../../config",
                    job_name="gen_entropy"):
        cfg = compose(config_name="config", overrides=overrides)

    logger.info("Building model...")
    with torch.no_grad():
        (policy, model, evaluator, _, _) = make_eval_model(cfg=cfg)
    model.to(device)

    # Load NAMM
    ckpt = torch.load(namm_ckpt, map_location="cpu", weights_only=False)
    evo = ckpt["evolution_state"]
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

    # Collect 2 prompts per task from val+test, filtering for long gold answers
    per_task = 2
    prompts: List[dict] = []
    for task in sorted(task_sampler.lb_prompts_per_task.keys()):
        task_prompts_all: List[dict] = []
        for split_name in ["val", "test"]:
            split_idxs = task_sampler.get_split_indices(split_name)
            if task in split_idxs:
                for oi in split_idxs[task]:
                    json_obj = task_sampler.lb_jsons_per_task[task][int(oi)]
                    answers = json_obj.get("answers", [])
                    # Filter: at least one gold answer must have enough tokens
                    max_ans_toks = max(
                        (len(tokenizer(a, add_special_tokens=False)["input_ids"])
                         for a in answers), default=0)
                    if max_ans_toks < args.min_answer_tokens:
                        continue
                    task_prompts_all.append({
                        "text": task_sampler.lb_prompts_per_task[task][int(oi)],
                        "task": task,
                        "split": split_name,
                        "answers": answers,
                        "max_answer_tokens": max_ans_toks,
                    })
        # Sort by longest gold answer first
        task_prompts_all.sort(key=lambda x: x["max_answer_tokens"], reverse=True)
        prompts.extend(task_prompts_all[:per_task])
    logger.info("  %d prompts (%d per task, val+test, min_answer_tokens=%d)",
                len(prompts), per_task, args.min_answer_tokens)

    bos = getattr(tokenizer, "bos_token", None) or ""

    # ── Three conditions ──────────────────────────────────────────────
    conditions = [
        ("M1_full",  m1_ckpt, False, "M1-matched, full cache"),
        ("M1_namm",  m1_ckpt, True,  "M1-matched, under NAMM"),
        ("A4_namm",  m4_ckpt, True,  "A4 (M4 weights), under NAMM"),
    ]

    all_results: Dict[str, Any] = {}

    for cond_label, ckpt_path, apply_mp, desc in conditions:
        logger.info("=" * 60)
        logger.info("%s (%s)", desc, cond_label)
        logger.info("=" * 60)
        swap_lora(model, ckpt_path, device)

        runs: List[Dict] = []
        for p_idx, p_info in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": p_info["text"]}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            run_data = generate_with_entropy(
                model, policy, input_ids, device,
                apply_memory_policy=apply_mp,
                max_gen_tokens=args.max_gen_tokens,
            )
            run_data["task"] = p_info["task"]
            run_data["split"] = p_info["split"]
            gen_text = tokenizer.decode(run_data["token_ids"],
                                        skip_special_tokens=True)
            run_data["text"] = gen_text
            runs.append(run_data)

            logger.info("  %d/%d done (gen_steps=%d, task=%s)",
                        p_idx + 1, len(prompts),
                        len(run_data["entropy"]), p_info["task"])
            if cond_label == conditions[0][0]:
                logger.info("-" * 60)
                logger.info("PROMPT [%s, %s, idx=%d]:\n%s",
                            p_info["task"], p_info["split"], p_idx,
                            p_info["text"])
                logger.info("GROUND-TRUTH ANSWERS: %s", p_info["answers"])
                logger.info("-" * 60)
            logger.info("GENERATED (%s): %s", cond_label, gen_text)

        agg = aggregate_per_step(runs)
        all_results[cond_label] = agg
        all_results[cond_label]["raw_runs"] = runs
        logger.info("  %s: mean entropy step 0=%.3f, step -1=%.3f",
                    cond_label, agg["mean_entropy"][0], agg["mean_entropy"][-1])

    # ── Save results ──────────────────────────────────────────────────
    # Save aggregated (without raw_runs which are large)
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {sk: sv for sk, sv in v.items() if sk != "raw_runs"}
    json_path = os.path.join(args.output_dir, "generation_entropy.json")
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("Saved %s", json_path)

    # Save raw per-prompt traces with decoded text and source prompts
    raw_path = os.path.join(args.output_dir, "generation_entropy_raw.json")
    raw_save = {"prompts": [{"text": p["text"], "task": p["task"],
                             "split": p["split"],
                             "answers": p["answers"]} for p in prompts]}
    for k, v in all_results.items():
        raw_save[k] = v.get("raw_runs", [])
    with open(raw_path, "w") as f:
        json.dump(raw_save, f)
    logger.info("Saved %s", raw_path)

    # ── Plots ─────────────────────────────────────────────────────────
    # Pick the median-entropy prompt at step 0 for M1_full as the highlight
    if "M1_full" in all_results:
        raw = all_results["M1_full"].get("raw_runs", [])
        step0_ents = [r["entropy"][0] for r in raw]
        median_val = float(np.median(step0_ents))
        highlight_idx = int(np.argmin([abs(e - median_val) for e in step0_ents]))
    else:
        highlight_idx = 0
    plot_spaghetti(all_results, args.output_dir, prompts=prompts,
                   highlight_idx=highlight_idx)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for cond in ["M1_full", "M1_namm", "A4_namm"]:
        if cond not in all_results:
            continue
        ent = all_results[cond]["mean_entropy"]
        logger.info("  %s: entropy step 0=%.3f  step 10=%.3f  last=%.3f  (n_steps=%d)",
                    cond, ent[0],
                    ent[min(10, len(ent) - 1)],
                    ent[-1], len(ent))

    if "M1_full" in all_results and "M1_namm" in all_results:
        b = np.array(all_results["M1_full"]["mean_entropy"])
        e = np.array(all_results["M1_namm"]["mean_entropy"])
        min_len = min(len(b), len(e))
        delta = e[:min_len] - b[:min_len]
        logger.info("  M1_namm - M1_full: mean delta=%.4f, max delta=%.4f",
                    delta.mean(), delta.max())

    if "M1_full" in all_results and "A4_namm" in all_results:
        b = np.array(all_results["M1_full"]["mean_entropy"])
        a = np.array(all_results["A4_namm"]["mean_entropy"])
        min_len = min(len(b), len(a))
        delta = a[:min_len] - b[:min_len]
        logger.info("  A4_namm - M1_full: mean delta=%.4f, max delta=%.4f",
                    delta.mean(), delta.max())

    logger.info("\nDone. Plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
