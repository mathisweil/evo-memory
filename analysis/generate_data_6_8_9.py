#!/usr/bin/env python3
"""Run NAMM-dependent analyses (Reports 6, 8, 9) using maskfix checkpoints.

These analyses require the full NAMM infrastructure (WrappedLlama + memory
policy) rather than merged LoRA weights, because they need eviction-aware
forward passes, NAMM token scores, and policy state.

The NAMM infrastructure works correctly after the attention mask fix + device
placement fix: make_eval_model -> mm.to(device) -> set_memory_params ->
set_params_batch_idxs -> forward pass with apply_memory_policy=True succeeds.

Produces:
  Report 6: Token importance alignment (NAMM scores vs attention weights)
            for M1 and M3-maskfix. Saves to analysis/report_6/.
  Report 8: Probing for residual knowledge of evicted content.
            Per-layer logistic regression probes on mean-pooled hidden states
            for M1 and M3-maskfix. Saves to analysis/report_8/.
  Report 9: Gradient flow and loss attribution under eviction.
            Forward+backward with and without NAMM eviction for M3-maskfix.
            Saves to analysis/report_9/.

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/generate_data_6_8_9.py

    # Regenerate plots from saved data (no GPU needed):
    PYTHONPATH=. .venv/bin/python analysis/generate_data_6_8_9.py --plot-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_HOME", ".hf_cache")

import matplotlib
matplotlib.use("Agg")

import numpy as np

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("analysis.maskfix_namm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M2_NAMM_MASKFIX_CKPT = ARTIFACTS / "M2_cs1024_maskfix" / "ckpt.pt"
M3_MASKFIX_LORA_CKPT = ARTIFACTS / "M3_cs1024_maskfix" / "best_ckpt.pt"
M3_BUGGY_LORA_CKPT = ARTIFACTS / "M3_cs1024" / "best_ckpt.pt"
M2_NAMM_BUGGY_CKPT = ARTIFACTS / "M2_cs1024" / "ckpt.pt"

REPORT_6_DIR = REPO_ROOT / "analysis" / "report_6"
REPORT_8_DIR = REPO_ROOT / "analysis" / "report_8"
REPORT_9_DIR = REPO_ROOT / "analysis" / "report_9"

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
MIN_CONDITIONING_LENGTH = 4096
NUM_LAYERS = 16
HIDDEN_DIM = 2048

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NAMM-dependent maskfix analyses (Reports 6, 8, 9)")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip inference; regenerate plots from saved data")
    p.add_argument("--report", type=int, nargs="*", default=[6, 8, 9],
                   help="Which reports to run (default: 6 8 9)")
    p.add_argument("--max-samples-r6", type=int, default=3,
                   help="Samples per task for Report 6 (default: 3)")
    p.add_argument("--max-samples-r8", type=int, default=40,
                   help="Max samples for Report 8 (default: 40)")
    p.add_argument("--max-samples-r9", type=int, default=40,
                   help="Max samples for Report 9 (default: 40)")
    return p.parse_args()


# ===========================================================================
# Shared model setup helpers
# ===========================================================================

def build_model_and_data(device: str = "cuda"):
    """Build model infrastructure via Hydra config.

    Returns (cfg, memory_policy, memory_model, memory_evaluator,
             task_sampler, tokenizer).
    """
    import torch
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    with torch.no_grad():
        memory_policy, memory_model, memory_evaluator, _, _ = make_eval_model(
            cfg=cfg)

    memory_model.to(device)  # MUST happen before loading weights
    memory_model.eval()

    tokenizer = memory_evaluator.tokenizer

    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    task_sampler.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT,
        val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=MIN_CONDITIONING_LENGTH,
        tokenizer=tokenizer,
    )

    return (cfg, memory_policy, memory_model, memory_evaluator,
            task_sampler, tokenizer)


def load_namm_weights(
    memory_model: Any,
    memory_policy: Any,
    namm_ckpt_path: str,
    device: str = "cuda",
) -> None:
    """Load NAMM scoring network weights from checkpoint."""
    import torch

    logger.info("Loading NAMM checkpoint: %s", namm_ckpt_path)
    ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
    evo_state = ckpt["evolution_state"]

    params_vec = evo_state.get("mean", evo_state["best_member"])
    params = params_vec.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    buffers_prefix = "stored_buffers_to_save."
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    logger.info("  NAMM loaded (%d params)", params_vec.shape[0])


def load_lora_weights(
    memory_model: Any,
    lora_ckpt_path: str,
    device: str = "cuda",
) -> None:
    """Load LoRA adapter weights into the wrapped memory model."""
    import torch

    logger.info("Loading LoRA checkpoint: %s", lora_ckpt_path)
    ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    lora_sd = ckpt["lora_state_dict"]

    if not memory_model.has_lora_adapters():
        rank = lora_cfg.get("rank", 8)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        memory_model.apply_lora_adapters(rank=rank, target_modules=target_modules)

    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_sd:
            p.data.copy_(lora_sd[n].to(p.device))
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            f"No LoRA weights loaded from {lora_ckpt_path}! "
            "Key format mismatch between checkpoint and model.")
    logger.info("  LoRA loaded (%d tensors, best_val=%s)",
                loaded, ckpt.get("best_val_score", "?"))


def reset_lora_weights(memory_model: Any) -> None:
    """Zero out LoRA weights to get base model behaviour."""
    for _n, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()
    logger.info("  Reset LoRA weights to zero (identity)")


def reset_policy_state(memory_model: Any) -> None:
    """Reset memory policy buffers between samples."""
    if hasattr(memory_model.memory_policy, "initialize_buffers"):
        memory_model.memory_policy.initialize_buffers()
    elif hasattr(memory_model.memory_policy, "reset"):
        memory_model.memory_policy.reset()


def cleanup_model(memory_model: Any) -> None:
    """Delete model and free GPU memory."""
    import torch
    del memory_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def get_test_prompts(
    task_sampler: Any,
    tokenizer: Any,
    num_per_task: int = 3,
    max_seq_len: int = 1024,
) -> list[dict]:
    """Get test prompts truncated to max_seq_len."""
    import torch

    test_idxs = task_sampler._test_idxs_per_task
    if test_idxs is None:
        raise RuntimeError("No test split available")

    prompts: list[dict] = []
    for task_name in sorted(test_idxs.keys()):
        idxs = test_idxs[task_name]
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        for idx in idxs[:num_per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len)
            prompts.append({
                "input_ids": ids["input_ids"],
                "task": task_name,
                "idx": int(idx),
                "seq_len": ids["input_ids"].shape[1],
            })

    logger.info("Prepared %d test prompts (seq_len range: %d-%d)",
                len(prompts),
                min(p["seq_len"] for p in prompts) if prompts else 0,
                max(p["seq_len"] for p in prompts) if prompts else 0)
    return prompts


def get_test_samples_with_answers(
    task_sampler: Any,
    tokenizer: Any,
    max_samples: int = 40,
    num_per_task: int = 10,
) -> list[dict]:
    """Get test-set samples with prompts, gold answers, and token IDs."""
    import torch

    test_idxs = task_sampler._test_idxs_per_task
    if test_idxs is None:
        raise RuntimeError("No test split available")

    samples: list[dict] = []
    for task_name in sorted(test_idxs.keys()):
        idxs = test_idxs[task_name]
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        task_jsons = task_sampler.lb_jsons_per_task[task_name]

        for idx in idxs[:num_per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=FILTER_BY_TOKENS)
            json_obj = task_jsons[idx]
            answers = json_obj.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            samples.append({
                "input_ids": ids["input_ids"],
                "task": task_name,
                "idx": int(idx),
                "seq_len": ids["input_ids"].shape[1],
                "answers": answers,
                "prompt_text": text,
            })

    samples.sort(key=lambda x: x["seq_len"])
    if len(samples) > max_samples:
        step = len(samples) / max_samples
        samples = [samples[int(i * step)] for i in range(max_samples)]

    logger.info("Prepared %d test samples (seq_len range: %d-%d)",
                len(samples),
                samples[0]["seq_len"] if samples else 0,
                samples[-1]["seq_len"] if samples else 0)
    return samples


def get_train_samples(
    task_sampler: Any,
    tokenizer: Any,
    max_samples: int = 40,
) -> list[dict]:
    """Build tokenised training samples with prompt+answer for CE loss."""
    import torch

    train_idxs = task_sampler._train_idxs_per_task
    if train_idxs is None:
        logger.error("No train split available")
        return []

    samples: list[dict] = []

    for task_name in sorted(train_idxs.keys()):
        idxs = train_idxs[task_name]
        prompts = task_sampler.lb_prompts_per_task[task_name]
        jsons = task_sampler.lb_jsons_per_task[task_name]

        for idx in idxs:
            if len(samples) >= max_samples:
                break

            prompt_text = prompts[idx]
            json_item = jsons[idx]
            answers = json_item.get("answers", json_item.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            answer = answers[0]

            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text},
                 {"role": "assistant", "content": answer}],
                add_generation_prompt=False,
                tokenize=True,
            )

            label_start = len(prompt_ids)
            if label_start >= len(full_ids):
                continue

            max_len = FILTER_BY_TOKENS
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            labels = [-100] * label_start + full_ids[label_start:]
            if len(labels) > len(full_ids):
                labels = labels[:len(full_ids)]

            n_answer = sum(1 for lbl in labels if lbl != -100)
            if n_answer == 0:
                continue

            input_ids_t = torch.tensor([full_ids], dtype=torch.long)
            labels_t = torch.tensor([labels], dtype=torch.long)

            samples.append({
                "input_ids": input_ids_t,
                "labels": labels_t,
                "task": task_name,
                "idx": int(idx),
                "seq_len": len(full_ids),
                "answer_tokens": n_answer,
            })

        if len(samples) >= max_samples:
            break

    logger.info(
        "Prepared %d training samples (seq_len range: %d-%d, answer tokens: %d-%d)",
        len(samples),
        min(s["seq_len"] for s in samples) if samples else 0,
        max(s["seq_len"] for s in samples) if samples else 0,
        min(s["answer_tokens"] for s in samples) if samples else 0,
        max(s["answer_tokens"] for s in samples) if samples else 0,
    )
    return samples


def find_answer_token_positions(
    tokenizer: Any,
    input_ids: "torch.Tensor",
    answers: list[str],
) -> list[int]:
    """Find token positions containing answer-relevant entities."""
    input_ids_flat = input_ids[0].tolist()
    seq_len = len(input_ids_flat)
    answer_positions: set[int] = set()

    for answer in answers:
        if not answer or not answer.strip():
            continue
        answer_ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            continue

        ans_len = len(answer_ids)
        for start in range(seq_len - ans_len + 1):
            if input_ids_flat[start:start + ans_len] == answer_ids:
                answer_positions.update(range(start, start + ans_len))

        for word in answer.strip().split():
            if len(word) < 3:
                continue
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if not word_ids:
                continue
            w_len = len(word_ids)
            for start in range(seq_len - w_len + 1):
                if input_ids_flat[start:start + w_len] == word_ids:
                    answer_positions.update(range(start, start + w_len))

    return sorted(answer_positions)


# ===========================================================================
# REPORT 6: Token Importance Alignment (maskfix)
# ===========================================================================

def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def extract_alignment_data(
    memory_model: Any,
    memory_policy: Any,
    prompts: list[dict],
    device: str = "cuda",
) -> list[dict]:
    """Extract NAMM scores and attention weights for each prompt.

    1. Forward pass with apply_memory_policy=False, output_attentions=True
    2. Call memory_policy.update_cache(analyze=True) for NAMM token scores
    3. Compute per-layer Spearman correlation and eviction regret
    """
    import torch

    results: list[dict] = []

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        reset_policy_state(memory_model)
        if hasattr(memory_policy, "record_eval_stats"):
            memory_policy.record_eval_stats = False

        # Pass 1: full-context forward (no eviction)
        with torch.no_grad():
            outputs = memory_model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                apply_memory_policy=False,
            )

        # Per-KV-position mean attention received
        attention_per_kv: list[np.ndarray] = []
        if outputs.attentions is not None:
            for layer_attn in outputs.attentions:
                per_kv = layer_attn[0].float().mean(dim=0).mean(dim=0)
                attention_per_kv.append(per_kv.cpu().numpy())

        # Convert KV cache to legacy format
        past_kv = outputs.past_key_values
        if hasattr(past_kv, "to_legacy_cache"):
            past_kv = past_kv.to_legacy_cache()
        elif not isinstance(past_kv, tuple):
            past_kv = tuple(
                (past_kv.key_cache[j], past_kv.value_cache[j])
                for j in range(len(past_kv.key_cache))
            )

        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Pass 2: NAMM analyze
        reset_policy_state(memory_model)
        try:
            _evicted_kv, analysis_dicts = memory_policy.update_cache(
                past_key_values=past_kv,
                num_new_tokens=seq_len,
                attn_weights_list=(
                    outputs.attentions
                    if memory_policy.requires_attn_scores else []),
                attention_mask=attention_mask,
                position_ids=position_ids,
                analyze=True,
            )
        except Exception as e:
            logger.warning("  Sample %d analyze failed: %s", i, e)
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Per-layer scores and eviction decisions
        sample_layers: list[dict] = []
        for layer_id, ad in enumerate(analysis_dicts):
            token_scores = ad.get("token_scores", None)
            retained_idxs = ad.get("retained_idxs", None)

            if token_scores is not None:
                scores_np = token_scores[0].float().detach().mean(dim=0).cpu().numpy()
            else:
                scores_np = np.array([])

            if retained_idxs is not None:
                retained = retained_idxs[0, 0].detach().cpu().numpy().tolist()
            else:
                retained = list(range(seq_len))

            all_idxs = set(range(len(scores_np)))
            evicted = sorted(all_idxs - set(retained))

            attn = (attention_per_kv[layer_id]
                    if layer_id < len(attention_per_kv)
                    else np.array([]))
            min_len = min(len(scores_np), len(attn))
            if min_len >= 3:
                rho = compute_spearman(scores_np[:min_len], attn[:min_len])
            else:
                rho = float("nan")

            if len(evicted) > 0 and len(attn) > 0:
                evicted_valid = [e for e in evicted if e < len(attn)]
                total_regret = (float(np.sum(attn[evicted_valid]))
                                if evicted_valid else 0.0)
                mean_regret = (float(np.mean(attn[evicted_valid]))
                               if evicted_valid else 0.0)
            else:
                total_regret = 0.0
                mean_regret = 0.0

            sample_layers.append({
                "layer_id": layer_id,
                "spearman_rho": rho,
                "total_regret": total_regret,
                "mean_regret": mean_regret,
                "num_tokens": len(scores_np),
                "num_retained": len(retained),
                "num_evicted": len(evicted),
            })

        results.append({
            "task": p["task"],
            "idx": p["idx"],
            "seq_len": seq_len,
            "layers": sample_layers,
        })

        logger.info("  Sample %d/%d (%s) done", i + 1, len(prompts), p["task"])

        del outputs, past_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_report_6(args: argparse.Namespace) -> dict | None:
    """Report 6: Token Importance Alignment (maskfix)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Use --plot-only on CPU.")

    # Validate checkpoints
    for path, name in [
        (M1_LORA_CKPT, "M1 LoRA"),
        (M3_MASKFIX_LORA_CKPT, "M3-maskfix LoRA"),
        (M2_NAMM_MASKFIX_CKPT, "M2-maskfix NAMM"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}\n"
                "Run scripts/download_artifacts.py first.")

    logger.info("=" * 60)
    logger.info("Report 6 (maskfix): Token Importance Alignment")
    logger.info("=" * 60)

    # Build model
    logger.info("Building model infrastructure...")
    cfg, memory_policy, memory_model, evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    logger.info("Preparing test prompts...")
    prompts = get_test_prompts(
        task_sampler, tokenizer,
        num_per_task=args.max_samples_r6,
        max_seq_len=1024,
    )

    # Load NAMM maskfix weights
    logger.info("Loading NAMM maskfix weights...")
    load_namm_weights(memory_model, memory_policy,
                      str(M2_NAMM_MASKFIX_CKPT), device)

    results: dict[str, Any] = {"cache_size": CACHE_SIZE, "conditions": {}}

    # -- M1: LoRA full-context weights + NAMM-maskfix scoring --
    logger.info("")
    logger.info("Condition: M1 (LoRA full-context) + NAMM-maskfix scoring")
    load_lora_weights(memory_model, str(M1_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16)

    m1_results = extract_alignment_data(
        memory_model, memory_policy, prompts, device)
    results["conditions"]["M1"] = {"samples": m1_results}
    logger.info("  M1: %d samples extracted", len(m1_results))

    # -- M3-maskfix: LoRA eviction-aware weights + NAMM-maskfix scoring --
    logger.info("")
    logger.info("Condition: M3-maskfix (LoRA + frozen NAMM-maskfix)")
    reset_lora_weights(memory_model)
    load_lora_weights(memory_model, str(M3_MASKFIX_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16)

    m3_results = extract_alignment_data(
        memory_model, memory_policy, prompts, device)
    results["conditions"]["M3_maskfix"] = {"samples": m3_results}
    logger.info("  M3-maskfix: %d samples extracted", len(m3_results))

    # Save and cleanup
    cleanup_model(memory_model)
    return results


def plot_report_6(data: dict) -> None:
    """Generate Report 6 maskfix plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    os.makedirs(REPORT_6_DIR, exist_ok=True)

    conditions = data.get("conditions", {})
    if not conditions:
        logger.warning("Report 6: No data, skipping plots")
        return

    colors = {"M1": "#d62728", "M3_maskfix": "#1f77b4"}
    labels = {
        "M1": "M1 (LoRA full-context)",
        "M3_maskfix": "M3-maskfix (LoRA + frozen NAMM)",
    }

    # -- Plot 1: Spearman correlation by layer --
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond_name, cond_data in conditions.items():
        layer_rhos: dict[int, list[float]] = {}
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                rho = ld["spearman_rho"]
                if not np.isnan(rho):
                    layer_rhos.setdefault(ld["layer_id"], []).append(rho)

        if not layer_rhos:
            continue

        layers = sorted(layer_rhos.keys())
        means = [np.mean(layer_rhos[l]) for l in layers]
        stds = [np.std(layer_rhos[l]) for l in layers]

        color = colors.get(cond_name, "#333")
        label = labels.get(cond_name, cond_name)
        ax.plot(layers, means, "-o", color=color, label=label,
                linewidth=2, markersize=5)
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_title("NAMM Score vs Attention Importance Alignment (maskfix)\n"
                 "(Spearman rho per layer, averaged across test samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_6_DIR / "score_attention_correlation_maskfix.png")
    plt.close(fig)
    logger.info("  Saved score_attention_correlation_maskfix.png")

    # -- Plot 2: Eviction regret by layer --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for cond_name, cond_data in conditions.items():
        layer_total: dict[int, list[float]] = {}
        layer_mean: dict[int, list[float]] = {}
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                layer_total.setdefault(ld["layer_id"], []).append(
                    ld["total_regret"])
                layer_mean.setdefault(ld["layer_id"], []).append(
                    ld["mean_regret"])

        layers = sorted(layer_total.keys())
        color = colors.get(cond_name, "#333")
        label = labels.get(cond_name, cond_name)

        ax1.plot(layers, [np.mean(layer_total[l]) for l in layers],
                 "-o", color=color, label=label, linewidth=2, markersize=5)
        ax2.plot(layers, [np.mean(layer_mean[l]) for l in layers],
                 "-o", color=color, label=label, linewidth=2, markersize=5)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Total Attention Mass on Evicted Tokens")
    ax1.set_title("Eviction Regret (Total, maskfix)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Attention per Evicted Token")
    ax2.set_title("Eviction Regret (Per Token, maskfix)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Eviction Regret: Attention Mass Lost to Evicted Tokens (maskfix)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT_6_DIR / "eviction_regret_maskfix.png",
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved eviction_regret_maskfix.png")

    # -- Plot 3: Per-task alignment shift (M1 vs M3-maskfix) --
    fig, ax = plt.subplots(figsize=(10, 6))

    task_cond_rhos: dict[str, dict[str, list[float]]] = {}
    for cond_name, cond_data in conditions.items():
        for sample in cond_data["samples"]:
            rhos = [ld["spearman_rho"] for ld in sample["layers"]
                    if not np.isnan(ld["spearman_rho"])]
            if rhos:
                task_cond_rhos.setdefault(sample["task"], {}).setdefault(
                    cond_name, []).append(np.mean(rhos))

    if task_cond_rhos:
        task_names = sorted(task_cond_rhos.keys())
        x = np.arange(len(task_names))
        width = 0.35
        cond_list = [c for c in ["M1", "M3_maskfix"] if c in conditions]

        for i, cond_name in enumerate(cond_list):
            vals = []
            errs = []
            for t in task_names:
                rho_list = task_cond_rhos.get(t, {}).get(cond_name, [])
                vals.append(np.mean(rho_list) if rho_list else 0.0)
                errs.append(np.std(rho_list) if rho_list else 0.0)

            color = colors.get(cond_name, "#333")
            label = labels.get(cond_name, cond_name)
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=label, color=color, edgecolor="white", capsize=3)

        task_display = {
            "qasper": "Qasper", "2wikimqa": "2WikiMQA",
            "qasper_e": "Qasper-E", "hotpotqa_e": "HotpotQA-E",
            "2wikimqa_e": "2WikiMQA-E",
        }
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([task_display.get(t, t) for t in task_names])

    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_ylabel("Mean Spearman rho (NAMM score vs attention)")
    ax.set_title("Alignment Shift: M1 vs M3-maskfix\n"
                 "(Does M3-maskfix fine-tuning improve NAMM-attention alignment?)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(REPORT_6_DIR / "alignment_shift_maskfix.png")
    plt.close(fig)
    logger.info("  Saved alignment_shift_maskfix.png")

    # Print summary
    logger.info("--- Report 6 (maskfix) Summary ---")
    for cond_name, cond_data in conditions.items():
        all_rhos: list[float] = []
        all_regrets: list[float] = []
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                if not np.isnan(ld["spearman_rho"]):
                    all_rhos.append(ld["spearman_rho"])
                all_regrets.append(ld["total_regret"])
        logger.info("  %s:", cond_name)
        logger.info("    Mean Spearman rho: %.4f +/- %.4f",
                     np.mean(all_rhos), np.std(all_rhos))
        logger.info("    Mean total regret: %.4f", np.mean(all_regrets))


# ===========================================================================
# REPORT 8: Probing (maskfix)
# ===========================================================================

def extract_hidden_states_no_eviction(
    memory_model: Any,
    samples: list[dict],
    device: str = "cuda",
) -> dict:
    """Extract mean-pooled hidden states with no eviction (M1 / Recency)."""
    import torch

    memory_model.eval()
    all_hidden: list[np.ndarray] = []
    all_seq_lens: list[int] = []

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        reset_policy_state(memory_model)

        try:
            with torch.no_grad():
                outputs = memory_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    apply_memory_policy=False,
                    use_cache=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                logger.warning("Sample %d: no hidden states returned, skipping", i)
                continue

            layer_means = []
            for layer_h in hidden_states:
                mean_h = layer_h[0].float().mean(dim=0)
                layer_means.append(mean_h.cpu().numpy())

            all_hidden.append(np.stack(layer_means))
            all_seq_lens.append(seq_len)

            if (i + 1) % 5 == 0 or i == 0:
                logger.info("  No-eviction: processed %d/%d samples",
                            i + 1, len(samples))

        except Exception as e:
            logger.error("  Sample %d (no-eviction) failed: %s", i, e)
            continue
        finally:
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "hidden_states": (np.stack(all_hidden) if all_hidden
                          else np.zeros((0, NUM_LAYERS + 1, HIDDEN_DIM))),
        "seq_lens": all_seq_lens,
    }


def extract_hidden_states_with_eviction(
    memory_model: Any,
    samples: list[dict],
    device: str = "cuda",
) -> dict:
    """Extract mean-pooled hidden states with NAMM eviction active (M3)."""
    import torch

    memory_model.eval()
    all_hidden: list[np.ndarray] = []
    all_answer_survival: list[float] = []
    all_retained: list[int] = []
    all_total: list[int] = []
    all_tasks: list[str] = []

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        answer_positions = set(sample.get("answer_positions", []))

        reset_policy_state(memory_model)

        try:
            with torch.no_grad():
                outputs = memory_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    apply_memory_policy=True,
                    use_cache=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                logger.warning("Sample %d: no hidden states returned, skipping", i)
                continue

            layer_means = []
            for layer_h in hidden_states:
                mean_h = layer_h[0].float().mean(dim=0)
                layer_means.append(mean_h.cpu().numpy())

            all_hidden.append(np.stack(layer_means))

            # Determine retained token count from KV cache
            past_kv = outputs.past_key_values
            if past_kv is not None:
                if isinstance(past_kv, tuple):
                    retained_count = past_kv[0][0].shape[-2]
                elif hasattr(past_kv, "key_cache"):
                    retained_count = past_kv.key_cache[0].shape[-2]
                else:
                    retained_count = seq_len
            else:
                retained_count = seq_len

            # Estimate answer survival
            if answer_positions and seq_len > retained_count:
                mid_start = min(CACHE_SIZE // 4, seq_len // 4)
                mid_end = seq_len - min(CACHE_SIZE // 4, seq_len // 4)
                answer_in_mid = sum(
                    1 for p in answer_positions if mid_start <= p < mid_end)
                total_in_mid = max(1, mid_end - mid_start)
                eviction_rate = min(1.0,
                    (seq_len - retained_count) / max(1, total_in_mid))
                survival_frac = 1.0 - (
                    answer_in_mid / max(1, len(answer_positions))) * eviction_rate
            elif not answer_positions:
                survival_frac = float("nan")
            else:
                survival_frac = 1.0

            all_answer_survival.append(survival_frac)
            all_retained.append(retained_count)
            all_total.append(seq_len)
            all_tasks.append(sample["task"])

            if (i + 1) % 5 == 0 or i == 0:
                logger.info("  With-eviction: processed %d/%d (retained %d/%d)",
                            i + 1, len(samples), retained_count, seq_len)

        except Exception as e:
            logger.error("  Sample %d (with-eviction) failed: %s", i, e)
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "hidden_states": (np.stack(all_hidden) if all_hidden
                          else np.zeros((0, NUM_LAYERS + 1, HIDDEN_DIM))),
        "answer_survival_fracs": all_answer_survival,
        "retained_counts": all_retained,
        "total_counts": all_total,
        "tasks": all_tasks,
    }


def extract_precise_eviction_info(
    memory_model: Any,
    memory_policy: Any,
    samples: list[dict],
    tokenizer: Any,
    device: str = "cuda",
    max_analyze_samples: int = 15,
) -> dict:
    """Two-pass analysis for precise eviction decisions."""
    import torch

    results: dict[str, Any] = {
        "answer_survival_exact": [],
        "per_task_survival": {},
        "sample_indices": [],
    }

    subset = samples[:max_analyze_samples]
    logger.info("Running precise eviction analysis on %d samples", len(subset))

    for i, sample in enumerate(subset):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        answer_positions = set(sample.get("answer_positions", []))
        task = sample["task"]

        reset_policy_state(memory_model)
        if hasattr(memory_policy, "record_eval_stats"):
            memory_policy.record_eval_stats = False

        try:
            # Pass 1: full-context forward (no eviction)
            with torch.no_grad():
                outputs = memory_model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                    apply_memory_policy=False,
                )

            # Get KV cache in legacy format
            past_kv = outputs.past_key_values
            if hasattr(past_kv, "to_legacy_cache"):
                past_kv = past_kv.to_legacy_cache()
            elif not isinstance(past_kv, tuple):
                past_kv = tuple(
                    (past_kv.key_cache[j], past_kv.value_cache[j])
                    for j in range(len(past_kv.key_cache))
                )

            attention_mask = torch.ones(1, seq_len, dtype=torch.long,
                                        device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Pass 2: NAMM analyze
            reset_policy_state(memory_model)
            try:
                _evicted_kv, analysis_dicts = memory_policy.update_cache(
                    past_key_values=past_kv,
                    num_new_tokens=seq_len,
                    attn_weights_list=(
                        outputs.attentions
                        if memory_policy.requires_attn_scores else []),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    analyze=True,
                )
            except Exception as e:
                logger.warning("  Sample %d analyze failed: %s", i, e)
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # Use layer 0 retained indices as representative
            if analysis_dicts and len(analysis_dicts) > 0:
                ad = analysis_dicts[0]
                retained_idxs = ad.get("retained_idxs", None)
                if retained_idxs is not None:
                    retained_set = set(
                        retained_idxs[0, 0].detach().cpu().numpy().tolist())
                else:
                    retained_set = set(range(seq_len))

                if answer_positions:
                    survived = answer_positions & retained_set
                    survival_frac = len(survived) / len(answer_positions)
                else:
                    survival_frac = float("nan")
            else:
                survival_frac = float("nan")

            results["answer_survival_exact"].append(survival_frac)
            results["sample_indices"].append(i)
            results["per_task_survival"].setdefault(task, []).append(
                survival_frac)

            logger.info("  Analyze %d/%d (%s): %.1f%% answer tokens survived",
                        i + 1, len(subset), task,
                        survival_frac * 100 if not np.isnan(survival_frac) else 0)

        except Exception as e:
            logger.error("  Analyze sample %d failed: %s", i, e)
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def train_probes(
    m1_hidden: np.ndarray,
    m3_hidden: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """Train logistic regression probes per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    n_samples = m1_hidden.shape[0]
    n_layers_plus_1 = m1_hidden.shape[1]

    n_positive = int(labels.sum())
    n_negative = n_samples - n_positive
    logger.info("Probe training: %d samples (%d positive, %d negative)",
                n_samples, n_positive, n_negative)

    if n_positive < 2 or n_negative < 2:
        logger.warning("Insufficient label balance. Falling back to LOO CV.")
        n_folds = n_samples

    random_accuracy = max(n_positive, n_negative) / n_samples

    m1_accuracies = np.zeros(n_layers_plus_1)
    m3_accuracies = np.zeros(n_layers_plus_1)
    m1_stds = np.zeros(n_layers_plus_1)
    m3_stds = np.zeros(n_layers_plus_1)

    for layer_idx in range(n_layers_plus_1):
        m1_X = m1_hidden[:, layer_idx, :]
        m3_X = m3_hidden[:, layer_idx, :]

        m1_fold_accs: list[float] = []
        m3_fold_accs: list[float] = []

        if n_folds >= n_samples:
            loo = LeaveOneOut()
            splits = list(loo.split(m1_X, labels))
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=42)
            splits = list(skf.split(m1_X, labels))

        for train_idx, test_idx in splits:
            scaler_m1 = StandardScaler()
            X_train_m1 = scaler_m1.fit_transform(m1_X[train_idx])
            X_test_m1 = scaler_m1.transform(m1_X[test_idx])

            clf_m1 = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
            clf_m1.fit(X_train_m1, labels[train_idx])
            m1_fold_accs.append(clf_m1.score(X_test_m1, labels[test_idx]))

            scaler_m3 = StandardScaler()
            X_train_m3 = scaler_m3.fit_transform(m3_X[train_idx])
            X_test_m3 = scaler_m3.transform(m3_X[test_idx])

            clf_m3 = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
            clf_m3.fit(X_train_m3, labels[train_idx])
            m3_fold_accs.append(clf_m3.score(X_test_m3, labels[test_idx]))

        m1_accuracies[layer_idx] = np.mean(m1_fold_accs)
        m3_accuracies[layer_idx] = np.mean(m3_fold_accs)
        m1_stds[layer_idx] = np.std(m1_fold_accs)
        m3_stds[layer_idx] = np.std(m3_fold_accs)

        if layer_idx % 4 == 0 or layer_idx == n_layers_plus_1 - 1:
            logger.info("  Layer %d: M1=%.3f, M3-maskfix=%.3f, random=%.3f",
                        layer_idx, m1_accuracies[layer_idx],
                        m3_accuracies[layer_idx], random_accuracy)

    return {
        "m1_accuracies": m1_accuracies,
        "m3_accuracies": m3_accuracies,
        "m1_stds": m1_stds,
        "m3_stds": m3_stds,
        "random_accuracy": random_accuracy,
        "n_samples": n_samples,
        "n_positive": n_positive,
    }


def run_report_8(args: argparse.Namespace) -> dict | None:
    """Report 8: Probing for Residual Knowledge (maskfix)."""
    import torch
    from namm.run_utils import make_eval_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Use --plot-only on CPU.")

    for path, name in [
        (M1_LORA_CKPT, "M1 LoRA"),
        (M3_MASKFIX_LORA_CKPT, "M3-maskfix LoRA"),
        (M2_NAMM_MASKFIX_CKPT, "M2-maskfix NAMM"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}\n"
                "Run scripts/download_artifacts.py first.")

    logger.info("=" * 60)
    logger.info("Report 8 (maskfix): Probing for Residual Knowledge")
    logger.info("=" * 60)

    # Build model infrastructure
    logger.info("Building model infrastructure...")
    cfg, memory_policy, memory_model, evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    # Get test samples with gold answers
    logger.info("Preparing test samples...")
    samples = get_test_samples_with_answers(
        task_sampler, tokenizer, max_samples=args.max_samples_r8)

    # Pre-compute answer token positions
    logger.info("Finding answer token positions...")
    for sample in samples:
        sample["answer_positions"] = find_answer_token_positions(
            tokenizer, sample["input_ids"], sample["answers"])

    n_with_answers = sum(1 for s in samples if s["answer_positions"])
    logger.info("Samples with answer tokens found: %d/%d",
                n_with_answers, len(samples))

    # -- Phase 1a: M1 (LoRA full context, no eviction) --
    logger.info("")
    logger.info("Phase 1a: M1 (full context, Recency passthrough)")

    # Swap to Recency policy for no eviction
    from namm.policy.base import Recency
    recency_policy = Recency(cache_size=None)
    memory_model.swap_memory_policy(recency_policy)

    load_lora_weights(memory_model, str(M1_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16)

    m1_data = extract_hidden_states_no_eviction(memory_model, samples, device)
    logger.info("M1: extracted %d samples", len(m1_data["seq_lens"]))

    # -- Phase 1b: M3-maskfix (LoRA + NAMM eviction) --
    logger.info("")
    logger.info("Phase 1b: M3-maskfix (with NAMM eviction)")

    # Rebuild model for M3 to get fresh NAMM policy
    cleanup_model(memory_model)

    logger.info("Rebuilding model for M3-maskfix...")
    from scripts.experiment_utils import load_hydra_config

    cfg_m3 = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    with torch.no_grad():
        memory_policy_m3, memory_model_m3, _, _, _ = make_eval_model(cfg=cfg_m3)

    memory_model_m3.to(device)
    memory_model_m3.eval()

    load_namm_weights(memory_model_m3, memory_policy_m3,
                      str(M2_NAMM_MASKFIX_CKPT), device)
    load_lora_weights(memory_model_m3, str(M3_MASKFIX_LORA_CKPT), device)
    memory_model_m3.to(dtype=torch.bfloat16)

    m3_data = extract_hidden_states_with_eviction(
        memory_model_m3, samples, device)
    logger.info("M3-maskfix: extracted %d samples",
                len(m3_data["retained_counts"]))

    # -- Phase 1c: Precise eviction analysis (subset) --
    logger.info("")
    logger.info("Phase 1c: Precise eviction analysis (two-pass)")

    reset_lora_weights(memory_model_m3)
    load_lora_weights(memory_model_m3, str(M3_MASKFIX_LORA_CKPT), device)

    eviction_data = extract_precise_eviction_info(
        memory_model_m3, memory_policy_m3, samples, tokenizer, device,
        max_analyze_samples=min(15, len(samples)),
    )

    # -- Phase 2: Construct labels and train probes --
    logger.info("")
    logger.info("Phase 2: Training probes")

    n_m1 = m1_data["hidden_states"].shape[0]
    n_m3 = m3_data["hidden_states"].shape[0]
    n_common = min(n_m1, n_m3)

    if n_common == 0:
        logger.error("No samples extracted for both conditions!")
        cleanup_model(memory_model_m3)
        return None

    m1_hidden = m1_data["hidden_states"][:n_common]
    m3_hidden = m3_data["hidden_states"][:n_common]

    labels = np.zeros(n_common, dtype=np.int32)
    for j in range(n_common):
        answer_pos = samples[j].get("answer_positions", [])
        if not answer_pos:
            labels[j] = 0
            continue

        seq_len = samples[j]["seq_len"]
        if j < len(m3_data["retained_counts"]):
            retained = m3_data["retained_counts"][j]
        else:
            retained = seq_len

        if retained >= seq_len:
            labels[j] = 0
        else:
            mid_start = min(CACHE_SIZE // 4, seq_len // 4)
            mid_end = seq_len - min(CACHE_SIZE // 4, seq_len // 4)
            answer_in_mid = sum(
                1 for p in answer_pos if mid_start <= p < mid_end)
            labels[j] = 1 if answer_in_mid > 0 else 0

    # Override with precise eviction data where available
    for k, sample_idx in enumerate(eviction_data.get("sample_indices", [])):
        if sample_idx < n_common:
            frac = eviction_data["answer_survival_exact"][k]
            if not np.isnan(frac):
                labels[sample_idx] = 1 if frac < 1.0 else 0

    logger.info("Labels: %d positive (answer evicted), %d negative, %d total",
                int(labels.sum()), int((1 - labels).sum()), n_common)

    probe_results = train_probes(m1_hidden, m3_hidden, labels)

    # Save data
    save_dict: dict[str, Any] = {
        "m1_accuracies": probe_results["m1_accuracies"],
        "m3_accuracies": probe_results["m3_accuracies"],
        "m1_stds": probe_results["m1_stds"],
        "m3_stds": probe_results["m3_stds"],
        "random_accuracy": np.array([probe_results["random_accuracy"]]),
        "n_samples": np.array([probe_results["n_samples"]]),
        "n_positive": np.array([probe_results["n_positive"]]),
        "labels": labels,
        "answer_survival_fracs": np.array(
            m3_data["answer_survival_fracs"][:n_common], dtype=np.float32),
        "retained_counts": np.array(
            m3_data["retained_counts"][:n_common], dtype=np.int32),
        "total_counts": np.array(
            m3_data["total_counts"][:n_common], dtype=np.int32),
        "task_names": np.array(
            m3_data["tasks"][:n_common], dtype=object),
    }

    if eviction_data.get("answer_survival_exact"):
        save_dict["answer_survival_exact"] = np.array(
            eviction_data["answer_survival_exact"], dtype=np.float32)
        for task, fracs in eviction_data.get("per_task_survival", {}).items():
            safe_name = task.replace("/", "_")
            save_dict[f"task_survival_{safe_name}"] = np.array(
                fracs, dtype=np.float32)

    cleanup_model(memory_model_m3)
    return save_dict


def plot_report_8(data: dict) -> None:
    """Generate Report 8 maskfix plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    os.makedirs(REPORT_8_DIR, exist_ok=True)

    m1_acc = np.asarray(data["m1_accuracies"])
    m3_acc = np.asarray(data["m3_accuracies"])
    m1_std = np.asarray(data["m1_stds"])
    m3_std = np.asarray(data["m3_stds"])
    random_acc_raw = data["random_accuracy"]
    random_acc = (float(random_acc_raw[0])
                  if hasattr(random_acc_raw, "__len__")
                  else float(random_acc_raw))

    n_layers_plus_1 = len(m1_acc)
    layers = np.arange(n_layers_plus_1)
    layer_labels = ["emb"] + [f"{i}" for i in range(n_layers_plus_1 - 1)]

    colors = {"M1": "#d62728", "M3": "#1f77b4", "random": "#7f7f7f"}

    # -- Plot 1: Probe accuracy per layer --
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, m1_acc, "-o", color=colors["M1"],
            label="M1 (full context)", linewidth=2, markersize=5)
    ax.fill_between(layers, m1_acc - m1_std, m1_acc + m1_std,
                    alpha=0.15, color=colors["M1"])

    ax.plot(layers, m3_acc, "-s", color=colors["M3"],
            label="M3-maskfix (with eviction)", linewidth=2, markersize=5)
    ax.fill_between(layers, m3_acc - m3_std, m3_acc + m3_std,
                    alpha=0.15, color=colors["M3"])

    ax.axhline(random_acc, color=colors["random"], linestyle="--",
               linewidth=1.5,
               label=f"Majority-class baseline ({random_acc:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy (CV)")
    ax.set_title(
        "Entity Presence Probe: Per-Layer Accuracy (maskfix)\n"
        "Can a linear probe on retained hidden states detect evicted answer entities?"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    n_samples_raw = data.get("n_samples", [0])
    n_pos_raw = data.get("n_positive", [0])
    n_samples = (int(n_samples_raw[0])
                 if hasattr(n_samples_raw, "__len__")
                 else int(n_samples_raw))
    n_positive = (int(n_pos_raw[0])
                  if hasattr(n_pos_raw, "__len__")
                  else int(n_pos_raw))
    ax.text(0.02, 0.02,
            f"n={n_samples}, {n_positive} positive / {n_samples - n_positive} negative",
            transform=ax.transAxes, fontsize=9, color="grey",
            verticalalignment="bottom")

    fig.tight_layout()
    path = REPORT_8_DIR / "probe_accuracy_maskfix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # -- Plot 2: Entity survival per task --
    fig, ax = plt.subplots(figsize=(10, 6))

    task_survival: dict[str, list[float]] = {}
    has_precise = False
    for key in data:
        if isinstance(key, str) and key.startswith("task_survival_"):
            task_name = key[len("task_survival_"):]
            fracs = data[key]
            valid = [float(f) for f in fracs if not np.isnan(float(f))]
            if valid:
                task_survival[task_name] = valid
                has_precise = True

    if not has_precise and "task_names" in data and "answer_survival_fracs" in data:
        tasks_arr = data["task_names"]
        survival_arr = data["answer_survival_fracs"]
        for j in range(len(tasks_arr)):
            task = str(tasks_arr[j])
            safe_task = task.replace("/", "_")
            frac = float(survival_arr[j])
            if not np.isnan(frac):
                task_survival.setdefault(safe_task, []).append(frac)

    if task_survival:
        task_names_sorted = sorted(task_survival.keys())
        x = np.arange(len(task_names_sorted))
        means = [np.mean(task_survival[t]) for t in task_names_sorted]
        stds = [np.std(task_survival[t]) for t in task_names_sorted]
        counts = [len(task_survival[t]) for t in task_names_sorted]

        ax.bar(x, means, yerr=stds, capsize=4,
               color="#2ca02c", edgecolor="white", alpha=0.85)

        for xi, cnt in zip(x, counts):
            ax.text(xi, 0.02, f"n={cnt}", ha="center", fontsize=8,
                    color="white", fontweight="bold")

        display = {
            "lb_qasper": "Qasper", "lb_2wikimqa": "2WikiMQA",
            "lb_qasper_e": "Qasper-E", "lb_hotpotqa_e": "HotpotQA-E",
            "lb_2wikimqa_e": "2WikiMQA-E",
        }
        ax.set_xticks(x)
        ax.set_xticklabels(
            [display.get(t, t) for t in task_names_sorted],
            rotation=15, ha="right")
    else:
        if "retained_counts" in data and "total_counts" in data:
            retained = np.asarray(data["retained_counts"])
            total = np.asarray(data["total_counts"])
            ratios = retained / np.maximum(total, 1)
            ax.hist(ratios, bins=20, color="#2ca02c", edgecolor="white",
                    alpha=0.85)
            ax.set_xlabel("Retention Ratio")
            ax.set_ylabel("Count")

    ax.set_ylabel("Answer Token Survival Fraction")
    ax.set_title(
        "Entity Survival After Eviction (maskfix)\n"
        "(per task, from NAMM-maskfix eviction analysis)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="No eviction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = REPORT_8_DIR / "entity_survival_maskfix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # -- Plot 3: Layer-wise information loss --
    fig, ax = plt.subplots(figsize=(12, 6))

    diff = m1_acc - m3_acc
    diff_std = np.sqrt(m1_std**2 + m3_std**2)

    bar_colors = ["#e74c3c" if d > 0 else "#27ae60" for d in diff]
    ax.bar(layers, diff, yerr=diff_std, capsize=3,
           color=bar_colors, edgecolor="white", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Difference (M1 - M3-maskfix)")
    ax.set_title(
        "Information Loss per Layer: Probe Accuracy Gap (maskfix)\n"
        "Positive = M1 retains more entity info than M3-maskfix")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.grid(True, alpha=0.3)

    max_layer = int(np.argmax(diff))
    if diff[max_layer] > 0:
        ax.annotate(
            f"Max gap: layer {max_layer}\n({diff[max_layer]:.3f})",
            xy=(max_layer, diff[max_layer]),
            xytext=(max_layer + 1.5, diff[max_layer] + 0.02),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=9, color="grey",
        )

    fig.tight_layout()
    path = REPORT_8_DIR / "layer_wise_information_maskfix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # Summary
    logger.info("--- Report 8 (maskfix) Summary ---")
    logger.info("  M1 mean probe accuracy: %.3f +/- %.3f",
                np.mean(m1_acc), np.mean(m1_std))
    logger.info("  M3-maskfix mean probe accuracy: %.3f +/- %.3f",
                np.mean(m3_acc), np.mean(m3_std))
    logger.info("  Random baseline: %.3f", random_acc)
    logger.info("  Mean accuracy gap (M1 - M3-maskfix): %.3f", np.mean(diff))
    logger.info("  Max gap at layer %d: %.3f", max_layer, diff[max_layer])

    if "retained_counts" in data and "total_counts" in data:
        retained = np.asarray(data["retained_counts"])
        total = np.asarray(data["total_counts"])
        mean_retention = np.mean(retained / np.maximum(total, 1))
        logger.info("  Mean token retention ratio: %.3f", mean_retention)


# ===========================================================================
# REPORT 9: Gradient Flow (maskfix)
# ===========================================================================

def _get_lora_param_info(memory_model: Any) -> list[dict]:
    """Return list of dicts describing each LoRA parameter tensor."""
    info: list[dict] = []
    for n, p in memory_model.model.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        layer_idx = -1
        module = "unknown"
        component = "unknown"
        for i_part, part in enumerate(parts):
            if part == "layers" and i_part + 1 < len(parts):
                try:
                    layer_idx = int(parts[i_part + 1])
                except ValueError:
                    pass
            if part in ("q_proj", "v_proj", "k_proj", "o_proj"):
                module = part
            if part in ("lora_A", "lora_B"):
                component = part
        info.append({
            "name": n,
            "layer_idx": layer_idx,
            "module": module,
            "component": component,
        })
    return info


def compute_answer_loss_and_grads(
    memory_model: Any,
    sample: dict,
    device: str,
    apply_memory_policy: bool,
) -> dict[str, Any] | None:
    """Run forward+backward on one sample, return loss and gradient info."""
    import torch
    import torch.nn.functional as F

    input_ids = sample["input_ids"].to(device)
    labels = sample["labels"].to(device)

    memory_model.model.zero_grad(set_to_none=True)

    reset_policy_state(memory_model)

    if apply_memory_policy and hasattr(memory_model.memory_policy, "record_eval_stats"):
        memory_model.memory_policy.record_eval_stats = True

    if apply_memory_policy:
        memory_model.memory_policy.set_params_batch_idxs(
            np.zeros([input_ids.shape[0]], dtype=np.int64))

    seq_len = input_ids.shape[1]

    answer_mask = (labels[0] != -100)
    if answer_mask.any():
        answer_start = answer_mask.nonzero(as_tuple=True)[0][0].item()
    else:
        return None

    chunk_align = getattr(memory_model, "max_new_tokens", 64) or 64
    context_end = (answer_start // chunk_align) * chunk_align
    context_end = max(context_end, 0)

    past_key_values = None
    retention_ratio = 1.0

    # Phase 1: context tokens under no_grad
    if context_end > 0:
        with torch.no_grad():
            ctx_outputs = memory_model(
                input_ids=input_ids[:, :context_end],
                use_cache=True,
                apply_memory_policy=apply_memory_policy,
                limit_new_tokens=None,
                output_hidden_states=False,
                skip_lm_head=True,
            )
        past_key_values = ctx_outputs.past_key_values

        if apply_memory_policy and past_key_values is not None:
            if isinstance(past_key_values, tuple):
                n_retained = past_key_values[0][0].shape[-2]
            else:
                n_retained = past_key_values.key_cache[0].shape[-2]
            retention_ratio = n_retained / context_end

        del ctx_outputs
        torch.cuda.empty_cache()

    # Phase 2: answer tokens with gradients
    phase2_input = input_ids[:, context_end:]
    phase2_pos = torch.arange(
        context_end, seq_len, device=device
    ).unsqueeze(0).expand(input_ids.shape[0], -1)

    outputs = memory_model(
        input_ids=phase2_input,
        position_ids=phase2_pos,
        past_key_values=past_key_values,
        use_cache=True,
        apply_memory_policy=apply_memory_policy,
        limit_new_tokens=None,
        output_hidden_states=True,
        skip_lm_head=True,
    )

    hidden_states = outputs.hidden_states[-1]
    phase2_labels = labels[:, context_end:]
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = phase2_labels[:, 1:].contiguous()

    lm_head = memory_model.lm_head
    ce_chunk_size = 512
    ce_seq_len = shift_hidden.shape[1]
    total_loss = torch.tensor(0.0, device=device)
    n_tokens = (shift_labels != -100).sum()

    if n_tokens == 0:
        return None

    for i_chunk in range(0, ce_seq_len, ce_chunk_size):
        chunk_h = shift_hidden[:, i_chunk:i_chunk + ce_chunk_size, :]
        chunk_logits = lm_head(chunk_h).float()
        chunk_labels = shift_labels[:, i_chunk:i_chunk + ce_chunk_size].contiguous().view(-1)
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.view(-1, chunk_logits.size(-1)),
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        del chunk_logits

    loss = total_loss / n_tokens.clamp(min=1)

    loss.backward()

    # Collect gradient norms and directions per layer
    param_info = _get_lora_param_info(memory_model)
    per_layer_grad_norms: dict[int, float] = {}
    per_param_grads: dict[str, list[float]] = {}

    for pinfo in param_info:
        name = pinfo["name"]
        layer_idx = pinfo["layer_idx"]
        param = dict(memory_model.model.named_parameters())[name]

        if param.grad is not None:
            grad_flat = param.grad.detach().float().cpu().flatten()
            grad_l2 = float(grad_flat.norm(2).item())

            if layer_idx not in per_layer_grad_norms:
                per_layer_grad_norms[layer_idx] = 0.0
            per_layer_grad_norms[layer_idx] += grad_l2 ** 2

            per_param_grads[name] = grad_flat.tolist()

    for layer_idx in per_layer_grad_norms:
        per_layer_grad_norms[layer_idx] = float(
            np.sqrt(per_layer_grad_norms[layer_idx]))

    loss_val = float(loss.item())

    if apply_memory_policy and hasattr(memory_model.memory_policy, "record_eval_stats"):
        memory_model.memory_policy.record_eval_stats = False

    del outputs, past_key_values, hidden_states
    memory_model.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return {
        "loss": loss_val,
        "per_layer_grad_norms": {str(k): v for k, v in per_layer_grad_norms.items()},
        "per_param_grads": per_param_grads,
        "retention_ratio": retention_ratio,
        "n_answer_tokens": int(n_tokens.item()),
    }


def run_report_9(args: argparse.Namespace) -> dict | None:
    """Report 9: Gradient Flow and Loss Attribution (maskfix)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Use --plot-only on CPU.")

    for path, name in [
        (M3_MASKFIX_LORA_CKPT, "M3-maskfix LoRA"),
        (M2_NAMM_MASKFIX_CKPT, "M2-maskfix NAMM"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}\n"
                "Run scripts/download_artifacts.py first.")

    logger.info("=" * 60)
    logger.info("Report 9 (maskfix): Gradient Flow and Loss Attribution")
    logger.info("=" * 60)

    # Build model
    logger.info("Building model infrastructure...")
    cfg, memory_policy, memory_model, evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    # Load NAMM maskfix weights
    load_namm_weights(memory_model, memory_policy,
                      str(M2_NAMM_MASKFIX_CKPT), device)

    # Load M3-maskfix LoRA weights
    load_lora_weights(memory_model, str(M3_MASKFIX_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16)

    # PEFT gradient fix: make embedding outputs require grad
    def make_inputs_require_grad(module: Any, inp: Any, out: Any) -> None:
        out.requires_grad_(True)

    try:
        embed_layer = memory_model.model.get_input_embeddings()
    except AttributeError:
        embed_layer = memory_model.get_input_embeddings()
    embed_layer.register_forward_hook(make_inputs_require_grad)
    logger.info("Registered PEFT embedding forward hook")

    # Get training samples
    logger.info("Preparing training samples...")
    train_samples = get_train_samples(
        task_sampler, tokenizer, max_samples=args.max_samples_r9)
    if not train_samples:
        logger.error("No training samples available")
        cleanup_model(memory_model)
        return None

    # -- Pass 1: WITH NAMM eviction --
    logger.info("")
    logger.info("Pass 1: Forward+backward WITH NAMM eviction (cache_size=%d)",
                CACHE_SIZE)

    evicted_results: list[dict] = []
    for i, sample in enumerate(train_samples):
        try:
            memory_model.train()
            result = compute_answer_loss_and_grads(
                memory_model, sample, device, apply_memory_policy=True)
            if result is not None:
                result["task"] = sample["task"]
                result["idx"] = sample["idx"]
                result["seq_len"] = sample["seq_len"]
                evicted_results.append(result)
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "  [evicted] %d/%d: loss=%.4f retention=%.4f",
                        i + 1, len(train_samples),
                        result["loss"], result["retention_ratio"],
                    )
        except Exception as e:
            logger.error("  Sample %d (evicted) failed: %s", i, e)
            torch.cuda.empty_cache()
            continue

    logger.info("Pass 1 complete: %d/%d samples succeeded",
                len(evicted_results), len(train_samples))

    # -- Pass 2: WITHOUT eviction (full context) --
    logger.info("")
    logger.info("Pass 2: Forward+backward WITHOUT eviction (full context)")

    from namm.policy.base import Recency
    original_policy = memory_model.memory_policy
    recency_policy = Recency(cache_size=None)
    memory_model.swap_memory_policy(recency_policy)

    saved_delay = getattr(memory_model, "memory_policy_fixed_delay", None)
    memory_model.memory_policy_fixed_delay = None

    full_results: list[dict] = []
    for i, sample in enumerate(train_samples):
        try:
            memory_model.train()
            result = compute_answer_loss_and_grads(
                memory_model, sample, device, apply_memory_policy=False)
            if result is not None:
                result["task"] = sample["task"]
                result["idx"] = sample["idx"]
                result["seq_len"] = sample["seq_len"]
                full_results.append(result)
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "  [full] %d/%d: loss=%.4f",
                        i + 1, len(train_samples), result["loss"],
                    )
        except Exception as e:
            logger.error("  Sample %d (full) failed: %s", i, e)
            torch.cuda.empty_cache()
            continue

    # Restore policy
    memory_model.memory_policy_fixed_delay = saved_delay
    memory_model.swap_memory_policy(original_policy)

    logger.info("Pass 2 complete: %d/%d samples succeeded",
                len(full_results), len(train_samples))

    # -- Compute cosine similarity between gradient directions --
    cosine_sims_per_layer: dict[str, list[float]] = {}
    for ev, fu in zip(evicted_results, full_results):
        if ev["idx"] != fu["idx"] or ev["task"] != fu["task"]:
            continue
        for param_name in ev["per_param_grads"]:
            if param_name not in fu["per_param_grads"]:
                continue
            ev_grad = np.array(ev["per_param_grads"][param_name],
                               dtype=np.float32)
            fu_grad = np.array(fu["per_param_grads"][param_name],
                               dtype=np.float32)
            ev_norm = np.linalg.norm(ev_grad)
            fu_norm = np.linalg.norm(fu_grad)
            if ev_norm > 1e-12 and fu_norm > 1e-12:
                cos_sim = float(np.dot(ev_grad, fu_grad) / (ev_norm * fu_norm))
            else:
                cos_sim = 0.0

            parts = param_name.split(".")
            layer_key = "unknown"
            for j, part in enumerate(parts):
                if part == "layers" and j + 1 < len(parts):
                    try:
                        layer_key = str(int(parts[j + 1]))
                        break
                    except ValueError:
                        pass
            cosine_sims_per_layer.setdefault(layer_key, []).append(cos_sim)

    # Strip per_param_grads before saving (too large for JSON)
    for r in evicted_results:
        del r["per_param_grads"]
    for r in full_results:
        del r["per_param_grads"]

    all_data: dict[str, Any] = {
        "evicted": evicted_results,
        "full_context": full_results,
        "cosine_sims_per_layer": cosine_sims_per_layer,
        "cache_size": CACHE_SIZE,
        "max_samples": args.max_samples_r9,
    }

    cleanup_model(memory_model)
    return all_data


def plot_report_9(data: dict) -> None:
    """Generate Report 9 maskfix plots."""
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    os.makedirs(REPORT_9_DIR, exist_ok=True)

    evicted = data.get("evicted", [])
    full = data.get("full_context", [])
    cosine_sims = data.get("cosine_sims_per_layer", {})

    # Summary stats
    if evicted:
        ev_losses = [s["loss"] for s in evicted]
        retentions = [s["retention_ratio"] for s in evicted]
        logger.info("  Evicted loss: %.4f +/- %.4f",
                     np.mean(ev_losses), np.std(ev_losses))
        logger.info("  Retention: %.4f +/- %.4f (median %.4f)",
                     np.mean(retentions), np.std(retentions),
                     np.median(retentions))

    if full:
        fu_losses = [s["loss"] for s in full]
        logger.info("  Full-context loss: %.4f +/- %.4f",
                     np.mean(fu_losses), np.std(fu_losses))

    if evicted and full:
        logger.info("  Loss increase: %.4f (%.1f%%)",
                     np.mean(ev_losses) - np.mean(fu_losses),
                     (np.mean(ev_losses) - np.mean(fu_losses)) /
                     max(np.mean(fu_losses), 1e-8) * 100)

    if cosine_sims:
        all_sims: list[float] = []
        for sims in cosine_sims.values():
            all_sims.extend(sims)
        logger.info("  Gradient cosine similarity: %.4f +/- %.4f",
                     np.mean(all_sims), np.std(all_sims))

    # -- Plot 1: Loss stratified by retention ratio --
    if evicted and full:
        retention_ratios = [s["retention_ratio"] for s in evicted]
        median_retention = float(np.median(retention_ratios))

        high_ret_losses = [s["loss"] for s in evicted
                           if s["retention_ratio"] >= median_retention]
        low_ret_losses = [s["loss"] for s in evicted
                          if s["retention_ratio"] < median_retention]
        full_losses = [s["loss"] for s in full]

        fig, ax = plt.subplots(figsize=(10, 6))

        box_data = [full_losses, high_ret_losses, low_ret_losses]
        box_labels = [
            f"Full context\n(n={len(full_losses)})",
            f"Evicted, high ret\n(>= {median_retention:.2f}, "
            f"n={len(high_ret_losses)})",
            f"Evicted, low ret\n(< {median_retention:.2f}, "
            f"n={len(low_ret_losses)})",
        ]
        colors = ["#2ca02c", "#1f77b4", "#d62728"]

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i_box, (vals, color) in enumerate(zip(box_data, colors)):
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.ones(len(vals)) * (i_box + 1) + jitter,
                vals, alpha=0.4, s=15, color=color, zorder=3,
            )

        ax.set_ylabel("Cross-Entropy Loss (answer tokens)", fontsize=12)
        ax.set_title(
            "Per-Sample Loss Stratified by Retention Ratio (maskfix)\n"
            f"(median retention = {median_retention:.2f}, "
            f"cache_size={data.get('cache_size', '?')})",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        path = REPORT_9_DIR / "loss_stratified_maskfix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("  Saved %s", path)
    else:
        logger.info("  Skipping loss_stratified_maskfix (no data)")

    # -- Plot 2: Per-layer gradient L2 norms --
    if evicted and full:
        evicted_layer_norms: dict[int, list[float]] = {}
        full_layer_norms: dict[int, list[float]] = {}

        for s in evicted:
            for layer_str, norm_val in s["per_layer_grad_norms"].items():
                evicted_layer_norms.setdefault(int(layer_str), []).append(
                    norm_val)

        for s in full:
            for layer_str, norm_val in s["per_layer_grad_norms"].items():
                full_layer_norms.setdefault(int(layer_str), []).append(
                    norm_val)

        layers = sorted(
            set(evicted_layer_norms.keys()) | set(full_layer_norms.keys()))

        if layers:
            ev_means = [np.mean(evicted_layer_norms.get(l, [0]))
                        for l in layers]
            ev_stds = [np.std(evicted_layer_norms.get(l, [0]))
                       for l in layers]
            fu_means = [np.mean(full_layer_norms.get(l, [0]))
                        for l in layers]
            fu_stds = [np.std(full_layer_norms.get(l, [0]))
                       for l in layers]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(layers))
            width = 0.35

            ax.bar(x - width / 2, ev_means, width, yerr=ev_stds,
                   label="Evicted (NAMM-maskfix cs1024)", color="#d62728",
                   edgecolor="white", capsize=3, alpha=0.8)
            ax.bar(x + width / 2, fu_means, width, yerr=fu_stds,
                   label="Full context", color="#1f77b4",
                   edgecolor="white", capsize=3, alpha=0.8)

            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("LoRA Gradient L2 Norm", fontsize=12)
            ax.set_title("Per-Layer LoRA Gradient Norms (maskfix)\n"
                         "(evicted vs full context, M3-maskfix checkpoint)",
                         fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in layers])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()

            path = REPORT_9_DIR / "grad_norms_maskfix.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("  Saved %s", path)
    else:
        logger.info("  Skipping grad_norms_maskfix (no data)")

    # -- Plot 3: Loss vs retention ratio scatter --
    if evicted:
        retentions = [s["retention_ratio"] for s in evicted]
        losses = [s["loss"] for s in evicted]
        tasks = [s["task"] for s in evicted]

        fig, ax = plt.subplots(figsize=(10, 7))

        unique_tasks = sorted(set(tasks))
        task_colors = {t: c for t, c in zip(
            unique_tasks,
            plt.cm.Set2(np.linspace(0, 1, max(len(unique_tasks), 1)))
        )}

        for task in unique_tasks:
            mask = [t == task for t in tasks]
            task_ret = [r for r, m in zip(retentions, mask) if m]
            task_loss = [lo for lo, m in zip(losses, mask) if m]
            ax.scatter(
                task_ret, task_loss, label=task, alpha=0.7, s=40,
                color=task_colors[task], edgecolors="white", linewidth=0.5,
            )

        if len(retentions) >= 3:
            slope, intercept, r_value, p_value, _std_err = \
                scipy_stats.linregress(retentions, losses)
            x_line = np.linspace(min(retentions), max(retentions), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "--", color="gray", alpha=0.7,
                    label=f"OLS: r={r_value:.3f}, p={p_value:.3f}")

        ax.set_xlabel("Retention Ratio", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss (answer tokens)", fontsize=12)
        ax.set_title(
            "Loss vs Retention Ratio Under NAMM Eviction (maskfix)\n"
            f"(M3-maskfix checkpoint, cache_size={data.get('cache_size', '?')})",
            fontsize=13,
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = REPORT_9_DIR / "loss_vs_retention_maskfix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("  Saved %s", path)
    else:
        logger.info("  Skipping loss_vs_retention_maskfix (no data)")

    # -- Plot 4: Gradient direction consistency --
    if cosine_sims:
        layer_data: list[tuple[int, list[float]]] = []
        for layer_str, sims in cosine_sims.items():
            try:
                layer_idx = int(layer_str)
            except ValueError:
                continue
            layer_data.append((layer_idx, sims))
        layer_data.sort(key=lambda x: x[0])

        if layer_data:
            layers_cos = [ld[0] for ld in layer_data]
            means = [np.mean(ld[1]) for ld in layer_data]
            stds = [np.std(ld[1]) for ld in layer_data]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            x = np.arange(len(layers_cos))
            bar_colors = [
                "#2ca02c" if m > 0.9 else "#ff7f0e" if m > 0.5 else "#d62728"
                for m in means
            ]
            ax1.bar(x, means, yerr=stds, color=bar_colors, edgecolor="white",
                    capsize=3, alpha=0.8)
            ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5,
                        label="Perfect alignment")
            ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
            ax1.set_xlabel("Layer", fontsize=12)
            ax1.set_ylabel("Cosine Similarity", fontsize=12)
            ax1.set_title("Gradient Direction Consistency (maskfix)\n"
                          "(evicted vs full context)", fontsize=13)
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(l) for l in layers_cos])
            ax1.set_ylim(-0.2, 1.15)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3, axis="y")

            box_data = [ld[1] for ld in layer_data]
            bp = ax2.boxplot(box_data,
                             labels=[str(l) for l in layers_cos],
                             patch_artist=True, widths=0.6)
            for patch in bp["boxes"]:
                patch.set_facecolor("#1f77b4")
                patch.set_alpha(0.6)
            ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
            ax2.set_xlabel("Layer", fontsize=12)
            ax2.set_ylabel("Cosine Similarity", fontsize=12)
            ax2.set_title("Per-Layer Distribution of\n"
                          "Gradient Cosine Similarity (maskfix)", fontsize=13)
            ax2.set_ylim(-0.2, 1.15)
            ax2.grid(True, alpha=0.3, axis="y")

            fig.suptitle(
                "Gradient Direction Alignment: Evicted vs Full Context (maskfix)\n"
                "(cos=1.0 means eviction does not change gradient direction)",
                fontsize=14, y=1.04,
            )
            fig.tight_layout()

            path = REPORT_9_DIR / "grad_direction_consistency_maskfix.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved %s", path)
    else:
        logger.info("  Skipping grad_direction_consistency_maskfix (no data)")

    # Summary
    logger.info("--- Report 9 (maskfix) Summary ---")
    if evicted:
        logger.info("  Evicted samples: %d", len(evicted))
    if full:
        logger.info("  Full-context samples: %d", len(full))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    reports_to_run = args.report

    # -- Report 6 --
    if 6 in reports_to_run:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# REPORT 6 (maskfix): Token Importance Alignment")
        logger.info("#" * 70)
        data_file_6 = REPORT_6_DIR / "maskfix_alignment_data.json"

        try:
            if args.plot_only:
                if not data_file_6.exists():
                    logger.error("Data file not found: %s", data_file_6)
                    logger.error("Run on GPU first (without --plot-only).")
                else:
                    with open(data_file_6) as f:
                        data_6 = json.load(f)
                    logger.info("Generating Report 6 maskfix plots...")
                    plot_report_6(data_6)
            else:
                data_6 = run_report_6(args)
                if data_6:
                    os.makedirs(REPORT_6_DIR, exist_ok=True)
                    logger.info("Saving data to %s", data_file_6)
                    with open(data_file_6, "w") as f:
                        json.dump(data_6, f, indent=2)
                    logger.info("Generating Report 6 maskfix plots...")
                    plot_report_6(data_6)
                else:
                    logger.error("Report 6: No data produced.")
        except Exception as e:
            logger.error("Report 6 failed: %s", e, exc_info=True)

        # Force GPU cleanup between sections
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- Report 8 --
    if 8 in reports_to_run:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# REPORT 8 (maskfix): Probing for Residual Knowledge")
        logger.info("#" * 70)
        data_file_8 = REPORT_8_DIR / "maskfix_probe_data.npz"

        try:
            if args.plot_only:
                if not data_file_8.exists():
                    logger.error("Data file not found: %s", data_file_8)
                    logger.error("Run on GPU first (without --plot-only).")
                else:
                    loaded = np.load(data_file_8, allow_pickle=True)
                    data_8 = {k: loaded[k] for k in loaded.files}
                    logger.info("Generating Report 8 maskfix plots...")
                    plot_report_8(data_8)
            else:
                data_8_dict = run_report_8(args)
                if data_8_dict:
                    os.makedirs(REPORT_8_DIR, exist_ok=True)
                    logger.info("Saving data to %s", data_file_8)
                    np.savez(data_file_8, **data_8_dict)
                    logger.info("  Saved (%d KB)",
                                data_file_8.stat().st_size // 1024)
                    logger.info("Generating Report 8 maskfix plots...")
                    plot_report_8(data_8_dict)
                else:
                    logger.error("Report 8: No data produced.")
        except Exception as e:
            logger.error("Report 8 failed: %s", e, exc_info=True)

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- Report 9 --
    if 9 in reports_to_run:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# REPORT 9 (maskfix): Gradient Flow and Loss Attribution")
        logger.info("#" * 70)
        data_file_9 = REPORT_9_DIR / "maskfix_gradient_data.json"

        try:
            if args.plot_only:
                if not data_file_9.exists():
                    logger.error("Data file not found: %s", data_file_9)
                    logger.error("Run on GPU first (without --plot-only).")
                else:
                    with open(data_file_9) as f:
                        data_9 = json.load(f)
                    logger.info("Generating Report 9 maskfix plots...")
                    plot_report_9(data_9)
            else:
                data_9 = run_report_9(args)
                if data_9:
                    os.makedirs(REPORT_9_DIR, exist_ok=True)
                    logger.info("Saving data to %s", data_file_9)
                    with open(data_file_9, "w") as f:
                        json.dump(data_9, f, indent=2)
                    logger.info("  Saved (%.1f KB)",
                                data_file_9.stat().st_size / 1024)
                    logger.info("Generating Report 9 maskfix plots...")
                    plot_report_9(data_9)
                else:
                    logger.error("Report 9: No data produced.")
        except Exception as e:
            logger.error("Report 9 failed: %s", e, exc_info=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("All requested maskfix NAMM analyses complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
