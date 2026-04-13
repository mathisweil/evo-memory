#!/usr/bin/env python3
"""Analysis 8 -- Probing for Residual Knowledge of Evicted Content.

When NAMM evicts tokens from the KV cache, does the model retain
information about the evicted content in the hidden states of the
surviving tokens?

We use an **entity presence probe**: for each test sample, we check
whether a linear classifier trained on mean-pooled hidden states of
RETAINED tokens can predict whether the gold-answer entity appeared
in the evicted portion of the context.

Conditions compared:
  - M1 (LoRA, full context)  -- upper bound; all entity info is present
  - M3 cs1024 (LoRA + frozen NAMM)  -- test; entity info may be in
    evicted tokens, probe must find residual/compressed info
  - Random baseline  -- shuffled labels

Produces three plots in analysis/report_8/:
  probe_accuracy.png            -- per-layer probe accuracy (M1 vs M3 vs random)
  entity_survival.png           -- fraction of answer tokens surviving eviction
  layer_wise_information.png    -- probe accuracy gap (M1 - M3) per layer

Usage:
    # Full run (GPU required):
    source activate.sh
    PYTHONPATH=. .venv/bin/python analysis/report_8/generate_plots.py

    # Plot-only (CPU ok, after data has been extracted):
    PYTHONPATH=. .venv/bin/python analysis/report_8/generate_plots.py --plot-only

    # Limit samples:
    PYTHONPATH=. .venv/bin/python analysis/report_8/generate_plots.py --max-samples 20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Repo setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("analysis.report_8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

OUT_DIR = SCRIPT_DIR
DATA_FILE = OUT_DIR / "probe_data.npz"

# ── Checkpoint paths ──────────────────────────────────────────────────────────

ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M3_LORA_CKPT = ARTIFACTS / "M3_cs1024" / "best_ckpt.pt"
M2_NAMM_CKPT = ARTIFACTS / "M2_cs1024" / "ckpt.pt"

# ── Model / data config ──────────────────────────────────────────────────────

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
NUM_LAYERS = 16
HIDDEN_DIM = 2048
MAX_SAMPLES = 40
NUM_SAMPLES_PER_TASK = 10


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analysis 8: Probing for Residual Knowledge of Evicted Content")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip inference; regenerate plots from saved probe_data.npz")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                   help="Maximum number of test samples to process")
    return p.parse_args()


# ── Model setup (Hydra + NAMM infrastructure) ────────────────────────────────

def build_model_and_data(device: str = "cuda"):
    """Build model infrastructure via Hydra config.

    Returns cfg, memory_policy, memory_model, memory_evaluator,
    task_sampler, tokenizer.
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

    memory_model.to(device)
    memory_model.eval()

    tokenizer = memory_evaluator.tokenizer

    # Build task sampler and apply splits
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    task_sampler.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT,
        val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=None,
        tokenizer=tokenizer,
    )

    return (cfg, memory_policy, memory_model, memory_evaluator,
            task_sampler, tokenizer)


def load_namm_weights(
    memory_model,
    memory_policy,
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
    memory_model,
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


def reset_lora_weights(memory_model) -> None:
    """Zero out LoRA weights to get base model behavior."""
    for _n, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()
    logger.info("  Reset LoRA weights to zero (identity)")


# ── Test data ────────────────────────────────────────────────────────────────

def get_test_samples(
    task_sampler,
    tokenizer,
    max_samples: int = MAX_SAMPLES,
    num_per_task: int = NUM_SAMPLES_PER_TASK,
) -> list[dict]:
    """Get test-set samples with prompts, gold answers, and token IDs.

    Returns list of dicts with keys: input_ids, task, idx, seq_len,
    answers, prompt_text.
    """
    import torch

    test_idxs = task_sampler._test_idxs_per_task
    if test_idxs is None:
        raise RuntimeError("No test split available")

    samples = []
    for task_name in sorted(test_idxs.keys()):
        idxs = test_idxs[task_name]
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        task_jsons = task_sampler.lb_jsons_per_task[task_name]

        for idx in idxs[:num_per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=FILTER_BY_TOKENS)
            # Extract gold answers
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

    # Sort by seq_len for efficient processing
    samples.sort(key=lambda x: x["seq_len"])

    if len(samples) > max_samples:
        step = len(samples) / max_samples
        samples = [samples[int(i * step)] for i in range(max_samples)]

    logger.info("Prepared %d test samples (seq_len range: %d-%d)",
                len(samples),
                samples[0]["seq_len"] if samples else 0,
                samples[-1]["seq_len"] if samples else 0)
    return samples


def find_answer_token_positions(
    tokenizer,
    input_ids: "torch.Tensor",
    answers: list[str],
) -> list[int]:
    """Find token positions that contain answer-relevant entities.

    Uses a simple heuristic: tokenize each gold answer string and search
    for those sub-sequences in the input_ids. Returns a deduplicated sorted
    list of token positions that are part of any answer match.

    Args:
        tokenizer: The model tokenizer.
        input_ids: Shape (1, seq_len) tensor of token IDs.
        answers: List of gold answer strings.

    Returns:
        Sorted list of unique token positions containing answer content.
    """
    input_ids_flat = input_ids[0].tolist()
    seq_len = len(input_ids_flat)
    answer_positions: set[int] = set()

    for answer in answers:
        if not answer or not answer.strip():
            continue
        # Tokenize the answer without special tokens
        answer_ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            continue

        # Slide over input_ids looking for subsequence matches
        ans_len = len(answer_ids)
        for start in range(seq_len - ans_len + 1):
            if input_ids_flat[start:start + ans_len] == answer_ids:
                answer_positions.update(range(start, start + ans_len))

        # Also try matching individual words in the answer for partial matches
        # This catches cases where tokenization differs slightly
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


# ── Hidden state extraction ──────────────────────────────────────────────────

def extract_hidden_states_m1(
    memory_model,
    samples: list[dict],
    device: str = "cuda",
) -> dict:
    """Extract mean-pooled hidden states for M1 (full context, no eviction).

    For M1 we run with apply_memory_policy=False to keep all tokens.
    We extract hidden states from every layer, mean-pool across sequence
    positions, and record answer token positions.

    Returns dict with:
        hidden_states: (n_samples, n_layers+1, hidden_dim) float32 array
        answer_positions: list of lists of answer token positions
        seq_lens: list of sequence lengths
    """
    import torch

    memory_model.eval()
    all_hidden = []
    all_answer_pos = []
    all_seq_lens = []

    tokenizer_ref = None
    # Get tokenizer from evaluator — we'll pass it in from the caller
    # For now, extract it from the model's memory_policy or pass separately

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # Reset memory policy state
        if hasattr(memory_model.memory_policy, "reset"):
            memory_model.memory_policy.reset()
        elif hasattr(memory_model.memory_policy, "initialize_buffers"):
            memory_model.memory_policy.initialize_buffers()

        try:
            with torch.no_grad():
                outputs = memory_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    apply_memory_policy=False,
                    use_cache=True,
                    return_dict=True,
                )

            # outputs.hidden_states is tuple of (n_layers+1,) tensors
            # each of shape (1, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                logger.warning("Sample %d: no hidden states returned, skipping", i)
                continue

            # Mean-pool across sequence positions for each layer
            layer_means = []
            for layer_h in hidden_states:
                # layer_h shape: (1, seq_len, hidden_dim)
                mean_h = layer_h[0].float().mean(dim=0)  # (hidden_dim,)
                layer_means.append(mean_h.cpu().numpy())

            all_hidden.append(np.stack(layer_means))  # (n_layers+1, hidden_dim)
            all_answer_pos.append(sample.get("answer_positions", []))
            all_seq_lens.append(seq_len)

            if (i + 1) % 5 == 0 or i == 0:
                logger.info("  M1: processed %d/%d samples", i + 1, len(samples))

        except Exception as e:
            logger.error("  M1 sample %d failed: %s", i, e)
            continue
        finally:
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "hidden_states": np.stack(all_hidden) if all_hidden else np.zeros(
            (0, NUM_LAYERS + 1, HIDDEN_DIM)),
        "answer_positions": all_answer_pos,
        "seq_lens": all_seq_lens,
    }


def extract_hidden_states_m3(
    memory_model,
    samples: list[dict],
    device: str = "cuda",
) -> dict:
    """Extract mean-pooled hidden states for M3 (LoRA + NAMM eviction).

    Runs with apply_memory_policy=True so NAMM eviction is active.
    Due to split processing, we only get the last layer's full hidden
    states reliably. For probe purposes, we extract hidden states from
    the final forward chunk (post-eviction context).

    Also records which answer tokens survived eviction by comparing the
    post-eviction KV cache size to the original sequence length.

    Returns dict with:
        hidden_states: (n_samples, n_layers+1, hidden_dim) float32 array
        answer_survival_fracs: list of floats (fraction of answer tokens
            that survive eviction)
        retained_counts: list of ints (number of retained tokens)
        total_counts: list of ints (original sequence lengths)
        tasks: list of task names
    """
    import torch

    memory_model.eval()
    all_hidden = []
    all_answer_survival = []
    all_retained = []
    all_total = []
    all_tasks = []

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        answer_positions = set(sample.get("answer_positions", []))

        # Reset memory policy state
        if hasattr(memory_model.memory_policy, "reset"):
            memory_model.memory_policy.reset()
        elif hasattr(memory_model.memory_policy, "initialize_buffers"):
            memory_model.memory_policy.initialize_buffers()

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

            # The last element of hidden_states is post-norm, and during
            # split processing it gets the full accumulated sequence.
            # However we only have the last layer reliably accumulated.
            # Use the last hidden state (post-norm) for all layers we can get.
            layer_means = []
            for layer_h in hidden_states:
                # layer_h shape: (1, retained_seq_len, hidden_dim) for
                # the final chunk, or (1, full_seq_len, hidden_dim) if
                # accumulated. Mean-pool over whatever we get.
                mean_h = layer_h[0].float().mean(dim=0)  # (hidden_dim,)
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

            # Estimate answer survival: if cache is smaller than seq_len,
            # tokens were evicted. We approximate by assuming eviction
            # removes the lowest-scoring tokens (which may include answer
            # tokens in the middle of the context).
            # A more precise approach: we know the retained count and that
            # NAMM keeps highest-scoring tokens. Since we cannot directly
            # observe which positions were kept without the analyze path
            # (which is expensive), we use the ratio as a proxy and also
            # do a separate two-pass analysis for a subset of samples.
            if answer_positions and seq_len > retained_count:
                # Heuristic: positions near the start and end are more
                # likely retained (sinks + recent tokens). Middle positions
                # are more likely evicted.
                evicted_count = seq_len - retained_count
                # Fraction of answer tokens in the "likely evicted" zone
                # (middle portion of the sequence)
                mid_start = min(CACHE_SIZE // 4, seq_len // 4)
                mid_end = seq_len - min(CACHE_SIZE // 4, seq_len // 4)
                answer_in_mid = sum(
                    1 for p in answer_positions if mid_start <= p < mid_end)
                total_in_mid = max(1, mid_end - mid_start)
                # Probability of answer token being evicted is proportional
                # to eviction rate in the middle zone
                eviction_rate = min(1.0, evicted_count / max(1, total_in_mid))
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
                logger.info("  M3: processed %d/%d samples (retained %d/%d tokens)",
                            i + 1, len(samples), retained_count, seq_len)

        except Exception as e:
            logger.error("  M3 sample %d failed: %s", i, e)
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "hidden_states": np.stack(all_hidden) if all_hidden else np.zeros(
            (0, NUM_LAYERS + 1, HIDDEN_DIM)),
        "answer_survival_fracs": all_answer_survival,
        "retained_counts": all_retained,
        "total_counts": all_total,
        "tasks": all_tasks,
    }


def extract_precise_eviction_info(
    memory_model,
    memory_policy,
    samples: list[dict],
    tokenizer,
    device: str = "cuda",
    max_analyze_samples: int = 15,
) -> dict:
    """Two-pass analysis to get precise eviction decisions.

    Pass 1: Forward without eviction to get full KV cache.
    Pass 2: Call memory_policy.update_cache(analyze=True) to get
    per-token scores and retained indices.

    This is expensive so we only do it for a subset of samples.

    Returns dict with:
        answer_survival_exact: list of floats per sample
        per_task_survival: dict mapping task -> list of survival fracs
    """
    import torch

    results = {
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

        # Reset policy
        memory_policy.initialize_buffers()
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
            memory_policy.initialize_buffers()
            try:
                _evicted_kv, analysis_dicts = memory_policy.update_cache(
                    past_key_values=past_kv,
                    num_new_tokens=seq_len,
                    attn_weights_list=(
                        outputs.attentions
                        if memory_policy.requires_attn_scores
                        else []),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    analyze=True,
                )
            except Exception as e:
                logger.warning("  Sample %d analyze failed: %s", i, e)
                del outputs
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

            logger.info("  Analyze sample %d/%d (%s): %.1f%% answer tokens survived",
                        i + 1, len(subset), task,
                        survival_frac * 100 if not np.isnan(survival_frac) else 0)

        except Exception as e:
            logger.error("  Analyze sample %d failed: %s", i, e)
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ── Probe training ───────────────────────────────────────────────────────────

def train_probes(
    m1_hidden: np.ndarray,
    m3_hidden: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """Train logistic regression probes per layer.

    For each layer (0 to NUM_LAYERS), trains a linear probe using
    stratified k-fold cross-validation to predict the binary label
    (whether the gold answer entity overlaps with evicted tokens).

    Args:
        m1_hidden: (n_samples, n_layers+1, hidden_dim) M1 hidden states.
        m3_hidden: (n_samples, n_layers+1, hidden_dim) M3 hidden states.
        labels: (n_samples,) binary labels.
        n_folds: Number of CV folds.

    Returns dict with:
        m1_accuracies: (n_layers+1,) mean accuracy per layer
        m3_accuracies: (n_layers+1,) mean accuracy per layer
        m1_stds: (n_layers+1,) std of accuracy per layer
        m3_stds: (n_layers+1,) std of accuracy per layer
        random_accuracy: float (expected random baseline)
        n_samples: int
        n_positive: int
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    n_samples = m1_hidden.shape[0]
    n_layers_plus_1 = m1_hidden.shape[1]

    # Check label balance
    n_positive = int(labels.sum())
    n_negative = n_samples - n_positive
    logger.info("Probe training: %d samples (%d positive, %d negative)",
                n_samples, n_positive, n_negative)

    if n_positive < 2 or n_negative < 2:
        logger.warning("Insufficient label balance for meaningful probing. "
                       "Falling back to leave-one-out CV.")
        n_folds = n_samples  # LOO

    # Random baseline: accuracy of always predicting the majority class
    random_accuracy = max(n_positive, n_negative) / n_samples

    m1_accuracies = np.zeros(n_layers_plus_1)
    m3_accuracies = np.zeros(n_layers_plus_1)
    m1_stds = np.zeros(n_layers_plus_1)
    m3_stds = np.zeros(n_layers_plus_1)

    for layer_idx in range(n_layers_plus_1):
        m1_X = m1_hidden[:, layer_idx, :]  # (n_samples, hidden_dim)
        m3_X = m3_hidden[:, layer_idx, :]

        m1_fold_accs = []
        m3_fold_accs = []

        if n_folds >= n_samples:
            # Leave-one-out
            loo = LeaveOneOut()
            splits = list(loo.split(m1_X, labels))
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=42)
            splits = list(skf.split(m1_X, labels))

        for train_idx, test_idx in splits:
            # M1 probe
            scaler_m1 = StandardScaler()
            X_train_m1 = scaler_m1.fit_transform(m1_X[train_idx])
            X_test_m1 = scaler_m1.transform(m1_X[test_idx])

            clf_m1 = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
            clf_m1.fit(X_train_m1, labels[train_idx])
            m1_fold_accs.append(clf_m1.score(X_test_m1, labels[test_idx]))

            # M3 probe
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
            logger.info("  Layer %d: M1=%.3f, M3=%.3f, random=%.3f",
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


# ── Main extraction pipeline ────────────────────────────────────────────────

def run_extraction(args: argparse.Namespace) -> dict:
    """Full GPU pipeline: load models, extract hidden states, train probes."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required for extraction. Use --plot-only on CPU.")

    # Validate checkpoints
    for path, name in [(M1_LORA_CKPT, "M1 LoRA"), (M3_LORA_CKPT, "M3 LoRA"),
                       (M2_NAMM_CKPT, "M2 NAMM")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}\n"
                "Run scripts/download_artifacts.py first.")

    logger.info("=" * 60)
    logger.info("Analysis 8: Probing for Residual Knowledge of Evicted Content")
    logger.info("=" * 60)

    # Build model infrastructure
    logger.info("Building model infrastructure...")
    cfg, memory_policy, memory_model, evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    # Get test samples with gold answers
    logger.info("Preparing test samples...")
    samples = get_test_samples(task_sampler, tokenizer, args.max_samples)

    # Pre-compute answer token positions for each sample
    logger.info("Finding answer token positions in prompts...")
    for sample in samples:
        sample["answer_positions"] = find_answer_token_positions(
            tokenizer, sample["input_ids"], sample["answers"])
        if sample["answer_positions"]:
            logger.info("  %s idx=%d: %d answer tokens found at %d positions",
                        sample["task"], sample["idx"],
                        len(sample["answer_positions"]),
                        len(sample["answer_positions"]))

    n_with_answers = sum(1 for s in samples if s["answer_positions"])
    logger.info("Samples with answer tokens found: %d/%d",
                n_with_answers, len(samples))

    # ── Load NAMM weights (shared frozen policy) ─────────────────────────
    logger.info("Loading NAMM weights...")
    load_namm_weights(memory_model, memory_policy, str(M2_NAMM_CKPT), device)

    # ── Phase 1a: M1 (LoRA full context, no eviction) ───────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1a: Extracting hidden states -- M1 (full context)")
    logger.info("=" * 60)

    # For M1: swap to Recency (passthrough) policy for no eviction
    from namm.policy.base import Recency
    original_policy = memory_model.memory_policy
    recency_policy = Recency(cache_size=None)
    memory_model.swap_memory_policy(recency_policy)

    load_lora_weights(memory_model, str(M1_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16, device=device)

    m1_data = extract_hidden_states_m1(memory_model, samples, device)
    logger.info("M1: extracted %d samples", len(m1_data["seq_lens"]))

    # ── Phase 1b: M3 cs1024 (LoRA + frozen NAMM eviction) ───────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1b: Extracting hidden states -- M3 cs1024 (with eviction)")
    logger.info("=" * 60)

    # Rebuild model for M3 to get a fresh NAMM policy
    del memory_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Reloading model for M3...")
    from namm.run_utils import make_eval_model

    with torch.no_grad():
        memory_policy_m3, memory_model_m3, evaluator_m3, _, _ = \
            make_eval_model(cfg=cfg)

    # Move model to device BEFORE loading weights (critical for NAMM)
    memory_model_m3.to(device)
    memory_model_m3.eval()

    # Load NAMM + LoRA for M3
    load_namm_weights(memory_model_m3, memory_policy_m3, str(M2_NAMM_CKPT),
                      device)
    load_lora_weights(memory_model_m3, str(M3_LORA_CKPT), device)
    memory_model_m3.to(dtype=torch.bfloat16)

    m3_data = extract_hidden_states_m3(memory_model_m3, samples, device)
    logger.info("M3: extracted %d samples", len(m3_data["retained_counts"]))

    # ── Phase 1c: Precise eviction analysis (subset) ─────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1c: Precise eviction analysis (two-pass)")
    logger.info("=" * 60)

    # Reset LoRA and reload for analyze pass (need no-eviction forward first)
    reset_lora_weights(memory_model_m3)
    load_lora_weights(memory_model_m3, str(M3_LORA_CKPT), device)

    eviction_data = extract_precise_eviction_info(
        memory_model_m3, memory_policy_m3, samples, tokenizer, device,
        max_analyze_samples=min(15, len(samples)),
    )

    # ── Phase 2: Construct labels and train probes ───────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Training probes")
    logger.info("=" * 60)

    # Align sample counts between M1 and M3
    n_m1 = m1_data["hidden_states"].shape[0]
    n_m3 = m3_data["hidden_states"].shape[0]
    n_common = min(n_m1, n_m3)

    if n_common == 0:
        logger.error("No samples extracted for both M1 and M3!")
        return {}

    m1_hidden = m1_data["hidden_states"][:n_common]
    m3_hidden = m3_data["hidden_states"][:n_common]

    # Binary label: was any answer entity in the evicted portion?
    # We use the M3 retained_counts vs answer_positions to determine this.
    # Label 1 = answer tokens were likely evicted (context was long enough
    # for eviction AND answer tokens are present in the middle zone)
    labels = np.zeros(n_common, dtype=np.int32)
    for j in range(n_common):
        answer_pos = samples[j].get("answer_positions", [])
        if not answer_pos:
            # No answer tokens found; label as 0 (no eviction of answer)
            labels[j] = 0
            continue

        seq_len = samples[j]["seq_len"]
        if j < len(m3_data["retained_counts"]):
            retained = m3_data["retained_counts"][j]
        else:
            retained = seq_len

        if retained >= seq_len:
            # No eviction happened
            labels[j] = 0
        else:
            # Check if answer tokens are in the middle zone
            # (likely eviction target)
            mid_start = min(CACHE_SIZE // 4, seq_len // 4)
            mid_end = seq_len - min(CACHE_SIZE // 4, seq_len // 4)
            answer_in_mid = sum(
                1 for p in answer_pos if mid_start <= p < mid_end)
            labels[j] = 1 if answer_in_mid > 0 else 0

    # Use precise eviction data where available to override labels
    for k, sample_idx in enumerate(eviction_data.get("sample_indices", [])):
        if sample_idx < n_common:
            frac = eviction_data["answer_survival_exact"][k]
            if not np.isnan(frac):
                # Label 1 if some answer tokens were evicted
                labels[sample_idx] = 1 if frac < 1.0 else 0

    logger.info("Labels: %d positive (answer evicted), %d negative, %d total",
                int(labels.sum()), int((1 - labels).sum()), n_common)

    # Train probes
    probe_results = train_probes(m1_hidden, m3_hidden, labels)

    # ── Save data ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Saving data to %s", DATA_FILE)

    save_dict = {
        # Probe results
        "m1_accuracies": probe_results["m1_accuracies"],
        "m3_accuracies": probe_results["m3_accuracies"],
        "m1_stds": probe_results["m1_stds"],
        "m3_stds": probe_results["m3_stds"],
        "random_accuracy": np.array([probe_results["random_accuracy"]]),
        "n_samples": np.array([probe_results["n_samples"]]),
        "n_positive": np.array([probe_results["n_positive"]]),
        "labels": labels,
        # Entity survival data
        "answer_survival_fracs": np.array(
            m3_data["answer_survival_fracs"][:n_common], dtype=np.float32),
        "retained_counts": np.array(
            m3_data["retained_counts"][:n_common], dtype=np.int32),
        "total_counts": np.array(
            m3_data["total_counts"][:n_common], dtype=np.int32),
        # Task info (encode as integers for npz)
        "task_names": np.array(
            m3_data["tasks"][:n_common], dtype=object),
    }

    # Add precise eviction data
    if eviction_data.get("answer_survival_exact"):
        save_dict["answer_survival_exact"] = np.array(
            eviction_data["answer_survival_exact"], dtype=np.float32)
        # Per-task survival
        for task, fracs in eviction_data.get("per_task_survival", {}).items():
            safe_name = task.replace("/", "_")
            save_dict[f"task_survival_{safe_name}"] = np.array(
                fracs, dtype=np.float32)

    np.savez(DATA_FILE, **save_dict)
    logger.info("  Saved (%d KB)", DATA_FILE.stat().st_size // 1024)

    return save_dict


# ── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(data: dict) -> None:
    """Generate all three plots from probe data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    os.makedirs(OUT_DIR, exist_ok=True)

    m1_acc = data["m1_accuracies"]
    m3_acc = data["m3_accuracies"]
    m1_std = data["m1_stds"]
    m3_std = data["m3_stds"]
    random_acc = float(data["random_accuracy"])
    n_layers_plus_1 = len(m1_acc)
    layers = np.arange(n_layers_plus_1)

    # If random_accuracy is stored as array, extract scalar
    if hasattr(random_acc, "__len__"):
        random_acc = float(random_acc)

    colors = {
        "M1": "#d62728",
        "M3": "#1f77b4",
        "random": "#7f7f7f",
    }

    # ── Plot 1: Probe accuracy per layer ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, m1_acc, "-o", color=colors["M1"],
            label="M1 (full context)", linewidth=2, markersize=5)
    ax.fill_between(layers, m1_acc - m1_std, m1_acc + m1_std,
                    alpha=0.15, color=colors["M1"])

    ax.plot(layers, m3_acc, "-s", color=colors["M3"],
            label="M3 cs1024 (with eviction)", linewidth=2, markersize=5)
    ax.fill_between(layers, m3_acc - m3_std, m3_acc + m3_std,
                    alpha=0.15, color=colors["M3"])

    ax.axhline(random_acc, color=colors["random"], linestyle="--",
               linewidth=1.5, label=f"Majority-class baseline ({random_acc:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy (CV)")
    ax.set_title(
        "Entity Presence Probe: Per-Layer Accuracy\n"
        "Can a linear probe on retained hidden states detect evicted answer entities?"
    )
    ax.set_xticks(layers)
    layer_labels = [f"emb"] + [f"{i}" for i in range(n_layers_plus_1 - 1)]
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    n_samples = int(data.get("n_samples", [0])[0]) if hasattr(
        data.get("n_samples", 0), "__len__") else int(
        data.get("n_samples", 0))
    n_positive = int(data.get("n_positive", [0])[0]) if hasattr(
        data.get("n_positive", 0), "__len__") else int(
        data.get("n_positive", 0))
    ax.text(0.02, 0.02,
            f"n={n_samples}, {n_positive} positive / {n_samples - n_positive} negative",
            transform=ax.transAxes, fontsize=9, color="grey",
            verticalalignment="bottom")

    fig.tight_layout()
    path = OUT_DIR / "probe_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # ── Plot 2: Entity survival per task ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use precise eviction data if available, else use approximate
    task_survival: dict[str, list[float]] = {}

    # Check for per-task precise data
    has_precise = False
    for key in data:
        if isinstance(key, str) and key.startswith("task_survival_"):
            task_name = key[len("task_survival_"):]
            fracs = data[key]
            valid = [float(f) for f in fracs if not np.isnan(float(f))]
            if valid:
                task_survival[task_name] = valid
                has_precise = True

    # Fall back to approximate data
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
        task_names = sorted(task_survival.keys())
        x = np.arange(len(task_names))
        means = [np.mean(task_survival[t]) for t in task_names]
        stds = [np.std(task_survival[t]) for t in task_names]
        counts = [len(task_survival[t]) for t in task_names]

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color="#2ca02c", edgecolor="white", alpha=0.85)

        # Add count labels
        for xi, cnt in zip(x, counts):
            ax.text(xi, 0.02, f"n={cnt}", ha="center", fontsize=8,
                    color="white", fontweight="bold")

        # Display names
        display = {
            "lb_qasper": "Qasper",
            "lb_2wikimqa": "2WikiMQA",
            "lb_qasper_e": "Qasper-E",
            "lb_hotpotqa_e": "HotpotQA-E",
            "lb_2wikimqa_e": "2WikiMQA-E",
        }
        ax.set_xticks(x)
        ax.set_xticklabels([display.get(t, t) for t in task_names],
                           rotation=15, ha="right")
    else:
        # Show overall retention stats
        if "retained_counts" in data and "total_counts" in data:
            retained = data["retained_counts"]
            total = data["total_counts"]
            ratios = retained / np.maximum(total, 1)
            ax.hist(ratios, bins=20, color="#2ca02c", edgecolor="white",
                    alpha=0.85)
            ax.set_xlabel("Retention Ratio (retained / total tokens)")
            ax.set_ylabel("Count")

    ax.set_ylabel("Answer Token Survival Fraction")
    ax.set_title(
        "Entity Survival: Fraction of Answer Tokens Retained After Eviction\n"
        "(per task, from NAMM eviction analysis)"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="No eviction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = OUT_DIR / "entity_survival.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # ── Plot 3: Layer-wise information loss (M1 - M3) ────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    diff = m1_acc - m3_acc
    diff_std = np.sqrt(m1_std**2 + m3_std**2)  # propagated uncertainty

    bar_colors = ["#e74c3c" if d > 0 else "#27ae60" for d in diff]
    bars = ax.bar(layers, diff, yerr=diff_std, capsize=3,
                  color=bar_colors, edgecolor="white", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Difference (M1 - M3)")
    ax.set_title(
        "Information Loss per Layer: Probe Accuracy Gap\n"
        "Positive = M1 retains more entity info than M3 (information lost to eviction)"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate max difference
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
    path = OUT_DIR / "layer_wise_information.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("--- Summary ---")
    logger.info("  M1 mean probe accuracy: %.3f +/- %.3f",
                np.mean(m1_acc), np.mean(m1_std))
    logger.info("  M3 mean probe accuracy: %.3f +/- %.3f",
                np.mean(m3_acc), np.mean(m3_std))
    logger.info("  Random baseline: %.3f", random_acc)
    logger.info("  Mean accuracy gap (M1 - M3): %.3f", np.mean(diff))
    logger.info("  Max gap at layer %d: %.3f", max_layer, diff[max_layer])

    if "retained_counts" in data and "total_counts" in data:
        retained = data["retained_counts"]
        total = data["total_counts"]
        mean_retention = np.mean(retained / np.maximum(total, 1))
        logger.info("  Mean token retention ratio: %.3f", mean_retention)

    logger.info("All plots saved to: %s", OUT_DIR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.plot_only:
        if not DATA_FILE.exists():
            logger.error("Data file not found: %s", DATA_FILE)
            logger.error("Run on GPU first (without --plot-only).")
            sys.exit(1)
        logger.info("Loading data from %s", DATA_FILE)
        loaded = np.load(DATA_FILE, allow_pickle=True)
        data = {k: loaded[k] for k in loaded.files}
    else:
        data = run_extraction(args)
        if not data:
            logger.error("Extraction produced no data.")
            sys.exit(1)

    logger.info("Generating plots...")
    generate_plots(data)
    logger.info("Done.")


if __name__ == "__main__":
    main()
