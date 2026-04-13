#!/usr/bin/env python3
"""Analysis 9 -- Gradient Flow and Loss Attribution Under Eviction.

Performs instrumented evaluation passes over training data to compare
gradient flow and per-sample loss between NAMM-evicted (cache_size=1024)
and full-context conditions using the M3 checkpoint (LoRA + frozen NAMM).

For each training sample:
  1. Forward pass WITH NAMM eviction -> per-token CE loss on answer tokens
     -> backward -> record LoRA gradient L2 norms and retention ratio
  2. Forward pass WITHOUT eviction (full context) -> per-token CE loss
     -> backward -> record LoRA gradient L2 norms

Results are stratified by retention ratio and compared.

Produces four plots in analysis/report_9/:
  loss_stratified.png             -- box plot of per-sample loss by retention stratum
  grad_norms.png                  -- per-layer gradient L2 norms, evicted vs full
  loss_vs_retention.png           -- scatter of loss vs retention ratio
  grad_direction_consistency.png  -- cosine similarity of gradient directions

Usage:
    # Full run (GPU required):
    source activate.sh
    PYTHONPATH=. .venv/bin/python analysis/report_9/generate_plots.py

    # With sample limit:
    PYTHONPATH=. .venv/bin/python analysis/report_9/generate_plots.py --max-samples 50

    # Plot-only (CPU, after data has been extracted):
    .venv/bin/python analysis/report_9/generate_plots.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Repo setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = SCRIPT_DIR
DATA_FILE = OUT_DIR / "gradient_data.json"
REPORT_FILE = OUT_DIR / "_report.md"

logger = logging.getLogger("analysis_9")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── Checkpoint paths ────────────────────────────────────────────────────────
ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M3_LORA_CKPT = ARTIFACTS / "M3_cs1024" / "best_ckpt.pt"
M2_NAMM_CKPT = ARTIFACTS / "M2_cs1024" / "ckpt.pt"

# ── Model / data config ────────────────────────────────────────────────────
RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
MIN_CONDITIONING_LENGTH = 4096
NUM_LAYERS = 16  # LLaMA 3.2-1B has 16 layers
MAX_SAMPLES_DEFAULT = 60


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_and_data() -> tuple:
    """Load the base model, task sampler, and evaluator via Hydra config.

    Returns:
        Tuple of (cfg, device, memory_policy, memory_model,
                  memory_evaluator, task_sampler).
    """
    import torch
    from es_finetuning.device import get_device
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    logger.info("Device: %s", device)

    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)

    task_sampler.filter_by_token_count(
        memory_evaluator.tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, FILTER_ANSWERS_BY_TOKENS)

    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT,
        val_frac=VAL_SPLIT,
        max_conditioning_length=cfg.get("max_conditioning_length", FILTER_BY_TOKENS),
        min_conditioning_length=MIN_CONDITIONING_LENGTH,
        tokenizer=memory_evaluator.tokenizer,
    )

    return (cfg, device, memory_policy, memory_model,
            memory_evaluator, task_sampler)


def load_namm_weights(
    memory_model: Any,
    memory_policy: Any,
    device: str,
    namm_ckpt_path: str,
    prefer_mean: bool = True,
) -> None:
    """Load NAMM scoring network weights from a checkpoint."""
    import torch

    logger.info("Loading NAMM checkpoint: %s", namm_ckpt_path)
    ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
    evo_state = ckpt["evolution_state"]

    if prefer_mean and "mean" in evo_state:
        params_vec = evo_state["mean"]
        param_source = "mean"
    else:
        params_vec = evo_state["best_member"]
        param_source = "best_member"

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
    logger.info("  Loaded NAMM %s (%d params)", param_source, params_vec.shape[0])


def load_lora_weights(
    memory_model: Any,
    lora_ckpt_path: str,
    device: str,
) -> None:
    """Load LoRA adapter weights from a checkpoint."""
    import torch

    logger.info("Loading LoRA checkpoint: %s", lora_ckpt_path)
    ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)

    if not memory_model.has_lora_adapters():
        lora_cfg = ckpt.get("lora_config", {})
        rank = lora_cfg.get("rank", 4)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        memory_model.apply_lora_adapters(
            rank=rank, target_modules=target_modules)

    lora_state = ckpt["lora_state_dict"]
    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_state:
            p.data.copy_(lora_state[n].to(p.device))
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            f"No LoRA weights loaded from {lora_ckpt_path}! "
            "Key format mismatch between checkpoint and model."
        )
    logger.info(
        "  Loaded %d LoRA tensors (best_step=%s, best_val=%s)",
        loaded,
        ckpt.get("best_step", "?"),
        ckpt.get("best_val_score", "?"),
    )


# ── Training sample preparation ────────────────────────────────────────────

def get_train_samples(
    task_sampler: Any,
    tokenizer: Any,
    max_samples: int = MAX_SAMPLES_DEFAULT,
) -> list[dict]:
    """Build tokenised training samples with prompt+answer for CE loss.

    Returns list of dicts with keys:
      input_ids: (1, seq_len) tensor
      labels: (1, seq_len) tensor (-100 for context, token ids for answer)
      task: str
      idx: int
      seq_len: int
      answer_tokens: int  (number of answer tokens)
    """
    import torch

    train_idxs = task_sampler._train_idxs_per_task
    if train_idxs is None:
        logger.error("No train split available")
        return []

    samples: list[dict] = []
    bos_id = getattr(tokenizer, "bos_token_id", None)

    for task_name in sorted(train_idxs.keys()):
        idxs = train_idxs[task_name]
        prompts = task_sampler.lb_prompts_per_task[task_name]
        jsons = task_sampler.lb_jsons_per_task[task_name]

        for idx in idxs:
            if len(samples) >= max_samples:
                break

            prompt_text = prompts[idx]
            json_item = jsons[idx]

            # Get the answer
            answers = json_item.get("answers", json_item.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            answer = answers[0]

            # Build prompt+answer using chat template (matching SFT training)
            user_content = prompt_text
            # If prompts already have chat template applied, use them directly;
            # otherwise, apply the chat template
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                add_generation_prompt=True,
                tokenize=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content},
                 {"role": "assistant", "content": answer}],
                add_generation_prompt=False,
                tokenize=True,
            )

            label_start = len(prompt_ids)
            if label_start >= len(full_ids):
                continue

            # Truncate to avoid OOM
            max_len = FILTER_BY_TOKENS
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            # Build labels: -100 for context tokens, real ids for answer tokens
            labels = [-100] * label_start + full_ids[label_start:]
            if len(labels) > len(full_ids):
                labels = labels[:len(full_ids)]

            n_answer = sum(1 for l in labels if l != -100)
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


# ── Gradient extraction ────────────────────────────────────────────────────

def _get_lora_param_info(memory_model: Any) -> list[dict]:
    """Return list of dicts describing each LoRA parameter tensor.

    Each dict has keys: name, layer_idx (int), module (str like q_proj/v_proj),
    component (str like lora_A/lora_B).
    """
    info = []
    for n, p in memory_model.model.named_parameters():
        if not p.requires_grad:
            continue
        # Typical name: base_model.model.layers.3.self_attn.q_proj.lora_A.default.weight
        parts = n.split(".")
        layer_idx = -1
        module = "unknown"
        component = "unknown"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
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
    """Run forward+backward on one sample, return loss and gradient info.

    Args:
        memory_model: The WrappedLlamaForCausalLM with LoRA.
        sample: Dict with input_ids, labels tensors.
        device: Torch device string.
        apply_memory_policy: If True, run with NAMM eviction active.

    Returns:
        Dict with:
          loss: float (mean CE on answer tokens)
          per_layer_grad_norms: dict[int, float] (layer_idx -> L2 norm)
          per_param_grads: dict[str, list[float]] (param name -> flattened grad)
          retention_ratio: float (only meaningful when apply_memory_policy=True)
        or None on failure.
    """
    import torch
    import torch.nn.functional as F

    input_ids = sample["input_ids"].to(device)
    labels = sample["labels"].to(device)

    # Zero existing gradients
    memory_model.model.zero_grad(set_to_none=True)

    # Reset memory policy state
    if hasattr(memory_model, "memory_policy"):
        if hasattr(memory_model.memory_policy, "initialize_buffers"):
            memory_model.memory_policy.initialize_buffers()
        elif hasattr(memory_model.memory_policy, "reset"):
            memory_model.memory_policy.reset()

    # Enable eval stats recording for retention tracking
    if apply_memory_policy and hasattr(memory_model.memory_policy, "record_eval_stats"):
        memory_model.memory_policy.record_eval_stats = True

    # Ensure batch_idxs set for NAMM
    if apply_memory_policy:
        memory_model.memory_policy.set_params_batch_idxs(
            np.zeros([input_ids.shape[0]], dtype=np.int64))

    seq_len = input_ids.shape[1]

    # Identify answer region for two-phase processing
    answer_mask = (labels[0] != -100)
    if answer_mask.any():
        answer_start = answer_mask.nonzero(as_tuple=True)[0][0].item()
    else:
        return None

    # Two-phase forward (matching trainer pattern for NAMM mode)
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

        # Compute retention ratio from KV cache
        if apply_memory_policy and past_key_values is not None:
            # Count retained tokens vs input tokens
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

    # Chunked cross-entropy (matching trainer)
    lm_head = memory_model.lm_head
    ce_chunk_size = 512
    ce_seq_len = shift_hidden.shape[1]
    total_loss = torch.tensor(0.0, device=device)
    n_tokens = (shift_labels != -100).sum()

    if n_tokens == 0:
        return None

    for i in range(0, ce_seq_len, ce_chunk_size):
        chunk_h = shift_hidden[:, i:i + ce_chunk_size, :]
        chunk_logits = lm_head(chunk_h).float()
        chunk_labels = shift_labels[:, i:i + ce_chunk_size].contiguous().view(-1)
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.view(-1, chunk_logits.size(-1)),
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        del chunk_logits

    loss = total_loss / n_tokens.clamp(min=1)

    # Backward
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

            # Accumulate per-layer norm (sum of squares, then sqrt at end)
            if layer_idx not in per_layer_grad_norms:
                per_layer_grad_norms[layer_idx] = 0.0
            per_layer_grad_norms[layer_idx] += grad_l2 ** 2

            per_param_grads[name] = grad_flat.tolist()

    # Convert sum-of-squares to L2 norms per layer
    for layer_idx in per_layer_grad_norms:
        per_layer_grad_norms[layer_idx] = float(
            np.sqrt(per_layer_grad_norms[layer_idx]))

    loss_val = float(loss.item())

    # Disable eval stats recording
    if apply_memory_policy and hasattr(memory_model.memory_policy, "record_eval_stats"):
        memory_model.memory_policy.record_eval_stats = False

    # Cleanup
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


# ── Main extraction pipeline ──────────────────────────────────────────────

def run_extraction(args: argparse.Namespace) -> dict:
    """Run the full gradient extraction pipeline."""
    import torch

    # Validate checkpoints
    for path, name in [
        (M3_LORA_CKPT, "M3 LoRA"),
        (M2_NAMM_CKPT, "M2 NAMM"),
    ]:
        if not path.exists():
            logger.error("%s checkpoint not found: %s", name, path)
            logger.error("Run scripts/download_artifacts.py to download checkpoints.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Analysis 9: Gradient Flow and Loss Attribution Under Eviction")
    logger.info("=" * 60)

    # Load model infrastructure
    (cfg, device, memory_policy, memory_model, memory_evaluator,
     task_sampler) = load_model_and_data()

    tokenizer = memory_evaluator.tokenizer

    # Load NAMM weights
    load_namm_weights(memory_model, memory_policy, device, str(M2_NAMM_CKPT))

    # Apply LoRA and load M3 weights
    load_lora_weights(memory_model, str(M3_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16, device=device)

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
        task_sampler, tokenizer, max_samples=args.max_samples)
    if not train_samples:
        logger.error("No training samples available")
        sys.exit(1)

    # ── Pass 1: WITH NAMM eviction ──────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pass 1: Forward+backward WITH NAMM eviction (cache_size=%d)", CACHE_SIZE)
    logger.info("=" * 60)

    evicted_results: list[dict] = []
    for i, sample in enumerate(train_samples):
        try:
            memory_model.train()  # Enable grad computation
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

    # ── Pass 2: WITHOUT eviction (full context) ─────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pass 2: Forward+backward WITHOUT eviction (full context)")
    logger.info("=" * 60)

    # Swap to Recency policy (no eviction)
    from namm.policy.base import Recency
    original_policy = memory_model.memory_policy
    recency_policy = Recency(cache_size=None)
    memory_model.swap_memory_policy(recency_policy)

    # Disable split processing for full context
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

    # ── Build output data ───────────────────────────────────────────────
    # Strip per_param_grads for JSON (too large); keep only what we need
    # for cosine similarity. Compute cosine similarity now.
    cosine_sims_per_layer: dict[str, list[float]] = {}
    for ev, fu in zip(evicted_results, full_results):
        if ev["idx"] != fu["idx"] or ev["task"] != fu["task"]:
            continue
        for param_name in ev["per_param_grads"]:
            if param_name not in fu["per_param_grads"]:
                continue
            ev_grad = np.array(ev["per_param_grads"][param_name], dtype=np.float32)
            fu_grad = np.array(fu["per_param_grads"][param_name], dtype=np.float32)
            ev_norm = np.linalg.norm(ev_grad)
            fu_norm = np.linalg.norm(fu_grad)
            if ev_norm > 1e-12 and fu_norm > 1e-12:
                cos_sim = float(np.dot(ev_grad, fu_grad) / (ev_norm * fu_norm))
            else:
                cos_sim = 0.0

            # Extract layer index from param name
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

    all_data = {
        "evicted": evicted_results,
        "full_context": full_results,
        "cosine_sims_per_layer": cosine_sims_per_layer,
        "cache_size": CACHE_SIZE,
        "max_samples": args.max_samples,
    }

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info("Saving gradient data to %s...", DATA_FILE)
    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info("  Saved (%.1f KB)", DATA_FILE.stat().st_size / 1024)

    return all_data


# ── Plotting (CPU) ─────────────────────────────────────────────────────────

def load_saved_data() -> dict:
    """Load previously extracted gradient data."""
    if not DATA_FILE.exists():
        logger.error("Data file not found: %s", DATA_FILE)
        logger.error("Run without --plot-only first to extract data.")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    logger.info("Loaded gradient data from %s", DATA_FILE)
    logger.info("  evicted: %d samples, full_context: %d samples",
                len(data["evicted"]), len(data["full_context"]))
    return data


def plot_loss_stratified(data: dict, out_dir: Path) -> None:
    """Plot 1: Box plot of per-sample loss, stratified by retention ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evicted = data["evicted"]
    full = data["full_context"]

    if not evicted or not full:
        logger.info("  Skipping loss_stratified (no data)")
        return

    # Compute median retention ratio for stratification
    retention_ratios = [s["retention_ratio"] for s in evicted]
    median_retention = float(np.median(retention_ratios))

    high_ret_losses = [s["loss"] for s in evicted if s["retention_ratio"] >= median_retention]
    low_ret_losses = [s["loss"] for s in evicted if s["retention_ratio"] < median_retention]
    full_losses = [s["loss"] for s in full]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [full_losses, high_ret_losses, low_ret_losses]
    box_labels = [
        f"Full context\n(n={len(full_losses)})",
        f"Evicted, high retention\n(>= {median_retention:.2f}, n={len(high_ret_losses)})",
        f"Evicted, low retention\n(< {median_retention:.2f}, n={len(low_ret_losses)})",
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (vals, color) in enumerate(zip(box_data, colors)):
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.ones(len(vals)) * (i + 1) + jitter,
            vals, alpha=0.4, s=15, color=color, zorder=3,
        )

    ax.set_ylabel("Cross-Entropy Loss (answer tokens)", fontsize=12)
    ax.set_title(
        "Per-Sample Loss Stratified by Retention Ratio\n"
        f"(median retention = {median_retention:.2f}, cache_size={data.get('cache_size', '?')})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = out_dir / "loss_stratified.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_grad_norms(data: dict, out_dir: Path) -> None:
    """Plot 2: Per-layer gradient L2 norms, evicted vs full context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evicted = data["evicted"]
    full = data["full_context"]

    if not evicted or not full:
        logger.info("  Skipping grad_norms (no data)")
        return

    # Collect per-layer norms across samples
    evicted_layer_norms: dict[int, list[float]] = {}
    full_layer_norms: dict[int, list[float]] = {}

    for s in evicted:
        for layer_str, norm_val in s["per_layer_grad_norms"].items():
            layer_idx = int(layer_str)
            evicted_layer_norms.setdefault(layer_idx, []).append(norm_val)

    for s in full:
        for layer_str, norm_val in s["per_layer_grad_norms"].items():
            layer_idx = int(layer_str)
            full_layer_norms.setdefault(layer_idx, []).append(norm_val)

    layers = sorted(set(evicted_layer_norms.keys()) | set(full_layer_norms.keys()))
    if not layers:
        logger.info("  Skipping grad_norms (no layer data)")
        return

    ev_means = [np.mean(evicted_layer_norms.get(l, [0])) for l in layers]
    ev_stds = [np.std(evicted_layer_norms.get(l, [0])) for l in layers]
    fu_means = [np.mean(full_layer_norms.get(l, [0])) for l in layers]
    fu_stds = [np.std(full_layer_norms.get(l, [0])) for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, ev_means, width, yerr=ev_stds,
           label="Evicted (NAMM cs1024)", color="#d62728",
           edgecolor="white", capsize=3, alpha=0.8)
    ax.bar(x + width / 2, fu_means, width, yerr=fu_stds,
           label="Full context", color="#1f77b4",
           edgecolor="white", capsize=3, alpha=0.8)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("LoRA Gradient L2 Norm", fontsize=12)
    ax.set_title("Per-Layer LoRA Gradient Norms\n"
                 "(evicted vs full context, M3 checkpoint)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = out_dir / "grad_norms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_loss_vs_retention(data: dict, out_dir: Path) -> None:
    """Plot 3: Scatter of per-sample loss vs retention ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    evicted = data["evicted"]
    if not evicted:
        logger.info("  Skipping loss_vs_retention (no data)")
        return

    retentions = [s["retention_ratio"] for s in evicted]
    losses = [s["loss"] for s in evicted]
    tasks = [s["task"] for s in evicted]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by task
    unique_tasks = sorted(set(tasks))
    task_colors = {t: c for t, c in zip(
        unique_tasks,
        plt.cm.Set2(np.linspace(0, 1, max(len(unique_tasks), 1)))
    )}

    for task in unique_tasks:
        mask = [t == task for t in tasks]
        task_ret = [r for r, m in zip(retentions, mask) if m]
        task_loss = [l for l, m in zip(losses, mask) if m]
        ax.scatter(
            task_ret, task_loss, label=task, alpha=0.7, s=40,
            color=task_colors[task], edgecolors="white", linewidth=0.5,
        )

    # Add trend line
    if len(retentions) >= 3:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            retentions, losses)
        x_line = np.linspace(min(retentions), max(retentions), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "--", color="gray", alpha=0.7,
                label=f"OLS: r={r_value:.3f}, p={p_value:.3f}")

    ax.set_xlabel("Retention Ratio", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss (answer tokens)", fontsize=12)
    ax.set_title(
        "Loss vs Retention Ratio Under NAMM Eviction\n"
        f"(M3 checkpoint, cache_size={data.get('cache_size', '?')})",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "loss_vs_retention.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_grad_direction_consistency(data: dict, out_dir: Path) -> None:
    """Plot 4: Cosine similarity between evicted and full-context gradients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosine_sims = data.get("cosine_sims_per_layer", {})
    if not cosine_sims:
        logger.info("  Skipping grad_direction_consistency (no data)")
        return

    # Parse layer indices and sort
    layer_data: list[tuple[int, list[float]]] = []
    for layer_str, sims in cosine_sims.items():
        try:
            layer_idx = int(layer_str)
        except ValueError:
            continue
        layer_data.append((layer_idx, sims))
    layer_data.sort(key=lambda x: x[0])

    if not layer_data:
        logger.info("  Skipping grad_direction_consistency (no layer data)")
        return

    layers = [ld[0] for ld in layer_data]
    means = [np.mean(ld[1]) for ld in layer_data]
    stds = [np.std(ld[1]) for ld in layer_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of mean cosine similarity per layer
    x = np.arange(len(layers))
    colors = ["#2ca02c" if m > 0.9 else "#ff7f0e" if m > 0.5 else "#d62728"
              for m in means]
    ax1.bar(x, means, yerr=stds, color=colors, edgecolor="white",
            capsize=3, alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect alignment")
    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax1.set_title("Gradient Direction Consistency\n(evicted vs full context)", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.set_ylim(-0.2, 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: box plot per layer
    box_data = [ld[1] for ld in layer_data]
    bp = ax2.boxplot(box_data, labels=[str(l) for l in layers],
                     patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.6)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title("Per-Layer Distribution of\nGradient Cosine Similarity", fontsize=13)
    ax2.set_ylim(-0.2, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Gradient Direction Alignment: Evicted vs Full Context\n"
        "(cos=1.0 means eviction does not change gradient direction)",
        fontsize=14, y=1.04,
    )
    fig.tight_layout()

    path = out_dir / "grad_direction_consistency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


def generate_plots(data: dict) -> dict:
    """Generate all plots and return summary statistics."""
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(OUT_DIR, exist_ok=True)

    evicted = data.get("evicted", [])
    full = data.get("full_context", [])
    cosine_sims = data.get("cosine_sims_per_layer", {})

    # Compute summary stats
    summary: dict[str, Any] = {}

    if evicted:
        ev_losses = [s["loss"] for s in evicted]
        retentions = [s["retention_ratio"] for s in evicted]
        summary["evicted_loss_mean"] = float(np.mean(ev_losses))
        summary["evicted_loss_std"] = float(np.std(ev_losses))
        summary["retention_mean"] = float(np.mean(retentions))
        summary["retention_std"] = float(np.std(retentions))
        summary["retention_median"] = float(np.median(retentions))
        summary["n_evicted_samples"] = len(evicted)

    if full:
        fu_losses = [s["loss"] for s in full]
        summary["full_loss_mean"] = float(np.mean(fu_losses))
        summary["full_loss_std"] = float(np.std(fu_losses))
        summary["n_full_samples"] = len(full)

    if evicted and full:
        summary["loss_increase_mean"] = float(
            np.mean(ev_losses) - np.mean(fu_losses))
        summary["loss_increase_pct"] = float(
            (np.mean(ev_losses) - np.mean(fu_losses)) / max(np.mean(fu_losses), 1e-8) * 100)

    if cosine_sims:
        all_sims = []
        for sims in cosine_sims.values():
            all_sims.extend(sims)
        summary["cosine_sim_mean"] = float(np.mean(all_sims))
        summary["cosine_sim_std"] = float(np.std(all_sims))

    logger.info("")
    logger.info("Summary statistics:")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # Generate plots
    logger.info("")
    logger.info("Generating plots...")
    plot_loss_stratified(data, OUT_DIR)
    plot_grad_norms(data, OUT_DIR)
    plot_loss_vs_retention(data, OUT_DIR)
    plot_grad_direction_consistency(data, OUT_DIR)

    logger.info("")
    logger.info("All plots saved to: %s", OUT_DIR)

    return summary


def write_report(summary: dict, data: dict) -> None:
    """Write _report.md with analysis findings."""
    evicted = data.get("evicted", [])
    full = data.get("full_context", [])

    lines = [
        "# Analysis 9: Gradient Flow and Loss Attribution Under Eviction",
        "",
        "## Overview",
        "",
        "This analysis compares gradient flow and per-sample loss between",
        "NAMM-evicted (cache_size=1024) and full-context conditions using",
        "the M3 checkpoint (LoRA + frozen NAMM). We perform instrumented",
        "evaluation passes over training data, computing per-token CE loss",
        "on answer tokens and recording LoRA gradient norms.",
        "",
        "## Results Summary",
        "",
    ]

    if summary:
        lines.append(f"- **Samples processed**: {summary.get('n_evicted_samples', '?')} "
                      f"evicted, {summary.get('n_full_samples', '?')} full context")
        lines.append(f"- **Mean retention ratio**: "
                      f"{summary.get('retention_mean', 0):.4f} +/- "
                      f"{summary.get('retention_std', 0):.4f} "
                      f"(median: {summary.get('retention_median', 0):.4f})")
        lines.append(f"- **Evicted loss**: "
                      f"{summary.get('evicted_loss_mean', 0):.4f} +/- "
                      f"{summary.get('evicted_loss_std', 0):.4f}")
        lines.append(f"- **Full-context loss**: "
                      f"{summary.get('full_loss_mean', 0):.4f} +/- "
                      f"{summary.get('full_loss_std', 0):.4f}")
        lines.append(f"- **Loss increase from eviction**: "
                      f"{summary.get('loss_increase_mean', 0):.4f} "
                      f"({summary.get('loss_increase_pct', 0):.1f}%)")
        lines.append(f"- **Gradient direction consistency**: "
                      f"mean cosine similarity = "
                      f"{summary.get('cosine_sim_mean', 0):.4f} +/- "
                      f"{summary.get('cosine_sim_std', 0):.4f}")

    lines.extend([
        "",
        "## Plots",
        "",
        "### 1. Loss Stratified by Retention Ratio",
        "![loss_stratified](loss_stratified.png)",
        "",
        "Box plot comparing per-sample CE loss across three conditions:",
        "full context, high-retention eviction, and low-retention eviction.",
        "",
        "### 2. Per-Layer Gradient Norms",
        "![grad_norms](grad_norms.png)",
        "",
        "Per-layer LoRA gradient L2 norms under eviction vs full context.",
        "Differences indicate which layers are most affected by eviction.",
        "",
        "### 3. Loss vs Retention Ratio",
        "![loss_vs_retention](loss_vs_retention.png)",
        "",
        "Scatter plot showing the relationship between retention ratio",
        "and CE loss. A negative trend would indicate that more aggressive",
        "eviction increases loss.",
        "",
        "### 4. Gradient Direction Consistency",
        "![grad_direction_consistency](grad_direction_consistency.png)",
        "",
        "Per-layer cosine similarity between gradient directions computed",
        "under eviction vs full context. Values near 1.0 indicate that",
        "eviction does not substantially alter the gradient signal.",
        "",
        "## Interpretation",
        "",
    ])

    # Add data-driven interpretation
    if summary:
        cos_mean = summary.get("cosine_sim_mean", 0)
        loss_pct = summary.get("loss_increase_pct", 0)

        if cos_mean > 0.9:
            lines.append(
                "Gradient directions are highly consistent between evicted and "
                "full-context conditions (mean cosine similarity > 0.9), suggesting "
                "that NAMM eviction preserves the overall gradient signal direction.")
        elif cos_mean > 0.5:
            lines.append(
                "Gradient directions show moderate consistency between conditions "
                "(mean cosine similarity 0.5-0.9), indicating partial disruption "
                "of gradient flow by eviction.")
        else:
            lines.append(
                "Gradient directions differ substantially between evicted and "
                "full-context conditions (mean cosine similarity < 0.5), suggesting "
                "significant gradient signal distortion from eviction.")

        lines.append("")

        if abs(loss_pct) < 5:
            lines.append(
                f"The loss increase from eviction is modest ({loss_pct:.1f}%), "
                "suggesting the evicted KV cache retains the most task-relevant "
                "information for answer prediction.")
        elif loss_pct > 0:
            lines.append(
                f"Eviction increases loss by {loss_pct:.1f}%, indicating that "
                "some answer-relevant information is lost during cache compression.")
        else:
            lines.append(
                f"Interestingly, eviction decreases loss by {abs(loss_pct):.1f}%, "
                "which may indicate that removing irrelevant context reduces "
                "distraction and improves answer-token prediction.")

    lines.extend(["", ""])

    report_text = "\n".join(lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    logger.info("Report written to %s", REPORT_FILE)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysis 9: Gradient Flow and Loss Attribution Under Eviction")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip inference; generate plots from saved data")
    parser.add_argument(
        "--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
        help=f"Max training samples to process (default: {MAX_SAMPLES_DEFAULT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot_only:
        data = load_saved_data()
    else:
        data = run_extraction(args)

    summary = generate_plots(data)
    write_report(summary, data)

    logger.info("")
    logger.info("Done. All outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
