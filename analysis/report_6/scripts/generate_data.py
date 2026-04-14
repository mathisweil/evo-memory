#!/usr/bin/env python3
"""Report 6 -- Token Importance Alignment data generation.

Requires NAMM infrastructure (GPU).  Extracts NAMM scores and attention
weights for M1 and M3, computes per-layer Spearman correlation and
eviction regret.

Saves to: analysis/report_6/data/maskfix_alignment_data.json

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/report_6/scripts/generate_data.py
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HOME", ".hf_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "maskfix_alignment_data.json"

ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M2_NAMM_MASKFIX_CKPT = ARTIFACTS / "M2_cs1024_maskfix" / "ckpt.pt"
M3_MASKFIX_LORA_CKPT = ARTIFACTS / "M3_cs1024_maskfix" / "best_ckpt.pt"

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
MIN_CONDITIONING_LENGTH = 4096
SAMPLES_PER_TASK = 73  # balanced across 5 tasks (min available)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model_and_data(device: str = "cuda"):
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
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    task_sampler.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT, val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=MIN_CONDITIONING_LENGTH,
        tokenizer=tokenizer,
    )
    return cfg, memory_policy, memory_model, memory_evaluator, task_sampler, tokenizer


def load_namm_weights(memory_model, memory_policy, ckpt_path, device="cuda"):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    evo = ckpt["evolution_state"]
    params = evo.get("mean", evo["best_member"]).unsqueeze(0).to(device)
    memory_model.set_memory_params(params)
    prefix = "stored_buffers_to_save."
    bufs = {k[len(prefix):]: v.to(device)
            for k, v in evo.items() if k.startswith(prefix)}
    if bufs:
        memory_model.load_buffers_dict(buffers_dict=bufs)
    memory_policy.set_params_batch_idxs(np.zeros([1]))


def load_lora_weights(memory_model, ckpt_path, device="cuda"):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    lora_sd = ckpt["lora_state_dict"]
    if not memory_model.has_lora_adapters():
        memory_model.apply_lora_adapters(
            rank=lora_cfg.get("rank", 8),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]))
    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_sd:
            p.data.copy_(lora_sd[n].to(p.device))
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No LoRA weights loaded from {ckpt_path}")
    logger.info("  LoRA loaded (%d tensors)", loaded)


def reset_lora_weights(memory_model):
    for _, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()


def reset_policy_state(memory_model):
    if hasattr(memory_model.memory_policy, "initialize_buffers"):
        memory_model.memory_policy.initialize_buffers()
    elif hasattr(memory_model.memory_policy, "reset"):
        memory_model.memory_policy.reset()


def get_prompts(task_sampler, tokenizer, per_task=SAMPLES_PER_TASK):
    """Get balanced prompts from all splits (train+val+test)."""
    import torch
    prompts = []
    for task_name in sorted(task_sampler.lb_prompts_per_task.keys()):
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        all_idxs = []
        for split_attr in ("_train_idxs_per_task", "_val_idxs_per_task",
                           "_test_idxs_per_task"):
            split_dict = getattr(task_sampler, split_attr, None)
            if split_dict and task_name in split_dict:
                all_idxs.extend(split_dict[task_name])
        for idx in all_idxs[:per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=FILTER_BY_TOKENS)
            prompts.append({
                "input_ids": ids["input_ids"],
                "task": task_name,
                "idx": int(idx),
                "seq_len": ids["input_ids"].shape[1],
            })
    return prompts


# ---------------------------------------------------------------------------
# Alignment extraction
# ---------------------------------------------------------------------------

def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def extract_alignment_data(memory_model, memory_policy, prompts, device="cuda"):
    import torch

    results = []
    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        reset_policy_state(memory_model)
        if hasattr(memory_policy, "record_eval_stats"):
            memory_policy.record_eval_stats = False

        with torch.no_grad():
            outputs = memory_model(
                input_ids=input_ids, use_cache=True,
                output_attentions=True, return_dict=True,
                apply_memory_policy=False,
            )

        attention_per_kv = []
        if outputs.attentions is not None:
            for layer_attn in outputs.attentions:
                per_kv = layer_attn[0].float().mean(dim=0).mean(dim=0)
                attention_per_kv.append(per_kv.cpu().numpy())

        past_kv = outputs.past_key_values
        if hasattr(past_kv, "to_legacy_cache"):
            past_kv = past_kv.to_legacy_cache()
        elif not isinstance(past_kv, tuple):
            past_kv = tuple(
                (past_kv.key_cache[j], past_kv.value_cache[j])
                for j in range(len(past_kv.key_cache)))

        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        reset_policy_state(memory_model)
        try:
            _, analysis_dicts = memory_policy.update_cache(
                past_key_values=past_kv, num_new_tokens=seq_len,
                attn_weights_list=(outputs.attentions
                                   if memory_policy.requires_attn_scores else []),
                attention_mask=attention_mask, position_ids=position_ids,
                analyze=True,
            )
        except Exception as e:
            logger.warning("  Sample %d analyze failed: %s", i, e)
            del outputs
            torch.cuda.empty_cache()
            continue

        sample_layers = []
        for layer_id, ad in enumerate(analysis_dicts):
            token_scores = ad.get("token_scores")
            retained_idxs = ad.get("retained_idxs")
            scores_np = (token_scores[0].float().detach().mean(dim=0).cpu().numpy()
                         if token_scores is not None else np.array([]))
            retained = (retained_idxs[0, 0].detach().cpu().numpy().tolist()
                        if retained_idxs is not None else list(range(seq_len)))
            evicted = sorted(set(range(len(scores_np))) - set(retained))
            attn = (attention_per_kv[layer_id]
                    if layer_id < len(attention_per_kv) else np.array([]))
            min_len = min(len(scores_np), len(attn))
            rho = (compute_spearman(scores_np[:min_len], attn[:min_len])
                   if min_len >= 3 else float("nan"))
            if len(evicted) > 0 and len(attn) > 0:
                ev_valid = [e for e in evicted if e < len(attn)]
                total_regret = float(np.sum(attn[ev_valid])) if ev_valid else 0.0
                mean_regret = float(np.mean(attn[ev_valid])) if ev_valid else 0.0
            else:
                total_regret = mean_regret = 0.0
            sample_layers.append({
                "layer_id": layer_id, "spearman_rho": rho,
                "total_regret": total_regret, "mean_regret": mean_regret,
                "num_tokens": len(scores_np),
                "num_retained": len(retained), "num_evicted": len(evicted),
            })

        results.append({
            "task": p["task"], "idx": p["idx"],
            "seq_len": seq_len, "layers": sample_layers,
        })
        logger.info("  Sample %d/%d (%s) done", i + 1, len(prompts), p["task"])
        del outputs, past_kv
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute():
    import torch

    if OUT_PATH.exists():
        logger.info("Data already exists: %s — skipping", OUT_PATH)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, memory_policy, memory_model, _, task_sampler, tokenizer = \
        build_model_and_data(device)
    prompts = get_prompts(task_sampler, tokenizer)

    load_namm_weights(memory_model, memory_policy, M2_NAMM_MASKFIX_CKPT, device)

    results: dict = {"cache_size": CACHE_SIZE, "conditions": {}}

    # M1
    logger.info("=== M1: LoRA full-context + NAMM scoring ===")
    load_lora_weights(memory_model, M1_LORA_CKPT, device)
    memory_model.to(dtype=torch.bfloat16)
    m1_data = extract_alignment_data(memory_model, memory_policy, prompts, device)
    results["conditions"]["M1"] = {"samples": m1_data}

    # M3
    logger.info("=== M3: LoRA eviction-aware + NAMM scoring ===")
    reset_lora_weights(memory_model)
    load_lora_weights(memory_model, M3_MASKFIX_LORA_CKPT, device)
    memory_model.to(dtype=torch.bfloat16)
    m3_data = extract_alignment_data(memory_model, memory_policy, prompts, device)
    results["conditions"]["M3"] = {"samples": m3_data}

    # M2 (NAMM only, no LoRA)
    logger.info("=== M2: No LoRA + NAMM scoring ===")
    reset_lora_weights(memory_model)
    memory_model.to(dtype=torch.bfloat16)
    m2_data = extract_alignment_data(memory_model, memory_policy, prompts, device)
    results["conditions"]["M2"] = {"samples": m2_data}

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", OUT_PATH)

    del memory_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    compute()
