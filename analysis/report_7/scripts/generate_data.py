#!/usr/bin/env python3
"""Report 7 -- Representation Similarity (CKA) data generation.

GPU required: loads base model, M1, and M3, runs forward passes to extract
hidden states, then computes linear CKA for all pairs:
  M1 vs M2, M1 vs M3, M2 vs M3.

M2 = base model (no LoRA). M1 = LoRA full-context. M3 = LoRA eviction-aware.

Uses balanced sampling: SAMPLES_PER_TASK from each of 5 tasks across all
splits (train+val+test).

Saves to: analysis/report_7/data/maskfix_data.npz

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/report_7/scripts/generate_data.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig

os.environ.setdefault("HF_HOME", ".hf_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from utils.hydra_helpers import LlamaCompatModel  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "maskfix_data.npz"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
M1_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M1/best_ckpt.pt"
M3_MASKFIX_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M3_cs1024_maskfix/best_ckpt.pt"

NUM_LAYERS = 16
SAMPLES_PER_TASK = 73  # balanced across 5 tasks (min available)
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
MIN_CONDITIONING_LENGTH = 4096
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024


def load_base_model(device: str = "cuda"):
    model = LlamaCompatModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model = model.to(device).eval()
    return model


def load_model_with_lora(ckpt_path: Path, device: str = "cuda"):
    model = LlamaCompatModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    lora_cfg = ckpt["lora_config"]
    lora_sd = ckpt["lora_state_dict"]
    config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=0.0,
    )
    model = PeftModel(model, config)
    remapped = {
        k.replace("base_model.model.layers.",
                   "base_model.model.model.layers."): v
        for k, v in lora_sd.items()
    }
    missing, _ = model.load_state_dict(remapped, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        raise RuntimeError(f"LoRA keys missing: {lora_missing[:5]}")
    model = model.merge_and_unload()
    model = model.to(device).eval()
    return model


def get_prompts(task_sampler: Any, tokenizer: Any,
                per_task: int = SAMPLES_PER_TASK) -> list[dict]:
    """Get balanced prompts from all splits (train+val+test)."""
    prompts = []
    for task_name in sorted(task_sampler.lb_prompts_per_task.keys()):
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        all_idxs: list[int] = []
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
    logger.info("Prepared %d prompts (%d per task, seq_len range: %d-%d)",
                len(prompts), per_task,
                min(p["seq_len"] for p in prompts) if prompts else 0,
                max(p["seq_len"] for p in prompts) if prompts else 0)
    return prompts


def build_task_sampler():
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_task_sampler
    cfg = load_hydra_config(RUN_CONFIG, extra_overrides=[
        f"cache_size={CACHE_SIZE}", f"max_memory_length={CACHE_SIZE}"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    task_sampler.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT, val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=MIN_CONDITIONING_LENGTH,
        tokenizer=tokenizer)
    return task_sampler, tokenizer


@torch.no_grad()
def extract_hidden(model, prompts, device="cuda"):
    all_hidden: list[list[torch.Tensor]] = []
    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        out = model(input_ids, output_hidden_states=True)
        hidden_pooled = []
        for hs in out.hidden_states:
            hidden_pooled.append(hs[0].float().mean(dim=0).cpu())
        all_hidden.append(hidden_pooled)
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Sample %d/%d done (seq_len=%d)",
                         i + 1, len(prompts), input_ids.shape[1])
        del out
        torch.cuda.empty_cache()
    logger.info("  Extracted %d samples", len(all_hidden))
    return all_hidden


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X
    num = (YTX ** 2).sum()
    denom = torch.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
    return (num / denom).item() if denom > 1e-10 else 0.0


def compute_cka_from_hidden(hidden_a, hidden_b, label=""):
    n_layers_total = len(hidden_a[0])
    n_samples = len(hidden_a)

    layer_cka = []
    for layer_idx in range(n_layers_total):
        reps_a = torch.stack([hidden_a[s][layer_idx] for s in range(n_samples)])
        reps_b = torch.stack([hidden_b[s][layer_idx] for s in range(n_samples)])
        cka = compute_cka(reps_a, reps_b)
        layer_cka.append(cka)

    cross_cka = np.zeros((n_layers_total, n_layers_total))
    for i in range(n_layers_total):
        reps_a_i = torch.stack([hidden_a[s][i] for s in range(n_samples)])
        for j in range(n_layers_total):
            reps_b_j = torch.stack([hidden_b[s][j] for s in range(n_samples)])
            cross_cka[i, j] = compute_cka(reps_a_i, reps_b_j)
        if (i + 1) % 5 == 0:
            logger.info("    %s cross-CKA row %d/%d done",
                         label, i, n_layers_total)

    return layer_cka, cross_cka


def compute():
    if OUT_PATH.exists():
        logger.info("Data already exists: %s — skipping", OUT_PATH)
        return

    for path, label in [(M1_CKPT, "M1"), (M3_MASKFIX_CKPT, "M3")]:
        if not path.exists():
            logger.error("Checkpoint not found: %s (%s)", path, label)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== Building task sampler ===")
    task_sampler, tokenizer = build_task_sampler()
    prompts = get_prompts(task_sampler, tokenizer)

    logger.info("=== Loading M2 (base model) ===")
    m2_model = load_base_model(device)
    m2_hidden = extract_hidden(m2_model, prompts, device)
    del m2_model
    torch.cuda.empty_cache()

    logger.info("=== Loading M1 ===")
    m1_model = load_model_with_lora(M1_CKPT, device)
    m1_hidden = extract_hidden(m1_model, prompts, device)
    del m1_model
    torch.cuda.empty_cache()

    logger.info("=== Loading M3 ===")
    m3_model = load_model_with_lora(M3_MASKFIX_CKPT, device)
    m3_hidden = extract_hidden(m3_model, prompts, device)
    del m3_model
    torch.cuda.empty_cache()

    logger.info("=== CKA: M1 vs M3 ===")
    cka_m1_m3, cross_m1_m3 = compute_cka_from_hidden(
        m1_hidden, m3_hidden, "M1vsM3")

    logger.info("=== CKA: M1 vs M2 ===")
    cka_m1_m2, cross_m1_m2 = compute_cka_from_hidden(
        m1_hidden, m2_hidden, "M1vsM2")

    logger.info("=== CKA: M2 vs M3 ===")
    cka_m2_m3, cross_m2_m3 = compute_cka_from_hidden(
        m2_hidden, m3_hidden, "M2vsM3")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(str(OUT_PATH),
             layer_cka_maskfix=np.array(cka_m1_m3),
             cross_cka_maskfix=cross_m1_m3,
             layer_cka_m1_m2=np.array(cka_m1_m2),
             cross_cka_m1_m2=cross_m1_m2,
             layer_cka_m2_m3=np.array(cka_m2_m3),
             cross_cka_m2_m3=cross_m2_m3)
    logger.info("Saved %s", OUT_PATH)


if __name__ == "__main__":
    compute()
