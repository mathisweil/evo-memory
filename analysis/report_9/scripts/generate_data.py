#!/usr/bin/env python3
"""Report 9 -- Gradient Flow and Loss Attribution Under Eviction.

Requires NAMM infrastructure (GPU).  Runs forward+backward passes on M3
with and without NAMM eviction, collecting per-sample loss, per-layer
gradient norms, and gradient direction cosine similarity.

Saves to: analysis/report_9/data/maskfix_gradient_data.json

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/report_9/scripts/generate_data.py
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
import torch
import torch.nn.functional as F

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
OUT_PATH = DATA_DIR / "maskfix_gradient_data.json"

ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
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
SAMPLES_PER_TASK = 51  # balanced across 5 tasks (min train available)


# ---------------------------------------------------------------------------
# Model setup helpers
# ---------------------------------------------------------------------------

def build_model_and_data(device="cuda"):
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    cfg = load_hydra_config(RUN_CONFIG, extra_overrides=[
        f"cache_size={CACHE_SIZE}", f"max_memory_length={CACHE_SIZE}"])
    with torch.no_grad():
        mp, mm, me, _, _ = make_eval_model(cfg=cfg)
    mm.to(device).eval()
    tokenizer = me.tokenizer
    ts = make_task_sampler(cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)
    ts.filter_by_token_count(tokenizer, FILTER_BY_TOKENS)
    ts.filter_answers_by_token_count(tokenizer, FILTER_ANSWERS_BY_TOKENS)
    ts.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT, val_frac=VAL_SPLIT,
        max_conditioning_length=FILTER_BY_TOKENS,
        min_conditioning_length=MIN_CONDITIONING_LENGTH, tokenizer=tokenizer)
    return cfg, mp, mm, me, ts, tokenizer


def load_namm_weights(mm, mp, ckpt_path, device="cuda"):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    evo = ckpt["evolution_state"]
    params = evo.get("mean", evo["best_member"]).unsqueeze(0).to(device)
    mm.set_memory_params(params)
    prefix = "stored_buffers_to_save."
    bufs = {k[len(prefix):]: v.to(device) for k, v in evo.items()
            if k.startswith(prefix)}
    if bufs:
        mm.load_buffers_dict(buffers_dict=bufs)
    mp.set_params_batch_idxs(np.zeros([1]))


def load_lora_weights(mm, ckpt_path, device="cuda"):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    lora_sd = ckpt["lora_state_dict"]
    if not mm.has_lora_adapters():
        mm.apply_lora_adapters(
            rank=lora_cfg.get("rank", 8),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]))
    loaded = 0
    for n, p in mm.model.named_parameters():
        if p.requires_grad and n in lora_sd:
            p.data.copy_(lora_sd[n].to(p.device))
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No LoRA weights loaded from {ckpt_path}")


def reset_policy_state(mm):
    if hasattr(mm.memory_policy, "initialize_buffers"):
        mm.memory_policy.initialize_buffers()
    elif hasattr(mm.memory_policy, "reset"):
        mm.memory_policy.reset()


# ---------------------------------------------------------------------------
# Training samples
# ---------------------------------------------------------------------------

def get_train_samples(task_sampler, tokenizer, per_task=SAMPLES_PER_TASK):
    """Get balanced training samples (per_task from each of 5 tasks)."""
    train_idxs = task_sampler._train_idxs_per_task
    if train_idxs is None:
        return []
    samples = []
    for task_name in sorted(train_idxs.keys()):
        idxs = train_idxs[task_name]
        prompts = task_sampler.lb_prompts_per_task[task_name]
        jsons = task_sampler.lb_jsons_per_task[task_name]
        task_count = 0
        for idx in idxs:
            if task_count >= per_task:
                break
            answers = jsons[idx].get("answers", jsons[idx].get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            answer = answers[0]
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[idx]}],
                add_generation_prompt=True, tokenize=True)
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[idx]},
                 {"role": "assistant", "content": answer}],
                add_generation_prompt=False, tokenize=True)
            label_start = len(prompt_ids)
            if label_start >= len(full_ids):
                continue
            if len(full_ids) > FILTER_BY_TOKENS:
                full_ids = full_ids[:FILTER_BY_TOKENS]
            labels = [-100] * label_start + full_ids[label_start:]
            labels = labels[:len(full_ids)]
            if sum(1 for l in labels if l != -100) == 0:
                continue
            samples.append({
                "input_ids": torch.tensor([full_ids], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
                "task": task_name, "idx": int(idx),
                "seq_len": len(full_ids),
                "answer_tokens": sum(1 for l in labels if l != -100),
            })
            task_count += 1
    return samples


# ---------------------------------------------------------------------------
# Loss + gradient extraction
# ---------------------------------------------------------------------------

def _get_lora_param_info(mm):
    info = []
    for name, param in mm.model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        layer_idx = -1
        for j, part in enumerate(parts):
            if part == "layers" and j + 1 < len(parts):
                try:
                    layer_idx = int(parts[j + 1])
                    break
                except ValueError:
                    pass
        module = "unknown"
        for mod in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            if mod in name:
                module = mod
                break
        component = "A" if "lora_A" in name else ("B" if "lora_B" in name else "?")
        info.append({"name": name, "layer_idx": layer_idx,
                      "module": module, "component": component})
    return info


def compute_loss_and_grads(mm, sample, device, apply_memory_policy):
    input_ids = sample["input_ids"].to(device)
    labels = sample["labels"].to(device)
    mm.model.zero_grad(set_to_none=True)
    reset_policy_state(mm)

    if apply_memory_policy and hasattr(mm.memory_policy, "record_eval_stats"):
        mm.memory_policy.record_eval_stats = True
    if apply_memory_policy:
        mm.memory_policy.set_params_batch_idxs(
            np.zeros([input_ids.shape[0]], dtype=np.int64))

    seq_len = input_ids.shape[1]
    answer_mask = (labels[0] != -100)
    if not answer_mask.any():
        return None
    answer_start = answer_mask.nonzero(as_tuple=True)[0][0].item()
    chunk_align = getattr(mm, "max_new_tokens", 64) or 64
    context_end = (answer_start // chunk_align) * chunk_align
    context_end = max(context_end, 0)

    past_kv = None
    retention_ratio = 1.0
    if context_end > 0:
        with torch.no_grad():
            ctx_out = mm(input_ids=input_ids[:, :context_end], use_cache=True,
                         apply_memory_policy=apply_memory_policy,
                         limit_new_tokens=None, output_hidden_states=False,
                         skip_lm_head=True)
        past_kv = ctx_out.past_key_values
        if apply_memory_policy and past_kv is not None:
            if isinstance(past_kv, tuple):
                n_ret = past_kv[0][0].shape[-2]
            else:
                n_ret = past_kv.key_cache[0].shape[-2]
            retention_ratio = n_ret / context_end
        del ctx_out
        torch.cuda.empty_cache()

    phase2_input = input_ids[:, context_end:]
    phase2_pos = torch.arange(context_end, seq_len, device=device).unsqueeze(0)
    outputs = mm(input_ids=phase2_input, position_ids=phase2_pos,
                 past_key_values=past_kv, use_cache=True,
                 apply_memory_policy=apply_memory_policy,
                 limit_new_tokens=None, output_hidden_states=True,
                 skip_lm_head=True)

    hidden = outputs.hidden_states[-1]
    phase2_labels = labels[:, context_end:]
    shift_h = hidden[:, :-1, :].contiguous()
    shift_l = phase2_labels[:, 1:].contiguous()
    lm_head = mm.lm_head
    n_tokens = (shift_l != -100).sum()
    if n_tokens == 0:
        return None

    total_loss = torch.tensor(0.0, device=device)
    for i_c in range(0, shift_h.shape[1], 512):
        ch = shift_h[:, i_c:i_c + 512, :]
        cl = shift_l[:, i_c:i_c + 512].contiguous().view(-1)
        total_loss = total_loss + F.cross_entropy(
            lm_head(ch).float().view(-1, lm_head.out_features),
            cl, ignore_index=-100, reduction="sum")
    loss = total_loss / n_tokens.clamp(min=1)
    loss.backward()

    param_info = _get_lora_param_info(mm)
    per_layer_norms: dict[int, float] = {}
    per_param_grads: dict[str, list[float]] = {}
    for pi in param_info:
        name = pi["name"]
        li = pi["layer_idx"]
        param = dict(mm.model.named_parameters())[name]
        if param.grad is not None:
            gf = param.grad.detach().float().cpu().flatten()
            gl2 = float(gf.norm(2).item())
            per_layer_norms[li] = per_layer_norms.get(li, 0.0) + gl2 ** 2
            per_param_grads[name] = gf.tolist()
    for li in per_layer_norms:
        per_layer_norms[li] = float(np.sqrt(per_layer_norms[li]))

    if apply_memory_policy and hasattr(mm.memory_policy, "record_eval_stats"):
        mm.memory_policy.record_eval_stats = False
    del outputs, past_kv, hidden
    mm.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return {
        "loss": float(loss.item()),
        "per_layer_grad_norms": {str(k): v for k, v in per_layer_norms.items()},
        "per_param_grads": per_param_grads,
        "retention_ratio": retention_ratio,
        "n_answer_tokens": int(n_tokens.item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute():
    if OUT_PATH.exists():
        logger.info("Data already exists: %s — skipping", OUT_PATH)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, mp, mm, _, ts, tokenizer = build_model_and_data(device)
    load_namm_weights(mm, mp, M2_NAMM_MASKFIX_CKPT, device)
    load_lora_weights(mm, M3_MASKFIX_LORA_CKPT, device)
    mm.to(dtype=torch.bfloat16)

    # PEFT gradient fix
    try:
        embed = mm.model.get_input_embeddings()
    except AttributeError:
        embed = mm.get_input_embeddings()
    embed.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

    train_samples = get_train_samples(ts, tokenizer)
    if not train_samples:
        logger.error("No training samples")
        return

    # Pass 1: with eviction
    logger.info("=== Pass 1: WITH eviction ===")
    evicted_results = []
    for i, s in enumerate(train_samples):
        try:
            mm.train()
            r = compute_loss_and_grads(mm, s, device, apply_memory_policy=True)
            if r:
                r.update(task=s["task"], idx=s["idx"], seq_len=s["seq_len"])
                evicted_results.append(r)
                if (i + 1) % 10 == 0:
                    logger.info("  [evict] %d/%d: loss=%.4f ret=%.4f",
                                i + 1, len(train_samples), r["loss"],
                                r["retention_ratio"])
        except Exception as e:
            logger.error("  Sample %d (evict) failed: %s", i, e)
            torch.cuda.empty_cache()

    # Pass 2: without eviction
    logger.info("=== Pass 2: WITHOUT eviction ===")
    from namm.policy.base import Recency
    orig_policy = mm.memory_policy
    mm.swap_memory_policy(Recency(cache_size=None))
    saved_delay = getattr(mm, "memory_policy_fixed_delay", None)
    mm.memory_policy_fixed_delay = None

    full_results = []
    for i, s in enumerate(train_samples):
        try:
            mm.train()
            r = compute_loss_and_grads(mm, s, device, apply_memory_policy=False)
            if r:
                r.update(task=s["task"], idx=s["idx"], seq_len=s["seq_len"])
                full_results.append(r)
                if (i + 1) % 10 == 0:
                    logger.info("  [full] %d/%d: loss=%.4f",
                                i + 1, len(train_samples), r["loss"])
        except Exception as e:
            logger.error("  Sample %d (full) failed: %s", i, e)
            torch.cuda.empty_cache()

    mm.memory_policy_fixed_delay = saved_delay
    mm.swap_memory_policy(orig_policy)

    # Cosine similarity
    cosine_sims: dict[str, list[float]] = {}
    for ev, fu in zip(evicted_results, full_results):
        if ev["idx"] != fu["idx"] or ev["task"] != fu["task"]:
            continue
        for pname in ev["per_param_grads"]:
            if pname not in fu["per_param_grads"]:
                continue
            eg = np.array(ev["per_param_grads"][pname], dtype=np.float32)
            fg = np.array(fu["per_param_grads"][pname], dtype=np.float32)
            en, fn = np.linalg.norm(eg), np.linalg.norm(fg)
            cs = float(np.dot(eg, fg) / (en * fn)) if en > 1e-12 and fn > 1e-12 else 0.0
            parts = pname.split(".")
            lk = "unknown"
            for j, part in enumerate(parts):
                if part == "layers" and j + 1 < len(parts):
                    try:
                        lk = str(int(parts[j + 1]))
                        break
                    except ValueError:
                        pass
            cosine_sims.setdefault(lk, []).append(cs)

    # Strip large grad arrays before saving
    for r in evicted_results:
        del r["per_param_grads"]
    for r in full_results:
        del r["per_param_grads"]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "evicted": evicted_results,
            "full_context": full_results,
            "cosine_sims_per_layer": cosine_sims,
            "cache_size": CACHE_SIZE,
            "max_samples": SAMPLES_PER_TASK,
        }, f, indent=2)
    logger.info("Saved %s", OUT_PATH)

    del mm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    compute()
