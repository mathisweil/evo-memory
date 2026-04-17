"""Reference drift bars: distributional shifts *not* from NAMM-toggle.

Produces two additional pairs:
  Pair 3: M1 (with M1 LoRA, full cache)  vs  B0 (base, no LoRA, full cache)
          → measures LoRA-fine-tuning effect on the output distribution
  Pair 4: B0 (base, full cache)           vs  M2 (base + M2 NAMM, cs1024)
          → measures raw NAMM effect on an un-fine-tuned model

Uses the same greedy-decode-with-logits machinery as analyze_distribution_drift.py
so the JS values are directly comparable to pair1/pair2 in drift_results.json.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device

from analyze_distribution_drift import (
    compare_logit_sequences,
    generate_with_logits,
    swap_lora,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_namm(model: Any, policy: Any, namm_ckpt_path: str, device: str) -> None:
    """Load NAMM evolution state into model + policy."""
    logger.info("Loading NAMM: %s", namm_ckpt_path)
    ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
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


def zero_lora(model: Any, device: str) -> None:
    """Zero every LoRA A matrix so LoRA's contribution is exactly zero.

    With A=0, the product A*B in `h = h + (x @ A @ B) * scale` is 0 for every
    adapter, making the model functionally equivalent to base. Cheaper than
    rebuilding a fresh non-LoRA model; reversible by reloading real weights.
    """
    n_zeroed = 0
    with torch.no_grad():
        for name, param in model.model.named_parameters():
            if "lora_A" in name:
                param.zero_()
                n_zeroed += 1
    logger.info("  Zeroed %d lora_A matrices -> base model behaviour", n_zeroed)
    model.to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_lora_checkpoint", type=str, required=True)
    p.add_argument("--m2_namm_checkpoint", type=str, required=True,
                   help="M2 NAMM checkpoint (base+NAMM, no LoRA).")
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--prompts_per_task", type=int, default=None)
    p.add_argument("--max_gen_tokens", type=int, default=128)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    overrides = [
        f"run@_global_={args.run_config}", "wandb_log=false",
        "wandb_project=Experiments",
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
    ]
    with initialize(version_base=None, config_path="../../../config",
                    job_name="drift_baselines"):
        cfg = compose(config_name="config", overrides=overrides)

    logger.info("Building model...")
    with torch.no_grad():
        (policy, model, _, _, _) = make_eval_model(cfg=cfg)
    model.to(device)
    model.apply_lora_adapters(rank=8, target_modules=["q_proj", "v_proj"])

    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    task_sampler.filter_answers_by_token_count(
        tokenizer, cfg.get("max_answer_tokens", cfg.get("max_new_tokens", 64)))
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get("train_frac", 0.7),
        val_frac=cfg.get("val_frac", 0.15),
        max_conditioning_length=cfg.get("split_max_conditioning_length")
            or cfg.get("max_conditioning_length", 6500),
        min_conditioning_length=cfg.get("split_min_conditioning_length")
            or cfg.get("min_conditioning_length"),
        tokenizer=tokenizer)
    split_idxs = task_sampler.get_split_indices(args.split)

    prompts, prompt_tasks, prompt_original_idxs = [], [], []
    for task in sorted(split_idxs.keys()):
        task_ois = list(split_idxs[task])
        if args.prompts_per_task is not None:
            task_ois = task_ois[:args.prompts_per_task]
        for oi in task_ois:
            prompts.append(task_sampler.lb_prompts_per_task[task][int(oi)])
            prompt_tasks.append(task)
            prompt_original_idxs.append(int(oi))
    if args.max_prompts and args.prompts_per_task is None:
        prompts = prompts[:args.max_prompts]
        prompt_tasks = prompt_tasks[:args.max_prompts]
        prompt_original_idxs = prompt_original_idxs[:args.max_prompts]
    logger.info("  %d prompts from split '%s'", len(prompts), args.split)

    # Need 3 conditions: M1 full, B0 (no LoRA, no NAMM), M2 (no LoRA, NAMM on)
    # Pair 3 = M1_full ↔ B0   ;   Pair 4 = B0 ↔ M2
    bos = getattr(tokenizer, "bos_token", None) or ""

    def run_condition(label: str, apply_mp: bool) -> List[Dict]:
        logger.info("\n%s\nCondition: %s (NAMM=%s)\n%s",
                    "=" * 60, label, apply_mp, "=" * 60)
        out = []
        for p_idx, raw_prompt in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            out.append(generate_with_logits(
                model, policy, input_ids, device,
                apply_memory_policy=apply_mp,
                max_gen_tokens=args.max_gen_tokens,
            ))
            if (p_idx + 1) % 10 == 0:
                logger.info("    %d/%d", p_idx + 1, len(prompts))
        return out

    # 1. M1 full — needs M1 LoRA loaded; NAMM not applied, so params don't matter
    swap_lora(model, args.m1_lora_checkpoint, device)
    results_m1 = run_condition("m1_full", apply_mp=False)

    # 2. B0 — zero LoRA to revert to base; NAMM not applied
    zero_lora(model, device)
    results_b0 = run_condition("b0_base", apply_mp=False)

    # 3. M2 — still zero-LoRA (base model), but load M2 NAMM params and apply
    load_namm(model, policy, args.m2_namm_checkpoint, device)
    results_m2 = run_condition("m2_base_namm", apply_mp=True)

    logger.info("\nComputing pairwise JS divergence ...")
    pair3 = []  # M1 full vs B0
    pair4 = []  # B0 vs M2
    for p_idx in range(len(prompts)):
        c3 = compare_logit_sequences(results_m1[p_idx], results_b0[p_idx])
        c3["task"] = prompt_tasks[p_idx]
        c3["original_idx"] = prompt_original_idxs[p_idx]
        pair3.append(c3)

        c4 = compare_logit_sequences(results_b0[p_idx], results_m2[p_idx])
        c4["task"] = prompt_tasks[p_idx]
        c4["original_idx"] = prompt_original_idxs[p_idx]
        pair4.append(c4)

    del results_m1, results_b0, results_m2
    torch.cuda.empty_cache()

    for name, comps in [("Pair 3 (M1 ↔ B0)", pair3),
                        ("Pair 4 (B0 ↔ M2)", pair4)]:
        js_vals = [c["mean_js"] for c in comps if c["mean_js"] is not None]
        logger.info("%s  n=%d  JS=%.4f (std %.4f)",
                    name, len(comps), np.mean(js_vals), np.std(js_vals))

    out_json = {
        "n_prompts": len(prompts),
        "max_gen_tokens": args.max_gen_tokens,
        "cache_size": args.cache_size,
        "split": args.split,
        "pair3_comparisons": pair3,
        "pair4_comparisons": pair4,
    }
    json_path = os.path.join(args.output_dir, "drift_baseline_results.json")
    with open(json_path, "w") as fp:
        json.dump(out_json, fp, default=str)
    logger.info("Saved %s", json_path)


if __name__ == "__main__":
    main()
