"""Distribution drift analysis: how much does adding/removing NAMM
change p(x_{t+1} | x_{1:t}) during generation?

Compares two pairs:
  Pair 1: M1 full cache  vs  M1 + post-hoc NAMM   (adding NAMM)
  Pair 2: M4 + NAMM      vs  A4 full cache        (removing NAMM)

Per generated token, captures JS divergence plus top-k overlap and token
match as auxiliary metrics. Output is a JSON consumed by
filter_and_replot.py, which produces the JS divergence bar plot.

Requires GPU (~40 min on 3090 Ti, 4 conditions × 70 prompts).
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a.float(), dim=-1)
    q = F.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="sum", log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction="sum", log_target=False)
    return float(0.5 * (kl_pm + kl_qm))


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    top_a = set(logits_a.topk(k).indices.tolist())
    top_b = set(logits_b.topk(k).indices.tolist())
    return len(top_a & top_b) / k


@torch.no_grad()
def generate_with_logits(
    model: Any,
    policy: Any,
    input_ids: torch.Tensor,
    device: str,
    apply_memory_policy: bool,
    max_gen_tokens: int = 128,
) -> Dict[str, Any]:
    reset_memory_policy_state(policy)
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        apply_memory_policy=apply_memory_policy,
        return_dict=True,
    )
    past_kv = outputs.past_key_values
    next_logits = outputs.logits[0, -1, :]

    all_logits: List[torch.Tensor] = []
    token_ids: List[int] = []

    for _ in range(max_gen_tokens):
        all_logits.append(next_logits.cpu().float())
        next_token_id = next_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token_id.item()))

        eos_id = getattr(getattr(model, "config", None), "eos_token_id", 2)
        eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
        if next_token_id.item() in eos_ids:
            break

        next_input = next_token_id.unsqueeze(0)
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
    return {"logits": all_logits, "token_ids": token_ids}


def compare_logit_sequences(result_a: Dict[str, Any],
                            result_b: Dict[str, Any]) -> Dict[str, Any]:
    logits_a = result_a["logits"]
    logits_b = result_b["logits"]
    n = min(len(logits_a), len(logits_b))

    js_divs, top5, top10 = [], [], []
    for t in range(n):
        js_divs.append(js_divergence(logits_a[t], logits_b[t]))
        top5.append(topk_overlap(logits_a[t], logits_b[t], k=5))
        top10.append(topk_overlap(logits_a[t], logits_b[t], k=10))

    return {
        "n_steps_compared": n,
        "n_steps_a": len(logits_a),
        "n_steps_b": len(logits_b),
        "tokens_a": result_a["token_ids"][:n],
        "tokens_b": result_b["token_ids"][:n],
        "token_match_rate": sum(
            1 for i in range(n)
            if result_a["token_ids"][i] == result_b["token_ids"][i]
        ) / max(n, 1),
        "js_divergence": js_divs,
        "top5_overlap": top5,
        "top10_overlap": top10,
        "mean_js": float(np.mean(js_divs)) if js_divs else None,
        "mean_top5": float(np.mean(top5)) if top5 else None,
        "mean_top10": float(np.mean(top10)) if top10 else None,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_lora_checkpoint", type=str, required=True)
    p.add_argument("--m4_lora_checkpoint", type=str, required=True)
    p.add_argument("--namm_checkpoint", type=str, required=True)
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
                    job_name="distribution_drift"):
        cfg = compose(config_name="config", overrides=overrides)

    logger.info("Building model...")
    with torch.no_grad():
        (policy, model, evaluator, _, _) = make_eval_model(cfg=cfg)
    model.to(device)

    logger.info("Loading NAMM: %s", args.namm_checkpoint)
    namm_ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                           weights_only=False)
    evo = namm_ckpt["evolution_state"]
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

    conditions = [
        ("m1_full", args.m1_lora_checkpoint, False),
        ("m1_namm", args.m1_lora_checkpoint, True),
        ("m4_namm", args.m4_lora_checkpoint, True),
        ("a4_full", args.m4_lora_checkpoint, False),
    ]

    bos = getattr(tokenizer, "bos_token", None) or ""
    results_by_condition: Dict[str, List[Dict]] = {}
    current_lora_path = None
    for cond_label, lora_path, apply_mp in conditions:
        logger.info("\n%s\nCondition: %s (NAMM=%s)\n%s",
                    "=" * 60, cond_label, apply_mp, "=" * 60)
        if lora_path != current_lora_path:
            swap_lora(model, lora_path, device)
            current_lora_path = lora_path

        cond_results = []
        for p_idx, raw_prompt in enumerate(prompts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True, tokenize=False)
            if bos and templated.startswith(bos):
                templated = templated[len(bos):]
            enc = tokenizer(templated, add_special_tokens=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            cond_results.append(generate_with_logits(
                model, policy, input_ids, device,
                apply_memory_policy=apply_mp,
                max_gen_tokens=args.max_gen_tokens,
            ))
            if (p_idx + 1) % 10 == 0:
                logger.info("    %d/%d", p_idx + 1, len(prompts))
        results_by_condition[cond_label] = cond_results

    logger.info("\nComputing pairwise JS divergence ...")
    pair1_comparisons, pair2_comparisons = [], []
    for p_idx in range(len(prompts)):
        c1 = compare_logit_sequences(
            results_by_condition["m1_full"][p_idx],
            results_by_condition["m1_namm"][p_idx])
        c1["task"] = prompt_tasks[p_idx]
        c1["original_idx"] = prompt_original_idxs[p_idx]
        pair1_comparisons.append(c1)

        c2 = compare_logit_sequences(
            results_by_condition["m4_namm"][p_idx],
            results_by_condition["a4_full"][p_idx])
        c2["task"] = prompt_tasks[p_idx]
        c2["original_idx"] = prompt_original_idxs[p_idx]
        pair2_comparisons.append(c2)

    del results_by_condition
    torch.cuda.empty_cache()

    for name, comps in [("Pair 1 (M1 toggle)", pair1_comparisons),
                        ("Pair 2 (M4 toggle)", pair2_comparisons)]:
        js_vals = [c["mean_js"] for c in comps if c["mean_js"] is not None]
        logger.info("%s  n=%d  JS=%.4f (std %.4f)",
                    name, len(comps), np.mean(js_vals), np.std(js_vals))

    all_results = {
        "n_prompts": len(prompts),
        "max_gen_tokens": args.max_gen_tokens,
        "cache_size": args.cache_size,
        "split": args.split,
        "pair1_comparisons": pair1_comparisons,
        "pair2_comparisons": pair2_comparisons,
    }
    json_path = os.path.join(args.output_dir, "drift_results.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, default=str)
    logger.info("Saved %s", json_path)


if __name__ == "__main__":
    main()
