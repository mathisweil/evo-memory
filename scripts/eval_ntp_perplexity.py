"""Evaluate next-token-prediction perplexity under different cache conditions.

For each long document, computes cross-entropy loss on the LAST chunk (256
tokens) after the rest of the prompt has been processed. This isolates the
effect of eviction on prediction quality — the last chunk's predictions
depend on the quality of the (possibly evicted) KV cache.

Three conditions:
  (a) Full context: all tokens processed, no eviction
  (b) NAMM eviction: all tokens processed, cache evicted to cache_size
  (c) Truncation: only the last N tokens fed to the model, no eviction

Supports --lora_checkpoint to compare M1, M4, M4-maskfix LoRAs.

Output: per-document perplexity + aggregated stats, saved as JSONL.

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/eval_ntp_perplexity.py \
        --namm_checkpoint <path> --cache_size 1024 \
        --lora_checkpoint <path>  # optional \
        --output_dir analysis_out/ntp_perplexity

    # Full comparison (run multiple times with different flags):
    # 1. Base model, full cache:       (no --namm_checkpoint, no --truncate_to)
    # 2. Base model, NAMM:             (--namm_checkpoint X --cache_size 1024)
    # 3. Base model, truncation:       (--truncate_to 1024)
    # 4. M1 LoRA, full cache:          (--lora_checkpoint M1)
    # 5. M1 LoRA, NAMM:               (--namm_checkpoint X --lora_checkpoint M1)
    # 6. M4 LoRA, NAMM:               (--namm_checkpoint X --lora_checkpoint M4)
    # 7. M4 maskfix LoRA, maskfix NAMM: (--namm_checkpoint Xf --lora_checkpoint M4f)
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from namm.policy.base import Recency
from es_finetuning.device import get_device


def parse_args():
    p = argparse.ArgumentParser(
        description="NTP perplexity under full/NAMM/truncation conditions")
    p.add_argument("--namm_checkpoint", type=str, default=None,
                   help="NAMM checkpoint. If omitted, runs without eviction "
                        "(full cache or truncation depending on --truncate_to).")
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--truncate_to", type=int, default=None,
                   help="If set, truncate input to last N tokens instead of "
                        "using NAMM eviction. Mutually exclusive with "
                        "--namm_checkpoint.")
    p.add_argument("--run_config", type=str,
                   default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--splits", nargs="+", default=["test", "extended_test"],
                   choices=["train", "val", "test", "extended_test"])
    p.add_argument("--extended_max_conditioning_length", type=int, default=8192)
    p.add_argument("--eval_chunk_size", type=int, default=256,
                   help="Compute loss on the last N tokens of each prompt. "
                        "Should match memory_policy_fixed_delay (256).")
    p.add_argument("--max_prompts_per_task", type=int, default=None)
    p.add_argument("--label", type=str, default=None,
                   help="Human-readable label for this run (used in output).")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def compute_chunk_perplexity(logits, labels):
    """Compute per-token cross-entropy and perplexity for a chunk.

    Args:
        logits: (seq_len, vocab_size)
        labels: (seq_len,) token IDs

    Returns:
        dict with per_token_loss, mean_loss, perplexity
    """
    # Shift: predict token t+1 from position t
    shift_logits = logits[:-1, :]
    shift_labels = labels[1:]

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, reduction='none')

    mean_loss = per_token_loss.mean().item()
    perplexity = np.exp(mean_loss)

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "n_tokens": int(shift_labels.shape[0]),
        "per_token_loss": per_token_loss.detach().cpu().tolist(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.namm_checkpoint and args.truncate_to:
        raise ValueError("Cannot use both --namm_checkpoint and --truncate_to")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    # Determine mode
    if args.namm_checkpoint:
        mode = "namm"
    elif args.truncate_to:
        mode = "truncation"
    else:
        mode = "full"

    # Build label
    lora_tag = ""
    if args.lora_checkpoint:
        if "m1" in args.lora_checkpoint.lower():
            lora_tag = "_m1"
        elif "maskfix" in args.lora_checkpoint.lower():
            lora_tag = "_m4fix"
        elif "m4" in args.lora_checkpoint.lower():
            lora_tag = "_m4"
        else:
            lora_tag = "_lora"

    label = args.label or f"{mode}_cs{args.cache_size}{lora_tag}"

    # Hydra config
    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false", "wandb_project=Experiments",
        f"filter_by_length={args.filter_by_length}",
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
        "+protected_tail_n=5",
    ]

    with initialize(version_base=None, config_path="../config",
                    job_name="ntp_perplexity"):
        cfg = compose(config_name="config", overrides=overrides)

    # Build model
    print(f"Building model (mode={mode}, label={label})...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)
    memory_evaluator.device = device

    # Load NAMM or swap to no-op
    if args.namm_checkpoint:
        print(f"Loading NAMM: {args.namm_checkpoint}")
        ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                          weights_only=False)
        evo = ckpt['evolution_state']
        prefer_mean = cfg.get('prefer_mean_to_best', True)
        params_vec = (evo['mean'] if (prefer_mean and 'mean' in evo)
                      else evo['best_member'])
        memory_model.set_memory_params(params_vec.unsqueeze(0).to(device))
        bp = 'stored_buffers_to_save.'
        bd = {k[len(bp):]: v.to(device) for k, v in evo.items()
              if k.startswith(bp)}
        if bd:
            memory_model.load_buffers_dict(buffers_dict=bd)
        batch_idxs = np.zeros([1])
        memory_policy.set_params_batch_idxs(batch_idxs)
        memory_policy.record_eval_stats = True
        memory_policy.initialize_stat_objects()
        apply_policy = True
    else:
        # No NAMM — swap to no-op policy
        noop = Recency(cache_size=None)
        memory_evaluator.swap_memory_policy(noop)
        memory_policy = noop
        apply_policy = False

    # Load LoRA
    if args.lora_checkpoint:
        print(f"Loading LoRA: {args.lora_checkpoint}")
        lora_ckpt = torch.load(args.lora_checkpoint, map_location="cpu",
                               weights_only=False)
        lora_cfg = lora_ckpt.get('lora_config', {})
        memory_model.apply_lora_adapters(
            rank=lora_cfg.get('rank', 8),
            target_modules=lora_cfg.get('target_modules', ['q_proj', 'v_proj']))
        memory_model.to(device)
        lora_sd = lora_ckpt['lora_state_dict']
        loaded = 0
        for n, p in memory_model.model.named_parameters():
            if n in lora_sd:
                p.data.copy_(lora_sd[n].to(p.device, dtype=p.dtype))
                loaded += 1
        print(f"  Loaded {loaded} LoRA tensors")

    # Task sampler
    print("Building task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get('train_frac', 0.7),
        val_frac=cfg.get('val_frac', 0.15),
        max_conditioning_length=cfg.get('split_max_conditioning_length',
                                        cfg.get('max_conditioning_length', 6500)),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        tokenizer=tokenizer,
        extended_max_conditioning_length=(
            args.extended_max_conditioning_length
            if "extended_test" in args.splits else None),
    )
    # Apply chat template to match eval conditions
    task_sampler.apply_chat_template_to_prompts(tokenizer)
    raw_prompts = task_sampler.lb_prompts_per_task

    eval_chunk = args.eval_chunk_size
    out_path = os.path.join(args.output_dir, f"ntp_{label}_{timestamp}.jsonl")
    print(f"Output: {out_path}")
    print(f"Eval chunk size: {eval_chunk} (loss computed on last {eval_chunk} tokens)")

    n_total = 0
    all_ppls = []

    with open(out_path, "w") as f_out:
        header = {
            "_header": True,
            "mode": mode,
            "label": label,
            "cache_size": args.cache_size,
            "truncate_to": args.truncate_to,
            "namm_checkpoint": (os.path.abspath(args.namm_checkpoint)
                                if args.namm_checkpoint else None),
            "lora_checkpoint": (os.path.abspath(args.lora_checkpoint)
                                if args.lora_checkpoint else None),
            "eval_chunk_size": eval_chunk,
            "timestamp": timestamp,
        }
        f_out.write(json.dumps(header) + "\n")

        for split_name in args.splits:
            split_idxs = task_sampler.get_split_indices(split_name)
            for task_name in sorted(split_idxs.keys()):
                task_indices = list(split_idxs[task_name])
                if args.max_prompts_per_task:
                    task_indices = task_indices[:args.max_prompts_per_task]
                print(f"\n  {split_name}/{task_name}: {len(task_indices)} prompts")

                for i, orig_idx in enumerate(task_indices):
                    prompt = raw_prompts[task_name][int(orig_idx)]
                    enc = tokenizer(prompt, add_special_tokens=True,
                                   return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    n_tok = int(input_ids.shape[-1])

                    if n_tok <= eval_chunk + 10:
                        continue  # too short

                    # Truncation mode: slice input
                    if args.truncate_to:
                        input_ids = input_ids[..., -args.truncate_to:]
                        attention_mask = attention_mask[..., -args.truncate_to:]
                        n_tok = int(input_ids.shape[-1])

                    # Reset NAMM state
                    if apply_policy:
                        memory_policy.initialize_stat_objects()

                    # Forward pass — get logits
                    with torch.no_grad():
                        outputs = memory_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            use_cache=True,
                            apply_memory_policy=apply_policy,
                        )

                    logits = outputs.logits[0]  # (returned_len, vocab_size)
                    # After split processing, logits only covers the last
                    # chunk (not the full prompt). Use all returned logits
                    # and align labels to match.
                    returned_len = logits.shape[0]
                    chunk_logits = logits
                    chunk_labels = input_ids[0, -returned_len:]

                    ppl_info = compute_chunk_perplexity(chunk_logits, chunk_labels)

                    record = {
                        "task": task_name,
                        "split": split_name,
                        "original_idx": int(orig_idx),
                        "n_tokens": n_tok,
                        "mode": mode,
                        "label": label,
                        "mean_loss": ppl_info["mean_loss"],
                        "perplexity": ppl_info["perplexity"],
                        "n_eval_tokens": ppl_info["n_tokens"],
                    }
                    f_out.write(json.dumps(record) + "\n")
                    n_total += 1
                    all_ppls.append(ppl_info["perplexity"])

                    if (i + 1) % 20 == 0:
                        print(f"    [{i+1}/{len(task_indices)}] "
                              f"ppl={ppl_info['perplexity']:.2f} "
                              f"loss={ppl_info['mean_loss']:.4f}")

                    del outputs
                    torch.cuda.empty_cache()

    print(f"\nDone: {n_total} prompts → {out_path}")

    # Summary
    if all_ppls:
        print(f"\n{'='*60}")
        print(f"  NTP Perplexity Summary: {label}")
        print(f"{'='*60}")
        print(f"  N prompts: {n_total}")
        print(f"  Mean perplexity: {np.mean(all_ppls):.2f}")
        print(f"  Median perplexity: {np.median(all_ppls):.2f}")
        print(f"  Std: {np.std(all_ppls):.2f}")
        print(f"  Min: {np.min(all_ppls):.2f}")
        print(f"  Max: {np.max(all_ppls):.2f}")

        # Per-split
        records = []
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("_header"):
                    records.append(r)

        for split in args.splits:
            split_ppls = [r["perplexity"] for r in records if r["split"] == split]
            if split_ppls:
                print(f"\n  {split}: mean={np.mean(split_ppls):.2f} "
                      f"median={np.median(split_ppls):.2f} n={len(split_ppls)}")

        # Per-task
        print(f"\n  Per-task (all splits):")
        tasks = sorted(set(r["task"] for r in records))
        for task in tasks:
            task_ppls = [r["perplexity"] for r in records if r["task"] == task]
            print(f"    {task.replace('lb/',''):15s}: "
                  f"ppl={np.mean(task_ppls):8.2f} (n={len(task_ppls)})")


if __name__ == "__main__":
    main()
