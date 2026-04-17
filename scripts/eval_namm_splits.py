"""Evaluate on the exact train/val/test splits used during NAMM training
(3-way 70/15/15 split with token-length filtering).

Supports NAMM checkpoints, LoRA checkpoints, recency baselines, truncation
baselines, and plain LLaMA baselines (--plain). This is the single canonical
eval entry point for all conditions.

Usage:
    # NAMM checkpoint:
    python eval_namm_splits.py \
        --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
        --splits train val test --cache_size 1024

    # LoRA + NAMM:
    python eval_namm_splits.py \
        --lora_checkpoint path/to/best_ckpt.pt \
        --namm_checkpoint path/to/namm.pt \
        --cache_size 1024 --splits test

    # Plain LLaMA baseline (no NAMM wrapper, no eviction):
    python eval_namm_splits.py --plain --splits test extended_test
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from hydra import compose, initialize
from namm.evaluation.evaluator import MemoryHFEvaluator
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device


def _snapshot_raw_eviction_arrays(memory_policy):
    """Snapshot the per-layer per-step buffers from a DynamicMemoryPolicy
    BEFORE get_param_stats(reset=True) clears them.

    Returns a dict of plain numpy arrays keyed by signal name. Each array is
    shape (num_layers, num_steps_layer_i) — ragged across layers (steps may
    differ if the policy fired at different rates), so stored as an object
    array of 1-D arrays per layer.

    Signals captured (when present on the policy):
      - dynamic_cache_sizes: cache size after eviction at each policy step.
      - final_dynamic_cache_sizes: cache size at the end of each prompt.
      - dynamic_mask_sample_sparsity: max-over-heads unmasked count per step.
      - dynamic_mask_head_sparsity: mean-over-heads unmasked count per step.
      - recorded_final_recencies: mean position-recency of kept tokens per step.
    """
    snap = {}

    def _to_obj_array(per_layer_lists):
        if not per_layer_lists:
            return None
        # Allocate (n_layers,)-shaped object array explicitly: when every
        # inner list is empty, np.array([...], dtype=object) collapses to
        # shape (n_layers, 0), which is wrong.
        n_layers = len(per_layer_lists)
        out = np.empty(n_layers, dtype=object)
        for i, layer in enumerate(per_layer_lists):
            out[i] = np.asarray(layer, dtype=np.float32)
        return out

    for attr in (
        'dynamic_cache_sizes',
        'final_dynamic_cache_sizes',
        'dynamic_mask_sample_sparsity',
        'dynamic_mask_head_sparsity',
        'recorded_final_recencies',
    ):
        if hasattr(memory_policy, attr):
            arr = _to_obj_array(getattr(memory_policy, attr))
            if arr is not None:
                snap[attr] = arr
    return snap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NAMM checkpoint on exact training splits "
                    "(3-way 70/15/15 with token filtering)")
    parser.add_argument("--plain", action="store_true",
                        help="Load plain LLaMA (no NAMM wrapper, no eviction). "
                             "Equivalent to the old eval_plain_llama.py script. "
                             "Incompatible with --namm_checkpoint and "
                             "--use_classic_recency.")
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Path to NAMM checkpoint (.pt file)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to a LoRA finetuning checkpoint (with "
                             "'lora_state_dict' and 'lora_config' keys, as "
                             "saved by LoRAGradTrainer). Loaded on top of the "
                             "NAMM/recency model. Use without --namm_checkpoint "
                             "to evaluate a plain LoRA at large cache_size.")
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b_5t",
                        help="Hydra run config override name")
    parser.add_argument("--cache_size", type=int, default=None,
                        help="Override cache size for NAMM eviction")
    parser.add_argument("--use_classic_recency", action="store_true",
                        help="Replace the (untrained) DeepMP scoring policy "
                             "with namm.policy.base.Recency, which deterministically "
                             "keeps only the last `cache_size` tokens. Use this "
                             "to reproduce the StreamingLLM-style 'recency' "
                             "baseline from the NAMM paper. Mutually exclusive "
                             "with --namm_checkpoint and --lora_checkpoint.")
    parser.add_argument("--truncate_input_to", type=int, default=None,
                        help="Truncate every prompt to its last N tokens "
                             "BEFORE feeding it to the model. The model never "
                             "sees the evicted tokens — there is no KV cache "
                             "eviction stack at all, no rotary offset bookkeeping, "
                             "no policy hooks. This is the cleanest 'naive "
                             "tail-only' baseline: 'what does the LM produce when "
                             "the input is shortened to its last N tokens?'. "
                             "Implemented by monkey-patching the evaluator's "
                             "tok_batch_encode to skip use_mid_cropping and "
                             "instead slice [:, -N:] after tokenization. "
                             "Compatible with --lora_checkpoint (LoRA modifies "
                             "LM weights, truncation only changes inputs). "
                             "Incompatible with --namm_checkpoint and "
                             "--use_classic_recency (those run the eviction "
                             "stack).")
    parser.add_argument("--filter_by_length", type=int, default=None,
                        help="Override filter_by_length in Hydra config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Inference batch size (default: use config value)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Parent directory for run subfolders. Defaults "
                             "to directory containing the checkpoint. Each "
                             "run creates a unique subfolder so results are "
                             "never overwritten.")
    parser.add_argument("--run_label", type=str, default=None,
                        help="Optional short label appended to the per-run "
                             "subfolder name (e.g. 'cs1024', 'lora_m4'). The "
                             "subfolder is always {label_}{timestamp}.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test", "extended_test"],
                        help="Which splits to evaluate (default: all three). "
                             "'extended_test' = test ∪ prompts in "
                             "(max_conditioning_length, "
                             "extended_max_conditioning_length] tokens.")
    parser.add_argument("--extended_max_conditioning_length", type=int,
                        default=8192,
                        help="Upper token bound for the extended_test split "
                             "(default: 8192). Only used if 'extended_test' "
                             "is in --splits.")
    parser.add_argument("--task_config", type=str, default=None,
                        help="Override task config (e.g. rh_ood_eval_3t for OOD eval)")
    parser.add_argument("--dump_namm_state", type=str, default=None,
                        help="Section C: per-prompt NAMM state dump directory. "
                             "When set, replaces the generation/F1 eval loop "
                             "with a forward-only per-prompt pass that saves "
                             "NAMM scores, retained indices, and the final "
                             "LLM-side attention to "
                             "<dir>/<task>__<idx:04d>.pt. Requires a NAMM "
                             "checkpoint; incompatible with --plain, "
                             "--use_classic_recency, --truncate_input_to.")
    parser.add_argument("--dump_max_prompts_per_task", type=int, default=None,
                        help="Section C: cap prompts per task in the dump "
                             "loop (smoke-test use).")
    parser.add_argument("--dump_condition_label", type=str, default=None,
                        help="Section C: short label ('B0', 'M1', 'M4') "
                             "stamped into every dump file's config metadata.")
    return parser.parse_args()


def _run_section_c_dump_loop(
    args,
    memory_policy,
    memory_model,
    memory_evaluator,
    task_sampler,
    tokenizer,
    device,
    timestamp,
    train_frac,
    val_frac,
    max_cond,
    min_cond,
    ext_max_cond,
):
    """Section C — per-prompt NAMM state dump (no F1 generation).

    Iterates the requested splits one prompt at a time, runs a forward-only
    pass with ``apply_memory_policy=True`` and ``output_attentions=True``,
    captures NAMM tensors from the policy, captures LLM self-attention
    from forward hooks, reduces attention to a per-KV-token scalar, and
    saves one ``.pt`` file per prompt under ``args.dump_namm_state``.

    Returns the list of written paths.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not hasattr(memory_policy, 'initialize_dump_buffers'):
        raise RuntimeError(
            "Memory policy does not support tensor dumping. "
            "--dump_namm_state requires the DeepMP policy (BAM-i1 NAMM).")

    memory_policy._record_dump_tensors = True
    memory_policy.initialize_dump_buffers()

    captured = {"per_chunk": []}

    def _make_hook(layer_idx):
        def hook_fn(_module, _inp, output):
            attn_weights = output[1]
            if attn_weights is not None:
                # (bs, n_heads, q_len, kv_len) — batch size 1 in dump mode.
                captured["per_chunk"].append(
                    (layer_idx, attn_weights[0].detach().to(torch.float16).cpu())
                )
        return hook_fn

    hooks = []
    for i, layer in enumerate(memory_model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(_make_hook(i)))

    out_root = os.path.abspath(args.dump_namm_state)
    os.makedirs(out_root, exist_ok=True)

    config_blob = {
        "condition": args.dump_condition_label or "unknown",
        "lora_path": (os.path.abspath(args.lora_checkpoint)
                      if args.lora_checkpoint else None),
        "namm_path": os.path.abspath(args.namm_checkpoint),
        "cache_size": args.cache_size,
        "run_config": args.run_config,
        "timestamp": timestamp,
        "protected_tail_n": 5,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "min_conditioning_length": min_cond,
        "max_conditioning_length": max_cond,
        "extended_max_conditioning_length": ext_max_cond,
    }

    written: List[str] = []
    total_prompts_seen = 0
    for split_name in args.splits:
        split_idxs = task_sampler.get_split_indices(split_name)
        for task_name in sorted(split_idxs.keys()):
            task_indices = sorted(int(i) for i in split_idxs[task_name])
            if args.dump_max_prompts_per_task is not None:
                task_indices = task_indices[:args.dump_max_prompts_per_task]
            print(f"  dump split={split_name} task={task_name}: "
                  f"{len(task_indices)} prompts")

            for orig_idx in task_indices:
                prompt_str = task_sampler.lb_prompts_per_task[task_name][orig_idx]
                enc = tokenizer(prompt_str, add_special_tokens=True,
                                return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                n_tok = int(input_ids.shape[-1])

                if args.cache_size and n_tok <= args.cache_size:
                    logger.info(
                        "Skipping task=%s idx=%d: n_tok=%d ≤ cache_size=%d "
                        "(no eviction fires)", task_name, orig_idx, n_tok,
                        args.cache_size)
                    continue

                # Between-prompt reset: policy auto-resets non-dump state on
                # new_sequences=True during the first chunk of the next
                # forward pass (see DeepMP.update_layer_cache_impl_).
                memory_policy.initialize_dump_buffers()
                captured["per_chunk"] = []

                with torch.no_grad():
                    memory_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        apply_memory_policy=True,
                        output_attentions=True,
                        use_cache=True,
                        skip_lm_head=True,
                    )

                dump = memory_policy.pop_dump_buffers()

                # Reduce final-chunk LLM attention to per-KV-token scalar
                # per layer. captured["per_chunk"] is a flat list of
                # (layer_idx, attn) pairs across chunks; we want only the
                # last occurrence of each layer_idx (= final chunk).
                final_attn_by_layer: Dict[int, torch.Tensor] = {}
                for layer_idx, attn in captured["per_chunk"]:
                    final_attn_by_layer[layer_idx] = attn

                if final_attn_by_layer:
                    n_layers = memory_policy.num_memory_layers
                    layer_means: List[Optional[torch.Tensor]] = []
                    layer_heads_means: List[Optional[torch.Tensor]] = []
                    for l in range(n_layers):
                        a = final_attn_by_layer.get(l)
                        if a is None:
                            layer_means.append(None)
                            layer_heads_means.append(None)
                            continue
                        a_f = a.float()
                        mean_per_head = a_f.mean(dim=-2)
                        layer_heads_means.append(mean_per_head.to(torch.float16))
                        layer_means.append(
                            mean_per_head.mean(dim=0).to(torch.float16)
                        )
                    # Per-layer kv_len can differ (same reason the NAMM-side
                    # dump stores per-layer lists instead of stacking).
                    if any(t is not None for t in layer_means):
                        final_attn_mean = layer_means
                        final_attn_per_head = layer_heads_means
                    else:
                        final_attn_mean = None
                        final_attn_per_head = None
                else:
                    final_attn_mean = None
                    final_attn_per_head = None

                record = dict(dump)
                record["final_attn_mean_per_token"] = final_attn_mean
                record["final_attn_mean_per_token_per_head"] = final_attn_per_head
                record["prompt_meta"] = {
                    "task": task_name,
                    "orig_idx": int(orig_idx),
                    "prompt_length_tokens": n_tok,
                    "split": split_name,
                    "n_steps": dump["n_steps"],
                    "protected_tail_n": 5,
                }
                record["config"] = dict(config_blob)

                fname = f"{task_name.replace('/', '__')}__{orig_idx:04d}.pt"
                out_path = os.path.join(out_root, fname)
                torch.save(record, out_path)
                written.append(out_path)
                total_prompts_seen += 1
                if total_prompts_seen % 5 == 0:
                    print(f"    dumped {total_prompts_seen} prompts "
                          f"(last: {fname})")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
    memory_policy._record_dump_tensors = False
    print(f"\nSection C: wrote {len(written)} dump files to {out_root}")
    return written


def main():
    args = parse_args()

    # ── Validate flag combinations ─────────────────────────────────────────
    if args.plain:
        if args.namm_checkpoint:
            raise ValueError("--plain is incompatible with --namm_checkpoint")
        if args.use_classic_recency:
            raise ValueError("--plain is incompatible with --use_classic_recency")
        if args.truncate_input_to is not None:
            raise ValueError("--plain is incompatible with --truncate_input_to")
        if args.lora_checkpoint:
            raise ValueError(
                "--plain + --lora_checkpoint is not supported. To evaluate "
                "a LoRA checkpoint without NAMM, omit --plain and omit "
                "--namm_checkpoint (the NAMM wrapper will use init params "
                "with a large cache_size, effectively no eviction).")

    if args.dump_namm_state:
        if args.plain:
            raise ValueError("--dump_namm_state is incompatible with --plain")
        if args.use_classic_recency:
            raise ValueError(
                "--dump_namm_state is incompatible with --use_classic_recency")
        if args.truncate_input_to is not None:
            raise ValueError(
                "--dump_namm_state is incompatible with --truncate_input_to")
        if not args.namm_checkpoint:
            raise ValueError(
                "--dump_namm_state requires --namm_checkpoint (Section C "
                "compares eviction under a fixed trained NAMM).")

    # ── Resolve output directory ────────────────────────────────────────────
    if args.output_dir:
        parent_dir = args.output_dir
    elif args.namm_checkpoint:
        ckpt_dir = os.path.dirname(os.path.abspath(args.namm_checkpoint))
        if os.path.basename(ckpt_dir) == "checkpoints":
            parent_dir = os.path.dirname(ckpt_dir)
        else:
            parent_dir = ckpt_dir
    else:
        parent_dir = os.path.join(REPO_ROOT, "eval_results", "plain_baseline")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = (f"{args.run_label}_{timestamp}"
                 if args.run_label else timestamp)
    output_dir = os.path.join(parent_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run output dir: {output_dir}")

    # ── Device setup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    # ── Load Hydra config (without changing cwd) ────────────────────────────
    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false",
        "wandb_project=Experiments",
    ]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
        # Also override eval_max_batch_size — the evaluator uses this for
        # actual batching, not `batch_size`. The 5t config sets it to 4.
        overrides.append(f"eval_max_batch_size={args.batch_size}")
    if args.filter_by_length is not None:
        overrides.append(f"filter_by_length={args.filter_by_length}")
    if args.cache_size is not None:
        overrides.append(f"cache_size={args.cache_size}")
        overrides.append(f"max_memory_length={args.cache_size}")
    # Protect the chat template tail from NAMM eviction. The generation
    # prompt (<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n)
    # is 5 tokens. Without protection, NAMM evicts these 100% of the time,
    # causing the model to produce verbose, non-stopping outputs.
    overrides.append("+protected_tail_n=5")
    if args.task_config is not None:
        overrides.append(f"task@_global_={args.task_config}")

    with initialize(version_base=None, config_path="../config",
                    job_name="eval_namm_splits"):
        cfg = compose(config_name="config", overrides=overrides)

    # ── Build model ──────────────────────────────────────────────────────────
    memory_policy = None
    memory_model = None
    if args.plain:
        # Plain LLaMA: no NAMM wrapper, no eviction policy.
        print("Loading plain LLaMA (no NAMM wrapper)...")
        with torch.no_grad():
            pretrained_llm = hydra.utils.call(
                cfg.pretrained_llm, _convert_="object")
            tokenizer_for_model = hydra.utils.call(cfg.tokenizer)
        pretrained_llm = pretrained_llm.to(device)
        pretrained_llm.eval()
        filter_by = cfg.get('filter_by_length', 8192)
        memory_evaluator = MemoryHFEvaluator(
            model=pretrained_llm,
            tokenizer=tokenizer_for_model,
            eval_max_batch_size=args.batch_size or cfg.get(
                'eval_max_batch_size', 4),
            batch_size=args.batch_size or cfg.get('batch_size', 4),
            max_conditioning_length=filter_by,
            max_memory_length=filter_by,
            max_gen_tokens=cfg.get("max_gen_tokens", 512),
            add_bos_token=cfg.get("add_bos_token", True),
            device=device,
        )
        eval_mode = "plain"
        print(f"  Model: {pretrained_llm.config._name_or_path} "
              f"(is_memory_model={memory_evaluator.is_memory_model})")
    else:
        print("Building NAMM model...")
        with torch.no_grad():
            (memory_policy, memory_model, memory_evaluator,
             evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)
        memory_model.to(device)
        memory_evaluator.device = device

    # ── NAMM policy setup (skipped for --plain) ─────────────────────────────
    if not args.plain:
        # Optional: swap to CLASSIC recency (last-N) policy.
        # The default --no-checkpoint path (eval_mode=recency_baseline) uses
        # untrained DeepMP at scoring_initializer=0, which is NOT the
        # StreamingLLM-style last-N baseline. --use_classic_recency replaces
        # the entire eviction policy with namm.policy.base.Recency.
        if args.use_classic_recency:
            if args.namm_checkpoint:
                raise ValueError(
                    "--use_classic_recency cannot be combined with "
                    "--namm_checkpoint.")
            from namm.policy.base import Recency
            cs = args.cache_size or cfg.get('cache_size', 1024)
            recency_policy = Recency(cache_size=cs)
            memory_evaluator.swap_memory_policy(recency_policy)
            memory_policy = recency_policy
            eval_mode = "classic_recency"
            print(f"  Swapped to classic Recency policy (last-{cs} tokens)")

        # Load NAMM checkpoint or use init params (recency baseline)
        if args.use_classic_recency or args.truncate_input_to is not None:
            pass  # Already handled above
        elif args.namm_checkpoint:
            print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
            ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                              weights_only=False)
            evo_state = ckpt['evolution_state']
            print(f"  Checkpoint iter: {ckpt['iter_num']}, "
                  f"best_val: {ckpt['best_val_loss']:.6f}")

            prefer_mean = cfg.get('prefer_mean_to_best', True)
            if prefer_mean and 'mean' in evo_state:
                params_vec = evo_state['mean']
                print(f"  Using CMA-ES mean "
                      f"(prefer_mean_to_best={prefer_mean})")
            else:
                params_vec = evo_state['best_member']
                print(f"  Using best_member")
            params = params_vec.unsqueeze(0).to(device)
            memory_model.set_memory_params(params)

            buffers_prefix = 'stored_buffers_to_save.'
            buffers_dict = {
                k[len(buffers_prefix):]: v.to(device)
                for k, v in evo_state.items()
                if k.startswith(buffers_prefix)
            }
            if buffers_dict:
                memory_model.load_buffers_dict(buffers_dict=buffers_dict)
            print(f"  NAMM loaded ({params_vec.shape[0]} params)")
            eval_mode = "namm"
        else:
            init_param = memory_policy.get_init_param_values()
            if init_param is None:
                # Stateless heuristic policy (e.g. Recency, H2O,
                # ScissorHands): no learnable parameters to set.
                print("  No checkpoint — stateless heuristic policy "
                      f"({type(memory_policy).__name__})")
                eval_mode = type(memory_policy).__name__.lower()
            else:
                # Learnable NAMM with no checkpoint → init params at
                # scoring_initializer=0 is NOT a recency baseline; it is
                # NAMM with random/zero scoring. With a LoRA adapter at
                # a cache_size smaller than the prompt length this
                # collapses to F1=0 on every prompt (see
                # docs/m1_recency_investigation.md). If cache_size is
                # large enough that no eviction happens (cs=None or
                # cs >= filter_by_length), the scoring is irrelevant and
                # the run is effectively "LoRA at full cache" — the
                # canonical M1 full-cache path uses cs=8192 with
                # filter_by_length=8192 and succeeds.
                cs = cfg.get('cache_size', None)
                max_ctx = cfg.get('filter_by_length', None)
                eviction_possible = (
                    cs is not None
                    and (max_ctx is None or cs < max_ctx)
                )
                if args.lora_checkpoint and eviction_possible:
                    raise ValueError(
                        "Refusing to run a learnable NAMM policy "
                        f"({type(memory_policy).__name__}) at its init "
                        "params alongside --lora_checkpoint with "
                        f"cache_size={cs} < filter_by_length={max_ctx}. "
                        "This is the landmine documented in "
                        "docs/m1_recency_investigation.md — untrained "
                        "NAMM scores evict tokens arbitrarily and F1 "
                        "collapses to 0.00. Pick one: (a) pass "
                        "--namm_checkpoint to load a trained NAMM, "
                        "(b) pass --use_classic_recency for a true "
                        "last-N recency baseline, (c) raise --cache_size "
                        "to >= filter_by_length to disable eviction, or "
                        "(d) switch --run_config to a stateless-policy "
                        "preset such as full_cache_baseline_llama32_1b "
                        "or recency_baseline_llama32_1b.")
                params = init_param.unsqueeze(0).to(device)
                memory_model.set_memory_params(params)
                if eviction_possible:
                    print("  No checkpoint — using learnable NAMM at "
                          "init params (NOT a true recency baseline).")
                    eval_mode = "namm_init_baseline"
                else:
                    print(f"  No checkpoint — cache_size={cs} >= "
                          f"filter_by_length={max_ctx}, no eviction "
                          "will occur (effective full-cache run).")
                    eval_mode = "full_cache"

        # Policy bookkeeping (DynamicMemoryPolicy only — classic Recency
        # and plain LLaMA have no learnable params/stat buffers).
        if (not args.use_classic_recency
                and args.truncate_input_to is None
                and memory_policy.get_init_param_values() is not None):
            batch_idxs = np.zeros([1])
            memory_policy.set_params_batch_idxs(batch_idxs)
            memory_policy.record_eval_stats = True
            memory_policy.initialize_stat_objects()

    # ── Load LoRA checkpoint (optional) ─────────────────────────────────────
    if args.lora_checkpoint:
        print(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
        lora_ckpt = torch.load(args.lora_checkpoint, map_location="cpu",
                               weights_only=False)
        if 'lora_state_dict' not in lora_ckpt:
            raise ValueError(
                f"Checkpoint at {args.lora_checkpoint} has no "
                f"'lora_state_dict' key. Top-level keys: "
                f"{list(lora_ckpt.keys())[:10]}")
        lora_cfg = lora_ckpt.get('lora_config', {})
        lora_rank = lora_cfg.get('rank', 8)
        lora_targets = lora_cfg.get('target_modules', ['q_proj', 'v_proj'])
        print(f"  lora_config: rank={lora_rank} targets={lora_targets} "
              f"step={lora_ckpt.get('best_step', '?')} "
              f"val={lora_ckpt.get('best_val_score', '?')}")

        # Inject LoRA adapters into memory_model.model.
        memory_model.apply_lora_adapters(
            rank=lora_rank, target_modules=lora_targets)
        memory_model.to(device)
        loaded = 0
        missing = []
        lora_sd = lora_ckpt['lora_state_dict']
        for n, p in memory_model.model.named_parameters():
            if n in lora_sd:
                p.data.copy_(lora_sd[n].to(p.device, dtype=p.dtype))
                loaded += 1
        for k in lora_sd:
            found = any(
                n == k
                for n, _ in memory_model.model.named_parameters())
            if not found:
                missing.append(k)
        print(f"  Loaded {loaded}/{len(lora_sd)} LoRA tensors")
        if missing:
            raise RuntimeError(
                f"LoRA checkpoint has {len(missing)} keys not present "
                f"in the model after apply_lora_adapters; first few: "
                f"{missing[:5]}")
        if loaded == 0:
            raise RuntimeError(
                "LoRA checkpoint had matching keys but 0 tensors were "
                "copied — check rank/targets and model wrapping.")
        eval_mode = ((eval_mode + "+lora") if args.namm_checkpoint
                     else "lora")

    # ── Optional: tail-only input truncation (StreamingLLM rolling-window) ──
    # When --truncate_input_to N is set, EVERY prompt is reduced to its last
    # N tokens *before* the model sees it. There is no KV cache eviction at
    # all — the truncation happens in input space, not cache space.
    #
    # Implementation: tokenize each prompt string with the same tokenizer
    # the evaluator will use, slice the last N token ids, decode back to a
    # string, and stash the truncated string back onto the task sampler.
    # The downstream eval pipeline (tok_batch_encode → model.generate)
    # then runs UNCHANGED on the shorter strings — no monkey-patching, no
    # mid-cropping interference, no fiddling with evaluator internals.
    #
    # The decode→re-encode round-trip can shift the token count by ±1 in
    # rare cases (BPE merges across the boundary), but never by more than
    # a handful, so the effective input length is N±a few. This is the
    # tradeoff for keeping the model path completely untouched.
    if args.truncate_input_to is not None:
        if args.namm_checkpoint:
            raise ValueError(
                "--truncate_input_to is incompatible with --namm_checkpoint. "
                "NAMM is meant to be evaluated on the full prompt; truncating "
                "the input would defeat its purpose.")
        if args.use_classic_recency:
            raise ValueError(
                "--truncate_input_to is incompatible with --use_classic_recency. "
                "Pick one: tail-only input truncation OR cache-side eviction.")
        # Note: this is set up here, BEFORE the task sampler is built.
        # The sampler builds prompts lazily, so we'll apply the truncation
        # inside a hook on get_lb_request_strings (see below).
        # Swap the memory policy to a no-op so NAMM's STFT/scoring/topk
        # pipeline is never invoked. Recency(cache_size=None) returns the
        # KV cache unchanged — no eviction, no overhead.
        from namm.policy.base import Recency
        noop_policy = Recency(cache_size=None)
        memory_evaluator.swap_memory_policy(noop_policy)
        memory_policy = noop_policy
        eval_mode = "trunc"
        print(f"  Will truncate every prompt to its last "
              f"{args.truncate_input_to} tokens. NAMM deactivated "
              f"(swapped to no-op policy, no eviction).")

    # ── Create task sampler with EXACT same filtering/splits as run_namm.py ─
    print("Creating task sampler (replicating run_namm.py filtering)...")
    task_sampler = make_task_sampler(cfg=cfg)
    # Capture raw model generations (pred + answers + length + prompt_idx) so
    # post-hoc analyses (BLEU, exact match, hallucination, qualitative) don't
    # require re-running the model.
    task_sampler.store_gen_outputs = True

    # Step 1: filter answers by token count (same as run_namm.py line 104)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    print(f"  max_answer_tokens: {max_answer_tok}")

    # Step 2: apply 3-way train/val/test split with token length filtering
    # (same as run_namm.py lines 106-121)
    train_frac = cfg.get('train_frac', 0.7)
    val_frac = cfg.get('val_frac', 0.15)
    max_cond = cfg.get('split_max_conditioning_length',
                       cfg.get('max_conditioning_length', 6500))
    min_cond = cfg.get('min_conditioning_length', None)
    ext_max_cond = (args.extended_max_conditioning_length
                    if "extended_test" in args.splits else None)
    if ext_max_cond is not None:
        # Stage-1 word filter inside TaskSampler.__init__ uses
        # filter_by_length / 1.3 as max words. To preserve prompts in
        # (max_cond, ext_max_cond] tokens, filter_by_length must be ≥ ext_max_cond.
        cur_filter = cfg.get('filter_by_length', None)
        if cur_filter is not None and cur_filter < ext_max_cond:
            print(f"WARNING: filter_by_length={cur_filter} < "
                  f"extended_max_conditioning_length={ext_max_cond}; "
                  f"prompts above ~{cur_filter/1.3:.0f} words were already "
                  f"dropped at construction. Consider passing "
                  f"--filter_by_length {ext_max_cond}.")
    task_sampler.apply_train_val_test_split(
        train_frac=train_frac,
        val_frac=val_frac,
        max_conditioning_length=max_cond,
        min_conditioning_length=min_cond,
        tokenizer=tokenizer,
        extended_max_conditioning_length=ext_max_cond,
    )
    print(f"  train_frac={train_frac}, val_frac={val_frac}")
    print(f"  min_conditioning_length={min_cond}, "
          f"max_conditioning_length={max_cond}")
    if ext_max_cond is not None:
        print(f"  extended_max_conditioning_length={ext_max_cond}")

    # Wrap prompts in the Llama 3 Instruct chat template so the model
    # sees the same framing it was instruction-tuned on. Without this, the
    # instruct model treats the input as raw text completion and doesn't
    # stop at <|eot_id|>, producing verbose outputs that depress F1.
    # This matches run_lora.py:276 which applies the same template during
    # LoRA training and eval.
    task_sampler.apply_chat_template_to_prompts(tokenizer)

    # Show per-task sample counts
    for task_n, n in task_sampler.num_prompts_per_lb_task.items():
        print(f"  Task: {task_n}, total eligible samples: {n}")

    # ── Section C: per-prompt dump mode (no generation, no F1) ──────────────
    if args.dump_namm_state:
        print(f"\nSection C dump mode → {args.dump_namm_state}")
        _run_section_c_dump_loop(
            args=args,
            memory_policy=memory_policy,
            memory_model=memory_model,
            memory_evaluator=memory_evaluator,
            task_sampler=task_sampler,
            tokenizer=tokenizer,
            device=device,
            timestamp=timestamp,
            train_frac=train_frac,
            val_frac=val_frac,
            max_cond=max_cond,
            min_cond=min_cond,
            ext_max_cond=ext_max_cond,
        )
        return

    # ── Apply tail-only input truncation at the STRING level ───────────────
    # If --truncate_input_to N is set, replace every prompt string with one
    # that decodes from its last N token ids. Done HERE (after all sampler
    # filtering is complete) so no other code path needs to know about it.
    # The downstream evaluator runs unchanged on the shorter strings — no
    # monkey-patching, no mid-cropping interference, nothing that could
    # affect the model path itself.
    if args.truncate_input_to is not None:
        n_truncate = args.truncate_input_to
        total_truncated = 0
        total_prompts = 0
        max_orig_len = 0
        max_new_len = 0
        for task_name, prompts in task_sampler.lb_prompts_per_task.items():
            new_prompts = []
            for p in prompts:
                ids = tokenizer(p, add_special_tokens=False).input_ids
                total_prompts += 1
                max_orig_len = max(max_orig_len, len(ids))
                if len(ids) > n_truncate:
                    tail_ids = ids[-n_truncate:]
                    new_p = tokenizer.decode(
                        tail_ids, skip_special_tokens=False)
                    new_prompts.append(new_p)
                    total_truncated += 1
                    max_new_len = max(max_new_len, len(tail_ids))
                else:
                    new_prompts.append(p)
                    max_new_len = max(max_new_len, len(ids))
            task_sampler.lb_prompts_per_task[task_name] = new_prompts
        print(f"  Tail-truncated {total_truncated}/{total_prompts} prompts "
              f"to last {n_truncate} tokens "
              f"(max orig len = {max_orig_len}, max new len ≤ {max_new_len}). "
              f"Decoded back to strings; downstream eval unchanged.")

    # ── Evaluate on requested splits ────────────────────────────────────────
    all_results = {}
    # Sidecar buffers — written to disk once after the eval loop:
    #   - generations_per_split: full model outputs per prompt per task per split
    #   - eviction_traces_per_split: raw per-layer per-step arrays (numpy)
    generations_per_split = {}
    eviction_traces_per_split = {}
    for split_name in args.splits:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on split: {split_name}")
        print('=' * 60)

        # train=True for train/val splits (uses lb_training_tasks),
        # train=False for test split (uses lb_test_tasks).
        is_train = split_name in ('train', 'val')

        with torch.no_grad():
            score_dicts = task_sampler.evaluate(
                lm=memory_evaluator,
                split=split_name,
                train=is_train,
                evolved_model=False,
                pop_reps=1,
                resample_requests=True,
                sampled_requests_per_task=None,
                performance_per_request=True,
            )
        scores = score_dicts[0]
        # Pull per-prompt F1 dict aside before the dict-vs-scalar split.
        # Layout: {task_name: {orig_lb_idx: f1_score}}
        per_prompt_f1 = scores.pop('performance_per_request', {})
        # Compute tasks_aggregate (same as NAMM training):
        # mean(per_task_F1 / 100) / 100
        task_f1s = [v for k, v in scores.items() if not isinstance(v, dict)]
        tasks_aggregate = np.mean([f / 100 for f in task_f1s]) / 100
        scores['tasks_aggregate'] = tasks_aggregate
        # mean_f1 = MACRO average (each task counts as 1/n_tasks).
        mean_f1 = np.mean(task_f1s)
        scores['mean_f1'] = mean_f1
        # micro_mean_f1 = prompt-count-weighted average — matches the
        # val_lb_avg_f1 metric printed during LoRA training so the two are
        # directly comparable. Computed from per_prompt_f1 so we don't have
        # to redo any scoring.
        # per_prompt_f1 values are raw 0-1 from get_score's all_scores list,
        # while task F1s and mean_f1 are 0-100. Multiply by 100 to match.
        all_prompt_scores = []
        for task_dict in per_prompt_f1.values():
            all_prompt_scores.extend(task_dict.values())
        if all_prompt_scores:
            scores['micro_mean_f1'] = float(np.mean(all_prompt_scores)) * 100.0
            scores['n_prompts_total'] = int(len(all_prompt_scores))
            scores['n_prompts_per_task'] = {
                task: len(d) for task, d in per_prompt_f1.items()}
        # Record per-prompt F1 inside the split's scores so it lands in JSON.
        scores['per_prompt_f1'] = per_prompt_f1

        # Snapshot generations BEFORE the next split overwrites them. Convert
        # numpy ints from prompt_idx to plain Python int for JSON safety.
        gens = task_sampler.last_gen_outputs or {}
        generations_per_split[split_name] = {
            task: [
                {**d, 'prompt_idx': int(d.get('prompt_idx', -1))}
                for d in dicts
            ]
            for task, dicts in gens.items()
        }

        # Snapshot RAW per-layer per-step eviction arrays BEFORE
        # get_param_stats(reset=True) clears them. Plain/Recency modes have
        # no such buffers, so the snapshot is empty there.
        has_policy_stats = (
            memory_policy is not None
            and not args.use_classic_recency
            and not args.plain
            and memory_policy.get_init_param_values() is not None
        )
        if has_policy_stats:
            eviction_traces_per_split[split_name] = (
                _snapshot_raw_eviction_arrays(memory_policy))
        else:
            eviction_traces_per_split[split_name] = {}

        # Collect eviction/cache stats from memory policy (DynamicMemoryPolicy
        # only — classic Recency and plain LLaMA have no stats).
        if has_policy_stats:
            mem_stats = memory_policy.get_param_stats(reset=True)
            memory_policy.initialize_stat_objects()
        else:
            mem_stats = {}
        eviction_summary = {}
        if mem_stats:
            dyn_cache = mem_stats.get('mem_stats/overall/dynamic_cache_sizes', None)
            final_cache = mem_stats.get('mem_stats/overall/final_dynamic_cache_sizes', None)
            unmasked = mem_stats.get('mem_stats/overall/unmasked_samples', None)
            unmasked_final = mem_stats.get('mem_stats/overall/unmasked_sample_final', None)
            cache_size_used = args.cache_size or cfg.get('cache_size', 1024)
            eviction_summary = {
                'avg_dynamic_cache_size': dyn_cache,
                'avg_final_cache_size': final_cache,
                'avg_unmasked_tokens': unmasked,
                'avg_final_unmasked_tokens': unmasked_final,
                'cache_budget': cache_size_used,
                'budget_utilization_pct': (dyn_cache / cache_size_used * 100) if dyn_cache else None,
            }
            scores['eviction_stats'] = eviction_summary

        all_results[split_name] = dict(scores)

        print(f"\nResults ({split_name}):")
        for k, v in sorted(scores.items()):
            if k in ('eviction_stats', 'per_prompt_f1',
                     'n_prompts_per_task'):
                continue
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        print(f"  --- tasks_aggregate: {tasks_aggregate:.6f} (mean_f1: {mean_f1:.2f})")
        if eviction_summary:
            print(f"\n  Eviction stats ({split_name}):")
            print(f"    Cache budget:              {eviction_summary['cache_budget']}")
            print(f"    Avg dynamic cache:         {eviction_summary['avg_dynamic_cache_size']:.1f} (across all steps)")
            print(f"    Avg final cache:           {eviction_summary['avg_final_cache_size']:.1f} (end of prompt)")
            print(f"    Avg unmasked tokens:       {eviction_summary['avg_unmasked_tokens']:.1f} (across all steps)")
            print(f"    Avg final unmasked tokens: {eviction_summary['avg_final_unmasked_tokens']:.1f} (end of prompt)")

    # ── Summary across splits ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    # Collect all metric keys seen across splits
    all_metric_keys = set()
    for scores in all_results.values():
        all_metric_keys.update(scores.keys())

    for metric in sorted(all_metric_keys):
        if metric in ('eviction_stats', 'per_prompt_f1',
                      'n_prompts_per_task'):
            continue
        vals = []
        for split_name in args.splits:
            if metric in all_results.get(split_name, {}):
                v = all_results[split_name][metric]
                if isinstance(v, (int, float)):
                    vals.append(f"{split_name}={v:.4f}")
        if vals:
            print(f"  {metric}: {', '.join(vals)}")

    # ── Save results as JSON ────────────────────────────────────────────────
    # Coerce numpy ints (used as prompt index keys) to plain Python ints so
    # the JSON serializer accepts them.
    for split_name, split_scores in all_results.items():
        ppf1 = split_scores.get('per_prompt_f1', {})
        split_scores['per_prompt_f1'] = {
            task: {int(k): float(v) for k, v in task_dict.items()}
            for task, task_dict in ppf1.items()
        }

    results_payload = {
        "type": "eval_namm_splits",
        "timestamp": timestamp,
        "config": {
            "namm_checkpoint": os.path.abspath(args.namm_checkpoint) if args.namm_checkpoint else None,
            "lora_checkpoint": os.path.abspath(args.lora_checkpoint) if args.lora_checkpoint else None,
            "eval_mode": eval_mode,
            "run_config": args.run_config,
            "cache_size": args.cache_size,
            "filter_by_length": args.filter_by_length,
            "batch_size": args.batch_size,
            "splits_evaluated": args.splits,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "max_conditioning_length": max_cond,
            "min_conditioning_length": min_cond,
            "max_answer_tokens": max_answer_tok,
            "extended_max_conditioning_length": ext_max_cond,
            "truncate_input_to": args.truncate_input_to,
            "use_classic_recency": args.use_classic_recency,
        },
        "scores_per_split": all_results,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Sidecar: full per-prompt generations (pred, answers, length, idx) ──
    # Kept separate from results.json because individual predictions can be
    # long. One file per run, all splits + all tasks inside.
    gens_path = os.path.join(output_dir, "generations.json")
    with open(gens_path, "w") as f:
        json.dump(generations_per_split, f, indent=2)
    print(f"Generations saved: {gens_path}")

    # ── Sidecar: raw per-layer per-step eviction arrays (.npz) ──────────────
    # Layout: keys are "{split}/{signal}", values are object arrays
    # of shape (num_layers,) where each entry is a 1-D float32 array of
    # per-step values for that layer. Load with `np.load(..., allow_pickle=True)`.
    has_eviction = any(traces for traces in eviction_traces_per_split.values())
    if has_eviction:
        npz_payload = {}
        for split_name, traces in eviction_traces_per_split.items():
            for signal_name, arr in traces.items():
                npz_payload[f"{split_name}/{signal_name}"] = arr
        npz_path = os.path.join(output_dir, "eviction_traces.npz")
        np.savez(npz_path, **npz_payload)
        print(f"Raw eviction traces saved: {npz_path}")


if __name__ == "__main__":
    main()
