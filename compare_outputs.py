#!/usr/bin/env python3
"""
compare_outputs.py — qualitative comparison of base vs LoRA model on QASPER.

Loads ONE model (base LLM), generates answers, then hot-swaps LoRA weights and
generates again. Keeps VRAM usage to a single model at a time.

Usage:
  python compare_outputs.py --ckpt results/m1/1337/ckpt.pt [--n 5] [--seed 42]

Options:
  --ckpt        Path to LoRA checkpoint (ckpt.pt from lora_grad trainer)
  --n           Number of QASPER examples to compare (default: 5)
  --seed        RNG seed for example selection (default: 42)
  --device      Device (default: cuda)
  --artifact-dir  Dir containing config.yaml; defaults to results/m1/{seed}/
"""
import argparse
import json
import os
import random
import sys
import yaml
import torch
from datasets import load_dataset

HF_CACHE = '/cs/student/project_msc/2025/csml/gmaralla/.hf_cache'
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# QASPER prompt template — matches LongBench/config/dataset2prompt.json
QASPER_TEMPLATE = (
    "You are given a scientific article and a question. Answer the question as concisely "
    "as you can, using a single phrase or sentence if possible. If the question cannot be "
    "answered based on the information in the article, write \"unanswerable\". If the "
    "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
    "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the "
    "above article as concisely as you can, using a single phrase or sentence if possible. "
    "If the question cannot be answered based on the information in the article, write "
    "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
    "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
)

SEP = '─' * 80


def parse_args():
    p = argparse.ArgumentParser(description='Base vs LoRA qualitative comparison on QASPER')
    p.add_argument('--ckpt', required=True, help='Path to LoRA checkpoint (ckpt.pt)')
    p.add_argument('--n', type=int, default=5, help='Number of examples to compare')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for example selection')
    p.add_argument('--device', default='cuda')
    p.add_argument('--artifact-dir', default=None,
                   help='Dir containing config.yaml (default: results/m1/{seed}/)')
    p.add_argument('--no-truncate', action='store_true',
                   help='Do not truncate context preview (may be very long)')
    return p.parse_args()


def load_qasper_examples(n: int, seed: int):
    """Load n random QASPER test examples from LongBench."""
    print(f"Loading QASPER test split from HF cache...")
    ds = load_dataset(
        'THUDM/LongBench',
        'qasper',
        split='test',
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:n]
    examples = []
    for i in selected:
        ex = ds[i]
        prompt = QASPER_TEMPLATE.format(context=ex['context'], input=ex['input'])
        examples.append({
            'prompt': prompt,
            'question': ex['input'],
            'answers': ex['answers'],
        })
    print(f"  Selected {len(examples)} examples (indices: {selected})")
    return examples


def load_base_model_and_evaluator(device: str):
    """Build WrappedLlamaForCausalLM + MemoryHFEvaluator via standard Hydra path."""
    from hydra.core.global_hydra import GlobalHydra
    from utils_hydra import initialize_cfg
    from main import make_eval_model

    GlobalHydra.instance().clear()
    cfg = initialize_cfg('cfgs', hydra_overrides=[
        'run@_global_=full_cache_baseline_llama32_1b',
        'wandb_log=false',
        'eval_only=true',
    ])

    (_, model, evaluator, _, _) = make_eval_model(cfg)
    model.to(device)
    model.eval()
    return model, evaluator


def apply_lora_weights(model, ckpt_path: str, device: str, artifact_dir: str):
    """
    Apply LoRA adapters to model and load trained weights from checkpoint.

    Uses lora_rank / lora_target_modules from artifact_dir/config.yaml if present;
    falls back to checkpoint's saved lora_config otherwise.
    """
    # Read saved trainer config for LoRA hyperparams
    saved_cfg = {}
    config_yaml = os.path.join(artifact_dir, 'config.yaml')
    if os.path.exists(config_yaml):
        with open(config_yaml) as f:
            saved_cfg = yaml.safe_load(f) or {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer checkpoint's own lora_config over saved_cfg (most reliable)
    if 'lora_config' in ckpt:
        lora_rank = ckpt['lora_config']['rank']
        lora_targets = ckpt['lora_config']['target_modules']
    else:
        lora_rank = saved_cfg.get('lora_rank', 8)
        lora_targets = saved_cfg.get('lora_target_modules', ['q_proj', 'v_proj'])

    print(f"  Applying LoRA: rank={lora_rank}, targets={lora_targets}")
    model.apply_lora_adapters(rank=lora_rank, target_modules=lora_targets)

    if 'lora_state_dict' in ckpt:
        loaded = 0
        for n, p in model.model.named_parameters():
            if p.requires_grad and n in ckpt['lora_state_dict']:
                p.data.copy_(ckpt['lora_state_dict'][n])
                loaded += 1
        print(f"  Loaded {loaded} LoRA weight tensors from {ckpt_path}")
    else:
        print(f"  WARNING: no 'lora_state_dict' in checkpoint. "
              f"Available keys: {list(ckpt.keys())}")

    model.eval()


def generate_answers(evaluator, prompts: list, max_gen_tokens: int = 128) -> list:
    """Run evaluate_lb on a list of prompt strings; returns list of generated strings."""
    # Stop at newline+Question to prevent the model from generating follow-up Q&A rounds
    stop_seqs = ['\nQuestion:', '\n\nQuestion:', '\n\n']
    return evaluator.evaluate_lb(
        dataset_samples=prompts,
        max_gen_tokens=max_gen_tokens,
        stop_gen=stop_seqs,
        disable_tqdm=False,
    )


def print_comparison(examples: list, base_answers: list, lora_answers: list,
                     truncate_ctx: bool = True):
    print(f"\n{'=' * 80}")
    print(f"  BASE vs LoRA — QASPER Qualitative Comparison")
    print(f"{'=' * 80}")
    for i, (ex, base_ans, lora_ans) in enumerate(zip(examples, base_answers, lora_answers)):
        print(f"\n{SEP}")
        print(f"  EXAMPLE {i + 1} / {len(examples)}")
        print(SEP)
        print(f"  Question  : {ex['question']}")
        refs = ex['answers']
        if isinstance(refs, list):
            refs_str = ' | '.join(refs) if refs else '(none)'
        else:
            refs_str = str(refs)
        print(f"  Reference : {refs_str}")
        print()
        print(f"  Base LLM  : {base_ans.strip()}")
        print(f"  LoRA      : {lora_ans.strip()}")
    print(f"\n{SEP}")
    print("  Done.")


def main():
    args = parse_args()
    artifact_dir = args.artifact_dir or os.path.join('results', 'm1', str(args.seed))

    examples = load_qasper_examples(args.n, args.seed)
    prompts = [e['prompt'] for e in examples]

    print("\n=== Step 1: Loading BASE model ===")
    model, evaluator = load_base_model_and_evaluator(args.device)

    print("\n=== Step 2: Generating BASE answers ===")
    base_answers = generate_answers(evaluator, prompts)

    print("\n=== Step 3: Applying LoRA weights ===")
    apply_lora_weights(model, args.ckpt, args.device, artifact_dir)

    print("\n=== Step 4: Generating LoRA answers ===")
    lora_answers = generate_answers(evaluator, prompts)

    print_comparison(examples, base_answers, lora_answers, truncate_ctx=not args.no_truncate)


if __name__ == '__main__':
    main()
