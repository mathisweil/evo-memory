"""Cross-reference NAMM retained positions with gold answer token positions.

For each prompt in the retention dump:
1. Find where the gold answer appears in the original prompt (token-level)
2. Check what fraction of answer tokens were retained by NAMM
3. Compare with what truncation would have kept (last N tokens)
4. Report per-task stats

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/_claude_analyze_retention_vs_answers.py \
        --dump analysis_out/retention_dumps/<file>.jsonl.gz \
        --cache_size 1024
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = "/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo"
sys.path.insert(0, PROJ)

from datasets import load_dataset
from transformers import AutoTokenizer


def find_answer_token_positions(prompt_text, answer_texts, tokenizer):
    """Find token-level positions of gold answers in the prompt.

    Returns a set of token indices where any gold answer appears.
    Uses character-level substring match → maps to token indices.
    """
    prompt_lower = prompt_text.lower()
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    # Build char→token mapping
    # Decode each token to get its character span
    char_to_tok = [None] * len(prompt_text)
    char_offset = 0
    for tok_idx, tok_id in enumerate(prompt_ids):
        tok_str = tokenizer.decode([tok_id])
        # Find this token's characters in the prompt
        # (approximate: BPE decode may not perfectly round-trip)
        start = prompt_text.find(tok_str, char_offset)
        if start == -1:
            # Fallback: assign based on position
            tok_len = len(tok_str)
            start = char_offset
        for c in range(start, min(start + len(tok_str), len(prompt_text))):
            char_to_tok[c] = tok_idx
        char_offset = start + len(tok_str)

    answer_token_positions = set()
    for ans in answer_texts:
        ans_lower = ans.strip().lower()
        if not ans_lower or ans_lower in ("unanswerable",):
            continue
        # Find all occurrences
        search_start = 0
        while True:
            idx = prompt_lower.find(ans_lower, search_start)
            if idx == -1:
                break
            for c in range(idx, min(idx + len(ans_lower), len(prompt_text))):
                if char_to_tok[c] is not None:
                    answer_token_positions.add(char_to_tok[c])
            search_start = idx + 1

    return answer_token_positions, len(prompt_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to .jsonl.gz retention dump")
    parser.add_argument("--cache_size", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Load LongBench data
    lb_data = {}
    prompt_templates = json.load(
        open(f"{PROJ}/data/longbench/dataset2prompt.json"))
    prompt_templates = {f"lb/{k}": v for k, v in prompt_templates.items()}

    for task in ["qasper", "qasper_e", "2wikimqa", "2wikimqa_e", "hotpotqa_e"]:
        ds = load_dataset("THUDM/LongBench", task, split="test",
                          trust_remote_code=True)
        lb_data[f"lb/{task}"] = ds

    # Read the retention dump
    records = []
    with gzip.open(args.dump, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_header"):
                print(f"Dump header: eval_mode={rec.get('eval_mode')}, "
                      f"cache_size={rec.get('cache_size')}")
                continue
            records.append(rec)

    print(f"Loaded {len(records)} prompt records")
    cs = args.cache_size

    # Per-task analysis
    task_stats = defaultdict(lambda: {
        "n": 0,
        "n_with_answer": 0,
        "namm_answer_recall": [],      # fraction of answer tokens retained
        "trunc_answer_recall": [],     # fraction kept by last-N truncation
        "namm_frac_first_third": [],
        "namm_frac_middle_third": [],
        "namm_frac_last_third": [],
        "prompt_lengths": [],
    })

    for rec in records:
        task = rec["task"]
        original_idx = rec["original_idx"]
        n_input = rec["n_input_tokens"]

        if task not in lb_data:
            continue

        # Get original prompt + answers
        ds = lb_data[task]
        if original_idx >= len(ds):
            continue
        example = ds[int(original_idx)]
        prompt_text = prompt_templates[task].format(**example)
        answers = example.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        # Find answer token positions in the raw prompt
        answer_toks, n_prompt_toks = find_answer_token_positions(
            prompt_text, answers, tokenizer)

        stats = task_stats[task]
        stats["n"] += 1
        stats["prompt_lengths"].append(n_input)

        if not answer_toks:
            continue
        stats["n_with_answer"] += 1

        # NAMM: get union of kept positions across all layers
        # Use layer 0 as representative (or union across layers)
        kept_per_layer_per_head = rec["kept_per_layer_per_head"]
        # Union across all heads in all layers
        namm_kept = set()
        for layer_heads in kept_per_layer_per_head:
            for head_positions in layer_heads:
                namm_kept.update(head_positions)

        # Also get per-layer thirds from the pre-computed metrics
        per_layer = rec.get("per_layer", [])
        if per_layer:
            # Average across layers
            stats["namm_frac_first_third"].append(
                np.mean([l["frac_first_third"] for l in per_layer]))
            stats["namm_frac_middle_third"].append(
                np.mean([l["frac_middle_third"] for l in per_layer]))
            stats["namm_frac_last_third"].append(
                np.mean([l["frac_last_third"] for l in per_layer]))

        # Truncation: keeps last cs tokens (indices n_input-cs .. n_input-1)
        # But note: the retention dump uses the CHAT-TEMPLATED prompt, so
        # n_input includes template tokens. Truncation baseline also operates
        # on the templated prompt. The answer positions are in the RAW prompt
        # space. We need to be careful here.
        # Actually, answer_toks are found in the RAW prompt. The retained
        # indices are in the TEMPLATED prompt. There's an offset from the
        # chat template header.
        # For a fair comparison, compute the offset:
        chat_header = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True, tokenize=False)
        bos = getattr(tokenizer, 'bos_token', '') or ''
        if bos and chat_header.startswith(bos):
            chat_header = chat_header[len(bos):]
        header_toks = len(tokenizer(chat_header, add_special_tokens=True).input_ids)
        # Shift answer positions by header_toks to align with templated positions
        answer_toks_shifted = {p + header_toks for p in answer_toks}

        # NAMM recall: how many answer tokens did NAMM keep?
        namm_kept_answers = answer_toks_shifted & namm_kept
        namm_recall = len(namm_kept_answers) / len(answer_toks_shifted)
        stats["namm_answer_recall"].append(namm_recall)

        # Truncation recall: last cs tokens
        trunc_kept = set(range(max(0, n_input - cs), n_input))
        trunc_kept_answers = answer_toks_shifted & trunc_kept
        trunc_recall = len(trunc_kept_answers) / len(answer_toks_shifted)
        stats["trunc_answer_recall"].append(trunc_recall)

    # Report
    print(f"\n{'='*90}")
    print(f"  NAMM vs Truncation: Answer Token Retention (cache_size={cs})")
    print(f"{'='*90}")
    print(f"{'Task':15s} {'N':>4s} {'w/ans':>5s} "
          f"{'NAMM recall':>12s} {'Trunc recall':>12s} {'Δ':>8s} "
          f"{'1st/3':>6s} {'mid/3':>6s} {'lst/3':>6s} {'avg_len':>8s}")
    print("-" * 90)

    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        if s["n"] == 0:
            continue
        t = task.replace("lb/", "")
        nr = np.mean(s["namm_answer_recall"]) if s["namm_answer_recall"] else 0
        tr = np.mean(s["trunc_answer_recall"]) if s["trunc_answer_recall"] else 0
        f1 = np.mean(s["namm_frac_first_third"]) if s["namm_frac_first_third"] else 0
        f2 = np.mean(s["namm_frac_middle_third"]) if s["namm_frac_middle_third"] else 0
        f3 = np.mean(s["namm_frac_last_third"]) if s["namm_frac_last_third"] else 0
        al = np.mean(s["prompt_lengths"])
        print(f"{t:15s} {s['n']:4d} {s['n_with_answer']:5d} "
              f"{nr:12.1%} {tr:12.1%} {nr-tr:+7.1%} "
              f"{f1:6.1%} {f2:6.1%} {f3:6.1%} {al:8.0f}")

    # Detailed per-prompt dump for qasper
    print(f"\n{'='*90}")
    print(f"  Per-prompt detail: qasper + qasper_e")
    print(f"{'='*90}")
    for rec in records:
        task = rec["task"]
        if "qasper" not in task:
            continue
        original_idx = rec["original_idx"]
        n_input = rec["n_input_tokens"]

        ds = lb_data.get(task)
        if ds is None or original_idx >= len(ds):
            continue
        example = ds[int(original_idx)]
        prompt_text = prompt_templates[task].format(**example)
        answers = example.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        answer_toks, _ = find_answer_token_positions(
            prompt_text, answers, tokenizer)

        chat_header = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True, tokenize=False)
        bos = getattr(tokenizer, 'bos_token', '') or ''
        if bos and chat_header.startswith(bos):
            chat_header = chat_header[len(bos):]
        header_toks = len(tokenizer(chat_header, add_special_tokens=True).input_ids)
        answer_toks_shifted = {p + header_toks for p in answer_toks}

        kept_all = set()
        for lh in rec["kept_per_layer_per_head"]:
            for h in lh:
                kept_all.update(h)
        trunc_kept = set(range(max(0, n_input - cs), n_input))

        if answer_toks_shifted:
            nr = len(answer_toks_shifted & kept_all) / len(answer_toks_shifted)
            tr = len(answer_toks_shifted & trunc_kept) / len(answer_toks_shifted)
        else:
            nr = tr = float('nan')

        ans_positions = sorted(answer_toks_shifted) if answer_toks_shifted else []
        ans_range = f"[{min(ans_positions)}-{max(ans_positions)}]" if ans_positions else "none"
        ans_rel = f"{np.mean(ans_positions)/n_input:.2f}" if ans_positions else "-"

        print(f"  {task.replace('lb/',''):10s} idx={original_idx:3d} "
              f"len={n_input:5d} ans_pos={ans_range:15s} "
              f"rel={ans_rel:5s} "
              f"NAMM={nr:5.1%} Trunc={tr:5.1%} "
              f"ans={repr(answers[0][:40]) if answers else '-'}")


if __name__ == "__main__":
    main()
