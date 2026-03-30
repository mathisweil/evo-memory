"""
datasets.py — NTP and SFT training datasets for LoRA gradient fine-tuning on LongBench.

Provides:
  - NTPDataset: tokenises LongBench 'context' fields for next-token prediction.
  - SFTDataset: builds prompt+answer sequences with answer-only loss masking.
  - pad_collate_fn: unified right-padding collate for both dataset types.
    Uses length-based padding masks to avoid masking real tokens when
    pad_token_id == eos_token_id (e.g. Llama fallback padding).
"""

import json
import os
import random
from typing import Optional

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

_LONGBENCH_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'longbench', 'dataset2prompt.json'
)


def _load_longbench_task(task_name: str, cache_dir: Optional[str]):
    """Load one LongBench HF test split, stripping the 'lb/' prefix."""
    return load_dataset(
        'THUDM/LongBench',
        task_name.removeprefix('lb/'),
        split='test',
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# NTP Dataset
# ---------------------------------------------------------------------------

class NTPDataset(Dataset):
    """Next-token-prediction dataset built from LongBench context fields.

    Tokenises each document's 'context' field and left-truncates to
    max_seq_len tokens.  Use pad_collate_fn for batching.

    Args:
        task_names  : LongBench task names (with or without 'lb/' prefix).
        tokenizer   : HuggingFace tokenizer.
        max_seq_len : Max tokens after left-truncation. Default 3500.
        cache_dir   : Optional HF datasets cache directory.
        seed        : Seed for one-time shuffle of samples.
    """

    def __init__(
        self,
        task_names: list,
        tokenizer,
        max_seq_len: int = 3500,
        cache_dir: Optional[str] = None,
        seed: int = 1337,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        samples = []
        for task_name in task_names:
            for item in _load_longbench_task(task_name, cache_dir):
                ids = tokenizer.encode(item['context'], add_special_tokens=True)
                if len(ids) > max_seq_len:
                    ids = ids[-max_seq_len:]  # left-truncate: keep most recent context
                samples.append(torch.tensor(ids, dtype=torch.long))

        random.Random(seed).shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        """Return {'input_ids': 1D LongTensor}."""
        return {'input_ids': self.samples[idx]}


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Supervised fine-tuning dataset from LongBench prompt+answer pairs.

    Applies apply_chat_template to each item, records label_start (the index
    of the first answer token), and truncates answer tokens when the full
    sequence exceeds max_seq_len.

    Filtering: items are dropped if the prompt exceeds max_conditioning_length
    tokens (matching TaskSampler's eligibility criteria) or if no answer exists.

    Split: eligible items are shuffled with seed before taking the first
    train_frac, giving a deterministic split that aligns with TaskSampler's
    seeded random split and prevents train/val leakage.

    When multiple task_names have unequal eligible counts, minority tasks are
    upsampled to the count of the largest task.

    Args:
        task_names              : LongBench task names.
        tokenizer               : HuggingFace tokenizer with apply_chat_template.
        max_seq_len             : Max total sequence length; answer tokens are truncated.
        max_conditioning_length : Max prompt tokens; longer prompts are discarded.
        min_conditioning_length : Min prompt tokens; shorter prompts are discarded.
        max_answer_tokens       : Max answer tokens; items whose shortest answer exceeds
                                  this are discarded.
        cache_dir               : Optional HF datasets cache directory.
        seed                    : Seed for split shuffle, upsampling, and final shuffle.
        train_frac              : Fraction of eligible samples used for training.
    """

    def __init__(
        self,
        task_names: list,
        tokenizer,
        max_seq_len: int = 3500,
        max_conditioning_length: int = 6500,
        min_conditioning_length: int = None,
        max_answer_tokens: int = None,
        cache_dir: Optional[str] = None,
        seed: int = 1337,
        train_frac: float = 0.8,
    ):
        super().__init__()
        self.min_conditioning_length = min_conditioning_length
        self.max_answer_tokens = max_answer_tokens

        with open(_LONGBENCH_PROMPT_PATH) as f:
            all_templates = json.load(f)

        bare_names = [t.removeprefix('lb/') for t in task_names]
        missing = [t for t in bare_names if t not in all_templates]
        if missing:
            raise ValueError(
                f"Tasks {missing} not found in {_LONGBENCH_PROMPT_PATH}. "
                f"Available: {list(all_templates.keys())}"
            )

        bos_id = getattr(tokenizer, 'bos_token_id', None)
        samples: list[dict] = []
        n_skipped_no_answer = 0
        n_skipped_too_long = 0
        n_skipped_answer_long = 0

        for task_name in bare_names:
            task_template = all_templates[task_name]
            items = list(_load_longbench_task(task_name, cache_dir))
            n_total = len(items)
            task_skip_long = 0
            task_skip_no_ans = 0
            eligible: list[dict] = []

            for item in items:
                if not item['answers']:
                    task_skip_no_ans += 1
                    n_skipped_no_answer += 1
                    continue

                answers = item['answers']
                if isinstance(answers, str):
                    answers = [answers]
                if max_answer_tokens is not None:
                    answer_lens = [(a, len(tokenizer.encode(a, add_special_tokens=False)))
                                   for a in answers]
                    shortest_len = min(l for _, l in answer_lens)
                    if shortest_len > max_answer_tokens:
                        n_skipped_answer_long += 1
                        continue
                    # Pick shortest answer that fits
                    answer = min(answer_lens, key=lambda x: x[1])[0]
                else:
                    answer = answers[0]

                user_content = task_template.format(**item)
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

                # Exclude BOS from prompt-length check to match TaskSampler's
                # add_special_tokens=False token counting.
                n_prompt_tok = label_start - (
                    1 if bos_id is not None and prompt_ids[0] == bos_id else 0
                )
                if n_prompt_tok > max_conditioning_length:
                    task_skip_long += 1
                    n_skipped_too_long += 1
                    continue
                if min_conditioning_length is not None and n_prompt_tok < min_conditioning_length:
                    task_skip_long += 1
                    n_skipped_too_long += 1
                    continue

                if label_start >= len(full_ids):
                    task_skip_no_ans += 1
                    n_skipped_no_answer += 1
                    continue

                if len(full_ids) > max_seq_len:
                    full_ids = full_ids[:max_seq_len]

                eligible.append({
                    'input_ids': torch.tensor(full_ids, dtype=torch.long),
                    'label_start': label_start,
                    'task_name': task_name,
                })

            n_eligible = len(eligible)

            # Deterministic seeded shuffle before split — aligns with TaskSampler's
            # seeded random split to prevent train/val leakage.
            random.Random(seed).shuffle(eligible)
            n_train = int(n_eligible * train_frac) if train_frac < 1.0 else n_eligible
            eligible = eligible[:n_train]

            lo = min_conditioning_length or 0
            filter_desc = f"(prompt outside [{lo}-{max_conditioning_length}] tok)"
            prompt_lens = [s['label_start'] for s in eligible]
            answer_lens = [len(s['input_ids']) - s['label_start'] for s in eligible]
            print(
                f"  {task_name}: {n_total} total, "
                f"{task_skip_long} filtered {filter_desc}, "
                f"{task_skip_no_ans} no answer -> "
                f"{n_eligible} eligible -> {n_train} train"
            )
            if prompt_lens:
                avg_p = sum(prompt_lens) / len(prompt_lens)
                avg_a = sum(answer_lens) / len(answer_lens)
                print(f"    prompt tokens:  min={min(prompt_lens)}, avg={avg_p:.0f}, max={max(prompt_lens)}")
                print(f"    answer tokens:  min={min(answer_lens)}, avg={avg_a:.0f}, max={max(answer_lens)}")

            samples.extend(eligible)

        print(
            f"\nSFTDataset: {len(samples)} training samples "
            f"({n_skipped_too_long} prompt-out-of-range, "
            f"{n_skipped_answer_long} answer>{max_answer_tokens} tok, "
            f"{n_skipped_no_answer} no-answer skipped)"
        )
        if samples:
            all_seq_lens = [len(s['input_ids']) for s in samples]
            avg_seq = sum(all_seq_lens) / len(all_seq_lens)
            print(f"  Sequence length: min={min(all_seq_lens)}, avg={avg_seq:.0f}, max={max(all_seq_lens)}")

        # Upsample minority tasks to balance per-task contribution.
        task_buckets: dict[str, list] = {}
        for s in samples:
            task_buckets.setdefault(s['task_name'], []).append(s)

        if len(task_buckets) > 1:
            max_count = max(len(v) for v in task_buckets.values())
            rng = random.Random(seed)
            balanced: list = []
            for t_name, t_samples in sorted(task_buckets.items()):
                n_orig = len(t_samples)
                if n_orig < max_count:
                    n_full = max_count // n_orig
                    n_rem = max_count % n_orig
                    upsampled = t_samples * n_full + rng.sample(t_samples, n_rem)
                    print(f"  Upsampling {t_name}: {n_orig} -> {len(upsampled)} (x{max_count/n_orig:.1f})")
                    balanced.extend(upsampled)
                else:
                    balanced.extend(t_samples)
            samples = balanced
            print(f"  Balanced: {len(samples)} total ({max_count} per task x {len(task_buckets)} tasks)")

        random.Random(seed).shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        """Return {'input_ids': 1D LongTensor, 'label_start': int}."""
        s = self.samples[idx]
        return {'input_ids': s['input_ids'], 'label_start': s['label_start']}


# ---------------------------------------------------------------------------
# Unified collate function
# ---------------------------------------------------------------------------

def pad_collate_fn(batch: list, pad_token_id: int, max_seq_len: int) -> dict:
    """Collate NTP or SFT samples into a padded batch.

    Detects mode from batch contents:
    - NTP (no 'label_start'): all non-padding positions are supervised.
    - SFT (has 'label_start'): only tokens from label_start onward are supervised.

    Uses a length-based padding mask rather than ``input_ids == pad_token_id``
    to avoid masking real tokens when pad_token_id reuses a vocab token (e.g.
    Llama's EOS fallback for pad_token_id).

    Args:
        batch        : List of dicts with 'input_ids' (1D LongTensor) and,
                       for SFT mode, 'label_start' (int).
        pad_token_id : Token ID used for right-padding.
        max_seq_len  : Hard cap on output sequence length.

    Returns:
        dict with:
          'input_ids'    : LongTensor [B, T]
          'labels'       : LongTensor [B, T]; -100 at masked positions
          'label_starts' : LongTensor [B] (SFT mode only)

    Raises:
        ValueError: (SFT mode) If all labels are -100 after masking.
    """
    is_sft = 'label_start' in batch[0]
    seqs = [x['input_ids'] for x in batch]

    # Record original lengths before padding/truncation for the mask.
    orig_lens = torch.tensor([min(len(s), max_seq_len) for s in seqs])

    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids[:, :max_seq_len]

    # Length-based padding mask: True at positions that are padding.
    # Safe when pad_token_id == eos_token_id or any other real vocab token.
    T = input_ids.size(1)
    padding_mask = torch.arange(T).unsqueeze(0) >= orig_lens.unsqueeze(1)  # [B, T]

    if is_sft:
        label_starts = [x['label_start'] for x in batch]
        labels = torch.full_like(input_ids, -100)
        for i, ls in enumerate(label_starts):
            labels[i, ls:] = input_ids[i, ls:]
        labels[padding_mask] = -100
        if not (labels != -100).any():
            raise ValueError(
                "pad_collate_fn: all labels are -100 — label_start may be >= sequence length."
            )
        return {
            'input_ids': input_ids,
            'labels': labels,
            'label_starts': torch.tensor(label_starts, dtype=torch.long),
        }
    else:
        labels = input_ids.clone()
        labels[padding_mask] = -100
        return {'input_ids': input_ids, 'labels': labels}
