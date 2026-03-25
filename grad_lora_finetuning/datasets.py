"""
datasets.py — NTP and SFT training datasets for LoRA gradient fine-tuning on LongBench.

Provides:
  - LongBenchNTPDataset: yields variable-length 1D input_ids tensors from LongBench 'context' fields.
  - ntp_pad_collate_fn: right-pads a batch to batch-max length, masks padding positions with -100 in labels.
  - LongBenchSFTDataset: yields dicts with 'input_ids' (full prompt+answer tensor) and
    'label_start' (first answer token index).
  - sft_pad_collate_fn: right-pads a batch, masks context+question positions with -100 in
    labels, asserts at least one non-masked label token exists.
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# Path to LongBench prompt templates (matches tpu branch data directory layout)
_LONGBENCH_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'longbench', 'dataset2prompt.json'
)


# ---------------------------------------------------------------------------
# NTP Dataset
# ---------------------------------------------------------------------------

class LongBenchNTPDataset(Dataset):
    """Teacher-forced NTP dataset built from LongBench context fields.

    Loads the 'test' split of one or more LongBench tasks, tokenises each
    document's 'context' field, and left-truncates to at most max_seq_len
    tokens.  Samples are returned as variable-length 1D LongTensors; batching
    and padding are handled by ntp_pad_collate_fn.

    Args:
        task_names  : List of LongBench task names, e.g. ['qasper', 'narrativeqa',
                      'passage_retrieval_en'].
        tokenizer   : HuggingFace tokenizer with .encode() and .pad_token_id.
        max_seq_len : Maximum token sequence length after left-truncation.
                      Default 3500 leaves headroom within the 4096-token RoPE window.
        cache_dir   : HF datasets cache directory (optional, defaults to HF default).
        seed        : Random seed for one-time shuffle of self.samples.
    """

    def __init__(
        self,
        task_names: list,
        tokenizer,
        max_seq_len: int = 3500,
        cache_dir: str = None,
        seed: int = 1337,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        samples = []
        for task_name in task_names:
            # Strip 'lb/' prefix added by TaskSampler (HF expects bare name)
            hf_name = task_name.removeprefix('lb/')
            dataset = load_dataset(
                'THUDM/LongBench',
                hf_name,
                split='test',
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            for item in dataset:
                ids = tokenizer.encode(item['context'], add_special_tokens=True)
                # Left-truncate: keep the last max_seq_len tokens so the model
                # sees the most recent (answer-relevant) content.
                if len(ids) > max_seq_len:
                    ids = ids[-max_seq_len:]
                samples.append(torch.tensor(ids, dtype=torch.long))

        # One-time shuffle with a fixed seed for reproducibility.
        random.Random(seed).shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> torch.Tensor:
        """Return a variable-length 1D LongTensor of token ids."""
        return self.samples[idx]


def ntp_pad_collate_fn(batch: list, pad_token_id: int, max_seq_len: int) -> dict:
    """Collate a list of variable-length 1D tensors into a padded batch.

    Right-pads each sequence to the length of the longest sequence in the
    batch (never exceeding max_seq_len).  Padding positions are masked
    with -100 in labels so that cross-entropy loss ignores them.

    Args:
        batch        : List of 1D LongTensors (variable length, all <= max_seq_len).
        pad_token_id : Token ID used for right-padding (from tokenizer.pad_token_id).
        max_seq_len  : Safety cap applied after pad_sequence.

    Returns:
        dict with keys:
          'input_ids' : LongTensor of shape [batch_size, seq_len]
          'labels'    : LongTensor of shape [batch_size, seq_len], -100 at pad positions
    """
    input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids[:, :max_seq_len]

    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100

    return {'input_ids': input_ids, 'labels': labels}


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class LongBenchSFTDataset(Dataset):
    """Supervised fine-tuning dataset built from LongBench task splits.

    Loads per-task prompt templates from data/longbench/dataset2prompt.json
    (same templates TaskSampler uses for evaluation).  Each item's prompt
    is wrapped in apply_chat_template for the instruct model, and the first
    gold answer becomes the assistant response.

    The 'label_start' index marks the boundary between the masked prompt and
    the supervised answer tokens; answer-only loss masking is applied by
    sft_pad_collate_fn.

    Args:
        task_names  : List of LongBench task names.
        tokenizer   : HuggingFace tokenizer with apply_chat_template support.
        max_seq_len : Maximum total token sequence length.  Samples exceeding this
                      are discarded (not truncated).
        cache_dir   : HF datasets cache directory (optional).
        seed        : Random seed for one-time shuffle.
        train_frac  : Fraction of each task's examples to use for training (default 0.8).
    """

    def __init__(
        self,
        task_names: list,
        tokenizer,
        max_seq_len: int = 3500,
        max_conditioning_length: int = 6500,
        cache_dir: str = None,
        seed: int = 1337,
        train_frac: float = 0.8,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_conditioning_length = max_conditioning_length
        self.train_frac = train_frac

        with open(_LONGBENCH_PROMPT_PATH) as f:
            all_templates = json.load(f)

        # Strip 'lb/' prefix added by TaskSampler (HF expects bare name)
        bare_names = [t.removeprefix('lb/') for t in task_names]

        for task in bare_names:
            if task not in all_templates:
                raise ValueError(
                    f"Task '{task}' not found in {_LONGBENCH_PROMPT_PATH}. "
                    f"Available: {list(all_templates.keys())}"
                )

        samples = []
        n_skipped_no_answer = 0
        n_skipped_too_long = 0

        for task_name in bare_names:
            task_template = all_templates[task_name]
            dataset = load_dataset(
                'THUDM/LongBench',
                task_name,
                split='test',
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            items = list(dataset)
            n_total_task = len(items)

            # --- Filter first, then split (matches task_sampler order) ---
            eligible = []
            task_skipped_long = 0
            task_skipped_no_answer = 0
            for item in items:
                if not item['answers']:
                    task_skipped_no_answer += 1
                    n_skipped_no_answer += 1
                    continue
                answer = item['answers'][0]

                user_content = task_template.format(**item)
                messages_prompt = [{"role": "user", "content": user_content}]
                messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

                prompt_ids = tokenizer.apply_chat_template(
                    messages_prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                full_ids = tokenizer.apply_chat_template(
                    messages_full,
                    add_generation_prompt=False,
                    tokenize=True,
                )

                label_start = len(prompt_ids)

                # Filter by PROMPT length to match task_sampler's eligibility.
                # task_sampler strips BOS and counts with add_special_tokens=False,
                # so subtract 1 for the BOS that apply_chat_template includes.
                bos_id = getattr(tokenizer, 'bos_token_id', None)
                n_prompt_tok = label_start - (1 if bos_id is not None and prompt_ids[0] == bos_id else 0)
                if n_prompt_tok > max_conditioning_length:
                    task_skipped_long += 1
                    n_skipped_too_long += 1
                    continue

                if label_start >= len(full_ids):
                    task_skipped_no_answer += 1
                    n_skipped_no_answer += 1
                    continue

                # Truncate answer if prompt + answer exceeds max_seq_len
                if len(full_ids) > max_seq_len:
                    full_ids = full_ids[:max_seq_len]

                n_answer_tokens = len(full_ids) - label_start

                eligible.append({
                    'full_ids': torch.tensor(full_ids, dtype=torch.long),
                    'label_start': label_start,
                    'task_name': task_name,
                    'n_answer_tokens': n_answer_tokens,
                })

            n_eligible = len(eligible)

            # Split AFTER filtering — take first train_frac of eligible samples.
            if train_frac < 1.0:
                n_train = int(n_eligible * train_frac)
                eligible = eligible[:n_train]
                print(f"  {task_name}: {n_total_task} total, {task_skipped_long} filtered "
                      f"(prompt>{max_conditioning_length} tok), {task_skipped_no_answer} no answer -> "
                      f"{n_eligible} eligible -> {n_train} train (first {train_frac:.0%})")
            else:
                print(f"  {task_name}: {n_total_task} total, {task_skipped_long} filtered "
                      f"(prompt>{max_conditioning_length} tok), {task_skipped_no_answer} no answer -> "
                      f"{n_eligible} eligible (using all)")

            task_prompt_lens = [s['label_start'] for s in eligible]
            task_answer_lens = [s['n_answer_tokens'] for s in eligible]

            if task_prompt_lens:
                avg_prompt = sum(task_prompt_lens) / len(task_prompt_lens)
                avg_answer = sum(task_answer_lens) / len(task_answer_lens)
                print(f"    prompt tokens:  min={min(task_prompt_lens)}, avg={avg_prompt:.0f}, max={max(task_prompt_lens)}")
                print(f"    answer tokens:  min={min(task_answer_lens)}, avg={avg_answer:.0f}, max={max(task_answer_lens)}")

            # Drop helper keys before adding to samples
            for s in eligible:
                del s['n_answer_tokens']
            samples.extend(eligible)

        total = len(samples) + n_skipped_no_answer + n_skipped_too_long
        print(f"\nLongBenchSFTDataset: {len(samples)} samples loaded from {total} total")
        print(f"  Skipped: {n_skipped_too_long} with prompt>{max_conditioning_length} tok, "
              f"{n_skipped_no_answer} no/empty answer")
        if len(samples) > 0:
            all_prompt_lens = [s['label_start'] for s in samples]
            all_answer_lens = [len(s['full_ids']) - s['label_start'] for s in samples]
            all_seq_lens = [len(s['full_ids']) for s in samples]
            print(f"  Overall prompt tokens:  min={min(all_prompt_lens)}, avg={sum(all_prompt_lens)/len(all_prompt_lens):.0f}, max={max(all_prompt_lens)}")
            print(f"  Overall answer tokens:  min={min(all_answer_lens)}, avg={sum(all_answer_lens)/len(all_answer_lens):.0f}, max={max(all_answer_lens)}")
            print(f"  Overall sequence len:   min={min(all_seq_lens)}, avg={sum(all_seq_lens)/len(all_seq_lens):.0f}, max={max(all_seq_lens)}")

        # --- Upsample minority tasks to balance per-task contribution ---
        task_buckets = {}
        for s in samples:
            t = s.get('task_name', '_unknown')
            task_buckets.setdefault(t, []).append(s)

        if len(task_buckets) > 1:
            max_count = max(len(v) for v in task_buckets.values())
            balanced_samples = []
            rng = random.Random(seed)
            for t_name, t_samples in sorted(task_buckets.items()):
                n_orig = len(t_samples)
                if n_orig < max_count:
                    n_full_copies = max_count // n_orig
                    n_remainder = max_count % n_orig
                    upsampled = t_samples * n_full_copies + rng.sample(t_samples, n_remainder)
                    print(f"  Upsampling {t_name}: {n_orig} -> {len(upsampled)} (x{max_count / n_orig:.1f})")
                    balanced_samples.extend(upsampled)
                else:
                    balanced_samples.extend(t_samples)
            samples = balanced_samples
            print(f"  Balanced dataset: {len(samples)} total samples ({max_count} per task x {len(task_buckets)} tasks)")

        random.Random(seed).shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        """Return a dict with 'input_ids' (1D LongTensor) and 'label_start' (int)."""
        item = self.samples[idx]
        return {'input_ids': item['full_ids'], 'label_start': item['label_start']}


def sft_pad_collate_fn(batch: list, pad_token_id: int, max_seq_len: int) -> dict:
    """Collate a list of SFT samples into a padded batch with answer-only labels.

    Right-pads each sequence to the length of the longest sequence in the batch.
    Context and question tokens are masked with -100 in labels; only the answer
    tokens (label_start onward) are supervised. Padding positions are also masked.

    Args:
        batch        : List of dicts with keys 'input_ids' (1D LongTensor) and
                       'label_start' (int, first answer token index).
        pad_token_id : Token ID used for right-padding.
        max_seq_len  : Safety cap applied after pad_sequence.

    Returns:
        dict with keys:
          'input_ids'    : LongTensor [batch_size, seq_len]
          'labels'       : LongTensor [batch_size, seq_len]; -100 at context/question and padding
          'label_starts' : LongTensor [batch_size], per-sample answer boundary
    """
    input_ids_list = [x['input_ids'] for x in batch]
    label_starts = [x['label_start'] for x in batch]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids[:, :max_seq_len]

    labels = torch.full_like(input_ids, -100)

    for i, ls in enumerate(label_starts):
        labels[i, ls:] = input_ids[i, ls:]

    labels[input_ids == pad_token_id] = -100

    assert (labels != -100).any(), (
        "sft_pad_collate_fn: all labels are -100 — check label_start boundary."
    )

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_starts': torch.tensor(label_starts, dtype=torch.long),
    }
