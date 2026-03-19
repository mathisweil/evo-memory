"""
lora_sft_dataset.py — SFT training dataset for LoRA finetuning on LongBench.

Multi-benchmark SFT dataset — loads per-task prompt templates from
data/longbench/config/dataset2prompt.json and formats them via apply_chat_template
for instruct models.  Supports any LongBench task that has context/input/answers.

Provides:
  - LongBenchSFTDataset: yields dicts with 'input_ids' (full prompt+answer tensor) and
    'label_start' (first answer token index). Answer-only loss masking is applied by
    sft_pad_collate_fn at batch construction time.
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

# Path to LongBench prompt templates (same file task_sampler.py uses)
_LONGBENCH_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), 'LongBench', 'config', 'dataset2prompt.json'
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LongBenchSFTDataset(Dataset):
    """Supervised fine-tuning dataset built from LongBench task splits.

    Loads per-task prompt templates from data/longbench/config/dataset2prompt.json
    (same templates task_sampler.py uses for evaluation).  Each item's prompt
    is wrapped in apply_chat_template for the instruct model, and the first
    gold answer becomes the assistant response.

    The 'label_start' index marks the boundary between the masked prompt and
    the supervised answer tokens; answer-only loss masking is applied by
    sft_pad_collate_fn.

    Left-truncation strategy: when prompt+answer exceeds max_seq_len, the
    combined token sequence is left-truncated (oldest context tokens dropped)
    so the answer tokens are always preserved at the right end.  Items where
    the answer tokens are entirely lost after truncation are skipped.

    Args:
        task_names  : List of LongBench task names (e.g. ['qasper', 'narrativeqa',
                      'passage_retrieval_en']).  Must exist in dataset2prompt.json.
        tokenizer   : HuggingFace tokenizer with apply_chat_template support.
        max_seq_len : Maximum total token sequence length.  Samples exceeding this
                      are discarded (not truncated), matching rhautier's approach.
        cache_dir   : HF datasets cache directory (optional, defaults to HF default).
        seed        : Random seed for one-time shuffle of self.samples.
        train_frac  : Fraction of each task's examples to use for training (default 0.8).
                      The first train_frac * N examples are used; the last (1-train_frac) * N
                      are held out for evaluation.  Set to 1.0 to use all examples.
    """

    def __init__(
        self,
        task_names: list,
        tokenizer,
        max_seq_len: int = 3500,
        cache_dir: str = None,
        seed: int = 1337,
        train_frac: float = 0.8,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.train_frac = train_frac

        # Load per-task prompt templates (same file task_sampler.py uses).
        with open(_LONGBENCH_PROMPT_PATH) as f:
            all_templates = json.load(f)

        for task in task_names:
            if task not in all_templates:
                raise ValueError(
                    f"Task '{task}' not found in {_LONGBENCH_PROMPT_PATH}. "
                    f"Available: {list(all_templates.keys())}"
                )

        samples = []
        n_skipped_no_answer = 0
        n_skipped_too_long = 0

        for task_name in task_names:
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
            if train_frac < 1.0:
                n_train = int(len(items) * train_frac)
                items = items[:n_train]
                print(f"  {task_name}: using first {n_train}/{n_total_task} examples (train_frac={train_frac})")
            task_count = 0
            task_skipped_long = 0
            task_prompt_lens = []
            task_answer_lens = []
            for item in items:
                # Use first gold answer; skip if no answers.
                if not item['answers']:
                    n_skipped_no_answer += 1
                    continue
                answer = item['answers'][0]

                # Format prompt using LongBench template, then wrap in chat format.
                user_content = task_template.format(**item)
                messages_prompt = [{"role": "user", "content": user_content}]
                messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

                # Tokenise via apply_chat_template so BOS/EOS/header tokens are set
                # correctly for the instruct model's built-in chat format.
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

                # Answer boundary: first index in full_ids that belongs to the answer.
                label_start = len(prompt_ids)
                n_answer_tokens = len(full_ids) - label_start

                # Discard samples exceeding max_seq_len (matching rhautier — no truncation)
                if len(full_ids) > max_seq_len:
                    n_skipped_too_long += 1
                    task_skipped_long += 1
                    continue

                # Skip samples where prompt fills entire sequence (no answer tokens)
                if label_start >= len(full_ids):
                    n_skipped_no_answer += 1
                    continue

                samples.append({
                    'full_ids': torch.tensor(full_ids, dtype=torch.long),
                    'label_start': label_start,
                })
                task_count += 1
                task_prompt_lens.append(label_start)
                task_answer_lens.append(n_answer_tokens)

            # Per-task stats
            if task_prompt_lens:
                avg_prompt = sum(task_prompt_lens) / len(task_prompt_lens)
                avg_answer = sum(task_answer_lens) / len(task_answer_lens)
                print(f"  {task_name}: {task_count} kept, {task_skipped_long} discarded (>{max_seq_len} tokens)")
                print(f"    prompt tokens:  min={min(task_prompt_lens)}, avg={avg_prompt:.0f}, max={max(task_prompt_lens)}")
                print(f"    answer tokens:  min={min(task_answer_lens)}, avg={avg_answer:.0f}, max={max(task_answer_lens)}")
            else:
                print(f"  {task_name}: 0 kept, {task_skipped_long} discarded")

        # Overall stats
        total = len(samples) + n_skipped_no_answer + n_skipped_too_long
        print(f"\nLongBenchSFTDataset: {len(samples)} samples loaded from {total} total")
        print(f"  Skipped: {n_skipped_too_long} exceeding max_seq_len={max_seq_len}, "
              f"{n_skipped_no_answer} no/empty answer")
        if len(samples) > 0:
            all_prompt_lens = [s['label_start'] for s in samples]
            all_answer_lens = [len(s['full_ids']) - s['label_start'] for s in samples]
            all_seq_lens = [len(s['full_ids']) for s in samples]
            print(f"  Overall prompt tokens:  min={min(all_prompt_lens)}, avg={sum(all_prompt_lens)/len(all_prompt_lens):.0f}, max={max(all_prompt_lens)}")
            print(f"  Overall answer tokens:  min={min(all_answer_lens)}, avg={sum(all_answer_lens)/len(all_answer_lens):.0f}, max={max(all_answer_lens)}")
            print(f"  Overall sequence len:   min={min(all_seq_lens)}, avg={sum(all_seq_lens)/len(all_seq_lens):.0f}, max={max(all_seq_lens)}")

        # One-time shuffle with a fixed seed for reproducibility.
        random.Random(seed).shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        """Return a dict with 'input_ids' (1D LongTensor) and 'label_start' (int)."""
        item = self.samples[idx]
        return {'input_ids': item['full_ids'], 'label_start': item['label_start']}


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def sft_pad_collate_fn(batch: list, pad_token_id: int, max_seq_len: int) -> dict:
    """Collate a list of SFT samples into a padded batch with answer-only labels.

    Right-pads each sequence to the length of the longest sequence in the batch
    (never exceeding max_seq_len).  Context and question tokens are masked with
    -100 in labels; only the answer tokens (label_start onward) are supervised.
    Padding positions are also masked with -100.

    Args:
        batch        : List of dicts with keys 'input_ids' (1D LongTensor) and
                       'label_start' (int, first answer token index).
        pad_token_id : Token ID used for right-padding (from tokenizer.pad_token_id).
        max_seq_len  : Safety cap applied after pad_sequence.

    Returns:
        dict with keys:
          'input_ids'    : LongTensor of shape [batch_size, seq_len]
          'labels'       : LongTensor of shape [batch_size, seq_len];
                           -100 at context/question and padding positions,
                           answer token ids elsewhere.
          'label_starts' : LongTensor of shape [batch_size], per-sample answer boundary.
    """
    input_ids_list = [x['input_ids'] for x in batch]
    label_starts = [x['label_start'] for x in batch]

    # Right-pad to the longest sequence in this batch.
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)

    # Safety trim to max_seq_len (left-truncation should make this a no-op).
    input_ids = input_ids[:, :max_seq_len]

    # Build labels: start with all -100 (everything masked by default).
    labels = torch.full_like(input_ids, -100)

    # For each sample, copy the answer tokens into labels starting at label_start.
    # Everything before label_start (context + question) stays -100.
    for i, ls in enumerate(label_starts):
        labels[i, ls:] = input_ids[i, ls:]

    # Re-mask padding positions that may have been filled by the loop above
    # (happens when a shorter sample's label_start is 0, unlikely but guarded).
    labels[input_ids == pad_token_id] = -100

    # Sanity assert: at least one non-masked label token must exist in the batch.
    # Catches wrong-direction masking (Pitfall 1) and empty-answer edge cases (Pitfall 3).
    assert (labels != -100).any(), (
        "sft_pad_collate_fn: all labels are -100 — check label_start boundary. "
        "This may indicate left-truncation removed all answer tokens or label_start "
        "is pointing in the wrong direction."
    )

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_starts': torch.tensor(label_starts, dtype=torch.long),
    }
