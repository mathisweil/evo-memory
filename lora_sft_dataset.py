"""
lora_sft_dataset.py — SFT training dataset for LoRA finetuning on QASPER.

SFT training dataset for LoRA finetuning — formats QASPER items as
context+question->answer with answer-only loss masking.

Provides:
  - LongBenchSFTDataset: yields dicts with 'input_ids' (full prompt+answer tensor) and
    'label_start' (first answer token index). Answer-only loss masking is applied by
    sft_pad_collate_fn at batch construction time.
  - sft_pad_collate_fn: right-pads a batch, masks context+question positions with -100 in
    labels, asserts at least one non-masked label token exists.

Usage example:
    from functools import partial
    dataset = LongBenchSFTDataset(
        task_names=['qasper'],
        tokenizer=tokenizer,
        max_seq_len=3500,
        cache_dir='/cs/student/project_msc/2025/csml/gmaralla/.hf_cache',
    )
    collate_fn = partial(sft_pad_collate_fn, pad_token_id=tokenizer.pad_token_id, max_seq_len=3500)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    for batch in loader:
        input_ids   = batch['input_ids']    # shape [batch, seq_len]
        labels      = batch['labels']       # shape [batch, seq_len], -100 for context+pad
        label_starts = batch['label_starts'] # shape [batch], first answer token index per sample
"""

import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Prompt template (instruct chat format)
# ---------------------------------------------------------------------------

# User turn content only — no "Answer:" suffix.
# apply_chat_template adds <|start_header_id|>assistant<|end_header_id|>\n\n
# as the generation prompt, so the answer starts immediately after.
QASPER_USER_TEMPLATE = (
    "You are given a scientific article and a question. Answer the question as concisely "
    "as you can, using a single phrase or sentence if possible. If the question cannot be "
    "answered based on the information in the article, write \"unanswerable\". If the "
    "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
    "provide any explanation.\n\nArticle: {context}\n\nQuestion: {input}"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LongBenchSFTDataset(Dataset):
    """Supervised fine-tuning dataset built from the QASPER split of LongBench.

    Each item formats the article context and question into the QASPER prompt
    template and appends the first gold answer.  The 'label_start' index marks
    the boundary between the masked prompt and the supervised answer tokens;
    answer-only loss masking is applied by sft_pad_collate_fn.

    Left-truncation strategy: when prompt+answer exceeds max_seq_len, the
    combined token sequence is left-truncated (oldest context tokens dropped)
    so the answer tokens are always preserved at the right end.  Items where
    the answer tokens are entirely lost after truncation are skipped.

    Args:
        task_names  : List of LongBench task names. Only ['qasper'] is supported
                      (Phase 9 QASPER-only scope). Raises ValueError for other tasks.
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

        # Phase 9 scope guard: only QASPER is supported for SFT.
        for task in task_names:
            if task != 'qasper':
                raise ValueError(
                    f"LongBenchSFTDataset only supports task_names=['qasper'] "
                    f"(Phase 9 QASPER-only scope). Got unsupported task: '{task}'. "
                    f"For other tasks use LongBenchNTPDataset."
                )

        samples = []
        n_skipped = 0

        for task_name in task_names:
            dataset = load_dataset(
                'THUDM/LongBench',
                task_name,
                split='test',
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            for item in dataset:
                # Use first gold answer; fall back to 'unanswerable' if empty.
                answer = item['answers'][0] if item['answers'] else 'unanswerable'

                # Build messages for instruct chat template.
                user_content = QASPER_USER_TEMPLATE.format(
                    context=item['context'],
                    input=item['input'],
                )
                messages_prompt = [{"role": "user", "content": user_content}]
                messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

                # Tokenise via apply_chat_template so BOS/EOS/header tokens are set
                # correctly for the instruct model's built-in chat format.
                # prompt_ids ends with the assistant generation header (\n\n), so
                # label_start = len(prompt_ids) correctly marks the first answer token.
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

                # Left-truncate: keep the last max_seq_len tokens, adjusting label_start.
                len_before = len(full_ids)
                if len_before > max_seq_len:
                    full_ids = full_ids[-max_seq_len:]
                    label_start = max(0, label_start - (len_before - max_seq_len))

                # Skip guard: if all answer tokens were truncated away, skip this item.
                if label_start >= len(full_ids):
                    n_skipped += 1
                    continue

                samples.append({
                    'full_ids': torch.tensor(full_ids, dtype=torch.long),
                    'label_start': label_start,
                })

        total = len(samples) + n_skipped
        print(
            f"LongBenchSFTDataset: {len(samples)} samples loaded, "
            f"{n_skipped} skipped (label_start >= seq_len)"
        )
        if total > 0 and n_skipped / total > 0.1:
            print(
                f"WARNING: LongBenchSFTDataset skipped {n_skipped}/{total} "
                f"({100 * n_skipped / total:.1f}%) samples — more than 10% were discarded "
                f"because all answer tokens fell outside max_seq_len={max_seq_len}. "
                f"Consider increasing max_seq_len."
            )

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
