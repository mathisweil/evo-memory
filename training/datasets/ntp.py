"""
lora_ntp_dataset.py — NTP training dataset wrapping LongBench for LoRA gradient training.

Provides:
  - LongBenchNTPDataset: yields variable-length 1D input_ids tensors from LongBench 'context' fields.
  - ntp_pad_collate_fn: right-pads a batch to batch-max length, masks padding positions with -100 in labels.

Usage example:
    from functools import partial
    dataset = LongBenchNTPDataset(
        task_names=['qasper'],
        tokenizer=tokenizer,
        max_seq_len=3500,
        cache_dir=None,  # set via HF_CACHE_DIR env var or Hydra config
    )
    collate_fn = partial(ntp_pad_collate_fn, pad_token_id=tokenizer.pad_token_id, max_seq_len=3500)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    for batch in loader:
        input_ids = batch['input_ids']  # shape [batch, seq_len]
        labels    = batch['labels']     # shape [batch, seq_len], -100 at pad positions
"""

import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


class LongBenchNTPDataset(Dataset):
    """Teacher-forced NTP dataset built from LongBench context fields.

    Loads the 'test' split of one or more LongBench tasks, tokenises each
    document's 'context' field, and left-truncates to at most max_seq_len
    tokens.  Samples are returned as variable-length 1D LongTensors; batching
    and padding are handled by ntp_pad_collate_fn.

    Args:
        task_names  : List of LongBench task names, e.g. ['qasper', 'narrativeqa',
                      'passage_retrieval_en'].  Supports a single-task subset for
                      EXP-06 dataset-variation experiments.
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
            dataset = load_dataset(
                'THUDM/LongBench',
                task_name,
                split='test',
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            for item in dataset:
                # Tokenise the context document (the long-range text to train on).
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
    batch (never exceeding max_seq_len, because LongBenchNTPDataset already
    left-truncates every sample to max_seq_len).  Padding positions are masked
    with -100 in labels so that cross-entropy loss ignores them.

    Args:
        batch        : List of 1D LongTensors (variable length, all <= max_seq_len).
        pad_token_id : Token ID used for right-padding (from tokenizer.pad_token_id).
        max_seq_len  : Safety cap applied after pad_sequence (should never trigger
                       given left-truncation, but guards against edge cases).

    Returns:
        dict with keys:
          'input_ids' : LongTensor of shape [batch_size, seq_len]
          'labels'    : LongTensor of shape [batch_size, seq_len], -100 at pad positions
    """
    # Right-pad to the longest sequence in this batch.
    input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

    # Safety trim to max_seq_len (left-truncation should make this a no-op).
    input_ids = input_ids[:, :max_seq_len]

    # Build labels: copy input_ids, then mask all padding positions with -100
    # so that cross-entropy ignores them during NTP training.
    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100

    return {'input_ids': input_ids, 'labels': labels}
