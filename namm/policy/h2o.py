"""H2O (Heavy-Hitter Oracle) KV-cache eviction policy.

Zhang et al., NeurIPS 2023 — https://arxiv.org/abs/2306.14048

Maintains a fixed budget of `cache_size` tokens per (layer, KV-head) by
splitting the budget between "heavy hitters" (highest accumulated post-softmax
attention) and a recent window. No learnable parameters: applied at inference
time on top of any pretrained model.
"""

import logging
from typing import Optional, Tuple

import torch

from .base import MemoryPolicy

logger = logging.getLogger(__name__)


class H2O(MemoryPolicy):
    """Heavy-Hitter Oracle eviction.

    Args:
        cache_size: Total KV cache budget per (layer, KV-head). When None,
            no eviction is performed (passthrough).
        heavy_hitter_ratio: Fraction of the budget reserved for heavy hitters.
            The remainder is the recent window. Default 0.5 matches the paper.
    """

    def __init__(self, cache_size: Optional[int],
                 heavy_hitter_ratio: float = 0.5):
        super().__init__(cache_size=cache_size)
        if cache_size is None:
            self.limit_cache = False
            self.k_hh = 0
            self.k_recent = 0
        else:
            self.limit_cache = True
            if not 0.0 <= heavy_hitter_ratio <= 1.0:
                raise ValueError(
                    f"heavy_hitter_ratio must be in [0, 1], got "
                    f"{heavy_hitter_ratio}")
            self.k_hh = int(round(cache_size * heavy_hitter_ratio))
            self.k_recent = cache_size - self.k_hh
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self._accumulated_scores: list[Optional[torch.Tensor]] = []

    @property
    def requires_attn_scores(self) -> bool:
        # Read with default because base.MemoryPolicy.__init__ probes this
        # property (via initialize_buffers) before our subclass __init__ has
        # finished assigning self.limit_cache.
        return getattr(self, 'limit_cache', False)

    def finalize_registration(self) -> None:
        super().finalize_registration()
        self._accumulated_scores = [None] * self.num_memory_layers

    def _reset_layer_state(self, layer_id: int) -> None:
        self._accumulated_scores[layer_id] = None

    def _is_new_sequence(self, num_new_tokens: int, kv_len: int) -> bool:
        return num_new_tokens == kv_len

    def update_layer_cache(
            self,
            layer_id: int,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            num_new_tokens: int,
            attn_weights: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.limit_cache:
            return key_cache, value_cache

        bs, n_kv_heads, kv_len, head_dim = key_cache.shape

        # Fold GQA groups so attention is per-KV-head, matching the cache.
        attn_weights = self.process_attn_weights(attn_weights)
        # attn_weights: [bs, n_kv_heads, q_len, kv_len_attn]
        # Trim/pad attention to align with the cache length: in some code
        # paths the attention's kv axis equals kv_len; in others (after
        # repeat_kv but before update) it may be off by 0. Slice to last
        # `kv_len` columns to be safe.
        if attn_weights.shape[-1] != kv_len:
            attn_weights = attn_weights[..., -kv_len:]

        # Sum attention over the query axis → per-key score for this step.
        step_scores = attn_weights.sum(dim=-2).to(torch.float32)
        # step_scores: [bs, n_kv_heads, kv_len]

        new_seq = self._is_new_sequence(
            num_new_tokens=num_new_tokens, kv_len=kv_len)
        prev = self._accumulated_scores[layer_id]
        if new_seq or prev is None:
            accumulated = step_scores
        else:
            prev_len = prev.shape[-1]
            if prev_len < kv_len:
                pad = torch.zeros(
                    bs, n_kv_heads, kv_len - prev_len,
                    dtype=prev.dtype, device=prev.device)
                prev = torch.cat([prev, pad], dim=-1)
            elif prev_len > kv_len:
                prev = prev[..., -kv_len:]
            accumulated = prev + step_scores

        if kv_len <= self.cache_size:
            self._accumulated_scores[layer_id] = accumulated
            return key_cache, value_cache

        keep_idxs = self._select_keep_indices(accumulated)
        # keep_idxs: [bs, n_kv_heads, cache_size] long, sorted ascending.

        gather_kv = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        new_key = key_cache.gather(dim=-2, index=gather_kv)
        new_value = value_cache.gather(dim=-2, index=gather_kv)
        new_accumulated = accumulated.gather(dim=-1, index=keep_idxs)

        self._accumulated_scores[layer_id] = new_accumulated
        return new_key, new_value

    def _select_keep_indices(self, accumulated: torch.Tensor) -> torch.Tensor:
        """Pick top-k_hh heavy hitters from non-recent positions plus the
        last k_recent positions. Returns ascending-sorted keep indices of
        shape [bs, n_kv_heads, cache_size]."""
        bs, n_kv_heads, n = accumulated.shape
        budget = self.cache_size
        k_recent = self.k_recent
        k_hh = self.k_hh

        device = accumulated.device
        recent_idxs = torch.arange(
            n - k_recent, n, device=device, dtype=torch.long
        ).view(1, 1, k_recent).expand(bs, n_kv_heads, -1)

        if k_hh == 0:
            return recent_idxs

        # Mask out the recent positions in the score view so they cannot be
        # picked twice — set to -inf.
        masked_scores = accumulated.clone()
        masked_scores[..., n - k_recent:] = float('-inf')
        hh_idxs = masked_scores.topk(k=k_hh, dim=-1, sorted=False).indices

        keep = torch.cat([hh_idxs, recent_idxs], dim=-1)
        keep, _ = torch.sort(keep, dim=-1)
        return keep
