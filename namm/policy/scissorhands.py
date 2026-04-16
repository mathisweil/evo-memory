"""ScissorHands KV-cache eviction policy.

Liu et al., NeurIPS 2023 — https://arxiv.org/abs/2305.17118

Persistence-of-importance heuristic: counts how often each cached token's
post-softmax attention falls below the average-mixing threshold (1 / current
sequence position) over a recent history window, then evicts the tokens with
the highest "unimportant" count when the cache exceeds budget B. The most
recent r tokens are protected from eviction. No learnable parameters.
"""

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import torch

from .base import MemoryPolicy

logger = logging.getLogger(__name__)


class ScissorHands(MemoryPolicy):
    """ScissorHands eviction.

    Args:
        cache_size: Total KV cache budget B per (layer, KV-head). When None,
            no eviction is performed (passthrough).
        history_window_ratio: Fraction of B used as the history window w.
            Default 0.5.
        recent_window_ratio: Fraction of B used as the protected recent
            window r. Default 0.25.
        drop_ratio: Fraction of B dropped on each compression event m.
            Default 0.5.
    """

    def __init__(self, cache_size: Optional[int],
                 history_window_ratio: float = 0.5,
                 recent_window_ratio: float = 0.25,
                 drop_ratio: float = 0.5):
        super().__init__(cache_size=cache_size)
        if cache_size is None:
            self.limit_cache = False
            self.history_window = 0
            self.recent_window = 0
            self.drop_count = 0
        else:
            self.limit_cache = True
            for name, val in (("history_window_ratio", history_window_ratio),
                              ("recent_window_ratio", recent_window_ratio),
                              ("drop_ratio", drop_ratio)):
                if not 0.0 < val <= 1.0:
                    raise ValueError(f"{name} must be in (0, 1], got {val}")
            self.history_window = max(1, int(round(
                cache_size * history_window_ratio)))
            self.recent_window = max(1, int(round(
                cache_size * recent_window_ratio)))
            self.drop_count = max(1, int(round(cache_size * drop_ratio)))
            if self.recent_window >= cache_size:
                raise ValueError(
                    f"recent_window ({self.recent_window}) must be < "
                    f"cache_size ({cache_size})")
        self.history_window_ratio = history_window_ratio
        self.recent_window_ratio = recent_window_ratio
        self.drop_ratio = drop_ratio
        self._attn_history: list[Deque[torch.Tensor]] = []
        self._tokens_seen: list[int] = []

    @property
    def requires_attn_scores(self) -> bool:
        # Read with default because base.MemoryPolicy.__init__ probes this
        # property (via initialize_buffers) before our subclass __init__ has
        # finished assigning self.limit_cache.
        return getattr(self, 'limit_cache', False)

    def finalize_registration(self) -> None:
        super().finalize_registration()
        self._attn_history = [
            deque(maxlen=self.history_window)
            for _ in range(self.num_memory_layers)
        ]
        self._tokens_seen = [0] * self.num_memory_layers

    def _reset_layer_state(self, layer_id: int) -> None:
        self._attn_history[layer_id] = deque(maxlen=self.history_window)
        self._tokens_seen[layer_id] = 0

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
        attn_weights = self.process_attn_weights(attn_weights)
        if attn_weights.shape[-1] != kv_len:
            attn_weights = attn_weights[..., -kv_len:]

        # Per-key score for this step: average attention over the q axis. We
        # use the mean (not sum) so the threshold 1/t is comparable across
        # steps regardless of how many query tokens were processed at once
        # (prefill vs. decode).
        step_scores = attn_weights.mean(dim=-2).to(torch.float32)

        new_seq = self._is_new_sequence(
            num_new_tokens=num_new_tokens, kv_len=kv_len)
        if new_seq:
            self._reset_layer_state(layer_id)
        history = self._attn_history[layer_id]

        # Pad existing history entries to current kv_len (new tokens get a 0
        # contribution to the unimportance counter, matching the paper: they
        # are inside the protected recent window anyway).
        for i, past in enumerate(history):
            if past.shape[-1] < kv_len:
                pad = torch.zeros(
                    bs, n_kv_heads, kv_len - past.shape[-1],
                    dtype=past.dtype, device=past.device)
                history[i] = torch.cat([past, pad], dim=-1)
            elif past.shape[-1] > kv_len:
                history[i] = past[..., -kv_len:]
        history.append(step_scores)

        self._tokens_seen[layer_id] += num_new_tokens
        if kv_len <= self.cache_size:
            return key_cache, value_cache

        t = max(self._tokens_seen[layer_id], 1)
        threshold = 1.0 / t
        # Stack history along a leading axis: [w_eff, bs, n_kv_heads, kv_len].
        hist_stack = torch.stack(list(history), dim=0)
        below = (hist_stack < threshold).to(torch.int32)
        importance = below.sum(dim=0)
        # Protect the recent window: clear unimportance counts so they cannot
        # be selected for eviction.
        importance[..., kv_len - self.recent_window:] = 0

        m = min(self.drop_count, kv_len - self.cache_size + self.drop_count)
        # Cap m so we never drop more than the non-recent slack.
        max_droppable = kv_len - self.recent_window
        m = min(m, max_droppable)
        keep_count = kv_len - m

        # Drop tokens with highest "unimportant" count → keep the lowest.
        # Use ascending top-k by negating, then sort ascending for stable
        # gather.
        keep_idxs = importance.topk(
            k=keep_count, dim=-1, largest=False, sorted=False).indices
        keep_idxs, _ = torch.sort(keep_idxs, dim=-1)

        gather_kv = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        new_key = key_cache.gather(dim=-2, index=gather_kv)
        new_value = value_cache.gather(dim=-2, index=gather_kv)
        # Re-gather history in lockstep so future importance counts align
        # with the new token positions.
        for i, past in enumerate(history):
            history[i] = past.gather(dim=-1, index=keep_idxs)

        return new_key, new_value
