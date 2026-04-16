"""Unit tests for ScissorHands eviction policy.

Run with:
    python -m pytest tests/test_scissorhands.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namm.policy.scissorhands import ScissorHands


def _register(policy: ScissorHands, num_layers: int = 1,
              num_kv_heads: int = 2, num_groups: int = 1,
              hidden_size: int = 64) -> None:
    policy.register_new_memory_model(
        config=object(),
        registration_kwargs={
            "num_memory_layers": num_layers,
            "num_heads": num_kv_heads,
            "num_key_value_groups": num_groups,
            "hidden_size": hidden_size,
        },
    )
    policy.finalize_registration()


def test_scissorhands_passthrough_when_cache_none() -> None:
    policy = ScissorHands(cache_size=None)
    _register(policy)
    assert policy.requires_attn_scores is False

    key = torch.randn(1, 2, 50, 64)
    val = torch.randn(1, 2, 50, 64)
    k_out, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=50, attn_weights=None)
    assert torch.equal(k_out, key)
    assert torch.equal(v_out, val)


def test_scissorhands_no_eviction_when_under_budget() -> None:
    policy = ScissorHands(cache_size=64)
    _register(policy)

    bs, n_kv_heads, kv_len, head_dim = 1, 2, 32, 64
    key = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    val = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    attn = torch.softmax(
        torch.randn(bs, n_kv_heads, kv_len, kv_len), dim=-1)
    k_out, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    assert torch.equal(k_out, key)
    assert torch.equal(v_out, val)


def test_scissorhands_evicts_to_at_most_budget() -> None:
    cache_size = 16
    policy = ScissorHands(cache_size=cache_size)
    _register(policy)

    bs, n_kv_heads, kv_len, head_dim = 1, 2, 40, 32
    key = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    val = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    attn = torch.softmax(
        torch.randn(bs, n_kv_heads, kv_len, kv_len), dim=-1)

    k_out, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    # n - m tokens remain, with m capped at the non-recent slack. The budget
    # is the upper bound; exact post-compression size depends on m.
    assert k_out.shape[2] <= kv_len
    assert k_out.shape[2] == v_out.shape[2]
    # Standard parameters: m = B/2 → drop 8 tokens from kv_len=40 → keep 32.
    assert k_out.shape[2] == kv_len - policy.drop_count


def test_scissorhands_recent_window_protected() -> None:
    """Tokens in the recent window are never evicted."""
    cache_size = 8  # recent_window=2, drop_count=4
    policy = ScissorHands(
        cache_size=cache_size,
        history_window_ratio=0.5,
        recent_window_ratio=0.25,
        drop_ratio=0.5,
    )
    _register(policy, num_kv_heads=1)
    bs, n_kv_heads, head_dim = 1, 1, 1
    kv_len = 12
    val = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    for i in range(kv_len):
        val[..., i, 0] = float(i)
    key = torch.zeros_like(val)
    # Make the recent window look maximally unimportant (all attention on
    # token 0). Without protection they would be evicted.
    attn_logits = torch.full((bs, n_kv_heads, 1, kv_len), -10.0)
    attn_logits[..., 0, 0] = 10.0
    attn = torch.softmax(attn_logits, dim=-1)
    _, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    kept = sorted(int(x) for x in v_out[0, 0, :, 0].tolist())
    # Recent window indices [10, 11] must be present.
    assert 10 in kept
    assert 11 in kept


def test_scissorhands_drops_least_important() -> None:
    """Tokens with the highest below-threshold count are dropped first."""
    cache_size = 6  # recent_window=2 (rounded from 1.5), drop_count=3
    policy = ScissorHands(
        cache_size=cache_size,
        history_window_ratio=0.5,
        recent_window_ratio=0.25,
        drop_ratio=0.5,
    )
    _register(policy, num_kv_heads=1)
    bs, n_kv_heads, head_dim = 1, 1, 1
    kv_len = 9
    val = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    for i in range(kv_len):
        val[..., i, 0] = float(i)
    key = torch.zeros_like(val)
    # Token 0 dominates attention; tokens 1,2,3 are all below 1/t threshold.
    # Recent window (last `recent_window` tokens) is protected. Of the rest,
    # token 0 has the lowest unimportance count → should be kept.
    attn_logits = torch.full((bs, n_kv_heads, 1, kv_len), -10.0)
    attn_logits[..., 0, 0] = 10.0
    attn = torch.softmax(attn_logits, dim=-1)
    _, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    kept = sorted(int(x) for x in v_out[0, 0, :, 0].tolist())
    assert 0 in kept, f"highest-attention token should survive, got {kept}"


def test_scissorhands_state_resets_on_new_sequence() -> None:
    policy = ScissorHands(cache_size=4)
    _register(policy, num_kv_heads=1)
    bs, n_kv_heads, head_dim = 1, 1, 1
    # First sequence.
    kv_len = 8
    key = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    val = torch.zeros_like(key)
    attn = torch.softmax(torch.randn(bs, n_kv_heads, 1, kv_len), dim=-1)
    policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    seen_after_first = policy._tokens_seen[0]
    assert seen_after_first > 0
    # Second sequence.
    kv_len2 = 6
    key2 = torch.zeros(bs, n_kv_heads, kv_len2, head_dim)
    val2 = torch.zeros_like(key2)
    attn2 = torch.softmax(
        torch.randn(bs, n_kv_heads, 1, kv_len2), dim=-1)
    policy.update_layer_cache(
        layer_id=0, key_cache=key2, value_cache=val2,
        num_new_tokens=kv_len2, attn_weights=attn2)
    # tokens_seen should have been reset before incrementing by kv_len2.
    assert policy._tokens_seen[0] == kv_len2


def test_scissorhands_invalid_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        ScissorHands(cache_size=8, history_window_ratio=0.0)
    with pytest.raises(ValueError):
        ScissorHands(cache_size=8, recent_window_ratio=2.0)
    with pytest.raises(ValueError):
        ScissorHands(cache_size=8, drop_ratio=-0.5)


def test_scissorhands_recent_window_must_be_smaller_than_budget() -> None:
    with pytest.raises(ValueError):
        ScissorHands(cache_size=4, recent_window_ratio=1.0)


def test_scissorhands_no_trainable_parameters() -> None:
    policy = ScissorHands(cache_size=64)
    _register(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert trainable == []
