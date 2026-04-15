"""Unit tests for H2O eviction policy.

Run with:
    python -m pytest tests/test_h2o.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namm.policy.h2o import H2O


def _register(policy: H2O, num_layers: int = 1, num_kv_heads: int = 2,
              num_groups: int = 1, hidden_size: int = 64) -> None:
    """Run the registration handshake the wrapper would normally perform."""
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


def test_h2o_passthrough_when_cache_none() -> None:
    """cache_size=None disables eviction entirely."""
    policy = H2O(cache_size=None)
    _register(policy)
    assert policy.requires_attn_scores is False

    key = torch.randn(1, 2, 50, 64)
    val = torch.randn(1, 2, 50, 64)
    k_out, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=50, attn_weights=None)
    assert torch.equal(k_out, key)
    assert torch.equal(v_out, val)


def test_h2o_no_eviction_when_under_budget() -> None:
    """When kv_len <= cache_size, no eviction occurs."""
    policy = H2O(cache_size=64)
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


def test_h2o_evicts_to_budget() -> None:
    """After eviction kv_len equals cache_size."""
    cache_size = 16
    policy = H2O(cache_size=cache_size)
    _register(policy)

    bs, n_kv_heads, kv_len, head_dim = 1, 2, 40, 32
    key = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    val = torch.randn(bs, n_kv_heads, kv_len, head_dim)
    attn = torch.softmax(
        torch.randn(bs, n_kv_heads, kv_len, kv_len), dim=-1)

    k_out, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    assert k_out.shape == (bs, n_kv_heads, cache_size, head_dim)
    assert v_out.shape == (bs, n_kv_heads, cache_size, head_dim)


def test_h2o_recent_window_always_kept() -> None:
    """The last k_recent positions are never evicted."""
    cache_size = 8  # k_hh=4, k_recent=4
    policy = H2O(cache_size=cache_size, heavy_hitter_ratio=0.5)
    _register(policy, num_kv_heads=1)
    head_dim = 4
    bs, n_kv_heads, kv_len = 1, 1, 16
    # Make recent tokens distinguishable: their value vectors carry their
    # original index.
    val = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    for i in range(kv_len):
        val[..., i, 0] = float(i)
    key = torch.zeros_like(val)
    # Heavy attention is concentrated on tokens 0 and 1; no recent token
    # earns much score.
    attn_logits = torch.full((bs, n_kv_heads, 1, kv_len), -10.0)
    attn_logits[..., 0, 0] = 5.0
    attn_logits[..., 0, 1] = 5.0
    attn = torch.softmax(attn_logits, dim=-1)

    _, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    kept = v_out[0, 0, :, 0].tolist()
    # All four recent indices [12,13,14,15] must appear.
    for idx in [12, 13, 14, 15]:
        assert idx in kept, f"recent token {idx} missing from {kept}"
    # Heavy hitters 0 and 1 must also appear.
    assert 0.0 in kept
    assert 1.0 in kept


def test_h2o_heavy_hitter_selection() -> None:
    """Tokens with highest accumulated attention are selected as HH."""
    cache_size = 6  # k_hh=3, k_recent=3
    policy = H2O(cache_size=cache_size, heavy_hitter_ratio=0.5)
    _register(policy, num_kv_heads=1)
    head_dim = 1
    bs, n_kv_heads, kv_len = 1, 1, 12
    val = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    for i in range(kv_len):
        val[..., i, 0] = float(i)
    key = torch.zeros_like(val)
    # Strong attention on tokens 2, 5, 7 (all outside recent window 9..11).
    attn_logits = torch.full((bs, n_kv_heads, 1, kv_len), -10.0)
    for hh in (2, 5, 7):
        attn_logits[..., 0, hh] = 5.0
    attn = torch.softmax(attn_logits, dim=-1)
    _, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    kept = sorted(int(x) for x in v_out[0, 0, :, 0].tolist())
    assert kept == [2, 5, 7, 9, 10, 11]


def test_h2o_per_head_independence() -> None:
    """Different heads accumulate independently and may keep different sets."""
    cache_size = 4  # k_hh=2, k_recent=2
    policy = H2O(cache_size=cache_size, heavy_hitter_ratio=0.5)
    _register(policy, num_kv_heads=2)
    head_dim = 1
    bs, n_kv_heads, kv_len = 1, 2, 8
    val = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    for i in range(kv_len):
        val[..., i, 0] = float(i)
    key = torch.zeros_like(val)
    # Head 0 prefers tokens 0,1; head 1 prefers tokens 3,4.
    attn_logits = torch.full((bs, n_kv_heads, 1, kv_len), -10.0)
    attn_logits[0, 0, 0, 0] = 5.0
    attn_logits[0, 0, 0, 1] = 5.0
    attn_logits[0, 1, 0, 3] = 5.0
    attn_logits[0, 1, 0, 4] = 5.0
    attn = torch.softmax(attn_logits, dim=-1)
    _, v_out = policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    head0 = sorted(int(x) for x in v_out[0, 0, :, 0].tolist())
    head1 = sorted(int(x) for x in v_out[0, 1, :, 0].tolist())
    assert head0 == [0, 1, 6, 7]
    assert head1 == [3, 4, 6, 7]
    assert head0 != head1


def test_h2o_state_resets_on_new_sequence() -> None:
    """A new prompt (num_new_tokens == kv_len) wipes accumulated scores."""
    policy = H2O(cache_size=16, heavy_hitter_ratio=0.5)
    _register(policy, num_kv_heads=1)
    bs, n_kv_heads, head_dim = 1, 1, 1
    # First sequence under budget — accumulator becomes step_scores_1.
    kv_len = 8
    key = torch.zeros(bs, n_kv_heads, kv_len, head_dim)
    val = torch.zeros_like(key)
    attn = torch.softmax(torch.randn(bs, n_kv_heads, 1, kv_len), dim=-1)
    policy.update_layer_cache(
        layer_id=0, key_cache=key, value_cache=val,
        num_new_tokens=kv_len, attn_weights=attn)
    # Second sequence (still under budget) with deterministic attention — if
    # the first sequence's accumulator carried over, the new scores would
    # not match a clean sum of the second-sequence step_scores.
    kv_len2 = 6
    key2 = torch.zeros(bs, n_kv_heads, kv_len2, head_dim)
    val2 = torch.zeros_like(key2)
    attn2 = torch.softmax(
        torch.full((bs, n_kv_heads, 1, kv_len2), 1.0), dim=-1)
    policy.update_layer_cache(
        layer_id=0, key_cache=key2, value_cache=val2,
        num_new_tokens=kv_len2, attn_weights=attn2)
    # Reset → accumulated equals step_scores from the second call (uniform).
    expected = attn2.sum(dim=-2).to(torch.float32)
    assert torch.allclose(policy._accumulated_scores[0], expected)
    assert policy._accumulated_scores[0].shape == (bs, n_kv_heads, kv_len2)


def test_h2o_invalid_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        H2O(cache_size=8, heavy_hitter_ratio=1.5)
    with pytest.raises(ValueError):
        H2O(cache_size=8, heavy_hitter_ratio=-0.1)


def test_h2o_no_trainable_parameters() -> None:
    policy = H2O(cache_size=64)
    _register(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert trainable == []
