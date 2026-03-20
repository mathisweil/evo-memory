"""
Tests for Bug 2 fix: swap to Recency policy when namm_active=False.

Verifies that:
  1. Recency policy is a lightweight passthrough (no scoring, no STFT)
  2. swap_memory_policy() correctly replaces a Deep policy with Recency
     on both the evaluator and the underlying model
  3. The run_lora.py swap logic triggers only when namm_active=False

Does NOT require CUDA or gated HuggingFace model access.

Run with:
    python -m pytest tests/test_recency_swap.py -v
"""

import os
import sys
import types

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namm.policy.base import MemoryPolicy, Recency


# ---------------------------------------------------------------------------
# 1. Recency policy unit tests
# ---------------------------------------------------------------------------

class TestRecencyPolicy:
    """Verify Recency is a lightweight passthrough."""

    def test_recency_none_cache_is_passthrough(self):
        """Recency(cache_size=None) passes KV cache through unchanged."""
        policy = Recency(cache_size=None)
        assert policy.limit_cache is False
        assert policy.requires_attn_scores is False

        key = torch.randn(1, 8, 100, 64)
        val = torch.randn(1, 8, 100, 64)
        k_out, v_out = policy.update_layer_cache(
            layer_id=0, key_cache=key, value_cache=val,
            num_new_tokens=10, attn_weights=None)
        assert torch.equal(k_out, key)
        assert torch.equal(v_out, val)

    def test_recency_with_cache_size_truncates(self):
        """Recency(cache_size=N) keeps only the last N tokens."""
        policy = Recency(cache_size=50)
        assert policy.limit_cache is True
        assert policy.requires_attn_scores is False

        key = torch.randn(1, 8, 100, 64)
        val = torch.randn(1, 8, 100, 64)
        k_out, v_out = policy.update_layer_cache(
            layer_id=0, key_cache=key, value_cache=val,
            num_new_tokens=10, attn_weights=None)
        assert k_out.shape[2] == 50
        assert v_out.shape[2] == 50
        assert torch.equal(k_out, key[..., -50:, :])
        assert torch.equal(v_out, val[..., -50:, :])

    def test_recency_has_no_parameters(self):
        """Recency should have no trainable parameters (no scoring network)."""
        policy = Recency(cache_size=1024)
        trainable = [p for p in policy.parameters() if p.requires_grad]
        assert len(trainable) == 0


# ---------------------------------------------------------------------------
# 2. swap_memory_policy integration test
# ---------------------------------------------------------------------------

class TestSwapMemoryPolicy:
    """Test that swap_memory_policy replaces policy on evaluator + model."""

    def _make_fake_model_with_deep_policy(self):
        """Create a minimal mock that simulates MemoryModelWrapper with a Deep policy."""
        # Fake Deep policy (has scoring overhead)
        deep_policy = MagicMock(spec=MemoryPolicy)
        deep_policy.requires_attn_scores = True
        deep_policy.requires_queries = False
        deep_policy.cache_size = 1024

        # Fake model wrapper
        model = MagicMock()
        model.memory_policy = deep_policy
        model.config = MagicMock()
        model.registration_kwargs = {
            'num_memory_layers': 16,
            'num_heads': 8,
            'num_key_value_groups': 1,
            'hidden_size': 2048,
        }

        # Implement swap_memory_policy matching MemoryModelWrapper.swap_memory_policy
        def swap_memory_policy(new_memory_policy):
            model.memory_policy = new_memory_policy
            new_memory_policy.register_new_memory_model(
                model.config, model.registration_kwargs)
            new_memory_policy.finalize_registration()
            model.memory_requires_attn = new_memory_policy.requires_attn_scores
            model.memory_requires_queries = new_memory_policy.requires_queries

        model.swap_memory_policy = swap_memory_policy
        return model, deep_policy

    def _make_fake_evaluator(self, model):
        """Create a minimal mock matching MemoryHFEvaluator."""
        evaluator = MagicMock()
        evaluator.model = model
        evaluator.memory_policy = model.memory_policy
        evaluator.device = 'cpu'
        evaluator.max_memory_length = 6500
        evaluator.max_conditioning_length = 6500
        evaluator.batch_size = "auto"

        # Implement swap_memory_policy matching MemoryHFEvaluator.swap_memory_policy
        def swap_memory_policy(new_memory_policy):
            evaluator.model.swap_memory_policy(new_memory_policy)
            evaluator.memory_policy = new_memory_policy
            if evaluator.model.memory_policy.cache_size is not None:
                evaluator.max_memory_length = min(
                    evaluator.max_memory_length,
                    evaluator.model.memory_policy.cache_size)
            new_memory_policy.to(evaluator.device)

        evaluator.swap_memory_policy = swap_memory_policy
        return evaluator

    def test_swap_replaces_deep_with_recency(self):
        """After swap, both evaluator and model reference the Recency policy."""
        model, deep_policy = self._make_fake_model_with_deep_policy()
        evaluator = self._make_fake_evaluator(model)

        # Before: Deep policy with scoring overhead
        assert evaluator.memory_policy.requires_attn_scores is True
        assert model.memory_policy.requires_attn_scores is True

        # Swap to Recency
        recency = Recency(cache_size=None)
        evaluator.swap_memory_policy(recency)

        # After: Recency with no scoring overhead
        assert evaluator.memory_policy is recency
        assert model.memory_policy is recency
        assert model.memory_policy.requires_attn_scores is False
        assert model.memory_requires_attn is False

    def test_swap_with_cache_size_updates_max_memory_length(self):
        """Swapping to Recency(cache_size=N) caps evaluator.max_memory_length."""
        model, _ = self._make_fake_model_with_deep_policy()
        evaluator = self._make_fake_evaluator(model)
        assert evaluator.max_memory_length == 6500

        recency = Recency(cache_size=2048)
        evaluator.swap_memory_policy(recency)

        assert evaluator.max_memory_length == 2048

    def test_swap_none_cache_preserves_max_memory_length(self):
        """Swapping to Recency(cache_size=None) does not cap max_memory_length."""
        model, _ = self._make_fake_model_with_deep_policy()
        evaluator = self._make_fake_evaluator(model)
        original = evaluator.max_memory_length

        recency = Recency(cache_size=None)
        evaluator.swap_memory_policy(recency)

        # cache_size is None → the min() branch is skipped
        assert evaluator.max_memory_length == original

    def test_recency_registers_with_model_config(self):
        """Recency.finalize_registration() creates rotary_offset buffer."""
        model, _ = self._make_fake_model_with_deep_policy()
        evaluator = self._make_fake_evaluator(model)

        recency = Recency(cache_size=None)
        evaluator.swap_memory_policy(recency)

        # After registration, Recency should have rotary_offset buffer
        assert hasattr(recency, 'rotary_offset')
        assert recency.num_memory_layers == 16


# ---------------------------------------------------------------------------
# 3. run_lora.py swap logic test
# ---------------------------------------------------------------------------

class TestRunLoraSwapLogic:
    """Test that run_lora.py's swap logic fires correctly."""

    def _apply_swap_logic(self, evaluator, namm_active, cache_size):
        """Replicate the exact swap logic from run_lora.py lines 209-227."""
        memory_policy = evaluator.memory_policy
        if not namm_active:
            recency_policy = Recency(cache_size=cache_size)
            evaluator.swap_memory_policy(recency_policy)
            memory_policy = recency_policy

            evaluator.max_memory_length = evaluator.max_conditioning_length
            if evaluator.batch_size == "auto":
                evaluator.batch_size = 1
        return memory_policy

    def test_swap_fires_when_namm_inactive(self):
        """When namm_active=False, the Recency swap should execute."""
        model, _ = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)

        memory_policy = self._apply_swap_logic(evaluator, namm_active=False, cache_size=None)

        assert isinstance(memory_policy, Recency)
        assert evaluator.memory_policy.requires_attn_scores is False
        assert model.memory_policy.requires_attn_scores is False

    def test_swap_does_not_fire_when_namm_active(self):
        """When namm_active=True, the Deep policy should remain."""
        model, deep_policy = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)

        memory_policy = self._apply_swap_logic(evaluator, namm_active=True, cache_size=None)

        assert memory_policy is deep_policy
        assert evaluator.memory_policy.requires_attn_scores is True
        # batch_size should remain "auto" for NAMM
        assert evaluator.batch_size == "auto"

    def test_swap_with_explicit_cache_size(self):
        """When namm_active=False and cache_size is set, Recency respects it."""
        model, _ = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)

        memory_policy = self._apply_swap_logic(evaluator, namm_active=False, cache_size=2048)

        assert memory_policy.cache_size == 2048
        assert memory_policy.limit_cache is True
        assert evaluator.max_memory_length == 6500  # set to max_conditioning_length

    def test_swap_sets_batch_size_to_1(self):
        """When namm_active=False, batch_size should be set to 1 (skip auto-detection)."""
        model, _ = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)
        assert evaluator.batch_size == "auto"

        self._apply_swap_logic(evaluator, namm_active=False, cache_size=None)

        assert evaluator.batch_size == 1

    def test_swap_preserves_explicit_batch_size(self):
        """When batch_size is already a fixed int, don't override it."""
        model, _ = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)
        evaluator.batch_size = 4  # user set --batch_size_eval 4

        self._apply_swap_logic(evaluator, namm_active=False, cache_size=None)

        assert evaluator.batch_size == 4  # preserved

    def test_swap_updates_max_memory_length_to_conditioning(self):
        """max_memory_length should be set to max_conditioning_length after swap."""
        model, _ = TestSwapMemoryPolicy()._make_fake_model_with_deep_policy()
        evaluator = TestSwapMemoryPolicy()._make_fake_evaluator(model)
        # Simulate NAMM config: small max_memory_length, large max_conditioning_length
        evaluator.max_memory_length = 1024
        evaluator.max_conditioning_length = 6500

        self._apply_swap_logic(evaluator, namm_active=False, cache_size=None)

        assert evaluator.max_memory_length == 6500
