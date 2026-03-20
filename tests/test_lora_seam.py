"""
LoRA seam correctness tests.

Tests the LORA-04 invariants:
  (a) test_lora_module_count       — correct LoRA module count after injection
  (b) test_base_weight_stability   — base weights unchanged after set_lora_params()
  (c) test_lora_weights_float32    — LoRA weights stored as float32 in checkpoint
  (d) test_round_trip_injection    — flat vector round-trip is exact (error = 0.0)

Plus:
  (e) test_set_lora_params_size_mismatch_raises — ValueError on size mismatch

Run with:
    pytest tests/test_lora_seam.py -v

NOTE: LLaMA 3.2-1B has 16 hidden layers.
q_proj + v_proj on 16 layers = 32 LoRA Linear modules.
"""

import os
import sys
import pytest
import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for LLaMA 3.2-1B tests"
)

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"


def _build_wrapped_llama():
    """Construct a WrappedLlamaForCausalLM using the same pattern as MemoryTrainer.

    Returns the wrapper on CUDA, in bfloat16, WITHOUT LoRA injected yet.
    """
    from utils.hydra_helpers import LlamaCompatModel
    from namm.llms.llama import WrappedLlamaForCausalLM
    from namm.policy import Recency

    base = LlamaCompatModel.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    policy = Recency(cache_size=None)

    wrapper = WrappedLlamaForCausalLM(
        model=base,
        memory_policy=policy,
    )
    wrapper.move_model_to(dtype=torch.bfloat16, device='cuda')
    return wrapper


@pytest.fixture(scope="module")
def llama_with_lora():
    """Module-scoped fixture: WrappedLlamaForCausalLM with LoRA injected."""
    wrapper = _build_wrapped_llama()
    wrapper.apply_lora_adapters(rank=4, target_modules=['q_proj', 'v_proj'])
    return wrapper


# ---------------------------------------------------------------------------
# Test (a): correct module count
# ---------------------------------------------------------------------------

@SKIP_NO_CUDA
def test_lora_module_count(llama_with_lora):
    """LoRA injected into exactly num_layers * len(target_modules) modules."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    wrapper = llama_with_lora
    lora_count = sum(
        1 for _, m in wrapper.model.named_modules()
        if isinstance(m, LoraLinear)
    )
    num_layers = wrapper.config.num_hidden_layers
    expected = num_layers * 2  # q_proj + v_proj
    assert lora_count == expected, (
        f"Expected {expected} LoRA modules, got {lora_count}."
    )


# ---------------------------------------------------------------------------
# Test (b): base weight stability
# ---------------------------------------------------------------------------

@SKIP_NO_CUDA
def test_base_weight_stability(llama_with_lora):
    """Base model weights are bit-for-bit identical after set_lora_params()."""
    wrapper = llama_with_lora

    base_params_before = {
        n: p.data.clone()
        for n, p in wrapper.model.named_parameters()
        if not p.requires_grad
    }
    assert len(base_params_before) > 0

    flat = wrapper.get_lora_params_flat()
    random_vec = torch.randn_like(flat)
    wrapper.set_lora_params(random_vec)

    named_params = dict(wrapper.model.named_parameters())
    for n, before in base_params_before.items():
        after = named_params[n].data
        assert torch.equal(before, after), (
            f"Base param '{n}' was modified by set_lora_params()"
        )

    wrapper.set_lora_params(flat)


# ---------------------------------------------------------------------------
# Test (c): float32 dtype in checkpoint
# ---------------------------------------------------------------------------

@SKIP_NO_CUDA
def test_lora_weights_float32_in_checkpoint(llama_with_lora):
    """LoRA weights stored as float32 (not bfloat16)."""
    wrapper = llama_with_lora

    lora_state_dict = {
        n: p.data.clone()
        for n, p in wrapper.model.named_parameters()
        if p.requires_grad
    }
    assert len(lora_state_dict) > 0

    for name, tensor in lora_state_dict.items():
        assert tensor.dtype == torch.float32, (
            f"LoRA param '{name}' has dtype {tensor.dtype}, expected torch.float32."
        )

    flat = wrapper.get_lora_params_flat()
    assert flat.dtype == torch.float32
    assert flat.device.type == 'cpu'


# ---------------------------------------------------------------------------
# Test (d): flat vector round-trip is exact
# ---------------------------------------------------------------------------

@SKIP_NO_CUDA
def test_round_trip_injection(llama_with_lora):
    """get_lora_params_flat() -> set_lora_params() -> get_lora_params_flat() is exact."""
    wrapper = llama_with_lora

    original_flat = wrapper.get_lora_params_flat()
    assert original_flat.numel() > 0

    test_vec = torch.randn_like(original_flat)
    wrapper.set_lora_params(test_vec)

    wrapper.set_lora_params(original_flat)
    restored_flat = wrapper.get_lora_params_flat()

    error = (original_flat - restored_flat).abs().max().item()
    assert error == 0.0, f"Round-trip error = {error} (expected 0.0)."


@SKIP_NO_CUDA
def test_set_lora_params_size_mismatch_raises(llama_with_lora):
    """set_lora_params() raises ValueError on size mismatch."""
    wrapper = llama_with_lora
    flat = wrapper.get_lora_params_flat()
    bad_vec = torch.zeros(flat.numel() + 1)
    with pytest.raises(ValueError, match="flat_vec size"):
        wrapper.set_lora_params(bad_vec)
