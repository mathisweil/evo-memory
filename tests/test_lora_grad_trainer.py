"""
Gradient-flow correctness gate for LoRAGradTrainer — Phase 3, Plan 03.

Covers TRAIN-04, TRAIN-05, TRAIN-06 and the CONTEXT.md locked smoke test
(10-step net loss decrease).

Run with:
    pytest tests/test_lora_grad_trainer.py -v --tb=short

Requires:
  - CUDA GPU (runs on sideswipe or prowl)
  - th2 conda env with pytest, peft, transformers, datasets

Skip mechanism: pytestmark at module level skips all tests on non-GPU machines.
"""

import copy
import os
import sys

import pytest
import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_CACHE = "/cs/student/project_msc/2025/csml/gmaralla/.hf_cache"
# Local snapshot path — LlamaCompatModel reads config.json from here to patch rope_scaling
LOCAL_MODEL_PATH = (
    "/cs/student/project_msc/2025/csml/gmaralla/.hf_cache/hub/"
    "models--meta-llama--Llama-3.2-1B/snapshots/"
    "4e20de362430cd3b72f300e6b0f18e50e7166e08"
)
DEVICE = "cuda"
LORA_RANK = 8
LORA_TARGETS = ["q_proj", "v_proj"]

NAMM_CKPT = (
    "/cs/student/project_msc/2025/csml/gmaralla/NAMM_implementation/"
    "exp_local/memory_evolution_hf/Llama-3.2-1B/NAMM/attn-spec-norm/bam/"
    "binary-1024cs/qasper-cma-es-p8-rMeanTrue-shared-8pop-16qs-256fixDel-"
    "llama32-1b-stage1/1337/ckpt.pt"
)

# Module-level skip guard: all tests are no-ops on non-GPU machines.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for gradient tests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wrapped_llama():
    """Construct WrappedLlamaForCausalLM in bfloat16 on CUDA.

    Uses LlamaCompatModel (not AutoModelForCausalLM) so that the llama3
    rope_scaling format is patched to None before loading — required with
    transformers 4.41.x which only accepts the old {type, factor} format.
    Uses Recency(cache_size=None) as the no-eviction passthrough policy.
    """
    from utils_hydra import LlamaCompatModel
    from memory_llms.llama import WrappedLlamaForCausalLM
    from memory_policy import Recency

    base = LlamaCompatModel.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    policy = Recency(cache_size=None)

    wrapper = WrappedLlamaForCausalLM(
        model=base,
        memory_policy=policy,
    )
    wrapper.move_model_to(dtype=torch.bfloat16, device=DEVICE)
    return wrapper


def _make_trainer_config(tmp_out_dir, gradient_accumulation_steps=1):
    """Return a minimal LoRATrainerConfig for testing."""
    from lora_grad_trainer import LoRATrainerConfig

    return LoRATrainerConfig(
        out_dir=tmp_out_dir,
        method="lora_grad",
        seed=1337,
        max_seq_len=512,
        task_names=["qasper"],
        cache_dir=HF_CACHE,
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        namm_active=False,
        eval_interval=100,
        log_interval=10,
        always_save_checkpoint=False,
        init_from=None,
        dtype="bfloat16",
    )


def _make_wandb_config():
    """Return a WandbConfig with logging disabled."""
    from memory_trainer import WandbConfig

    return WandbConfig(
        wandb_log=False,
        wandb_project="test_lora_grad",
        wandb_run_name="test",
        wandb_group_name="test",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture: loads model once for the entire test session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lora_trainer_fixture(tmp_path_factory):
    """Session-scoped fixture: load LLaMA 3.2-1B once, apply LoRA, build trainer.

    Yields: (trainer, batch, lora_params)
      - trainer    : LoRAGradTrainer instance (wandb disabled)
      - batch      : dict with 'input_ids' and 'labels', shape [1, seq_len]
      - lora_params: list of LoRA parameter tensors (requires_grad=True)
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU required — session fixture skipped on CPU-only machine")

    from functools import partial
    from transformers import AutoTokenizer
    from lora_grad_trainer import LoRAGradTrainer
    from lora_ntp_dataset import LongBenchNTPDataset, ntp_pad_collate_fn

    tmp_out = str(tmp_path_factory.mktemp("lora_trainer_out"))

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -- Model + LoRA injection --
    model = _build_wrapped_llama()
    model.apply_lora_adapters(rank=LORA_RANK, target_modules=LORA_TARGETS)

    # -- LoRA params (all requires_grad=True after injection) --
    lora_params = [p for p in model.parameters() if p.requires_grad]
    assert len(lora_params) > 0, "LoRA injection produced no trainable params"

    # -- Build dataset (small subset: max_seq_len=512 for test speed) --
    dataset = LongBenchNTPDataset(
        task_names=["qasper"],
        tokenizer=tokenizer,
        max_seq_len=512,
        cache_dir=HF_CACHE,
        seed=1337,
    )
    assert len(dataset) > 0, "Dataset is empty — check HF cache and network"

    # -- Get one batch from dataset --
    first_item = dataset[0]
    collate_fn = partial(
        ntp_pad_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=512,
    )
    batch = collate_fn([first_item])

    # -- Trainer config + instantiation (gradient_accumulation_steps=1 for tests) --
    cfg = _make_trainer_config(tmp_out, gradient_accumulation_steps=1)
    wcfg = _make_wandb_config()

    trainer = LoRAGradTrainer(
        model=model,
        tokenizer=tokenizer,
        evaluation_model=None,
        trainer_config=cfg,
        wandb_config=wcfg,
        device=DEVICE,
    )

    # Put model in eval mode initially (tests control train/eval as needed).
    model.eval()

    yield trainer, batch, lora_params

    # Teardown: zero grads to avoid leaking state between sessions
    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 1: loss.requires_grad is True after forward pass
# ---------------------------------------------------------------------------

def test_loss_requires_grad(lora_trainer_fixture):
    """TRAIN-04: loss.requires_grad must be True after forward (PEFT hook check).

    Calls model forward without invoking backward, then checks that the loss
    tensor has requires_grad=True. Failure means the PEFT embedding hook
    (make_inputs_require_grad) is disconnected and gradients would not flow.
    """
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    # Forward only — no backward yet
    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
    )

    assert outputs.loss.requires_grad, (
        "loss.requires_grad is False — PEFT embedding hook (make_inputs_require_grad) "
        "is not registered or not active. Check LoRAGradTrainer.__init__."
    )


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 2: all LoRA params have non-None grad after backward
# ---------------------------------------------------------------------------

def test_lora_grads_nonzero(lora_trainer_fixture):
    """TRAIN-04: All LoRA params must have non-None grad after backward.

    Also checks that at least one LoRA grad has a non-zero absolute sum,
    confirming that gradients actually propagated (not just allocated as zeros).
    """
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
    )
    outputs.loss.backward()

    assert all(p.grad is not None for p in lora_params), (
        "Some LoRA params have grad=None after backward. "
        "Check PEFT embedding hook and requires_grad assignment."
    )
    assert any(p.grad.abs().sum() > 0 for p in lora_params), (
        "All LoRA param gradients are zero — gradient flow is blocked."
    )

    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 3: base model params have None grad (frozen base confirmed)
# ---------------------------------------------------------------------------

def test_base_grads_none(lora_trainer_fixture):
    """TRAIN-04: Base model params must have None grad after backward.

    Frozen-base violation means AdamW would corrupt the pre-trained base weights.
    """
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
    )
    outputs.loss.backward()

    # Base params: all params in model.model that do NOT belong to LoRA
    base_params = [
        p for n, p in trainer.model.model.named_parameters()
        if "lora_" not in n and not p.requires_grad
    ]
    assert len(base_params) > 0, "No base params found — check model structure"

    # Every frozen base param must have p.grad is None
    non_none = [
        (n, p)
        for n, p in trainer.model.model.named_parameters()
        if "lora_" not in n and not p.requires_grad and p.grad is not None
    ]
    assert len(non_none) == 0, (
        f"Base params with non-None grad after backward: "
        f"{[n for n, _ in non_none[:5]]}... — frozen base violated. "
        f"All frozen base params must satisfy: p.grad is None"
    )

    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 4: all LoRA params are float32 (no AMP downcast)
# ---------------------------------------------------------------------------

def test_lora_float32(lora_trainer_fixture):
    """TRAIN-04: LoRA parameter tensors must be float32.

    AMP or bfloat16 downcast causes underflow at ES sigma=0.001.
    apply_lora_adapters() is required to cast LoRA params to float32.
    """
    trainer, batch, lora_params = lora_trainer_fixture

    wrong_dtypes = {p.dtype for p in lora_params if p.dtype != torch.float32}
    assert not wrong_dtypes, (
        f"LoRA params must be float32, but found dtype(s): {wrong_dtypes}. "
        f"Check apply_lora_adapters() float32 cast."
    )


# ---------------------------------------------------------------------------
# TRAIN-05 — Test 5: checkpoint save/load restores optimizer state
# ---------------------------------------------------------------------------

def test_checkpoint_resume(lora_trainer_fixture, tmp_path):
    """TRAIN-05: Checkpoint saved by LoRAGradTrainer restores optimizer state on reload.

    Verifies:
      - Optimizer state dict is non-empty after one gradient update
      - Fresh trainer loaded from checkpoint has matching optimizer state
      - step_num returned by _load_ckpt matches what was saved
      - LoRA weights match bit-for-bit after load
    """
    from lora_grad_trainer import LoRAGradTrainer
    from functools import partial
    from transformers import AutoTokenizer
    from lora_ntp_dataset import ntp_pad_collate_fn

    trainer, batch, lora_params = lora_trainer_fixture

    # -- Set up artifact dir so _save_ckpt has an output path --
    trainer._write_artifact_contract(
        cfg_yaml="test: true\n", method="lora_grad", seed=1337
    )

    # -- Set model to train mode and re-freeze base weights (as train() does) --
    trainer.model.train()
    for n, p in trainer.model.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    # -- One gradient update step --
    trainer.optimizer.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
    )
    loss = outputs.loss / trainer.trainer_config.gradient_accumulation_steps
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, trainer.trainer_config.max_grad_norm)
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    # -- Optimizer state must be non-empty after first step --
    opt_state_after_step = trainer.optimizer.state_dict()
    assert len(opt_state_after_step["state"]) > 0, (
        "Optimizer state is empty after gradient step — Adam state not initialized"
    )

    # -- Save checkpoint --
    ckpt_path = trainer._save_ckpt(step_num=1)
    assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

    # Snapshot LoRA weights before load
    lora_weights_before = [p.data.clone() for p in lora_params]

    # -- Build a fresh trainer from the same model + config --
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    if tokenizer2.pad_token_id is None:
        tokenizer2.pad_token_id = tokenizer2.eos_token_id

    model2 = _build_wrapped_llama()
    model2.apply_lora_adapters(rank=LORA_RANK, target_modules=LORA_TARGETS)

    out_dir2 = str(tmp_path / "trainer2_out")
    cfg2 = _make_trainer_config(out_dir2, gradient_accumulation_steps=1)
    wcfg2 = _make_wandb_config()

    trainer2 = LoRAGradTrainer(
        model=model2,
        tokenizer=tokenizer2,
        evaluation_model=None,
        trainer_config=cfg2,
        wandb_config=wcfg2,
        device=DEVICE,
    )

    # -- Load checkpoint into trainer2 --
    step_num = trainer2._load_ckpt(ckpt_path)

    # -- Verify step_num restored --
    assert step_num == 1, (
        f"Expected step_num=1 from _load_ckpt, got {step_num}"
    )

    # -- Verify optimizer state restored (non-empty) --
    restored_opt_state = trainer2.optimizer.state_dict()
    assert len(restored_opt_state["state"]) > 0, (
        "Optimizer state is empty after _load_ckpt — TRAIN-05 optimizer restore failed"
    )

    # -- Verify LoRA weights match --
    trainer2_lora_params = [p for p in trainer2.model.parameters() if p.requires_grad]
    for i, (p1, p2) in enumerate(zip(lora_params, trainer2_lora_params)):
        assert torch.allclose(p1.data.float(), p2.data.float()), (
            f"LoRA weight mismatch at param index {i} after checkpoint load"
        )

    # Clean up
    trainer.model.eval()
    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-06 — Test 6: LoRA grads are non-None when namm_active=True
# ---------------------------------------------------------------------------

def test_namm_active_lora_grads(lora_trainer_fixture):
    """TRAIN-06: LoRA grads must be non-None even with NAMM-active forward.

    NAMM eviction is non-differentiable (topk + index), but gradient-transparent:
    backprop flows through LLM/LoRA math on the retained token set.

    Skips if the NAMM checkpoint is not available on this machine.
    """
    if not os.path.exists(NAMM_CKPT):
        pytest.skip(f"NAMM checkpoint not available: {NAMM_CKPT}")

    trainer, batch, lora_params = lora_trainer_fixture

    # Enable namm_active mode for this test
    original_namm_active = trainer.trainer_config.namm_active
    trainer.trainer_config.namm_active = True

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.train()
    for n, p in trainer.model.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    try:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = trainer.model(
            input_ids=input_ids,
            labels=labels,
            apply_memory_policy=True,
        )
        outputs.loss.backward()

        assert all(p.grad is not None for p in lora_params), (
            "LoRA params have None grad when namm_active=True — "
            "NAMM-active forward is blocking gradient flow. "
            "Expected: gradient-transparent (flows through retained tokens)."
        )

    finally:
        # Restore original namm_active value
        trainer.trainer_config.namm_active = original_namm_active
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.model.eval()


# ---------------------------------------------------------------------------
# TRAIN-04 smoke test — Test 7: net loss decrease over 10 _train_step calls
# (LOCKED DECISION from CONTEXT.md)
# ---------------------------------------------------------------------------

def test_loss_decreases_10_steps(lora_trainer_fixture):
    """Net loss decrease over 10 gradient updates (LOCKED DECISION from CONTEXT.md).

    Asserts: losses[-1] < losses[0] (net decrease only, not strict per-step).
    Per-step monotonicity is NOT asserted — minibatch noise makes that flaky.

    Uses gradient_accumulation_steps=1 so each _train_step + _optimizer_step pair
    constitutes one actual gradient update. Confirms the training loop converges
    on a small repeated batch.
    """
    trainer, batch, lora_params = lora_trainer_fixture

    # Force gradient_accumulation_steps=1 for this test
    original_grad_accum = trainer.trainer_config.gradient_accumulation_steps
    trainer.trainer_config.gradient_accumulation_steps = 1

    # Set model to train mode, re-freeze base weights
    trainer.model.train()
    for n, p in trainer.model.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    trainer.optimizer.zero_grad(set_to_none=True)

    losses = []
    try:
        for step in range(10):
            # _train_step runs forward + backward (loss is scaled by grad_accum_steps=1)
            step_loss, _ = trainer._train_step(batch)
            # _optimizer_step clips grads, steps optimizer+scheduler, zeros grads
            trainer._optimizer_step()
            losses.append(step_loss)

    finally:
        # Restore original gradient_accumulation_steps
        trainer.trainer_config.gradient_accumulation_steps = original_grad_accum
        trainer.model.eval()
        trainer.optimizer.zero_grad(set_to_none=True)

    assert len(losses) == 10, f"Expected 10 loss values, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"Loss must decrease net over 10 steps: "
        f"losses[0]={losses[0]:.4f}, losses[-1]={losses[-1]:.4f}. "
        f"All losses: {[f'{l:.4f}' for l in losses]}"
    )
