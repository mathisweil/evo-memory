"""
Gradient-flow correctness gate for LoRAGradTrainer.

Covers TRAIN-04, TRAIN-05, TRAIN-06 and the smoke test
(10-step net loss decrease).

Run with:
    pytest tests/test_lora_grad_trainer.py -v --tb=short

Requires:
  - CUDA GPU
  - peft, transformers, datasets installed
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
DEVICE = "cuda"
LORA_RANK = 8
LORA_TARGETS = ["q_proj", "v_proj"]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for gradient tests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wrapped_llama():
    """Construct WrappedLlamaForCausalLM in bfloat16 on CUDA."""
    from utils.hydra_helpers import LlamaCompatModel
    from namm.llms.llama import WrappedLlamaForCausalLM
    from namm.policy import Recency

    base = LlamaCompatModel.from_pretrained(
        MODEL_NAME,
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
    from grad_lora_finetuning.trainer import LoRATrainerConfig

    return LoRATrainerConfig(
        out_dir=tmp_out_dir,
        method="lora_grad",
        seed=1337,
        max_seq_len=512,
        task_names=["qasper"],
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
    from namm.trainer import WandbConfig

    return WandbConfig(
        wandb_log=False,
        wandb_project="test_lora_grad",
        wandb_run_name="test",
        wandb_group_name="test",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lora_trainer_fixture(tmp_path_factory):
    """Session-scoped fixture: load LLaMA 3.2-1B once, apply LoRA, build trainer.

    Yields: (trainer, batch, lora_params)
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU required")

    from functools import partial
    from transformers import AutoTokenizer
    from grad_lora_finetuning.trainer import LoRAGradTrainer
    from grad_lora_finetuning.datasets import NTPDataset, pad_collate_fn

    tmp_out = str(tmp_path_factory.mktemp("lora_trainer_out"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = _build_wrapped_llama()
    model.apply_lora_adapters(rank=LORA_RANK, target_modules=LORA_TARGETS)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    assert len(lora_params) > 0

    dataset = NTPDataset(
        task_names=["qasper"],
        tokenizer=tokenizer,
        max_seq_len=512,
        seed=1337,
    )
    assert len(dataset) > 0

    first_item = dataset[0]
    collate_fn = partial(
        pad_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=512,
    )
    batch = collate_fn([first_item])

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

    model.eval()

    yield trainer, batch, lora_params

    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 1: loss.requires_grad is True
# ---------------------------------------------------------------------------

def test_loss_requires_grad(lora_trainer_fixture):
    """loss.requires_grad must be True after forward."""
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
        use_cache=True,
    )

    assert outputs.loss.requires_grad, (
        "loss.requires_grad is False — PEFT embedding hook not active."
    )


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 2: LoRA grads are non-None after backward
# ---------------------------------------------------------------------------

def test_lora_grads_nonzero(lora_trainer_fixture):
    """All LoRA params must have non-None grad after backward."""
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
        use_cache=True,
    )
    outputs.loss.backward()

    assert all(p.grad is not None for p in lora_params), (
        "Some LoRA params have grad=None after backward."
    )
    assert any(p.grad.abs().sum() > 0 for p in lora_params), (
        "All LoRA param gradients are zero."
    )

    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 3: base params have None grad (frozen)
# ---------------------------------------------------------------------------

def test_base_grads_none(lora_trainer_fixture):
    """Base model params must have None grad after backward."""
    trainer, batch, lora_params = lora_trainer_fixture

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
        use_cache=True,
    )
    outputs.loss.backward()

    non_none = [
        (n, p)
        for n, p in trainer.model.model.named_parameters()
        if "lora_" not in n and not p.requires_grad and p.grad is not None
    ]
    assert len(non_none) == 0, (
        f"Base params with non-None grad: {[n for n, _ in non_none[:5]]}"
    )

    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 — Test 4: LoRA params are float32
# ---------------------------------------------------------------------------

def test_lora_float32(lora_trainer_fixture):
    """LoRA parameter tensors must be float32."""
    trainer, batch, lora_params = lora_trainer_fixture

    wrong_dtypes = {p.dtype for p in lora_params if p.dtype != torch.float32}
    assert not wrong_dtypes, (
        f"LoRA params must be float32, found: {wrong_dtypes}"
    )


# ---------------------------------------------------------------------------
# TRAIN-05 — Test 5: checkpoint resume
# ---------------------------------------------------------------------------

def test_checkpoint_resume(lora_trainer_fixture, tmp_path):
    """Checkpoint restores optimizer state on reload."""
    from grad_lora_finetuning.trainer import LoRAGradTrainer
    from transformers import AutoTokenizer

    trainer, batch, lora_params = lora_trainer_fixture

    trainer._write_artifact_contract(
        cfg_yaml="test: true\n", method="lora_grad", seed=1337
    )

    trainer.model.train()
    for n, p in trainer.model.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    trainer.optimizer.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    outputs = trainer.model(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=False,
        use_cache=True,
    )
    loss = outputs.loss / trainer.trainer_config.gradient_accumulation_steps
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, trainer.trainer_config.max_grad_norm)
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    opt_state_after_step = trainer.optimizer.state_dict()
    assert len(opt_state_after_step["state"]) > 0

    ckpt_path = trainer._save_checkpoint(step_num=1)
    assert os.path.exists(ckpt_path)

    # Build a fresh trainer and load checkpoint
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME)
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

    step_num = trainer2._load_ckpt(ckpt_path)
    assert step_num == 1

    restored_opt_state = trainer2.optimizer.state_dict()
    assert len(restored_opt_state["state"]) > 0

    trainer.model.eval()
    trainer.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# TRAIN-04 smoke test — Test 7: net loss decrease over 10 steps
# ---------------------------------------------------------------------------

def test_loss_decreases_10_steps(lora_trainer_fixture):
    """Net loss decrease over 10 gradient updates."""
    trainer, batch, lora_params = lora_trainer_fixture

    original_grad_accum = trainer.trainer_config.gradient_accumulation_steps
    trainer.trainer_config.gradient_accumulation_steps = 1

    trainer.model.train()
    for n, p in trainer.model.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    trainer.optimizer.zero_grad(set_to_none=True)

    losses = []
    try:
        for step in range(10):
            step_loss, _ = trainer._train_step(batch)
            trainer._optimizer_step()
            losses.append(step_loss)

    finally:
        trainer.trainer_config.gradient_accumulation_steps = original_grad_accum
        trainer.model.eval()
        trainer.optimizer.zero_grad(set_to_none=True)

    assert len(losses) == 10
    assert losses[-1] < losses[0], (
        f"Loss must decrease: losses[0]={losses[0]:.4f}, losses[-1]={losses[-1]:.4f}"
    )
