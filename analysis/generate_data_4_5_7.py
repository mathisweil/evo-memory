"""Run GPU analyses (Reports 4, 5, 7) using maskfix checkpoints.

Loads M1, M3-maskfix, and M3-buggy models via LlamaCompatModel + PEFT merge
(NOT the full NAMM infrastructure, which has device placement issues).

Produces:
  Report 4: LoRA weight norms, SVD spectra, subspace overlap (CPU-only, from
            checkpoint lora_state_dict directly).
  Report 5: Per-layer per-head attention entropy and sink fractions.
  Report 7: Per-layer CKA between mean-pooled hidden states.

Reports 6, 8, 9 require the full NAMM infrastructure for eviction-aware
forward passes and are NOT included here -- they need separate handling due
to device placement issues with the NAMM policy + WrappedLlama stack.

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python analysis/generate_data_4_5_7.py

    # Regenerate plots from saved data (no GPU needed):
    PYTHONPATH=. .venv/bin/python analysis/generate_data_4_5_7.py --plot-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.hydra_helpers import LlamaCompatModel  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

M1_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M1/best_ckpt.pt"
M3_MASKFIX_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M3_cs1024_maskfix/best_ckpt.pt"
M3_BUGGY_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt"

N_SAMPLES = 10
MAX_LENGTH = 1024
NUM_LAYERS = 16
RANK = 8
PROJS = ["q_proj", "v_proj"]

REPORT_4_DIR = REPO_ROOT / "analysis" / "report_4"
REPORT_5_DIR = REPO_ROOT / "analysis" / "report_5"
REPORT_7_DIR = REPO_ROOT / "analysis" / "report_7"

# Plot colours
M1_COLOR = "#2196F3"
M3_MF_COLOR = "#4CAF50"      # maskfix = green
M3_BUGGY_COLOR = "#FF5722"   # buggy   = red/orange

# ---------------------------------------------------------------------------
# Model loading (same pattern as run_gpu_analyses.py)
# ---------------------------------------------------------------------------

def load_model_with_lora(ckpt_path: str | Path, device: str = "cuda"):
    """Load base model + apply LoRA, merge into base weights to save memory."""
    logger.info("Loading base model from %s", MODEL_ID)
    model = LlamaCompatModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    lora_config_dict = ckpt["lora_config"]
    lora_sd = ckpt["lora_state_dict"]

    config = LoraConfig(
        r=lora_config_dict.get("r", 8),
        lora_alpha=lora_config_dict.get("lora_alpha", 16),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=0.0,
    )
    model = PeftModel(model, config)

    # Remap checkpoint keys: checkpoint uses "base_model.model.layers.X..."
    # but PeftModel wrapping LlamaCompatModel expects
    # "base_model.model.model.layers.X..."
    remapped_sd = {}
    for k, v in lora_sd.items():
        new_key = k.replace(
            "base_model.model.layers.", "base_model.model.model.layers."
        )
        remapped_sd[new_key] = v

    missing, _ = model.load_state_dict(remapped_sd, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        raise RuntimeError(
            f"LoRA weights not loaded! Missing keys: {lora_missing[:5]}"
        )
    logger.info(
        "  LoRA weights loaded: %d tensors, %d lora missing",
        len(remapped_sd), len(lora_missing),
    )

    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    logger.info(
        "  Model on device. GPU mem: %.2f GB",
        torch.cuda.memory_allocated() / 1e9,
    )
    return model


# ---------------------------------------------------------------------------
# Test prompts (same as run_gpu_analyses.py)
# ---------------------------------------------------------------------------

def get_test_prompts(tokenizer, n: int = N_SAMPLES):
    """Load a sample of test prompts from LongBench."""
    from datasets import load_dataset

    prompts: list[dict] = []
    for task in ["qasper", "2wikimqa", "hotpotqa_e"]:
        ds = load_dataset(
            "THUDM/LongBench", task, split="test", trust_remote_code=True,
        )
        for i, sample in enumerate(ds):
            if len(prompts) >= n:
                break
            text = (
                sample["context"][:8000]
                + "\n\nQuestion: " + sample["input"]
                + "\nAnswer:"
            )
            ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
            )
            prompts.append({"input_ids": ids["input_ids"], "task": task, "idx": i})
        if len(prompts) >= n:
            break
    return prompts[:n]


# ---------------------------------------------------------------------------
# Inference: extract attention + hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_and_hidden(model, prompts, device: str = "cuda"):
    """Run inference on prompts, compute entropy stats inline, store
    mean-pooled hidden states for CKA.

    Returns:
        entropy_avg: (n_layers, n_heads)
        sink_avg:    (n_layers, n_heads)
        all_hidden:  list of lists — all_hidden[sample][layer] = (hidden_dim,)
    """
    entropy_sum = None
    sink_sum = None
    all_hidden: list[list[torch.Tensor]] = []
    count = 0

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        out = model(input_ids, output_attentions=True, output_hidden_states=True)

        for layer_idx, layer_attn in enumerate(out.attentions):
            attn = layer_attn[0].float()  # (n_heads, seq, seq)
            n_heads = attn.shape[0]
            if entropy_sum is None:
                entropy_sum = torch.zeros(NUM_LAYERS, n_heads)
                sink_sum = torch.zeros(NUM_LAYERS, n_heads)

            # Use last query position (answer-generation position)
            attn_last = attn[:, -1, :]  # (n_heads, seq)
            attn_clamped = attn_last.clamp(min=1e-12)
            h = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (n_heads,)
            entropy_sum[layer_idx] += h.cpu()

            sink = attn_last[:, :5].sum(dim=-1)  # (n_heads,)
            sink_sum[layer_idx] += sink.cpu()

        hidden_pooled = []
        for hs in out.hidden_states:
            hidden_pooled.append(hs[0].float().mean(dim=0).cpu())  # (hidden_dim,)
        all_hidden.append(hidden_pooled)

        count += 1
        seq_len = input_ids.shape[1]
        logger.info("  Sample %d/%d done (seq_len=%d)", i + 1, len(prompts), seq_len)

        del out
        torch.cuda.empty_cache()

    entropy_avg = (entropy_sum / count).numpy()
    sink_avg = (sink_sum / count).numpy()
    return entropy_avg, sink_avg, all_hidden


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two representation matrices (n, d1) and (n, d2)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X
    numerator = (YTX ** 2).sum()
    denominator = torch.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
    if denominator < 1e-10:
        return 0.0
    return (numerator / denominator).item()


# ===========================================================================
# Report 4: LoRA weight analysis (CPU only -- reads checkpoint dicts)
# ===========================================================================

def get_BA(state_dict: dict, layer: int, proj: str) -> np.ndarray:
    """Build effective LoRA update B @ A for a given layer and projection."""
    prefix = f"base_model.model.layers.{layer}.self_attn.{proj}"
    A = state_dict[f"{prefix}.lora_A.default.weight"]  # (rank, d_in)
    B = state_dict[f"{prefix}.lora_B.default.weight"]  # (d_out, rank)
    return (B @ A).float().numpy()


def compute_report4_data() -> dict:
    """Compute Report 4 metrics from checkpoint lora_state_dicts.

    Compares M1 vs M3-maskfix and M1 vs M3-buggy.
    """
    logger.info("=== Report 4: loading checkpoint state dicts (CPU) ===")
    m1_sd = torch.load(str(M1_CKPT), map_location="cpu")["lora_state_dict"]
    m3mf_sd = torch.load(str(M3_MASKFIX_CKPT), map_location="cpu")["lora_state_dict"]
    m3bug_sd = torch.load(str(M3_BUGGY_CKPT), map_location="cpu")["lora_state_dict"]

    results: dict = {
        tag: {
            "norms": {p: [] for p in PROJS},
            "sv": {p: [] for p in PROJS},
        }
        for tag in ["m1", "m3_maskfix", "m3_buggy"]
    }
    results["ratio_maskfix"] = {p: [] for p in PROJS}
    results["ratio_buggy"] = {p: [] for p in PROJS}
    results["overlap_maskfix"] = {p: [] for p in PROJS}
    results["overlap_buggy"] = {p: [] for p in PROJS}

    sds = {"m1": m1_sd, "m3_maskfix": m3mf_sd, "m3_buggy": m3bug_sd}

    for layer in range(NUM_LAYERS):
        for proj in PROJS:
            BAs = {}
            Us = {}
            Ss = {}
            for tag, sd in sds.items():
                ba = get_BA(sd, layer, proj)
                BAs[tag] = ba
                norm = np.linalg.norm(ba, "fro")
                results[tag]["norms"][proj].append(float(norm))

                U, S, _ = np.linalg.svd(ba, full_matrices=False)
                Us[tag] = U[:, :RANK]
                Ss[tag] = S[:RANK]
                results[tag]["sv"][proj].append(S[:RANK].tolist())

            # Ratios and overlaps vs M1
            m1_norm = results["m1"]["norms"][proj][-1]
            for suffix, tag in [("maskfix", "m3_maskfix"), ("buggy", "m3_buggy")]:
                other_norm = results[tag]["norms"][proj][-1]
                results[f"ratio_{suffix}"][proj].append(
                    other_norm / m1_norm if m1_norm > 1e-12 else float("nan")
                )
                cos_angles = np.linalg.svd(
                    Us["m1"].T @ Us[tag], compute_uv=False,
                )
                cos_angles = np.clip(cos_angles, 0.0, 1.0)
                results[f"overlap_{suffix}"][proj].append(float(cos_angles.mean()))

        logger.info("  layer %2d done", layer)

    # Convert to numpy arrays for saving
    for tag in ["m1", "m3_maskfix", "m3_buggy"]:
        for proj in PROJS:
            results[tag]["norms"][proj] = np.array(results[tag]["norms"][proj])
    for suffix in ["maskfix", "buggy"]:
        for proj in PROJS:
            results[f"ratio_{suffix}"][proj] = np.array(results[f"ratio_{suffix}"][proj])
            results[f"overlap_{suffix}"][proj] = np.array(results[f"overlap_{suffix}"][proj])

    return results


def save_report4_data(r4: dict) -> None:
    """Save Report 4 data as .npz."""
    os.makedirs(REPORT_4_DIR, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for tag in ["m1", "m3_maskfix", "m3_buggy"]:
        for proj in PROJS:
            flat[f"{tag}_norms_{proj}"] = np.array(r4[tag]["norms"][proj])
            flat[f"{tag}_sv_{proj}"] = np.array(r4[tag]["sv"][proj])
    for suffix in ["maskfix", "buggy"]:
        for proj in PROJS:
            flat[f"ratio_{suffix}_{proj}"] = np.array(r4[f"ratio_{suffix}"][proj])
            flat[f"overlap_{suffix}_{proj}"] = np.array(r4[f"overlap_{suffix}"][proj])
    path = REPORT_4_DIR / "maskfix_data.npz"
    np.savez(str(path), **flat)
    logger.info("Saved %s", path)


def load_report4_data() -> dict:
    """Load Report 4 data from .npz and reconstruct nested dict."""
    path = REPORT_4_DIR / "maskfix_data.npz"
    data = np.load(str(path), allow_pickle=False)
    r4: dict = {
        tag: {"norms": {}, "sv": {}}
        for tag in ["m1", "m3_maskfix", "m3_buggy"]
    }
    for tag in ["m1", "m3_maskfix", "m3_buggy"]:
        for proj in PROJS:
            r4[tag]["norms"][proj] = data[f"{tag}_norms_{proj}"]
            r4[tag]["sv"][proj] = data[f"{tag}_sv_{proj}"]
    for suffix in ["maskfix", "buggy"]:
        r4[f"ratio_{suffix}"] = {}
        r4[f"overlap_{suffix}"] = {}
        for proj in PROJS:
            r4[f"ratio_{suffix}"][proj] = data[f"ratio_{suffix}_{proj}"]
            r4[f"overlap_{suffix}"][proj] = data[f"overlap_{suffix}_{proj}"]
    return r4


# ===========================================================================
# Report 4: plots
# ===========================================================================

def plot_report4(r4: dict) -> None:
    """Generate all Report 4 plots with _maskfix suffix."""
    logger.info("=== Generating Report 4 plots ===")
    os.makedirs(REPORT_4_DIR, exist_ok=True)
    layers = np.arange(NUM_LAYERS)

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    })

    # ── Plot 1: weight magnitude ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, proj in zip(axes, PROJS):
        w = 0.25
        ax.bar(
            layers - w, r4["m1"]["norms"][proj], w,
            label="M1 (full ctx)", color=M1_COLOR, alpha=0.85,
        )
        ax.bar(
            layers, r4["m3_maskfix"]["norms"][proj], w,
            label="M3 maskfix", color=M3_MF_COLOR, alpha=0.85,
        )
        ax.bar(
            layers + w, r4["m3_buggy"]["norms"][proj], w,
            label="M3 buggy", color=M3_BUGGY_COLOR, alpha=0.85,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("||B @ A||_F")
        ax.set_title(f"Effective LoRA Update Norm -- {proj}")
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Per-Layer LoRA Weight Magnitude: M1 vs M3-maskfix vs M3-buggy",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(REPORT_4_DIR / "weight_magnitude_maskfix.png"))
    plt.close(fig)
    logger.info("  Saved weight_magnitude_maskfix.png")

    # ── Plot 2: singular values (q_proj, 4x4 grid) ───────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    axes_flat = axes.flatten()
    for layer in range(NUM_LAYERS):
        ax = axes_flat[layer]
        sv_m1 = np.array(r4["m1"]["sv"]["q_proj"][layer])
        sv_mf = np.array(r4["m3_maskfix"]["sv"]["q_proj"][layer])
        sv_bug = np.array(r4["m3_buggy"]["sv"]["q_proj"][layer])
        x = np.arange(1, len(sv_m1) + 1)
        w = 0.25
        ax.bar(
            x - w, sv_m1, w,
            label="M1" if layer == 0 else None,
            color=M1_COLOR, alpha=0.85,
        )
        ax.bar(
            x, sv_mf, w,
            label="M3 maskfix" if layer == 0 else None,
            color=M3_MF_COLOR, alpha=0.85,
        )
        ax.bar(
            x + w, sv_bug, w,
            label="M3 buggy" if layer == 0 else None,
            color=M3_BUGGY_COLOR, alpha=0.85,
        )
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xticks(x)
        if layer >= 12:
            ax.set_xlabel("SV index")
        if layer % 4 == 0:
            ax.set_ylabel("Singular value")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Singular Value Spectrum of B@A (q_proj): M1 vs M3-maskfix vs M3-buggy",
        fontsize=13, fontweight="bold",
    )
    fig.legend(
        ["M1 (full ctx)", "M3 maskfix", "M3 buggy"],
        loc="upper right", fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(REPORT_4_DIR / "singular_values_maskfix.png"))
    plt.close(fig)
    logger.info("  Saved singular_values_maskfix.png")

    # ── Plot 3: subspace overlap ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, proj in zip(axes, PROJS):
        w = 0.35
        ax.bar(
            layers - w / 2, r4["overlap_maskfix"][proj], w,
            label="M1 vs M3-maskfix", color=M3_MF_COLOR, alpha=0.85,
        )
        ax.bar(
            layers + w / 2, r4["overlap_buggy"][proj], w,
            label="M1 vs M3-buggy", color=M3_BUGGY_COLOR, alpha=0.85,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine of principal angles")
        ax.set_title(f"Subspace Overlap -- {proj}")
        ax.set_xticks(layers)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Subspace Overlap Between M1 and M3 LoRA Column Spaces",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(REPORT_4_DIR / "subspace_overlap_maskfix.png"))
    plt.close(fig)
    logger.info("  Saved subspace_overlap_maskfix.png")

    # ── Plot 4: norm ratio ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for proj, marker in zip(PROJS, ["o", "s"]):
        ax.plot(
            layers, r4["ratio_maskfix"][proj], f"{marker}-",
            color=M3_MF_COLOR, label=f"maskfix {proj}",
            markersize=7, linewidth=2,
        )
        ax.plot(
            layers, r4["ratio_buggy"][proj], f"{marker}--",
            color=M3_BUGGY_COLOR, label=f"buggy {proj}",
            markersize=7, linewidth=2, alpha=0.7,
        )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="ratio = 1")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||M3 B@A||_F / ||M1 B@A||_F")
    ax.set_title(
        "Norm Ratio (M3 / M1): maskfix vs buggy",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(REPORT_4_DIR / "norm_ratio_maskfix.png"))
    plt.close(fig)
    logger.info("  Saved norm_ratio_maskfix.png")


# ===========================================================================
# Report 5: attention entropy plots
# ===========================================================================

def save_report5_data(
    m1_entropy: np.ndarray, m1_sinks: np.ndarray,
    m3mf_entropy: np.ndarray, m3mf_sinks: np.ndarray,
    m3bug_entropy: np.ndarray, m3bug_sinks: np.ndarray,
) -> None:
    os.makedirs(REPORT_5_DIR, exist_ok=True)
    path = REPORT_5_DIR / "maskfix_data.npz"
    np.savez(
        str(path),
        m1_entropy=m1_entropy, m1_sinks=m1_sinks,
        m3mf_entropy=m3mf_entropy, m3mf_sinks=m3mf_sinks,
        m3bug_entropy=m3bug_entropy, m3bug_sinks=m3bug_sinks,
    )
    logger.info("Saved %s", path)


def load_report5_data():
    path = REPORT_5_DIR / "maskfix_data.npz"
    d = np.load(str(path))
    return (
        d["m1_entropy"], d["m1_sinks"],
        d["m3mf_entropy"], d["m3mf_sinks"],
        d["m3bug_entropy"], d["m3bug_sinks"],
    )


def plot_report5(
    m1_entropy: np.ndarray, m1_sinks: np.ndarray,
    m3mf_entropy: np.ndarray, m3mf_sinks: np.ndarray,
    m3bug_entropy: np.ndarray, m3bug_sinks: np.ndarray,
) -> None:
    """Generate Report 5 plots with _maskfix suffix."""
    logger.info("=== Generating Report 5 plots ===")
    os.makedirs(REPORT_5_DIR, exist_ok=True)
    x = np.arange(NUM_LAYERS)

    # ── Plot 5.1: Entropy by layer (bar chart, 3-way) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, m1_entropy.mean(axis=1), w, label="M1 (full ctx)", color=M1_COLOR)
    ax.bar(x, m3mf_entropy.mean(axis=1), w, label="M3 maskfix", color=M3_MF_COLOR)
    ax.bar(x + w, m3bug_entropy.mean(axis=1), w, label="M3 buggy", color=M3_BUGGY_COLOR)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title("Attention Entropy by Layer (last query position)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)

    ax = axes[1]
    ax.bar(x - w, m1_sinks.mean(axis=1), w, label="M1", color=M1_COLOR)
    ax.bar(x, m3mf_sinks.mean(axis=1), w, label="M3 maskfix", color=M3_MF_COLOR)
    ax.bar(x + w, m3bug_sinks.mean(axis=1), w, label="M3 buggy", color=M3_BUGGY_COLOR)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Mass on First 5 Tokens")
    ax.set_title("Attention Sink Fraction by Layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)

    fig.suptitle(
        "Attention Patterns: M1 vs M3-maskfix vs M3-buggy",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(REPORT_5_DIR / "attention_entropy_maskfix.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved attention_entropy_maskfix.png")

    # ── Plot 5.2: Entropy heatmap (M1 vs M3-maskfix) ─────────────────────
    fig, (ax1, ax2, cax) = plt.subplots(
        1, 3, figsize=(15, 6),
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
    )
    vmin = min(m1_entropy.min(), m3mf_entropy.min())
    vmax = max(m1_entropy.max(), m3mf_entropy.max())

    ax1.imshow(m1_entropy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("M1 Attention Entropy")

    im = ax2.imshow(m3mf_entropy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_title("M3 maskfix Attention Entropy")

    fig.colorbar(im, cax=cax, label="Entropy (nats)")
    fig.tight_layout()
    fig.savefig(str(REPORT_5_DIR / "entropy_heatmap_maskfix.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved entropy_heatmap_maskfix.png")

    # ── Plot 5.3: Entropy diff (M3-maskfix minus M1) ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    diff = m3mf_entropy - m1_entropy
    max_abs_diff = np.abs(diff).max()
    entropy_scale = max(m1_entropy.max() - m1_entropy.min(), 1e-6)
    vbound = max(max_abs_diff * 1.2, entropy_scale * 0.1)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vbound, vmax=vbound)
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"Attention Entropy Difference (M3-maskfix \u2212 M1)\n"
        f"Max |diff| = {max_abs_diff:.1e} nats"
        + (" \u2014 identical within floating point" if max_abs_diff < 1e-6 else ""),
        fontsize=13,
    )
    fig.colorbar(im, label="Entropy diff (nats)")
    ax.set_xticks(range(0, diff.shape[1], 4))
    ax.set_yticks(range(diff.shape[0]))
    if max_abs_diff < 1e-6:
        ax.text(
            0.5, 0.5,
            "All differences = 0.0\n(models identical on full context)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white",
                alpha=0.8, edgecolor="#cccccc",
            ),
        )
    fig.tight_layout()
    fig.savefig(str(REPORT_5_DIR / "entropy_diff_maskfix.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved entropy_diff_maskfix.png")


# ===========================================================================
# Report 7: CKA plots
# ===========================================================================

def save_report7_data(
    layer_cka_maskfix: list[float], cross_cka_maskfix: np.ndarray,
    layer_cka_buggy: list[float], cross_cka_buggy: np.ndarray,
) -> None:
    os.makedirs(REPORT_7_DIR, exist_ok=True)
    path = REPORT_7_DIR / "maskfix_data.npz"
    np.savez(
        str(path),
        layer_cka_maskfix=np.array(layer_cka_maskfix),
        cross_cka_maskfix=cross_cka_maskfix,
        layer_cka_buggy=np.array(layer_cka_buggy),
        cross_cka_buggy=cross_cka_buggy,
    )
    logger.info("Saved %s", path)


def load_report7_data():
    path = REPORT_7_DIR / "maskfix_data.npz"
    d = np.load(str(path))
    return (
        d["layer_cka_maskfix"].tolist(),
        d["cross_cka_maskfix"],
        d["layer_cka_buggy"].tolist(),
        d["cross_cka_buggy"],
    )


def _plot_cka_bar(layer_cka_mf, layer_cka_bug, n_layers_total, out_path):
    """Bar chart of CKA by layer (maskfix and buggy side by side)."""
    labels = ["Emb"] + [str(i) for i in range(NUM_LAYERS)]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_layers_total)
    w = 0.35
    ax.bar(x - w / 2, layer_cka_mf, w, label="M1 vs M3-maskfix", color=M3_MF_COLOR)
    ax.bar(x + w / 2, layer_cka_bug, w, label="M1 vs M3-buggy", color=M3_BUGGY_COLOR, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title(
        "Representation Similarity (CKA) by Layer: maskfix vs buggy",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _plot_cka_heatmap(cross_cka, title_suffix, out_path):
    """Cross-layer CKA heatmap."""
    labels = ["Emb"] + [str(i) for i in range(NUM_LAYERS)]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cross_cka, cmap="viridis", aspect="equal", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"M3 {title_suffix} Layer")
    ax.set_ylabel("M1 Layer")
    ax.set_title(f"Cross-Layer CKA: M1 vs M3 {title_suffix}", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA")
    # Annotate values
    if n <= 17:
        for i in range(n):
            for j in range(n):
                val = cross_cka[i, j]
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], "r--", linewidth=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_report7(
    layer_cka_maskfix: list[float], cross_cka_maskfix: np.ndarray,
    layer_cka_buggy: list[float], cross_cka_buggy: np.ndarray,
) -> None:
    """Generate Report 7 plots with _maskfix suffix."""
    logger.info("=== Generating Report 7 plots ===")
    os.makedirs(REPORT_7_DIR, exist_ok=True)
    n_layers_total = NUM_LAYERS + 1  # embedding + 16 decoder layers

    _plot_cka_bar(
        layer_cka_maskfix, layer_cka_buggy, n_layers_total,
        REPORT_7_DIR / "cka_by_layer_maskfix.png",
    )
    logger.info("  Saved cka_by_layer_maskfix.png")

    _plot_cka_heatmap(
        cross_cka_maskfix, "maskfix",
        REPORT_7_DIR / "cka_heatmap_maskfix.png",
    )
    logger.info("  Saved cka_heatmap_maskfix.png")

    _plot_cka_heatmap(
        cross_cka_buggy, "buggy (reference)",
        REPORT_7_DIR / "cka_heatmap_buggy_ref.png",
    )
    logger.info("  Saved cka_heatmap_buggy_ref.png")


# ===========================================================================
# Compute CKA from hidden states
# ===========================================================================

def compute_cka_from_hidden(
    m1_hidden: list[list[torch.Tensor]],
    other_hidden: list[list[torch.Tensor]],
) -> tuple[list[float], np.ndarray]:
    """Compute layer-wise and cross-layer CKA between two sets of hidden states.

    Args:
        m1_hidden: list[sample][layer] of (hidden_dim,) tensors.
        other_hidden: same structure.

    Returns:
        (layer_cka, cross_cka) where layer_cka is a list of floats and
        cross_cka is a (n_layers+1, n_layers+1) numpy array.
    """
    n_layers_total = len(m1_hidden[0])
    n_samples = len(m1_hidden)

    # Layer-wise CKA
    layer_cka: list[float] = []
    for layer_idx in range(n_layers_total):
        m1_reps = torch.stack([m1_hidden[s][layer_idx] for s in range(n_samples)])
        other_reps = torch.stack([other_hidden[s][layer_idx] for s in range(n_samples)])
        cka_val = compute_cka(m1_reps, other_reps)
        layer_cka.append(cka_val)
        label = "Emb" if layer_idx == 0 else f"L{layer_idx - 1}"
        logger.info("  %s: CKA = %.4f", label, cka_val)

    # Cross-layer CKA heatmap
    logger.info("  Computing cross-layer CKA heatmap...")
    cross_cka = np.zeros((n_layers_total, n_layers_total))
    for i in range(n_layers_total):
        m1_i = torch.stack([m1_hidden[s][i] for s in range(n_samples)])
        for j in range(n_layers_total):
            other_j = torch.stack([other_hidden[s][j] for s in range(n_samples)])
            cross_cka[i, j] = compute_cka(m1_i, other_j)
        logger.info("    Row %d/%d done", i, n_layers_total)

    return layer_cka, cross_cka


# ===========================================================================
# Summary printing
# ===========================================================================

def print_summary(
    r4: dict,
    m1_entropy: np.ndarray, m3mf_entropy: np.ndarray, m3bug_entropy: np.ndarray,
    m1_sinks: np.ndarray, m3mf_sinks: np.ndarray, m3bug_sinks: np.ndarray,
    layer_cka_mf: list[float], layer_cka_bug: list[float],
) -> None:
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Report 4
    logger.info("--- Report 4: LoRA Weight Analysis ---")
    for proj in PROJS:
        r_mf = r4["ratio_maskfix"][proj]
        r_bug = r4["ratio_buggy"][proj]
        o_mf = r4["overlap_maskfix"][proj]
        o_bug = r4["overlap_buggy"][proj]
        logger.info(
            "%s maskfix: ratio mean=%.4f std=%.4f | overlap mean=%.4f std=%.4f",
            proj, r_mf.mean(), r_mf.std(), o_mf.mean(), o_mf.std(),
        )
        logger.info(
            "%s buggy:   ratio mean=%.4f std=%.4f | overlap mean=%.4f std=%.4f",
            proj, r_bug.mean(), r_bug.std(), o_bug.mean(), o_bug.std(),
        )

    # Report 5
    logger.info("--- Report 5: Attention Entropy ---")
    logger.info("  M1       mean entropy=%.4f  mean sink=%.4f",
                m1_entropy.mean(), m1_sinks.mean())
    logger.info("  M3-mfix  mean entropy=%.4f  mean sink=%.4f",
                m3mf_entropy.mean(), m3mf_sinks.mean())
    logger.info("  M3-buggy mean entropy=%.4f  mean sink=%.4f",
                m3bug_entropy.mean(), m3bug_sinks.mean())

    # Report 7
    logger.info("--- Report 7: CKA ---")
    logger.info("  maskfix: mean CKA=%.4f  min=%.4f (layer %d)  max=%.4f (layer %d)",
                np.mean(layer_cka_mf), min(layer_cka_mf),
                np.argmin(layer_cka_mf), max(layer_cka_mf), np.argmax(layer_cka_mf))
    logger.info("  buggy:   mean CKA=%.4f  min=%.4f (layer %d)  max=%.4f (layer %d)",
                np.mean(layer_cka_bug), min(layer_cka_bug),
                np.argmin(layer_cka_bug), max(layer_cka_bug), np.argmax(layer_cka_bug))


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_full_analysis() -> None:
    """Run the complete extraction + analysis pipeline."""
    # ── Validate checkpoints ──────────────────────────────────────────────
    for path, label in [
        (M1_CKPT, "M1"), (M3_MASKFIX_CKPT, "M3 maskfix"),
        (M3_BUGGY_CKPT, "M3 buggy"),
    ]:
        if not path.exists():
            logger.error("Checkpoint not found: %s (%s)", path, label)
            sys.exit(1)

    # ==================================================================
    # Report 4: LoRA weight analysis (CPU only, no model loading needed)
    # ==================================================================
    r4 = compute_report4_data()
    save_report4_data(r4)
    plot_report4(r4)

    # ==================================================================
    # GPU analyses: load models one at a time
    # ==================================================================
    logger.info("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("=== Loading test prompts ===")
    prompts = get_test_prompts(tokenizer)
    logger.info("  %d prompts loaded", len(prompts))

    # ── M1 ────────────────────────────────────────────────────────────
    logger.info("=== Loading M1 model ===")
    m1_model = load_model_with_lora(M1_CKPT)
    logger.info("  Extracting M1 attention + hidden states...")
    m1_entropy, m1_sinks, m1_hidden = extract_attention_and_hidden(m1_model, prompts)
    del m1_model
    torch.cuda.empty_cache()
    logger.info("  GPU mem after M1 cleanup: %.2f GB",
                torch.cuda.memory_allocated() / 1e9)

    # ── M3 maskfix ────────────────────────────────────────────────────
    logger.info("=== Loading M3 maskfix model ===")
    m3mf_model = load_model_with_lora(M3_MASKFIX_CKPT)
    logger.info("  Extracting M3-maskfix attention + hidden states...")
    m3mf_entropy, m3mf_sinks, m3mf_hidden = extract_attention_and_hidden(
        m3mf_model, prompts,
    )
    del m3mf_model
    torch.cuda.empty_cache()
    logger.info("  GPU mem after M3-maskfix cleanup: %.2f GB",
                torch.cuda.memory_allocated() / 1e9)

    # ── M3 buggy ──────────────────────────────────────────────────────
    logger.info("=== Loading M3 buggy model ===")
    m3bug_model = load_model_with_lora(M3_BUGGY_CKPT)
    logger.info("  Extracting M3-buggy attention + hidden states...")
    m3bug_entropy, m3bug_sinks, m3bug_hidden = extract_attention_and_hidden(
        m3bug_model, prompts,
    )
    del m3bug_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Save Report 5 data
    # ==================================================================
    save_report5_data(
        m1_entropy, m1_sinks,
        m3mf_entropy, m3mf_sinks,
        m3bug_entropy, m3bug_sinks,
    )
    plot_report5(
        m1_entropy, m1_sinks,
        m3mf_entropy, m3mf_sinks,
        m3bug_entropy, m3bug_sinks,
    )

    # ==================================================================
    # Report 7: CKA computation
    # ==================================================================
    logger.info("=== Computing CKA: M1 vs M3-maskfix ===")
    layer_cka_mf, cross_cka_mf = compute_cka_from_hidden(m1_hidden, m3mf_hidden)

    logger.info("=== Computing CKA: M1 vs M3-buggy ===")
    layer_cka_bug, cross_cka_bug = compute_cka_from_hidden(m1_hidden, m3bug_hidden)

    save_report7_data(layer_cka_mf, cross_cka_mf, layer_cka_bug, cross_cka_bug)
    plot_report7(layer_cka_mf, cross_cka_mf, layer_cka_bug, cross_cka_bug)

    # ==================================================================
    # Summary
    # ==================================================================
    print_summary(
        r4,
        m1_entropy, m3mf_entropy, m3bug_entropy,
        m1_sinks, m3mf_sinks, m3bug_sinks,
        layer_cka_mf, layer_cka_bug,
    )

    logger.info("=== All maskfix GPU analyses complete ===")


def run_plot_only() -> None:
    """Regenerate plots from saved .npz data (no GPU needed)."""
    logger.info("=== Plot-only mode: loading saved data ===")

    # Report 4
    r4 = load_report4_data()
    plot_report4(r4)

    # Report 5
    (m1_entropy, m1_sinks,
     m3mf_entropy, m3mf_sinks,
     m3bug_entropy, m3bug_sinks) = load_report5_data()
    plot_report5(
        m1_entropy, m1_sinks,
        m3mf_entropy, m3mf_sinks,
        m3bug_entropy, m3bug_sinks,
    )

    # Report 7
    (layer_cka_mf, cross_cka_mf,
     layer_cka_bug, cross_cka_bug) = load_report7_data()
    plot_report7(layer_cka_mf, cross_cka_mf, layer_cka_bug, cross_cka_bug)

    # Summary
    print_summary(
        r4,
        m1_entropy, m3mf_entropy, m3bug_entropy,
        m1_sinks, m3mf_sinks, m3bug_sinks,
        layer_cka_mf, layer_cka_bug,
    )

    logger.info("=== Plot-only mode complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maskfix GPU analyses (Reports 4, 5, 7)",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip inference; regenerate plots from saved .npz data",
    )
    args = parser.parse_args()

    if args.plot_only:
        run_plot_only()
    else:
        run_full_analysis()


if __name__ == "__main__":
    main()
