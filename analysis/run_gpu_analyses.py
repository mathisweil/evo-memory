"""Run GPU-requiring analyses (5: attention entropy, 7: CKA).

Loads M1 and M3 cs1024 models, runs inference on a sample of test prompts,
and extracts attention weights and hidden states for analysis.

Saves intermediate data to analysis/report_5/ and analysis/report_7/,
then generates plots.

Usage:
    source activate.sh
    PYTHONPATH=. HF_HOME=.hf_cache python analysis/run_gpu_analyses.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.hydra_helpers import LlamaCompatModel

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
M1_CKPT = "experiment_artifacts/gcs/M1/best_ckpt.pt"
M3_CKPT = "experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt"
N_SAMPLES = 10  # test samples to process
MAX_LENGTH = 1024  # truncate prompts to fit GPU memory with attention output

REPORT_5_DIR = "analysis/report_5"
REPORT_7_DIR = "analysis/report_7"


def load_model_with_lora(ckpt_path, device="cuda"):
    """Load base model + apply LoRA, merge into base weights to save memory."""
    model = LlamaCompatModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lora_config_dict = ckpt["lora_config"]
    lora_sd = ckpt["lora_state_dict"]

    config = LoraConfig(
        r=lora_config_dict.get("r", 8),
        lora_alpha=lora_config_dict.get("lora_alpha", 16),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=0.0,  # no dropout at inference
    )
    model = PeftModel(model, config)

    # Remap checkpoint keys: checkpoint uses "base_model.model.layers.X..."
    # but PeftModel wrapping LlamaCompatModel expects "base_model.model.model.layers.X..."
    remapped_sd = {}
    for k, v in lora_sd.items():
        new_key = k.replace("base_model.model.layers.", "base_model.model.model.layers.")
        remapped_sd[new_key] = v

    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        raise RuntimeError(f"LoRA weights not loaded! Missing keys: {lora_missing[:5]}")
    print(f"  LoRA weights loaded: {len(remapped_sd)} tensors, {len(lora_missing)} lora missing")

    model = model.merge_and_unload()  # merge LoRA into base weights, free adapter memory
    model = model.to(device)
    model.eval()
    print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


def get_test_prompts(tokenizer, n=N_SAMPLES):
    """Load a sample of test prompts from LongBench."""
    from datasets import load_dataset
    prompts = []
    for task in ["qasper", "2wikimqa", "hotpotqa_e"]:
        ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        for i, sample in enumerate(ds):
            if len(prompts) >= n:
                break
            text = sample["context"][:8000] + "\n\nQuestion: " + sample["input"] + "\nAnswer:"
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            prompts.append({"input_ids": ids["input_ids"], "task": task, "idx": i})
        if len(prompts) >= n:
            break
    return prompts[:n]


@torch.no_grad()
def extract_attention_and_hidden(model, prompts, device="cuda"):
    """Run inference, compute entropy stats inline, store compressed hidden states."""
    n_layers = 16
    # Accumulators for entropy stats
    entropy_sum = None
    sink_sum = None
    # Hidden states: store mean-pooled per layer per sample for CKA
    all_hidden = []  # list of (n_layers+1, hidden_dim) mean-pooled vectors
    count = 0

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        out = model(input_ids, output_attentions=True, output_hidden_states=True)

        # Compute entropy inline (don't store full attention matrices)
        for layer_idx, layer_attn in enumerate(out.attentions):
            attn = layer_attn[0].float()  # (n_heads, seq, seq)
            n_heads = attn.shape[0]
            if entropy_sum is None:
                entropy_sum = torch.zeros(n_layers, n_heads)
                sink_sum = torch.zeros(n_layers, n_heads)

            attn_clamped = attn.clamp(min=1e-10)
            h = -(attn_clamped * attn_clamped.log()).sum(dim=-1).mean(dim=-1)  # (n_heads,)
            entropy_sum[layer_idx] += h.cpu()

            sink = attn[:, :, :5].sum(dim=-1).mean(dim=-1)  # (n_heads,)
            sink_sum[layer_idx] += sink.cpu()

        # Store hidden states (mean-pooled to save memory)
        hidden_pooled = []
        for h in out.hidden_states:
            hidden_pooled.append(h[0].float().mean(dim=0).cpu())  # (hidden_dim,)
        all_hidden.append(hidden_pooled)

        count += 1
        seq_len = input_ids.shape[1]
        print(f"  Sample {i+1}/{len(prompts)} done (seq_len={seq_len})")

        del out
        torch.cuda.empty_cache()

    entropy_avg = entropy_sum / count
    sink_avg = sink_sum / count
    return entropy_avg.numpy(), sink_avg.numpy(), all_hidden


def compute_cka(X, Y):
    """Linear CKA between two representation matrices.
    X: (n, d1), Y: (n, d2)
    """
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


def run_analysis():
    print("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n=== Loading test prompts ===")
    prompts = get_test_prompts(tokenizer)
    print(f"  {len(prompts)} prompts loaded")

    # --- M1 ---
    print("\n=== Loading M1 model ===")
    m1_model = load_model_with_lora(M1_CKPT)
    print("  Extracting M1 attention + hidden states...")
    m1_entropy, m1_sinks, m1_hidden = extract_attention_and_hidden(m1_model, prompts)
    del m1_model
    torch.cuda.empty_cache()
    print(f"  GPU mem after M1 cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- M3 ---
    print("\n=== Loading M3 cs1024 model ===")
    m3_model = load_model_with_lora(M3_CKPT)
    print("  Extracting M3 attention + hidden states...")
    m3_entropy, m3_sinks, m3_hidden = extract_attention_and_hidden(m3_model, prompts)
    del m3_model
    torch.cuda.empty_cache()

    # === Analysis 5: Attention Entropy ===
    print("\n=== Generating attention plots (Report 5) ===")

    # Save data
    os.makedirs(REPORT_5_DIR, exist_ok=True)
    np.savez(f"{REPORT_5_DIR}/attention_data.npz",
             m1_entropy=m1_entropy, m3_entropy=m3_entropy,
             m1_sinks=m1_sinks, m3_sinks=m3_sinks)
    print("  Saved attention_data.npz")

    # Plot 5.1: Entropy by layer (averaged over heads)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(16)
    ax.bar(x - 0.2, m1_entropy.mean(axis=1), 0.4, label="M1 (full context)", color="#d62728")
    ax.bar(x + 0.2, m3_entropy.mean(axis=1), 0.4, label="M3 cs1024 (evicted)", color="#1f77b4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title("Attention Entropy by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - 0.2, m1_sinks.mean(axis=1), 0.4, label="M1", color="#d62728")
    ax.bar(x + 0.2, m3_sinks.mean(axis=1), 0.4, label="M3 cs1024", color="#1f77b4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Mass on First 5 Tokens")
    ax.set_title("Attention Sink Fraction by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{REPORT_5_DIR}/attention_entropy.png", dpi=150)
    plt.close(fig)
    print("  Saved attention_entropy.png")

    # Plot 5.2: Entropy heatmap (layer x head)
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(15, 6),
                                         gridspec_kw={"width_ratios": [1, 1, 0.05]})
    vmin = min(m1_entropy.min(), m3_entropy.min())
    vmax = max(m1_entropy.max(), m3_entropy.max())

    ax1.imshow(m1_entropy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("M1 Attention Entropy")

    im = ax2.imshow(m3_entropy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_title("M3 cs1024 Attention Entropy")

    fig.colorbar(im, cax=cax, label="Entropy (nats)")
    fig.tight_layout()
    fig.savefig(f"{REPORT_5_DIR}/entropy_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved entropy_heatmap.png")

    # Plot 5.3: Entropy difference
    fig, ax = plt.subplots(figsize=(10, 6))
    diff = m3_entropy - m1_entropy
    max_abs_diff = np.abs(diff).max()
    # Use entropy range as scale when diff is near-zero
    entropy_scale = max(m1_entropy.max() - m1_entropy.min(), 1e-6)
    vbound = max(max_abs_diff * 1.2, entropy_scale * 0.1)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vbound, vmax=vbound)
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"Attention Entropy Difference (M3 \u2212 M1)\n"
                 f"Max |diff| = {max_abs_diff:.1e} nats"
                 + (" \u2014 identical within floating point" if max_abs_diff < 1e-6 else ""),
                 fontsize=13)
    fig.colorbar(im, label="Entropy diff (nats)")
    ax.set_xticks(range(0, diff.shape[1], 4))
    ax.set_yticks(range(diff.shape[0]))
    if max_abs_diff < 1e-6:
        ax.text(0.5, 0.5, "All differences = 0.0\n(models identical on full context)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontweight="bold", color="#555555",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          alpha=0.8, edgecolor="#cccccc"))
    fig.tight_layout()
    fig.savefig(f"{REPORT_5_DIR}/entropy_diff.png", dpi=150)
    plt.close(fig)
    print("  Saved entropy_diff.png")

    # === Analysis 7: CKA ===
    print("\n=== Computing CKA (Report 7) ===")
    n_layers = len(m1_hidden[0])  # 17 (embedding + 16 layers)

    # Layer-wise CKA using mean-pooled hidden states
    # Each m1_hidden[s][layer] is a (hidden_dim,) vector
    layer_cka = []
    for layer_idx in range(n_layers):
        m1_reps = torch.stack([m1_hidden[s][layer_idx] for s in range(len(m1_hidden))])  # (n_samples, hidden_dim)
        m3_reps = torch.stack([m3_hidden[s][layer_idx] for s in range(len(m3_hidden))])
        cka_val = compute_cka(m1_reps, m3_reps)
        layer_cka.append(cka_val)
        print(f"  Layer {layer_idx}: CKA = {cka_val:.4f}")

    # Cross-layer CKA heatmap
    print("  Computing cross-layer CKA...")
    cross_cka = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        m1_i = torch.stack([m1_hidden[s][i] for s in range(len(m1_hidden))])
        for j in range(n_layers):
            m3_j = torch.stack([m3_hidden[s][j] for s in range(len(m3_hidden))])
            cross_cka[i, j] = compute_cka(m1_i, m3_j)
        print(f"  Row {i}/{n_layers} done")

    # Save data
    os.makedirs(REPORT_7_DIR, exist_ok=True)
    np.savez(f"{REPORT_7_DIR}/cka_data.npz",
             layer_cka=np.array(layer_cka), cross_cka=cross_cka)
    print("  Saved cka_data.npz")

    # Plot 7.1: CKA by layer
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_labels = ["Emb"] + [str(i) for i in range(16)]
    ax.bar(range(n_layers), layer_cka, color="#1f77b4")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("M1 vs M3 cs1024 — Representation Similarity (CKA) by Layer")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Identical")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{REPORT_7_DIR}/cka_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved cka_by_layer.png")

    # Plot 7.2: Cross-layer CKA heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cross_cka, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("M3 Layer")
    ax.set_ylabel("M1 Layer")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_title("Cross-Layer CKA: M1 Layer i vs M3 Layer j")
    fig.colorbar(im, label="CKA")
    fig.tight_layout()
    fig.savefig(f"{REPORT_7_DIR}/cka_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved cka_heatmap.png")

    print("\n=== All GPU analyses complete ===")

    # Print summary for reports
    print("\n--- Report 5 Summary ---")
    print(f"M1 mean entropy: {m1_entropy.mean():.4f}")
    print(f"M3 mean entropy: {m3_entropy.mean():.4f}")
    print(f"M1 mean sink fraction: {m1_sinks.mean():.4f}")
    print(f"M3 mean sink fraction: {m3_sinks.mean():.4f}")
    print(f"Layers where M3 entropy < M1 (sharper): {(m3_entropy.mean(1) < m1_entropy.mean(1)).sum()}/16")

    print("\n--- Report 7 Summary ---")
    print(f"Mean CKA (all layers): {np.mean(layer_cka):.4f}")
    print(f"CKA embedding: {layer_cka[0]:.4f}")
    print(f"CKA layer 0: {layer_cka[1]:.4f}")
    print(f"CKA layer 15: {layer_cka[16]:.4f}")
    print(f"Min CKA: {min(layer_cka):.4f} at layer {np.argmin(layer_cka)}")
    print(f"Max CKA: {max(layer_cka):.4f} at layer {np.argmax(layer_cka)}")


if __name__ == "__main__":
    run_analysis()
