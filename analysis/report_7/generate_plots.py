"""Analysis 7 -- Representation Similarity (CKA).

Compares internal representations of M1 (LoRA-only, full context) and
M3 cs1024 (LoRA + frozen NAMM) by computing linear Centered Kernel Alignment
(CKA) between hidden states at each of the 16 LLaMA 3.2-1B layers.

Produces three plots:
  cka_by_layer.png     -- Bar chart of linear CKA at each layer
  cka_heatmap.png      -- Cross-layer CKA heatmap (M1 layer i vs M3 layer j)
  cka_vs_retention.png -- CKA vs layer retention ratio (from report 3)

IMPORTANT: The extraction phase (default mode) requires a GPU to load models
and run inference.  Use --plot-only to regenerate plots from saved data on a
CPU-only node.

Usage:
    # Full pipeline (GPU required):
    python analysis/report_7/generate_plots.py

    # Plot-only from saved data (CPU ok):
    python analysis/report_7/generate_plots.py --plot-only
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR = SCRIPT_DIR
DATA_FILE = os.path.join(OUT_DIR, "cka_data.json")

# Paths to downloaded checkpoints (via scripts/download_artifacts.py)
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "experiment_artifacts", "gcs")
M1_CKPT = os.path.join(ARTIFACTS_DIR, "M1", "best_ckpt.pt")
M3_CKPT = os.path.join(ARTIFACTS_DIR, "M3_cs1024", "best_ckpt.pt")
NAMM_CKPT = os.path.join(ARTIFACTS_DIR, "M2_cs1024", "ckpt.pt")

# Model constants (LLaMA 3.2-1B-Instruct)
NUM_LAYERS = 16

# Data split parameters (must match training)
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
SPLIT_SEED = 42

# Approximate per-layer retention ratios from M2 cs1024 analysis.
# These are placeholders -- replace with actual data from report_3 if available.
# Layer 0 has highest retention (early layers less evicted), later layers lower.
DEFAULT_RETENTION_RATIOS = None  # Will try to load from report_3


# ── Linear CKA ──────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    Args:
        X: np.ndarray of shape (n_samples, d1)
        Y: np.ndarray of shape (n_samples, d2)

    Returns:
        float: CKA similarity in [0, 1]. 1 = identical (up to linear transform),
               0 = completely unrelated.

    Reference: Kornblith et al. (2019), "Similarity of Neural Network
    Representations Revisited", ICML 2019. Eq. (3) -- HSIC formulation
    with linear kernel.
    """
    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute cross-covariance and self-covariance Frobenius norms
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YTX = Y.T @ X
    XTX = X.T @ X
    YTY = Y.T @ Y

    numerator = np.sum(YTX ** 2)
    denominator = np.sqrt(np.sum(XTX ** 2) * np.sum(YTY ** 2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


# ── Model loading helpers ────────────────────────────────────────────────────

def _add_repo_to_path():
    """Ensure repo root and scripts/ are on sys.path for imports."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    scripts_dir = os.path.join(REPO_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def load_model_for_hidden_states(ckpt_path, namm_ckpt_path=None,
                                  namm_active=False, cache_size=None):
    """Load a LoRA-fine-tuned model and return (memory_model, tokenizer, cfg).

    Follows the patterns in scripts/run_lora.py and scripts/run_eval.py.

    Args:
        ckpt_path: Path to LoRA checkpoint (best_ckpt.pt).
        namm_ckpt_path: Path to NAMM checkpoint (for M3 only).
        namm_active: Whether NAMM eviction should be active during inference.
        cache_size: KV cache size override (for NAMM).

    Returns:
        (memory_model, tokenizer, cfg) tuple.
    """
    import torch
    from hydra import compose, initialize_config_dir
    from namm.run_utils import make_eval_model, make_task_sampler

    _add_repo_to_path()
    from experiment_utils import load_hydra_config

    # Build Hydra config -- use 5-task config to match training
    run_config = "namm_bam_i1_llama32_1b_5t"
    extra_overrides = []
    if cache_size is not None:
        extra_overrides.append(f"cache_size={cache_size}")
        extra_overrides.append(f"max_memory_length={cache_size}")

    cfg = load_hydra_config(run_config, extra_overrides=extra_overrides)

    # Build model
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    tokenizer = memory_evaluator.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Swap to recency policy if NAMM is not active (M1)
    if not namm_active:
        from namm.policy.base import Recency
        recency_policy = Recency(cache_size=cache_size)
        memory_evaluator.swap_memory_policy(recency_policy)
        memory_policy = recency_policy
        memory_evaluator.max_memory_length = memory_evaluator.max_conditioning_length
        if memory_evaluator.batch_size == "auto":
            memory_evaluator.batch_size = 1

    # Load NAMM checkpoint (M3 only)
    if namm_ckpt_path is not None and namm_active:
        print(f"Loading NAMM checkpoint: {namm_ckpt_path}")
        ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
        evo_state = ckpt["evolution_state"]

        # Prefer mean (matches training eval behaviour)
        if "mean" in evo_state:
            params_vec = evo_state["mean"]
        else:
            params_vec = evo_state["best_member"]
        params = params_vec.unsqueeze(0).to(device)
        memory_model.set_memory_params(params)

        buffers_prefix = "stored_buffers_to_save."
        buffers_dict = {
            k[len(buffers_prefix):]: v.to(device)
            for k, v in evo_state.items()
            if k.startswith(buffers_prefix)
        }
        if buffers_dict:
            memory_model.load_buffers_dict(buffers_dict=buffers_dict)

        batch_idxs = np.zeros([1])
        memory_policy.set_params_batch_idxs(batch_idxs)
        print(f"  NAMM loaded ({params_vec.shape[0]} params)")

    # Move model to device and cast to bfloat16
    memory_model.to(dtype=torch.bfloat16, device=device)

    # Apply LoRA adapters (rank=8, q_proj + v_proj -- matches training config)
    memory_model.apply_lora_adapters(
        rank=8,
        target_modules=["q_proj", "v_proj"],
    )

    # Load LoRA weights from checkpoint
    print(f"Loading LoRA checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "lora_state_dict" in ckpt:
        loaded = 0
        for n, p in memory_model.model.named_parameters():
            if p.requires_grad and n in ckpt["lora_state_dict"]:
                p.data.copy_(ckpt["lora_state_dict"][n].to(p.device))
                loaded += 1
        print(f"  Loaded {loaded} LoRA tensors")
    else:
        print("  WARNING: no lora_state_dict in checkpoint")

    memory_model.eval()
    return memory_model, tokenizer, cfg


def get_test_inputs(tokenizer, cfg):
    """Get tokenised test prompts using the same split as training.

    Returns:
        List of dicts, each with 'input_ids' (torch.LongTensor) and 'prompt' (str).
    """
    import torch
    from namm.run_utils import make_task_sampler

    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_FRAC, split_seed=SPLIT_SEED)

    # Apply token-based filtering to match training
    max_cond = cfg.get("max_conditioning_length", 6500)
    min_cond = cfg.get("min_conditioning_length", None)

    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        max_conditioning_length=max_cond,
        min_conditioning_length=min_cond,
        tokenizer=tokenizer,
    )

    # Filter answers by token count
    task_sampler.filter_answers_by_token_count(tokenizer, 64)

    test_indices = task_sampler.get_split_indices("test")

    test_inputs = []
    for task_name, indices in test_indices.items():
        prompts = task_sampler.lb_prompts_per_task[task_name]
        for idx in indices:
            prompt = prompts[idx]
            # Apply chat template (matches SFT training format)
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
            test_inputs.append({
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "task": task_name,
                "idx": int(idx),
            })

    print(f"Collected {len(test_inputs)} test samples across "
          f"{len(test_indices)} tasks")
    return test_inputs


def extract_hidden_states(model, test_inputs, device, max_seq_len=6500):
    """Run inference and extract hidden states at all layers.

    Args:
        model: WrappedLlamaForCausalLM with LoRA loaded.
        test_inputs: List of dicts with 'input_ids'.
        device: torch device.
        max_seq_len: Maximum sequence length to process (truncate if longer).

    Returns:
        List of np.ndarray, one per layer, each of shape (n_samples, hidden_dim).
        We take the mean-pooled hidden state across the sequence for each sample.
    """
    import torch

    n_layers = NUM_LAYERS
    # +1 because output_hidden_states returns embedding layer + all decoder layers
    all_hidden = [[] for _ in range(n_layers + 1)]

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(test_inputs):
            input_ids = item["input_ids"].unsqueeze(0).to(device)

            # Truncate if too long
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, :max_seq_len]

            # Reset memory policy state for each sample
            model.memory_policy.reset()

            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )

            # outputs.hidden_states is a tuple of (n_layers + 1) tensors
            # Each tensor has shape (batch=1, seq_len, hidden_dim)
            # Mean-pool across sequence dimension for a fixed-size representation
            for layer_idx, hs in enumerate(outputs.hidden_states):
                pooled = hs.float().mean(dim=1).cpu().numpy()  # (1, hidden_dim)
                all_hidden[layer_idx].append(pooled[0])

            if (i + 1) % 10 == 0 or (i + 1) == len(test_inputs):
                print(f"  Processed {i + 1}/{len(test_inputs)} samples")

            # Free GPU memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Stack into (n_samples, hidden_dim) arrays per layer
    hidden_arrays = [np.stack(layer_list, axis=0) for layer_list in all_hidden]
    return hidden_arrays


# ── Extraction pipeline ──────────────────────────────────────────────────────

def run_extraction():
    """Load models, extract hidden states, compute CKA, save to JSON."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Extraction will be very slow or fail.")
        print("Use --plot-only to generate plots from pre-computed data.")

    # Verify checkpoints exist
    for path, label in [(M1_CKPT, "M1"), (M3_CKPT, "M3 cs1024"),
                        (NAMM_CKPT, "M2 cs1024 (NAMM)")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} checkpoint not found at {path}")
            print("Run: python scripts/download_artifacts.py")
            sys.exit(1)

    # ── Load M1 model ──
    print("\n=== Loading M1 (LoRA-only, full context) ===")
    m1_model, tokenizer, cfg = load_model_for_hidden_states(
        M1_CKPT, namm_active=False, cache_size=None)

    # ── Get test inputs ──
    print("\n=== Preparing test inputs ===")
    test_inputs = get_test_inputs(tokenizer, cfg)

    # ── Extract M1 hidden states ──
    print(f"\n=== Extracting M1 hidden states ({len(test_inputs)} samples) ===")
    m1_hidden = extract_hidden_states(m1_model, test_inputs, device)
    print(f"  M1: {len(m1_hidden)} layer outputs, "
          f"shape per layer: {m1_hidden[0].shape}")

    # Free M1 model memory
    del m1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load M3 model ──
    print("\n=== Loading M3 (LoRA + frozen NAMM, cs1024) ===")
    m3_model, _, _ = load_model_for_hidden_states(
        M3_CKPT, namm_ckpt_path=NAMM_CKPT,
        namm_active=True, cache_size=1024)

    # ── Extract M3 hidden states ──
    print(f"\n=== Extracting M3 hidden states ({len(test_inputs)} samples) ===")
    m3_hidden = extract_hidden_states(m3_model, test_inputs, device)
    print(f"  M3: {len(m3_hidden)} layer outputs, "
          f"shape per layer: {m3_hidden[0].shape}")

    del m3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute layer-wise CKA ──
    print("\n=== Computing layer-wise CKA ===")
    # Layer indices 0..16: index 0 is the embedding output, 1..16 are decoder layers
    layer_cka = []
    for layer_idx in range(NUM_LAYERS + 1):
        cka_val = linear_cka(m1_hidden[layer_idx], m3_hidden[layer_idx])
        layer_label = "emb" if layer_idx == 0 else f"L{layer_idx - 1}"
        layer_cka.append(cka_val)
        print(f"  {layer_label}: CKA = {cka_val:.4f}")

    # ── Compute cross-layer CKA heatmap ──
    print("\n=== Computing cross-layer CKA heatmap ===")
    n_total = NUM_LAYERS + 1  # embedding + 16 decoder layers
    cross_cka = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            cross_cka[i, j] = linear_cka(m1_hidden[i], m3_hidden[j])
    print(f"  Heatmap shape: {cross_cka.shape}")

    # ── Save results ──
    results = {
        "layer_cka": layer_cka,
        "cross_cka": cross_cka.tolist(),
        "n_samples": m1_hidden[0].shape[0],
        "n_layers": NUM_LAYERS,
        "m1_checkpoint": M1_CKPT,
        "m3_checkpoint": M3_CKPT,
        "namm_checkpoint": NAMM_CKPT,
        "layer_labels": ["emb"] + [f"L{i}" for i in range(NUM_LAYERS)],
    }

    with open(DATA_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CKA data to {DATA_FILE}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def load_retention_ratios():
    """Try to load per-layer retention ratios from report_3 data."""
    report3_dir = os.path.join(SCRIPT_DIR, "..", "report_3")
    # Try common data file names
    for fname in ["retention_data.json", "data.json", "layer_retention.json"]:
        path = os.path.join(report3_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if "retention_ratios" in data:
                return data["retention_ratios"]
            if "layer_retention" in data:
                return data["layer_retention"]

    # If no report_3 data, use a synthetic proxy:
    # Early layers retain more, later layers retain less (illustrative only).
    print("  NOTE: No report_3 retention data found. Using synthetic proxy.")
    return None


def plot_cka_by_layer(data, out_dir):
    """Bar chart of linear CKA at each layer."""
    layer_cka = data["layer_cka"]
    labels = data["layer_labels"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.RdYlGn(np.array(layer_cka))
    bars = ax.bar(range(n), layer_cka, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=45)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Linear CKA", fontsize=12)
    ax.set_title("Representation Similarity: M1 vs M3 (cs1024)\n"
                 "Linear CKA per Layer", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="grey", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation for expected pattern
    ax.annotate(f"n = {data['n_samples']} test samples",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.8))

    fig.tight_layout()
    path = os.path.join(out_dir, "cka_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cka_heatmap(data, out_dir):
    """Cross-layer CKA heatmap (M1 layer i vs M3 layer j)."""
    cross_cka = np.array(data["cross_cka"])
    labels = data["layer_labels"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(cross_cka, cmap="viridis", vmin=0, vmax=1,
                   aspect="equal", origin="lower")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("M3 (LoRA + NAMM cs1024) Layer", fontsize=12)
    ax.set_ylabel("M1 (LoRA-only, full context) Layer", fontsize=12)
    ax.set_title("Cross-Layer CKA Heatmap: M1 vs M3\n"
                 "Off-diagonal peaks suggest computational shifts", fontsize=13)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA", fontsize=11)

    # Annotate values on the heatmap (only if small enough to read)
    if n <= 17:
        for i in range(n):
            for j in range(n):
                val = cross_cka[i, j]
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    # Draw diagonal line
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], "r--", linewidth=1, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "cka_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cka_vs_retention(data, out_dir):
    """Scatter plot of CKA vs layer retention ratio."""
    retention = load_retention_ratios()

    # Skip embedding layer for this analysis (retention only applies to decoder layers)
    decoder_cka = data["layer_cka"][1:]  # L0..L15
    decoder_labels = data["layer_labels"][1:]

    if retention is None:
        # Generate a synthetic retention proxy for illustration
        # More aggressive eviction at later layers -> lower retention
        retention = np.linspace(0.95, 0.55, NUM_LAYERS).tolist()
        retention_source = "synthetic proxy (no report_3 data)"
    else:
        # Ensure we have the right number of values
        if len(retention) > NUM_LAYERS:
            retention = retention[:NUM_LAYERS]
        elif len(retention) < NUM_LAYERS:
            print(f"  WARNING: retention has {len(retention)} values, "
                  f"expected {NUM_LAYERS}. Padding with NaN.")
            retention = retention + [np.nan] * (NUM_LAYERS - len(retention))
        retention_source = "report_3 data"

    retention = np.array(retention)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color points by layer index
    colors = plt.cm.coolwarm(np.linspace(0, 1, NUM_LAYERS))
    for i in range(NUM_LAYERS):
        if not np.isnan(retention[i]):
            ax.scatter(retention[i], decoder_cka[i], c=[colors[i]], s=80,
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(decoder_labels[i], (retention[i], decoder_cka[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, ha="left")

    # Fit a trend line if we have enough points
    valid = ~np.isnan(retention)
    if valid.sum() >= 3:
        z = np.polyfit(retention[valid], np.array(decoder_cka)[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(retention[valid].min(), retention[valid].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=1.5,
                label=f"Linear fit (slope={z[0]:.2f})")

        # Compute correlation
        corr = np.corrcoef(retention[valid],
                           np.array(decoder_cka)[valid])[0, 1]
        ax.annotate(f"r = {corr:.3f}", xy=(0.02, 0.02),
                    xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              alpha=0.8))

    ax.set_xlabel("Retention Ratio (fraction of tokens kept)", fontsize=12)
    ax.set_ylabel("Linear CKA (M1 vs M3)", fontsize=12)
    ax.set_title("CKA vs Layer Retention Ratio\n"
                 f"({retention_source})", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if valid.sum() >= 3:
        ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "cka_vs_retention.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def generate_plots(data):
    """Generate all plots from loaded CKA data."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n=== Generating Plots ===")
    plot_cka_by_layer(data, OUT_DIR)
    plot_cka_heatmap(data, OUT_DIR)
    plot_cka_vs_retention(data, OUT_DIR)
    print(f"\nAll plots saved to: {OUT_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 7: Representation Similarity (CKA)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, generate plots from saved data")
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(DATA_FILE):
            print(f"ERROR: No saved data at {DATA_FILE}")
            print("Run without --plot-only on a GPU node first.")
            sys.exit(1)
        print(f"Loading saved CKA data from {DATA_FILE}")
        with open(DATA_FILE) as f:
            data = json.load(f)
        print(f"  {data['n_samples']} samples, {data['n_layers']} layers")
    else:
        data = run_extraction()

    generate_plots(data)

    # Print summary
    print("\n=== Summary ===")
    layer_cka = data["layer_cka"]
    labels = data["layer_labels"]
    print(f"{'Layer':<8} {'CKA':>8}")
    print("-" * 18)
    for label, cka in zip(labels, layer_cka):
        print(f"{label:<8} {cka:>8.4f}")

    diag_cka = [data["cross_cka"][i][i] for i in range(len(labels))]
    print(f"\nDiagonal mean CKA:   {np.mean(diag_cka):.4f}")
    print(f"Embedding CKA:       {layer_cka[0]:.4f}")
    print(f"First decoder CKA:   {layer_cka[1]:.4f}")
    print(f"Last decoder CKA:    {layer_cka[-1]:.4f}")


if __name__ == "__main__":
    main()
