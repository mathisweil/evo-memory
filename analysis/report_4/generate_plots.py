"""
Analysis 4: LoRA Weight Comparison (M1 vs M3 cs1024)

Compare learned LoRA adapter weights between:
  - M1: full context, no eviction
  - M3 cs1024: frozen NAMM active during training

Produces:
  1. weight_magnitude.png  -- per-layer ||B@A||_F for q_proj and v_proj
  2. singular_values.png   -- SVD spectra of B@A (q_proj) per layer
  3. subspace_overlap.png  -- cosine of principal angles between M1/M3 subspaces
  4. norm_ratio.png        -- ||M3||_F / ||M1||_F ratio per layer
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = ROOT

M1_PATH = "/cs/student/project_msc/2025/csml/sruppage/evo-memory/experiment_artifacts/gcs/M1/best_ckpt.pt"
M3_PATH = "/cs/student/project_msc/2025/csml/sruppage/evo-memory/experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt"

NUM_LAYERS = 16
RANK = 8
PROJS = ["q_proj", "v_proj"]

# ── load checkpoints ──────────────────────────────────────────────────────────
print("Loading M1 checkpoint...")
m1_ckpt = torch.load(M1_PATH, map_location="cpu")
m1_sd = m1_ckpt["lora_state_dict"]

print("Loading M3 cs1024 checkpoint...")
m3_ckpt = torch.load(M3_PATH, map_location="cpu")
m3_sd = m3_ckpt["lora_state_dict"]


# ── helper: build effective LoRA update B @ A ─────────────────────────────────
def get_BA(state_dict, layer, proj):
    prefix = f"base_model.model.layers.{layer}.self_attn.{proj}"
    A = state_dict[f"{prefix}.lora_A.default.weight"]  # (rank, d_in)
    B = state_dict[f"{prefix}.lora_B.default.weight"]  # (d_out, rank)
    return (B @ A).float().numpy()  # (d_out, d_in)


# ── compute per-layer metrics ─────────────────────────────────────────────────
results = {
    "m1_norms": {p: [] for p in PROJS},
    "m3_norms": {p: [] for p in PROJS},
    "ratio": {p: [] for p in PROJS},
    "m1_sv": {p: [] for p in PROJS},
    "m3_sv": {p: [] for p in PROJS},
    "subspace_overlap": {p: [] for p in PROJS},
}

for layer in range(NUM_LAYERS):
    for proj in PROJS:
        BA_m1 = get_BA(m1_sd, layer, proj)
        BA_m3 = get_BA(m3_sd, layer, proj)

        # Frobenius norms
        norm_m1 = np.linalg.norm(BA_m1, "fro")
        norm_m3 = np.linalg.norm(BA_m3, "fro")
        results["m1_norms"][proj].append(norm_m1)
        results["m3_norms"][proj].append(norm_m3)
        results["ratio"][proj].append(norm_m3 / norm_m1 if norm_m1 > 1e-12 else float("nan"))

        # SVD
        U_m1, S_m1, Vt_m1 = np.linalg.svd(BA_m1, full_matrices=False)
        U_m3, S_m3, Vt_m3 = np.linalg.svd(BA_m3, full_matrices=False)
        results["m1_sv"][proj].append(S_m1[:RANK])
        results["m3_sv"][proj].append(S_m3[:RANK])

        # Subspace overlap via principal angles
        # Use top-rank left singular vectors as basis for column space
        U1 = U_m1[:, :RANK]  # (d_out, rank)
        U3 = U_m3[:, :RANK]  # (d_out, rank)
        # Cosines of principal angles = singular values of U1^T @ U3
        cos_angles = np.linalg.svd(U1.T @ U3, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0.0, 1.0)
        results["subspace_overlap"][proj].append(cos_angles.mean())

    print(f"  layer {layer:2d}  done")

# Convert to arrays
for key in ["m1_norms", "m3_norms", "ratio", "subspace_overlap"]:
    for proj in PROJS:
        results[key][proj] = np.array(results[key][proj])

layers = np.arange(NUM_LAYERS)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})
M1_COLOR = "#2196F3"
M3_COLOR = "#FF5722"

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Weight Magnitude
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, proj in zip(axes, PROJS):
    width = 0.35
    ax.bar(layers - width / 2, results["m1_norms"][proj], width, label="M1 (full ctx)", color=M1_COLOR, alpha=0.85)
    ax.bar(layers + width / 2, results["m3_norms"][proj], width, label="M3 cs1024", color=M3_COLOR, alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||B @ A||_F")
    ax.set_title(f"Effective LoRA Update Norm -- {proj}")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Per-Layer LoRA Weight Magnitude: M1 vs M3 cs1024", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOT_DIR, "weight_magnitude.png"))
plt.close(fig)
print("Saved weight_magnitude.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Singular Values (q_proj only, 4x4 grid)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True)
axes_flat = axes.flatten()

for layer in range(NUM_LAYERS):
    ax = axes_flat[layer]
    sv_m1 = results["m1_sv"]["q_proj"][layer]
    sv_m3 = results["m3_sv"]["q_proj"][layer]
    x = np.arange(1, len(sv_m1) + 1)
    ax.bar(x - 0.2, sv_m1, 0.4, label="M1" if layer == 0 else None, color=M1_COLOR, alpha=0.85)
    ax.bar(x + 0.2, sv_m3, 0.4, label="M3 cs1024" if layer == 0 else None, color=M3_COLOR, alpha=0.85)
    ax.set_title(f"Layer {layer}", fontsize=9)
    ax.set_xticks(x)
    if layer >= 12:
        ax.set_xlabel("SV index")
    if layer % 4 == 0:
        ax.set_ylabel("Singular value")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Singular Value Spectrum of B@A (q_proj): M1 vs M3 cs1024", fontsize=13, fontweight="bold")
fig.legend(["M1 (full ctx)", "M3 cs1024"], loc="upper right", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOT_DIR, "singular_values.png"))
plt.close(fig)
print("Saved singular_values.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Subspace Overlap
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
ax.bar(layers - width / 2, results["subspace_overlap"]["q_proj"], width, label="q_proj", color=M1_COLOR, alpha=0.85)
ax.bar(layers + width / 2, results["subspace_overlap"]["v_proj"], width, label="v_proj", color=M3_COLOR, alpha=0.85)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean cosine of principal angles")
ax.set_title("Subspace Overlap Between M1 and M3 cs1024 LoRA Column Spaces", fontsize=12, fontweight="bold")
ax.set_xticks(layers)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "subspace_overlap.png"))
plt.close(fig)
print("Saved subspace_overlap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Norm Ratio
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(layers, results["ratio"]["q_proj"], "o-", color=M1_COLOR, label="q_proj", markersize=7, linewidth=2)
ax.plot(layers, results["ratio"]["v_proj"], "s-", color=M3_COLOR, label="v_proj", markersize=7, linewidth=2)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="ratio = 1 (equal)")
ax.set_xlabel("Layer")
ax.set_ylabel("||M3 B@A||_F / ||M1 B@A||_F")
ax.set_title("Norm Ratio (M3 / M1): Which Layers Work Harder Under Eviction?", fontsize=12, fontweight="bold")
ax.set_xticks(layers)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "norm_ratio.png"))
plt.close(fig)
print("Saved norm_ratio.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary tables for the report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY TABLES")
print("=" * 80)

print("\n--- Per-Layer Frobenius Norms and Ratio (q_proj) ---")
print(f"{'Layer':>5} | {'M1 ||B@A||_F':>14} | {'M3 ||B@A||_F':>14} | {'Ratio M3/M1':>12}")
print("-" * 55)
for i in range(NUM_LAYERS):
    print(f"{i:5d} | {results['m1_norms']['q_proj'][i]:14.6f} | {results['m3_norms']['q_proj'][i]:14.6f} | {results['ratio']['q_proj'][i]:12.4f}")

print("\n--- Per-Layer Frobenius Norms and Ratio (v_proj) ---")
print(f"{'Layer':>5} | {'M1 ||B@A||_F':>14} | {'M3 ||B@A||_F':>14} | {'Ratio M3/M1':>12}")
print("-" * 55)
for i in range(NUM_LAYERS):
    print(f"{i:5d} | {results['m1_norms']['v_proj'][i]:14.6f} | {results['m3_norms']['v_proj'][i]:14.6f} | {results['ratio']['v_proj'][i]:12.4f}")

print("\n--- Subspace Overlap (mean cosine of principal angles) ---")
print(f"{'Layer':>5} | {'q_proj':>10} | {'v_proj':>10}")
print("-" * 35)
for i in range(NUM_LAYERS):
    print(f"{i:5d} | {results['subspace_overlap']['q_proj'][i]:10.4f} | {results['subspace_overlap']['v_proj'][i]:10.4f}")

# Also print aggregate stats
print("\n--- Aggregate Stats ---")
for proj in PROJS:
    r = results["ratio"][proj]
    o = results["subspace_overlap"][proj]
    print(f"{proj}: ratio mean={r.mean():.4f}, std={r.std():.4f}, min={r.min():.4f} (L{r.argmin()}), max={r.max():.4f} (L{r.argmax()})")
    print(f"{proj}: overlap mean={o.mean():.4f}, std={o.std():.4f}, min={o.min():.4f} (L{o.argmin()}), max={o.max():.4f} (L{o.argmax()})")

print("\nDone.")
