#!/usr/bin/env python3
"""Report 4 -- LoRA Weight Comparison data generation.

CPU-only: reads M1 and M3 checkpoint lora_state_dicts and computes per-layer
norms, SVD spectra, subspace overlap, and norm ratios.

Saves to: analysis/report_4/data/maskfix_data.npz

Usage:
    PYTHONPATH=. .venv/bin/python analysis/report_4/scripts/generate_data.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

M1_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M1/best_ckpt.pt"
M3_MASKFIX_CKPT = REPO_ROOT / "experiment_artifacts/gcs/M3_cs1024_maskfix/best_ckpt.pt"

NUM_LAYERS = 16
RANK = 8
PROJS = ["q_proj", "v_proj"]


def get_BA(state_dict: dict, layer: int, proj: str) -> np.ndarray:
    prefix = f"base_model.model.layers.{layer}.self_attn.{proj}"
    A = state_dict[f"{prefix}.lora_A.default.weight"]
    B = state_dict[f"{prefix}.lora_B.default.weight"]
    return (B @ A).float().numpy()


def compute() -> None:
    out_path = DATA_DIR / "maskfix_data.npz"
    if out_path.exists():
        logger.info("Data already exists: %s — skipping", out_path)
        return

    for path, label in [(M1_CKPT, "M1"), (M3_MASKFIX_CKPT, "M3")]:
        if not path.exists():
            logger.error("Checkpoint not found: %s (%s)", path, label)
            sys.exit(1)

    logger.info("Loading checkpoint state dicts (CPU)...")
    m1_sd = torch.load(str(M1_CKPT), map_location="cpu")["lora_state_dict"]
    m3_sd = torch.load(str(M3_MASKFIX_CKPT), map_location="cpu")["lora_state_dict"]

    results: dict = {
        tag: {"norms": {p: [] for p in PROJS}, "sv": {p: [] for p in PROJS}}
        for tag in ["m1", "m3_maskfix"]
    }
    results["ratio_maskfix"] = {p: [] for p in PROJS}
    results["overlap_maskfix"] = {p: [] for p in PROJS}

    sds = {"m1": m1_sd, "m3_maskfix": m3_sd}

    for layer in range(NUM_LAYERS):
        for proj in PROJS:
            BAs, Us = {}, {}
            for tag, sd in sds.items():
                ba = get_BA(sd, layer, proj)
                BAs[tag] = ba
                norm = float(np.linalg.norm(ba, "fro"))
                results[tag]["norms"][proj].append(norm)

                U, S, _ = np.linalg.svd(ba, full_matrices=False)
                Us[tag] = U[:, :RANK]
                results[tag]["sv"][proj].append(S[:RANK].tolist())

            m1_norm = results["m1"]["norms"][proj][-1]
            m3_norm = results["m3_maskfix"]["norms"][proj][-1]
            results["ratio_maskfix"][proj].append(
                m3_norm / m1_norm if m1_norm > 1e-12 else float("nan")
            )
            cos_angles = np.linalg.svd(
                Us["m1"].T @ Us["m3_maskfix"], compute_uv=False,
            )
            cos_angles = np.clip(cos_angles, 0.0, 1.0)
            results["overlap_maskfix"][proj].append(float(cos_angles.mean()))

        logger.info("  layer %2d done", layer)

    flat: dict[str, np.ndarray] = {}
    for tag in ["m1", "m3_maskfix"]:
        for proj in PROJS:
            flat[f"{tag}_norms_{proj}"] = np.array(results[tag]["norms"][proj])
            flat[f"{tag}_sv_{proj}"] = np.array(results[tag]["sv"][proj])
    for proj in PROJS:
        flat[f"ratio_maskfix_{proj}"] = np.array(results["ratio_maskfix"][proj])
        flat[f"overlap_maskfix_{proj}"] = np.array(results["overlap_maskfix"][proj])

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(str(out_path), **flat)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    compute()
