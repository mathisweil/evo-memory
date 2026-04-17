"""Section C — Analyse eviction-mask drift from per-prompt NAMM dumps.

Consumes dumps written by ``scripts/eval_namm_splits.py --dump_namm_state`` for
three conditions (B0 base Llama, M1 base+M1-LoRA, M4 base+M4-LoRA) under the
same frozen NAMM checkpoint on the same 70-prompt FAIR-01 test split.

Produces:
- ``eval_results/section_c_metrics.json``: all numeric results.
- ``figures/section_c/C1_mask_overlap.{png,pdf}``: headline IoU heatmap.
- ``figures/section_c/C2_retention_by_layer.{png,pdf}``: per-layer retention.
- ``figures/section_c/C3_score_distributions.{png,pdf}``: score KDE + KS.
- ``figures/section_c/C4_iou_by_layer.{png,pdf}``: per-layer IoU drift.

All randomness (cross-prompt pair sampling, KDE subsampling) is seeded to
``numpy.random.default_rng(seed=0)``.

Usage:
    venv/bin/python scripts/analyze_mask_drift.py \\
        --dumps_root eval_results/section_c_dumps \\
        --metrics_out eval_results/section_c_metrics.json \\
        --figures_out figures/section_c
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_paper_figures import COLORS, save_figure, setup_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITION_ORDER: Tuple[str, ...] = ("B0", "M1", "M4")
CONDITION_COLOR: Dict[str, str] = {
    "B0": COLORS["baseline"],
    "M1": COLORS["lora"],
    "M4": COLORS["joint"],
}
CONDITION_LABEL: Dict[str, str] = {
    "B0": "B0 (base)",
    "M1": "M1 (+LoRA)",
    "M4": "M4 (+LoRA, joint)",
}


@dataclass
class PromptRecord:
    """One condition's dump for a single prompt."""

    task: str
    orig_idx: int
    prompt_length: int
    protected_tail_n: int
    n_steps: int
    n_layers: int
    n_heads: int
    cache_size: int
    final_retained_positions: torch.Tensor  # (n_layers, n_heads, k) int
    final_scores: torch.Tensor  # (n_layers, n_heads, n_kv) fp16
    final_new_mask: torch.Tensor  # (n_layers, n_heads, k) bool
    final_position_ids: Optional[torch.Tensor]  # (n_layers, n_kv) int (head-0)
    final_attn_mean_per_token: Optional[torch.Tensor]  # (n_layers, n_kv) fp16
    per_step_n_kv: List[int]
    per_step_retained_count_per_head: torch.Tensor  # (n_steps, n_layers, n_heads)


# ── Loading ──────────────────────────────────────────────────────────────────


def _parse_filename(path: Path) -> Tuple[str, int]:
    """Parse ``{task}__{orig_idx:04d}.pt``.

    Tasks can embed double underscores (e.g. ``rh__qasper``); the right-most
    ``__{4-digit-int}.pt`` suffix identifies the prompt index.
    """
    stem = path.stem
    base, _, idx_part = stem.rpartition("__")
    if not base or not idx_part:
        raise ValueError(f"Cannot parse dump filename: {path}")
    return base, int(idx_part)


def load_condition_dumps(condition_dir: Path) -> Dict[Tuple[str, int], PromptRecord]:
    """Load all ``.pt`` dumps in ``condition_dir`` into a key→record map."""
    records: Dict[Tuple[str, int], PromptRecord] = {}
    files = sorted(condition_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt dumps under {condition_dir}")
    for f in files:
        task, orig_idx = _parse_filename(f)
        blob = torch.load(f, map_location="cpu", weights_only=False)
        meta = blob["prompt_meta"]
        retained_positions = blob.get("final_retained_positions")
        if retained_positions is None:
            # Fall back to head-0 position_ids translated through retained_idxs.
            # This loses per-head divergence but lets old dumps still load.
            pos = blob.get("final_position_ids")
            retained_idxs = blob["final_retained_idxs"]
            if pos is None:
                raise RuntimeError(
                    f"{f}: missing both final_retained_positions and "
                    "final_position_ids; cannot translate to prompt space."
                )
            # pos: (n_layers, n_kv); retained_idxs: (n_layers, n_heads, k)
            retained_positions = torch.gather(
                pos.unsqueeze(1).expand(-1, retained_idxs.shape[1], -1).long(),
                dim=-1,
                index=retained_idxs.long(),
            ).to(torch.int32)
        scores = blob["final_scores"]
        n_layers, n_heads, _ = scores.shape
        records[(task, orig_idx)] = PromptRecord(
            task=task,
            orig_idx=orig_idx,
            prompt_length=int(meta["prompt_length_tokens"]),
            protected_tail_n=int(meta.get("protected_tail_n", 0)),
            n_steps=int(meta.get("n_steps", blob.get("n_steps", 0))),
            n_layers=n_layers,
            n_heads=n_heads,
            cache_size=int(blob["config"]["cache_size"]),
            final_retained_positions=retained_positions.to(torch.int32),
            final_scores=scores,
            final_new_mask=blob["final_new_mask"],
            final_position_ids=blob.get("final_position_ids"),
            final_attn_mean_per_token=blob.get("final_attn_mean_per_token"),
            per_step_n_kv=list(blob["per_step_n_kv"]),
            per_step_retained_count_per_head=blob[
                "per_step_retained_count_per_head"
            ],
        )
    return records


def load_all_dumps(
    dumps_root: Path,
) -> Tuple[Dict[str, Dict[Tuple[str, int], PromptRecord]], List[Tuple[str, int]]]:
    """Load dumps for every condition subdirectory and intersect prompt keys."""
    by_condition: Dict[str, Dict[Tuple[str, int], PromptRecord]] = {}
    for cond in CONDITION_ORDER:
        cond_dir = dumps_root / cond
        if not cond_dir.exists():
            raise RuntimeError(f"Missing condition dir: {cond_dir}")
        by_condition[cond] = load_condition_dumps(cond_dir)
        logger.info("Loaded %d dumps for %s", len(by_condition[cond]), cond)

    shared_keys = set.intersection(
        *(set(d.keys()) for d in by_condition.values())
    )
    if not shared_keys:
        raise RuntimeError("No prompts common across all conditions.")
    missing: Dict[str, List[Tuple[str, int]]] = {
        c: sorted(set(by_condition[c].keys()) - shared_keys)
        for c in CONDITION_ORDER
    }
    for c, ms in missing.items():
        if ms:
            logger.warning(
                "%s has %d prompts absent from another condition (skipped)",
                c, len(ms),
            )
    keys_sorted = sorted(shared_keys)
    return by_condition, keys_sorted


# ── Per-(layer, head) IoU in prompt-position space ───────────────────────────


def _retained_positions_masked(
    record: PromptRecord,
) -> torch.Tensor:
    """Return retained prompt positions with the protected tail removed.

    The protected tail is the last ``protected_tail_n`` prompt positions,
    forced to be kept by ``+protected_tail_n=5``. Including them inflates
    IoU by a trivial +5/1024 of agreement across conditions, so we drop them.

    Returns:
        Int tensor (n_layers, n_heads, k) where the protected-tail positions
        have been replaced by ``-1``. IoU is computed on the remaining values.
    """
    positions = record.final_retained_positions.clone().to(torch.int64)
    if record.protected_tail_n > 0:
        tail_cutoff = record.prompt_length - record.protected_tail_n
        positions = torch.where(
            positions >= tail_cutoff,
            torch.full_like(positions, -1),
            positions,
        )
    return positions


def _pair_iou_per_layer_head(
    positions_a: torch.Tensor,
    positions_b: torch.Tensor,
    prompt_length: int,
) -> np.ndarray:
    """Compute IoU per (layer, head) between two retained-position tensors.

    Uses a dense boolean buffer over the prompt length — O(L * H * P) work
    per prompt, which at 16 × 32 × 6500 ≈ 3.3 M booleans is fast.

    Args:
        positions_a: (n_layers, n_heads, k) int64 with -1 for excluded entries.
        positions_b: same shape.
        prompt_length: P such that valid positions are in [0, P).

    Returns:
        Numpy array shape (n_layers, n_heads) of IoU floats in [0, 1].
    """
    n_layers, n_heads, _ = positions_a.shape
    mask_a = _positions_to_dense_mask(positions_a, prompt_length)
    mask_b = _positions_to_dense_mask(positions_b, prompt_length)
    inter = (mask_a & mask_b).sum(dim=-1).to(torch.float64)
    union = (mask_a | mask_b).sum(dim=-1).to(torch.float64)
    iou = torch.where(
        union > 0, inter / union, torch.zeros_like(inter)
    )
    return iou.cpu().numpy()  # (n_layers, n_heads)


def _positions_to_dense_mask(
    positions: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    """Scatter (n_layers, n_heads, k) position indices into a dense bool mask."""
    valid = positions >= 0
    safe_positions = positions.clamp(min=0, max=prompt_length - 1)
    dense = torch.zeros(
        positions.shape[0],
        positions.shape[1],
        prompt_length,
        dtype=torch.bool,
    )
    dense.scatter_(dim=-1, index=safe_positions, src=valid)
    return dense


def compute_pairwise_iou(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
) -> Dict[str, Dict]:
    """Mean pairwise IoU per condition pair, overall and per-layer."""
    pairs = list(combinations(CONDITION_ORDER, 2)) + [(c, c) for c in CONDITION_ORDER]
    per_layer: Dict[str, List[np.ndarray]] = {
        f"{a}__{b}": [] for a, b in pairs
    }
    overall: Dict[str, List[float]] = {f"{a}__{b}": [] for a, b in pairs}

    for key in keys:
        rec_by_cond = {c: dumps[c][key] for c in CONDITION_ORDER}
        prompt_length = rec_by_cond["B0"].prompt_length
        pos_by_cond = {
            c: _retained_positions_masked(rec) for c, rec in rec_by_cond.items()
        }
        # Cross-pair IoU (the headline metric).
        for a, b in combinations(CONDITION_ORDER, 2):
            iou = _pair_iou_per_layer_head(
                pos_by_cond[a], pos_by_cond[b], prompt_length
            )
            per_layer[f"{a}__{b}"].append(iou)
            overall[f"{a}__{b}"].append(float(iou.mean()))
        # Self IoU (should be 1.0 — sanity check).
        for c in CONDITION_ORDER:
            iou = _pair_iou_per_layer_head(
                pos_by_cond[c], pos_by_cond[c], prompt_length
            )
            per_layer[f"{c}__{c}"].append(iou)
            overall[f"{c}__{c}"].append(float(iou.mean()))

    summary: Dict[str, Dict] = {}
    for name in per_layer:
        stacked = np.stack(per_layer[name], axis=0)  # (n_prompts, L, H)
        summary[name] = {
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
            "n_prompts": int(stacked.shape[0]),
            "per_layer_mean": stacked.mean(axis=(0, 2)).tolist(),
            "per_layer_std": stacked.std(axis=(0, 2)).tolist(),
            "per_prompt_mean": [float(v) for v in overall[name]],
        }
    return summary


# ── Per-layer union-over-heads retention ─────────────────────────────────────


def compute_per_layer_retention(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
) -> Dict[str, Dict]:
    """For each condition × layer, mean fraction of unique prompt positions
    retained by at least one head at the final step (excluding the protected
    tail).
    """
    out: Dict[str, Dict] = {}
    for cond in CONDITION_ORDER:
        per_layer_prompts: List[np.ndarray] = []
        for key in keys:
            rec = dumps[cond][key]
            positions = _retained_positions_masked(rec)  # (L, H, k)
            prompt_length = rec.prompt_length
            mask = _positions_to_dense_mask(positions, prompt_length)
            any_head = mask.any(dim=1)  # (L, prompt_length)
            # Exclude the protected tail from the denominator too.
            n_valid = max(prompt_length - rec.protected_tail_n, 1)
            retention = any_head.sum(dim=-1).to(torch.float64).numpy() / n_valid
            per_layer_prompts.append(retention)
        stacked = np.stack(per_layer_prompts, axis=0)  # (n_prompts, n_layers)
        out[cond] = {
            "per_layer_mean": stacked.mean(axis=0).tolist(),
            "per_layer_std": stacked.std(axis=0).tolist(),
            "n_prompts": int(stacked.shape[0]),
        }
    return out


# ── Score distribution KS ────────────────────────────────────────────────────


def compute_ks_distance(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
    rng: np.random.Generator,
    max_samples_per_layer: int = 200_000,
) -> Dict[str, Dict]:
    """Pairwise Kolmogorov-Smirnov distance between per-layer score distributions.

    Pools ``final_scores`` values for each (condition, layer) across all
    prompts and heads, subsamples to ``max_samples_per_layer`` for tractable
    KS, and reports distances per (pair, layer) plus an overall mean.
    """
    scores_by_cond_layer: Dict[str, List[np.ndarray]] = {c: [] for c in CONDITION_ORDER}
    n_layers = next(iter(dumps["B0"].values())).n_layers
    for cond in CONDITION_ORDER:
        for layer in range(n_layers):
            parts: List[np.ndarray] = []
            for key in keys:
                rec = dumps[cond][key]
                layer_scores = rec.final_scores[layer].float().flatten().numpy()
                parts.append(layer_scores)
            flat = np.concatenate(parts)
            if flat.size > max_samples_per_layer:
                idx = rng.choice(
                    flat.size, size=max_samples_per_layer, replace=False
                )
                flat = flat[idx]
            scores_by_cond_layer[cond].append(flat)

    out: Dict[str, Dict] = {}
    for a, b in combinations(CONDITION_ORDER, 2):
        per_layer: List[float] = []
        per_layer_p: List[float] = []
        for layer in range(n_layers):
            stat = ks_2samp(
                scores_by_cond_layer[a][layer],
                scores_by_cond_layer[b][layer],
            )
            per_layer.append(float(stat.statistic))
            per_layer_p.append(float(stat.pvalue))
        out[f"{a}__{b}"] = {
            "per_layer_ks": per_layer,
            "per_layer_pvalue": per_layer_p,
            "mean_ks": float(np.mean(per_layer)),
        }
    out["_samples_kept"] = {
        cond: int(sum(a.size for a in scores_by_cond_layer[cond]))
        for cond in CONDITION_ORDER
    }
    return out


# ── Spearman ρ(NAMM scores, LLM attention) reproduction ──────────────────────


def compute_spearman_per_condition(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
) -> Dict[str, Dict]:
    """Reproduce paper §5.4 Spearman ρ between NAMM final scores (mean over
    heads) and per-token mean LLM attention, averaged over (prompts, layers).

    Returns a dict keyed by condition with mean/std and per-prompt values.
    """
    out: Dict[str, Dict] = {}
    for cond in CONDITION_ORDER:
        per_prompt_rho: List[float] = []
        for key in keys:
            rec = dumps[cond][key]
            if rec.final_attn_mean_per_token is None:
                continue
            rhos: List[float] = []
            n_layers = rec.n_layers
            attn = rec.final_attn_mean_per_token.float()
            scores = rec.final_scores.float()
            # Attention captured on the final chunk may have a smaller
            # kv_len than the NAMM n_kv at the eviction call. Align on the
            # minimum length from the end.
            for layer in range(n_layers):
                attn_l = attn[layer]
                score_l = scores[layer].mean(dim=0)
                m = min(attn_l.shape[-1], score_l.shape[-1])
                if m < 4:
                    continue
                a = attn_l[-m:].numpy()
                s = score_l[-m:].numpy()
                if np.std(a) < 1e-12 or np.std(s) < 1e-12:
                    continue
                rho = spearmanr(s, a).statistic
                if np.isfinite(rho):
                    rhos.append(float(rho))
            if rhos:
                per_prompt_rho.append(float(np.mean(rhos)))
        arr = np.asarray(per_prompt_rho)
        out[cond] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "n_prompts": int(arr.size),
            "per_prompt": per_prompt_rho,
        }
    paper_ref = {"B0": None, "M1": -0.115, "M4": -0.168}
    warnings: List[str] = []
    for cond, ref in paper_ref.items():
        if ref is None:
            continue
        m = out[cond]["mean"]
        if np.isfinite(m) and abs(m - ref) > 0.03:
            warnings.append(
                f"{cond}: reproduction mean={m:+.3f} vs paper §5.4={ref:+.3f} "
                f"(delta={m - ref:+.3f}); check if §5.4 used a different split."
            )
    out["_paper_reference"] = paper_ref
    out["_warnings"] = warnings
    return out


# ── Cross-prompt & analytic baselines ────────────────────────────────────────


def cross_prompt_iou_baseline(
    condition_dumps: Dict[Tuple[str, int], PromptRecord],
    keys: Sequence[Tuple[str, int]],
    rng: np.random.Generator,
    n_pairs: int = 200,
) -> Dict[str, float]:
    """Mean IoU between different-prompt mask pairs under one condition.

    Quantifies NAMM's natural prompt-to-prompt variability so C1 has a
    meaningful lower reference. IoU is only defined where both prompts have
    the same length — we enforce this by only pairing within prompts of
    equal ``prompt_length``.
    """
    keys_list = list(keys)
    by_length: Dict[int, List[Tuple[str, int]]] = {}
    for key in keys_list:
        L = condition_dumps[key].prompt_length
        by_length.setdefault(L, []).append(key)
    # Prefer the largest bucket so we get stable statistics.
    eligible_buckets = [ks for ks in by_length.values() if len(ks) >= 2]
    if not eligible_buckets:
        logger.warning(
            "Cross-prompt IoU: no two prompts share a length; "
            "falling back to min-length truncation."
        )
        return {"mean": float("nan"), "std": float("nan"), "n_pairs": 0}

    sampled_ious: List[float] = []
    for _ in range(n_pairs):
        bucket = eligible_buckets[rng.integers(len(eligible_buckets))]
        a_key, b_key = rng.choice(len(bucket), size=2, replace=False)
        a = condition_dumps[bucket[a_key]]
        b = condition_dumps[bucket[b_key]]
        positions_a = _retained_positions_masked(a)
        positions_b = _retained_positions_masked(b)
        iou = _pair_iou_per_layer_head(positions_a, positions_b, a.prompt_length)
        sampled_ious.append(float(iou.mean()))
    arr = np.asarray(sampled_ious)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "n_pairs": int(arr.size),
    }


def analytic_random_iou_baseline(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
) -> float:
    """IoU of two random top-k masks at the same retention rate (closed form).

    For per-head k out of P positions, E[IoU(A, B)] = k / (2P - k) if A and B
    are drawn uniformly at random. We average this over prompts and report
    a single analytic number.
    """
    ious: List[float] = []
    for key in keys:
        rec = dumps["B0"][key]
        k = rec.cache_size
        p = rec.prompt_length - rec.protected_tail_n
        if p <= k:
            continue
        ious.append(k / (2 * p - k))
    return float(np.mean(ious)) if ious else float("nan")


# ── Figures ──────────────────────────────────────────────────────────────────


def _plot_c1_mask_overlap(
    pairwise: Dict[str, Dict],
    cross_prompt_b0: Dict[str, float],
    random_iou: float,
    out_dir: Path,
) -> str:
    """3×3 heatmap of mean pairwise IoU (averaged over prompts, heads, layers)."""
    n = len(CONDITION_ORDER)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(CONDITION_ORDER):
        for j, b in enumerate(CONDITION_ORDER):
            key = f"{a}__{b}" if i <= j else f"{b}__{a}"
            matrix[i, j] = pairwise[key]["mean"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER])
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{matrix[i, j]:.3f}",
                ha="center", va="center",
                color="white" if matrix[i, j] < 0.55 else "black",
                fontsize=10,
            )
    ax.set_title(
        "Eviction-mask IoU\n"
        f"cross-prompt B0 baseline={cross_prompt_b0['mean']:.3f}, "
        f"random={random_iou:.3f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="IoU")
    fig.tight_layout()
    return save_figure(fig, "C1_mask_overlap", out_dir)


def _plot_c2_retention_by_layer(
    retention: Dict[str, Dict], out_dir: Path
) -> str:
    """Per-layer union-over-heads retention rate, one line per condition."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for cond in CONDITION_ORDER:
        mean = np.asarray(retention[cond]["per_layer_mean"])
        std = np.asarray(retention[cond]["per_layer_std"])
        x = np.arange(len(mean))
        ax.plot(x, mean, marker="o", ms=3.5,
                color=CONDITION_COLOR[cond], label=CONDITION_LABEL[cond])
        ax.fill_between(
            x, mean - std, mean + std,
            color=CONDITION_COLOR[cond], alpha=0.15,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of prompt positions retained\n(union over heads)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, "C2_retention_by_layer", out_dir)


def _plot_c3_score_distributions(
    dumps: Dict[str, Dict[Tuple[str, int], PromptRecord]],
    keys: Sequence[Tuple[str, int]],
    ks_metrics: Dict[str, Dict],
    rng: np.random.Generator,
    out_dir: Path,
    max_samples: int = 100_000,
) -> str:
    """KDE of final NAMM score values per condition (pooled across layers/heads)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for cond in CONDITION_ORDER:
        parts: List[np.ndarray] = []
        for key in keys:
            parts.append(dumps[cond][key].final_scores.float().flatten().numpy())
        flat = np.concatenate(parts)
        if flat.size > max_samples:
            idx = rng.choice(flat.size, size=max_samples, replace=False)
            flat = flat[idx]
        ax.hist(
            flat, bins=200, density=True, histtype="step",
            color=CONDITION_COLOR[cond], label=CONDITION_LABEL[cond],
            linewidth=1.4,
        )
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.7, alpha=0.6,
               label="threshold (s=0)")
    caption_lines = [
        f"KS({a},{b})={ks_metrics[f'{a}__{b}']['mean_ks']:.3f}"
        for a, b in combinations(CONDITION_ORDER, 2)
    ]
    ax.set_title("NAMM score distributions (pooled, final step)\n"
                 + "; ".join(caption_lines), fontsize=10)
    ax.set_xlabel("NAMM token score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, "C3_score_distributions", out_dir)


def _plot_c4_iou_by_layer(pairwise: Dict[str, Dict], out_dir: Path) -> str:
    """Per-layer IoU for the three cross-condition pairs."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    pair_color = {
        "B0__M1": COLORS["lora"],
        "B0__M4": COLORS["joint"],
        "M1__M4": COLORS["sequential"],
    }
    for pair, color in pair_color.items():
        per_layer = np.asarray(pairwise[pair]["per_layer_mean"])
        per_layer_std = np.asarray(pairwise[pair]["per_layer_std"])
        x = np.arange(len(per_layer))
        ax.plot(x, per_layer, marker="o", ms=3.5, color=color, label=pair.replace("__", " vs "))
        ax.fill_between(x, per_layer - per_layer_std, per_layer + per_layer_std,
                        color=color, alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, "C4_iou_by_layer", out_dir)


# ── Driver ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dumps_root", type=str, required=True,
        help="Directory containing B0/, M1/, M4/ sub-directories of .pt dumps",
    )
    parser.add_argument(
        "--metrics_out", type=str,
        default=str(REPO_ROOT / "eval_results" / "section_c_metrics.json"),
    )
    parser.add_argument(
        "--figures_out", type=str,
        default=str(REPO_ROOT / "figures" / "section_c"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ks_max_samples_per_layer", type=int, default=200_000,
        help="Subsample cap per (condition, layer) before KS-2samp.",
    )
    parser.add_argument(
        "--cross_prompt_pairs", type=int, default=200,
        help="Number of (prompt_a, prompt_b) pairs for B0 cross-prompt baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    dumps_root = Path(args.dumps_root).resolve()
    figures_out = Path(args.figures_out).resolve()
    metrics_out = Path(args.metrics_out).resolve()
    figures_out.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    dumps, keys = load_all_dumps(dumps_root)
    logger.info("Analysing %d prompts shared across %s",
                len(keys), ", ".join(CONDITION_ORDER))

    pairwise = compute_pairwise_iou(dumps, keys)
    retention = compute_per_layer_retention(dumps, keys)
    ks = compute_ks_distance(
        dumps, keys, rng, max_samples_per_layer=args.ks_max_samples_per_layer,
    )
    spearman = compute_spearman_per_condition(dumps, keys)
    cross_b0 = cross_prompt_iou_baseline(
        dumps["B0"], keys, rng, n_pairs=args.cross_prompt_pairs,
    )
    random_iou = analytic_random_iou_baseline(dumps, keys)

    metrics: Dict[str, object] = {
        "pairwise_iou": pairwise,
        "per_layer_retention": retention,
        "ks_distance_scores": ks,
        "spearman_namm_vs_attn": spearman,
        "reference_cross_prompt_iou_B0": cross_b0,
        "reference_random_baseline_iou": random_iou,
        "meta": {
            "conditions": list(CONDITION_ORDER),
            "n_prompts": len(keys),
            "prompt_keys": [[t, int(i)] for t, i in keys],
            "seed": args.seed,
            "dumps_root": str(dumps_root),
        },
    }

    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("Wrote metrics → %s", metrics_out)

    setup_style()
    c1 = _plot_c1_mask_overlap(pairwise, cross_b0, random_iou, figures_out)
    c2 = _plot_c2_retention_by_layer(retention, figures_out)
    c3 = _plot_c3_score_distributions(dumps, keys, ks, rng, figures_out)
    c4 = _plot_c4_iou_by_layer(pairwise, figures_out)
    logger.info("Wrote figures → %s, %s, %s, %s", c1, c2, c3, c4)

    for w in spearman.get("_warnings", []):
        logger.warning("Spearman cross-check: %s", w)


if __name__ == "__main__":
    main()
