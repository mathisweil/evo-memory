#!/usr/bin/env python3
"""Generate all paper figures and tables for the evo-memory research paper.

Reads experiment results from the specified directories, produces a summary
CSV, then renders every figure and table needed for the paper.

Usage:
    python scripts/generate_paper_figures.py \\
        --experiment_dir experiments/experiment_1 \\
        --outputs_dir outputs \\
        --paper_figures_dir paper_figures
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    import orjson

    def _load_json(path: Path) -> dict:
        """Load JSON using orjson."""
        return orjson.loads(path.read_bytes())

except ImportError:
    import json

    def _load_json(path: Path) -> dict:  # type: ignore[misc]
        """Load JSON using stdlib json."""
        with path.open() as f:
            return json.load(f)


logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# max_conditioning_length from namm_bam_i1_llama32_1b.yaml
FULL_CACHE_TOKENS: int = 4086

# Llama-3.2-1B architecture constants for LoRA parameter count formula
_LORA_HIDDEN: int = 2048
_LORA_LAYERS: int = 16

# Condition metadata: (display_name, optimizer, has_namm, cache_size_str, color_key)
CONDITION_META: dict[str, tuple[str, str, bool, str, str]] = {
    "B0": ("Base model (full cache)", "none", False, "full", "baseline"),
    "B1": ("Base + recency (1024)", "recency", False, "1024", "baseline"),
    "M1-r4": ("LoRA only r=4", "LoRA", False, "full", "lora"),
    "M1-r8": ("LoRA only r=8", "LoRA", False, "full", "lora"),
    "M1-r16": ("LoRA only r=16", "LoRA", False, "full", "lora"),
    "M1-ES": ("ES only", "ES", False, "full", "es"),
    "M2": ("Standalone NAMM", "CMA-ES", True, "1024", "namm"),
    "M3-LoRA": ("Sequential: LoRA → NAMM", "LoRA+NAMM", True, "1024", "sequential"),
    "M3-ES": ("Sequential: ES → NAMM", "ES+NAMM", True, "1024", "sequential"),
    "M4-LoRA": ("Joint: LoRA + NAMM", "LoRA+NAMM", True, "1024", "joint"),
    "M4-ES": ("Joint: ES + NAMM", "ES+NAMM", True, "1024", "joint"),
    "A2-512": ("NAMM (cache=512)", "CMA-ES", True, "512", "namm"),
    "A2-2048": ("NAMM (cache=2048)", "CMA-ES", True, "2048", "namm"),
    "A4-on": ("M4-LoRA + NAMM at eval", "LoRA+NAMM", True, "1024", "joint"),
    "A4-off": ("M4-LoRA, NAMM disabled at eval", "LoRA", False, "1024", "joint"),
    "A5-LoRA": ("LoRA + frozen NAMM (train)", "LoRA+NAMM", True, "1024", "frozen_namm"),
    "A5-ES": ("ES + frozen NAMM (train)", "ES+NAMM", True, "1024", "frozen_namm"),
}

COLORS: dict[str, str] = {
    "baseline": "#888888",
    "lora": "#2166ac",
    "lora_light": "#74add1",
    "es": "#f46d43",
    "es_light": "#fdae61",
    "namm": "#1a9850",
    "sequential": "#762a83",
    "joint": "#d73027",
    "frozen_namm": "#01665e",
}

# Path patterns to search relative to experiment_dir (Path.glob syntax)
_EXPERIMENT_PATTERNS: dict[str, list[str]] = {
    "B0": ["**/baseline/results.json"],
    "B1": ["**/es_recency/b1_recency/results.json"],
    "M1-r4": ["**/m1_lora_only/m1_r4/results.json"],
    "M1-r8": ["**/m1_lora_only/m1_r8/results.json"],
    "M1-r16": ["**/m1_lora_only/m1_r16/results.json"],
    "M1-ES": ["**/es_only/m1_es/results.json"],
    "M4-LoRA": ["**/joint_lora/m4_joint_lora/results.json"],
    "M4-ES": ["**/joint_es/m4_joint_es/results.json"],
    "A2-512": ["**/a2_cache/m2_cache512/results.json"],
    "A2-2048": ["**/a2_cache/m2_cache2048/results.json"],
    "A4-on": ["**/a4_modularity/m4_namm_on/results.json"],
    "A4-off": ["**/a4_modularity/m4_namm_off/results.json"],
    "A5-LoRA": [
        "**/rh_m4_frozen/a5_lora_frozen_namm/results.json",
        "**/m1_lora_only/a5_lora_frozen_namm/results.json",
    ],
    "A5-ES": ["**/es_namm/a5_es_frozen_namm/results.json"],
}

# Conditions found by wandb_run_name (searched across all dirs)
_NAMM_RUN_NAMES: dict[str, str] = {
    "M2": "m2_namm_standalone",
    "M3-LoRA": "m3_namm_on_lora",
    "M3-ES": "m3_namm_on_es",
}

# Hydra output subdirectory used by run_namm.py
_NAMM_RUNS_SUBDIR = Path("experiments/namm_only_runs")


# ── Trainable parameter formula ───────────────────────────────────────────────


def _lora_trainable_params(rank: int) -> int:
    """Compute LoRA trainable parameter count for Llama-3.2-1B.

    Uses the formula from experiment_specification.md:
        params = 2 × num_layers × 2 × hidden_size × rank
    (two targets q_proj + v_proj, each with A and B LoRA matrices
    approximated at hidden_size × rank each.)

    Args:
        rank: LoRA rank r.

    Returns:
        Total trainable parameter count.
    """
    return 2 * _LORA_LAYERS * 2 * _LORA_HIDDEN * rank


# ── JSON loading ──────────────────────────────────────────────────────────────


def _safe_load(path: Path) -> dict | None:
    """Load a JSON file, returning None on any error.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict, or None if the file is missing or malformed.
    """
    try:
        return _load_json(path)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None


# ── Metric extraction ─────────────────────────────────────────────────────────


def _normalise_f1(value: float) -> float:
    """Normalise an F1 value to 0–100 scale.

    Args:
        value: Raw F1 value, either 0–1 or 0–100.

    Returns:
        F1 on 0–100 scale.
    """
    return value * 100.0 if value < 1.5 else value


def extract_f1(result: dict | None) -> float | None:
    """Extract Qasper token-level F1 from any results.json schema.

    Supports the standard schema, the run_eval.py eval schema, the
    joint/history schema, and list-valued joint results.

    Args:
        result: Parsed results.json dict, or None.

    Returns:
        F1 score on 0–100 scale, or None if not extractable.
    """
    if result is None:
        return None

    # List schema: joint runs return list of per-loop entries
    if isinstance(result, list):
        return extract_f1(result[-1]) if result else None

    # Standard schema: {"f1": 0.XXX}
    if "f1" in result:
        return _normalise_f1(float(result["f1"]))

    # run_eval.py schema: {"type": "eval", "scores": {"lb/qasper": N}}
    scores = result.get("scores", {})
    if "lb/qasper" in scores:
        return _normalise_f1(float(scores["lb/qasper"]))

    # MemoryTrainer eval JSON: {"lb/qasper": N, "iter": N, ...}
    if "lb/qasper" in result:
        return _normalise_f1(float(result["lb/qasper"]))

    # Joint history schema: {"history": {"eval_scores": [{"lb/qasper": N}]}}
    eval_scores = result.get("history", {}).get("eval_scores", [])
    if eval_scores:
        last_qasper = eval_scores[-1].get("lb/qasper")
        if last_qasper is not None:
            return _normalise_f1(float(last_qasper))

    return None


def extract_exact_match(result: dict | None) -> float | None:
    """Extract exact match score from results.json.

    Args:
        result: Parsed results.json dict, or None.

    Returns:
        Exact match on 0–100 scale, or None.
    """
    if result is None:
        return None
    if isinstance(result, list):
        return extract_exact_match(result[-1]) if result else None
    if "exact_match" in result:
        return _normalise_f1(float(result["exact_match"]))
    return None


def extract_num_samples(result: dict | None) -> int | None:
    """Extract the number of evaluated samples from results.json.

    Args:
        result: Parsed results.json dict, or None.

    Returns:
        Sample count, or None.
    """
    if result is None:
        return None
    if isinstance(result, list):
        return extract_num_samples(result[-1]) if result else None
    for key in ("num_samples", "num_data_samples"):
        if key in result:
            return int(result[key])
    cfg = result.get("config", {})
    if "num_samples" in cfg:
        return int(cfg["num_samples"])
    return None


# ── Result loading ────────────────────────────────────────────────────────────


def _find_by_patterns(base: Path, patterns: list[str]) -> dict | None:
    """Return the parsed JSON from the most recent match of any glob pattern.

    Args:
        base: Root directory to search.
        patterns: Glob patterns to try in order.

    Returns:
        Parsed dict from the most recently modified match, or None.
    """
    candidates: list[tuple[float, Path]] = []
    for pattern in patterns:
        for path in base.glob(pattern):
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        return None
    _, best = max(candidates, key=lambda x: x[0])
    return _safe_load(best)


def _find_by_run_name(search_dirs: list[Path], run_name: str) -> dict | None:
    """Search for a results.json associated with a given wandb run name.

    Checks the file path first (fast), then JSON content (slow fallback).

    Args:
        search_dirs: Directories to search recursively.
        run_name: Run name to match (e.g. 'm2_namm_standalone').

    Returns:
        Parsed dict from the most recently modified match, or None.
    """
    candidates: list[tuple[float, Path]] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for path in base.rglob("results.json"):
            if run_name in str(path):
                candidates.append((path.stat().st_mtime, path))
                continue
            data = _safe_load(path)
            if data is None:
                continue
            cfg = data.get("config", {})
            if run_name in (
                data.get("wandb_run_name", ""),
                data.get("run_name", ""),
                cfg.get("wandb_run_name", ""),
            ):
                candidates.append((path.stat().st_mtime, path))
    if not candidates:
        return None
    _, best = max(candidates, key=lambda x: x[0])
    return _safe_load(best)


def load_all_results(experiment_dir: Path, outputs_dir: Path) -> dict[str, dict | None]:
    """Load all canonical experiment results into a dict.

    Args:
        experiment_dir: Root of the experiments/experiment_N directory.
        outputs_dir: Root of the outputs/ directory (Hydra outputs).

    Returns:
        Dict mapping condition_id to parsed results dict (or None).
    """
    results: dict[str, dict | None] = {}

    for cond_id, patterns in _EXPERIMENT_PATTERNS.items():
        data = _find_by_patterns(experiment_dir, patterns)
        if data is None:
            logger.warning(
                "WARNING: %s — results.json not found (patterns: %s)",
                cond_id,
                patterns,
            )
        results[cond_id] = data

    search_dirs = [experiment_dir, outputs_dir, _NAMM_RUNS_SUBDIR]
    for cond_id, run_name in _NAMM_RUN_NAMES.items():
        data = _find_by_run_name(search_dirs, run_name)
        if data is None:
            logger.warning(
                "WARNING: %s — results.json not found (searched for wandb_run_name=%r in %s)",
                cond_id,
                run_name,
                [str(d) for d in search_dirs],
            )
        results[cond_id] = data

    return results


# ── DataFrame & CSV ───────────────────────────────────────────────────────────


def build_dataframe(results: dict[str, dict | None]) -> pl.DataFrame:
    """Build a polars DataFrame summarising all loaded results.

    Args:
        results: Mapping from condition_id to parsed result dict or None.

    Returns:
        DataFrame with columns: condition_id, condition_name, optimizer,
        namm, cache_size, f1, exact_match, num_samples.
    """
    rows = [
        {
            "condition_id": cid,
            "condition_name": meta[0],
            "optimizer": meta[1],
            "namm": "yes" if meta[2] else "no",
            "cache_size": meta[3],
            "f1": extract_f1(results.get(cid)),
            "exact_match": extract_exact_match(results.get(cid)),
            "num_samples": extract_num_samples(results.get(cid)),
        }
        for cid, meta in CONDITION_META.items()
    ]
    return pl.DataFrame(rows)


# ── Style & figure saving ─────────────────────────────────────────────────────


def setup_style() -> None:
    """Configure matplotlib for a clean academic figure style."""
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> str:
    """Save a figure as both PDF and PNG and close it.

    Args:
        fig: Matplotlib figure to save.
        stem: Output filename stem (no extension).
        out_dir: Output directory.

    Returns:
        PDF output path as a string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    return str(pdf_path)


def _placeholder(message: str) -> plt.Figure:
    """Create a figure displaying a centred placeholder message.

    Args:
        message: Text to display.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="grey",
        style="italic",
    )
    ax.axis("off")
    return fig


# ── Shared plotting helper ────────────────────────────────────────────────────


def _hbar(
    ax: plt.Axes,
    names: list[str],
    values: list[float | None],
    colors: list[str],
    ref_val: float | None = None,
) -> None:
    """Draw a horizontal bar chart.

    Args:
        ax: Target axes.
        names: Y-tick labels (bottom to top order).
        values: F1 values; None renders as a zero bar labelled 'n/a'.
        colors: Bar fill colours, one per entry.
        ref_val: If given, draw a vertical dashed reference line at this x.
    """
    y = np.arange(len(names))
    bar_vals = [v if v is not None else 0.0 for v in values]

    ax.barh(y, bar_vals, color=colors, edgecolor="white", height=0.6)

    for i, (val, bv) in enumerate(zip(values, bar_vals)):
        label = f"{val:.1f}" if val is not None else "n/a"
        ax.text(bv + 0.3, i, label, va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Qasper token-level F1 (0–100)")

    max_val = max((v for v in bar_vals if v > 0), default=50.0)
    ax.set_xlim(0, max_val * 1.18)

    if ref_val is not None:
        ax.axvline(
            ref_val,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label=f"B0 reference ({ref_val:.1f})",
        )
        ax.legend(loc="lower right")


def _get_f1(df: pl.DataFrame, cond_id: str) -> float | None:
    """Return the F1 value for a given condition ID, or None.

    Args:
        df: Summary DataFrame.
        cond_id: Condition identifier.

    Returns:
        F1 score, or None if the condition is missing or has no data.
    """
    rows = df.filter(pl.col("condition_id") == cond_id)
    if rows.is_empty():
        return None
    val = rows["f1"][0]
    return float(val) if val is not None else None


# ── Figures ───────────────────────────────────────────────────────────────────


def fig1_main_results(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 1: main results bar chart (gradient conditions, FAIR-01).

    Args:
        df: Summary DataFrame from build_dataframe.
        out_dir: Output directory for paper figures.

    Returns:
        PDF output path.
    """
    ids = ["B0", "B1", "M1-r8", "M2", "M3-LoRA", "M4-LoRA"]
    names = [CONDITION_META[c][0] for c in ids]
    values = [_get_f1(df, c) for c in ids]
    colors = [COLORS[CONDITION_META[c][4]] for c in ids]

    if all(v is None for v in values):
        logger.warning("WARNING: fig1 — all values missing, producing placeholder")
        return save_figure(
            _placeholder("Fig 1: No data available yet\n(run experiments first)"),
            "fig1_main_results",
            out_dir,
        )

    b0_val = _get_f1(df, "B0")
    fig, ax = plt.subplots(figsize=(7, 4))
    _hbar(ax, names, values, colors, ref_val=b0_val)
    ax.set_title("Figure 1 — Main Results (Gradient Conditions, FAIR-01)")
    fig.tight_layout()
    return save_figure(fig, "fig1_main_results", out_dir)


def fig2_es_results(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 2: ES variants bar chart.

    Args:
        df: Summary DataFrame.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    ids = ["B0", "B1", "M1-ES", "M2", "M3-ES", "M4-ES"]
    names = [CONDITION_META[c][0] for c in ids]
    values = [_get_f1(df, c) for c in ids]
    colors = [COLORS[CONDITION_META[c][4]] for c in ids]

    if all(v is None for v in values):
        logger.warning("WARNING: fig2 — all ES values missing, producing placeholder")
        return save_figure(
            _placeholder(
                "Fig 2: No ES data available yet\n"
                "(† ES is not compute-equivalent to gradient methods)"
            ),
            "fig2_es_results",
            out_dir,
        )

    b0_val = _get_f1(df, "B0")
    fig, ax = plt.subplots(figsize=(7, 4))
    _hbar(ax, names, values, colors, ref_val=b0_val)
    ax.set_title("Figure 2 — ES Variants\n(† Not compute-equivalent to gradient methods)")
    fig.tight_layout()
    return save_figure(fig, "fig2_es_results", out_dir)


def fig3_rank_ablation(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 3: LoRA rank ablation (A1).

    Args:
        df: Summary DataFrame.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    entries = [("M1-r4", 4), ("M1-r8", 8), ("M1-r16", 16)]
    labels = [f"r={r}" for _, r in entries]
    values = [_get_f1(df, cid) for cid, _ in entries]

    if all(v is None for v in values):
        logger.warning("WARNING: fig3 — rank ablation data missing")
        return save_figure(
            _placeholder("Fig 3: No rank ablation data available yet"),
            "fig3_rank_ablation",
            out_dir,
        )

    bar_vals = [v if v is not None else 0.0 for v in values]
    bar_colors = [COLORS["lora_light"], COLORS["lora"], COLORS["lora_light"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, bar_vals, color=bar_colors, edgecolor="white", width=0.5)

    for bar, val, (_, rank) in zip(bars, values, entries):
        text = f"{val:.1f}" if val is not None else "n/a"
        if rank == 8:
            text += " *"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if rank == 8 else "normal",
        )

    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Qasper F1 (0–100)")
    ax.set_ylim(0, max(bar_vals) * 1.18 if any(bar_vals) else 50)
    ax.set_title("Figure 3 — LoRA Rank Ablation (A1)\n(* = main table condition)")
    fig.tight_layout()
    return save_figure(fig, "fig3_rank_ablation", out_dir)


def fig4_cache_sweep(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 4: cache size sweep / accuracy–memory Pareto (A2).

    Args:
        df: Summary DataFrame.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    cache_sizes = [512, 1024, 2048]
    cond_ids = ["A2-512", "M2", "A2-2048"]
    values = [_get_f1(df, cid) for cid in cond_ids]
    b0_val = _get_f1(df, "B0")
    b1_val = _get_f1(df, "B1")

    if all(v is None for v in values):
        logger.warning("WARNING: fig4 — cache sweep data missing")
        return save_figure(
            _placeholder("Fig 4: No cache sweep data available yet"),
            "fig4_cache_sweep",
            out_dir,
        )

    xs = list(range(len(cache_sizes)))
    valid_xy = [(x, v) for x, v in zip(xs, values) if v is not None]

    fig, ax = plt.subplots(figsize=(6, 4))

    if valid_xy:
        vx, vy = zip(*valid_xy)
        ax.plot(
            vx,
            vy,
            color=COLORS["namm"],
            marker="o",
            linewidth=2,
            markersize=8,
            label="NAMM (M2)",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in cache_sizes])
    ax.set_xlabel("Cache size (tokens)")
    ax.set_ylabel("Qasper F1 (0–100)")

    if b0_val is not None:
        ax.axhline(
            b0_val,
            color=COLORS["baseline"],
            linestyle="--",
            linewidth=1.2,
            label=f"B0 full cache ({b0_val:.1f})",
        )
    if b1_val is not None:
        ax.axhline(
            b1_val,
            color=COLORS["baseline"],
            linestyle=":",
            linewidth=1.2,
            label=f"B1 recency ({b1_val:.1f})",
        )
    ax.legend(fontsize=8)

    # Secondary x-axis: compression ratio
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{FULL_CACHE_TOKENS / c:.1f}\u00d7" for c in cache_sizes])
    ax2.set_xlabel("Compression ratio (full / cache)")

    ax.set_title("Figure 4 — Cache Size Sweep / Accuracy–Memory Pareto (A2)")
    fig.tight_layout()
    return save_figure(fig, "fig4_cache_sweep", out_dir)


def fig5_modularity(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 5: modularity test (A4).

    Args:
        df: Summary DataFrame.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    ids = ["M1-r8", "A4-off", "A4-on"]
    labels = [
        "M1-LoRA (r=8)",
        "M4-LoRA\n(NAMM off at eval)",
        "M4-LoRA\n(NAMM on at eval)",
    ]
    colors = [COLORS["lora"], COLORS["joint"], COLORS["joint"]]
    hatches = ["", "//", ""]
    values = [_get_f1(df, cid) for cid in ids]

    if all(v is None for v in values):
        logger.warning("WARNING: fig5 — modularity data missing")
        return save_figure(
            _placeholder("Fig 5: No modularity data available yet"),
            "fig5_modularity",
            out_dir,
        )

    bar_vals = [v if v is not None else 0.0 for v in values]
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for i, (bv, color, hatch) in enumerate(zip(bar_vals, colors, hatches)):
        ax.bar(i, bv, color=color, edgecolor="white", hatch=hatch, width=0.55)
        lbl = f"{values[i]:.1f}" if values[i] is not None else "n/a"
        ax.text(i, bv + 0.2, lbl, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Qasper F1 (0–100)")
    ax.set_ylim(0, max(bar_vals) * 1.18 if any(bar_vals) else 50)
    ax.set_title("Figure 5 — Modularity Test (A4)")
    fig.tight_layout()
    return save_figure(fig, "fig5_modularity", out_dir)


def fig6_frozen_namm(df: pl.DataFrame, out_dir: Path) -> str:
    """Figure 6: frozen NAMM ablation (A5).

    Args:
        df: Summary DataFrame.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    pairs = [
        ("M1-r8", "A5-LoRA", "LoRA", COLORS["lora"], COLORS["frozen_namm"]),
        ("M1-ES", "A5-ES", "ES", COLORS["es"], COLORS["frozen_namm"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    any_data = False

    for ax, (base_id, a5_id, label, base_color, a5_color) in zip(axes, pairs):
        base_f1 = _get_f1(df, base_id)
        a5_f1 = _get_f1(df, a5_id)
        vals = [base_f1, a5_f1]
        bvals = [v if v is not None else 0.0 for v in vals]
        clrs = [base_color, a5_color]
        xlabels = [
            f"{label} only\n(no NAMM train)",
            f"{label} + frozen NAMM\n(train-time)",
        ]

        bars = ax.bar(xlabels, bvals, color=clrs, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            lbl = f"{val:.1f}" if val is not None else "n/a"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                lbl,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("Qasper F1 (0–100)")
        ax.set_ylim(0, max(bvals) * 1.2 if any(bvals) else 50)
        ax.set_title(f"{label} — Frozen NAMM Effect")
        ax.tick_params(axis="x", labelsize=8)
        if any(v is not None for v in vals):
            any_data = True

    if not any_data:
        plt.close(fig)
        logger.warning("WARNING: fig6 — all frozen NAMM data missing")
        fig = _placeholder("Fig 6: No frozen NAMM data available yet")

    fig.suptitle("Figure 6 — Frozen NAMM Ablation (A5)", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, "fig6_frozen_namm", out_dir)


def fig7_namm_curve(search_dirs: list[Path], out_dir: Path) -> str:
    """Figure 7: NAMM training curve (M2 CMA-ES fitness over generations).

    Searches for per-iteration eval JSON files written by MemoryTrainer
    (pattern: ``{out_dir}/eval_{iter}.json``), reconstructs the curve, and
    plots it. Produces a placeholder if no training logs are found.

    Args:
        search_dirs: Directories to search for NAMM eval log files.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    # Collect all eval_NNN.json files from known NAMM output dirs
    eval_files: list[tuple[int, Path]] = []
    eval_pattern = re.compile(r"eval_(\d+)\.json$")

    for base in search_dirs:
        if not base.exists():
            continue
        for path in base.rglob("eval_*.json"):
            m = eval_pattern.search(path.name)
            if m and "m2" in str(path).lower():
                eval_files.append((int(m.group(1)), path))

    eval_files.sort(key=lambda x: x[0])

    if not eval_files:
        logger.warning(
            "WARNING: fig7 — No NAMM per-iteration eval logs found "
            "(MemoryTrainer eval_*.json files). "
            "Run M2 with store_eval_results_locally=true to generate this figure."
        )
        return save_figure(
            _placeholder("Fig 7: NAMM training curve\n(Run M2 to generate)"),
            "fig7_namm_training_curve",
            out_dir,
        )

    iterations, fitness_best, fitness_mean = [], [], []
    for it, path in eval_files:
        data = _safe_load(path)
        if data is None:
            continue
        # MemoryTrainer logs: {"iter": N, "lb/qasper": F1, "val_tasks_aggregate": F}
        f1 = extract_f1(data)
        if f1 is None:
            continue
        iterations.append(it)
        fitness_best.append(f1)
        # Population mean is logged as pop/mean_score or similar
        mean_val = data.get("pop/mean_score") or data.get("mean_score")
        if mean_val is not None:
            fitness_mean.append(_normalise_f1(float(mean_val)))

    if not iterations:
        logger.warning("WARNING: fig7 — eval log files found but no extractable F1")
        return save_figure(
            _placeholder("Fig 7: NAMM training curve\n(No F1 values in eval logs)"),
            "fig7_namm_training_curve",
            out_dir,
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        iterations,
        fitness_best,
        color=COLORS["namm"],
        linewidth=2,
        label="Best F1 (eval checkpoint)",
    )

    if len(fitness_mean) == len(iterations):
        ax.plot(
            iterations,
            fitness_mean,
            color=COLORS["namm"],
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            label="Population mean F1",
        )

    ax.set_xlabel("CMA-ES Generation")
    ax.set_ylabel("Qasper F1 (0–100)")
    ax.set_title("Figure 7 — NAMM Training Curve (M2, Standalone CMA-ES)")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, "fig7_namm_training_curve", out_dir)


def fig8_joint_curve(raw_results: dict[str, dict | None], out_dir: Path) -> str:
    """Figure 8: joint training learning curve (M4-LoRA).

    Reads per-outer-loop eval scores from the M4-LoRA results.json history.

    Args:
        raw_results: Dict from load_all_results.
        out_dir: Output directory.

    Returns:
        PDF output path.
    """
    result = raw_results.get("M4-LoRA")
    eval_scores: list[dict] = []
    if result is not None:
        eval_scores = result.get("history", {}).get("eval_scores", [])

    if not eval_scores:
        logger.warning(
            "WARNING: fig8 — M4-LoRA results.json has no eval_scores history. "
            "Run M4-LoRA with --eval_after_each_loop to generate this figure."
        )
        return save_figure(
            _placeholder("Fig 8: Joint learning curve\n(Run M4-LoRA to generate)"),
            "fig8_joint_learning_curve",
            out_dir,
        )

    loop_f1 = [
        _normalise_f1(float(s["lb/qasper"]))
        if "lb/qasper" in s and s["lb/qasper"] is not None
        else None
        for s in eval_scores
    ]
    loops = list(range(1, len(loop_f1) + 1))
    valid = [(x, y) for x, y in zip(loops, loop_f1) if y is not None]

    fig, ax = plt.subplots(figsize=(6, 4))
    if valid:
        xs, ys = zip(*valid)
        ax.plot(
            xs,
            ys,
            color=COLORS["joint"],
            marker="o",
            linewidth=2,
            markersize=8,
            label="F1 after outer loop",
        )
        ax.set_xticks(loops)
        ax.set_xlim(0.5, len(loops) + 0.5)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.3, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Outer loop index")
    ax.set_ylabel("Qasper F1 (0–100)")
    ax.set_title("Figure 8 — Joint Training Learning Curve (M4-LoRA)")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, "fig8_joint_learning_curve", out_dir)


# ── Table rendering ───────────────────────────────────────────────────────────


def _render_table(
    rows: list[list[str]],
    col_labels: list[str],
    title: str,
    best_col: int | None,
    footnote: str | None = None,
) -> plt.Figure:
    """Render tabular data as a matplotlib figure.

    Args:
        rows: Data rows; each is a list of strings, one per column.
        col_labels: Column header labels.
        title: Figure title shown above the table.
        best_col: Column index whose highest numeric value is bolded and highlighted.
        footnote: Optional footnote text placed below the table.

    Returns:
        Matplotlib figure containing the rendered table.
    """
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_h = max(2.5, 0.38 * (n_rows + 2) + (0.6 if footnote else 0.1))
    fig_w = max(6.0, 1.7 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Identify the best row in best_col
    best_row: int | None = None
    if best_col is not None:
        best_val = -1.0
        for i, row in enumerate(rows):
            try:
                v = float(row[best_col])
                if v > best_val:
                    best_val, best_row = v, i
            except (ValueError, IndexError):
                pass

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))

    for col in range(n_cols):
        cell = table[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for row_i in range(n_rows):
        bg = "#f4f6f8" if row_i % 2 == 0 else "white"
        for col in range(n_cols):
            cell = table[row_i + 1, col]
            cell.set_facecolor(bg)
            if row_i == best_row and col == best_col:
                cell.set_text_props(fontweight="bold", color="#c0392b")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    if footnote:
        fig.text(
            0.02,
            0.01,
            footnote,
            fontsize=7,
            style="italic",
            va="bottom",
        )

    fig.tight_layout()
    return fig


def _fmt(val: float | None) -> str:
    """Format an F1 value for table display.

    Args:
        val: F1 score or None.

    Returns:
        Two-decimal string, or an em dash for missing data.
    """
    return f"{val:.2f}" if val is not None else "\u2014"


def tab1_main_results(df: pl.DataFrame, tables_dir: Path) -> str:
    """Table 1: main results (gradient conditions, FAIR-01).

    Args:
        df: Summary DataFrame.
        tables_dir: Output directory for tables.

    Returns:
        PDF output path.
    """
    ids = ["B0", "B1", "M1-r8", "M2", "M3-LoRA", "M4-LoRA"]
    col_labels = ["Condition", "Fine-tuning", "Memory Policy", "Cache", "Qasper F1"]
    rows = [
        [
            CONDITION_META[cid][0],
            CONDITION_META[cid][1],
            "NAMM" if CONDITION_META[cid][2] else "none",
            CONDITION_META[cid][3],
            _fmt(_get_f1(df, cid)),
        ]
        for cid in ids
    ]
    fig = _render_table(
        rows,
        col_labels,
        "Table 1 \u2014 Main Results (Gradient Conditions, FAIR-01)",
        best_col=4,
    )
    return save_figure(fig, "tab1_main_results", tables_dir)


def tab2_es_results(df: pl.DataFrame, tables_dir: Path) -> str:
    """Table 2: ES variants results.

    Args:
        df: Summary DataFrame.
        tables_dir: Output directory for tables.

    Returns:
        PDF output path.
    """
    ids = ["B0", "B1", "M1-ES", "M2", "M3-ES", "M4-ES"]
    col_labels = ["Condition", "Optimizer", "Memory Policy", "Cache", "Qasper F1"]
    rows = [
        [
            CONDITION_META[cid][0],
            CONDITION_META[cid][1],
            "NAMM" if CONDITION_META[cid][2] else "none",
            CONDITION_META[cid][3],
            _fmt(_get_f1(df, cid)),
        ]
        for cid in ids
    ]
    footnote = (
        "\u2020 ES is not compute-equivalent to gradient methods "
        "(\u224838,400 vs \u22481,600 forward passes); direct comparison is invalid."
    )
    fig = _render_table(
        rows,
        col_labels,
        "Table 2 \u2014 ES Variants",
        best_col=4,
        footnote=footnote,
    )
    return save_figure(fig, "tab2_es_results", tables_dir)


def tab3_rank_ablation(df: pl.DataFrame, tables_dir: Path) -> str:
    """Table 3: LoRA rank ablation (A1).

    Args:
        df: Summary DataFrame.
        tables_dir: Output directory for tables.

    Returns:
        PDF output path.
    """
    entries = [("M1-r4", 4), ("M1-r8", 8), ("M1-r16", 16)]
    col_labels = ["Rank r", "# Trainable Params", "Qasper F1", "Note"]
    rows = [
        [
            str(r),
            f"{_lora_trainable_params(r):,}",
            _fmt(_get_f1(df, cid)),
            "* main condition" if r == 8 else "",
        ]
        for cid, r in entries
    ]
    fig = _render_table(
        rows,
        col_labels,
        "Table 3 \u2014 LoRA Rank Ablation (A1)\n"
        "Params = 2 \u00d7 16 layers \u00d7 2 targets \u00d7 2048 hidden \u00d7 r",
        best_col=2,
    )
    return save_figure(fig, "tab3_rank_ablation", tables_dir)


def tab4_full_results(df: pl.DataFrame, tables_dir: Path) -> str:
    """Table 4: full results across all 17 conditions.

    Args:
        df: Summary DataFrame.
        tables_dir: Output directory for tables.

    Returns:
        PDF output path.
    """
    col_labels = ["ID", "Condition", "Optimizer", "NAMM", "Cache", "F1"]
    rows = [
        [
            cid,
            CONDITION_META[cid][0],
            CONDITION_META[cid][1],
            "yes" if CONDITION_META[cid][2] else "no",
            CONDITION_META[cid][3],
            _fmt(_get_f1(df, cid)),
        ]
        for cid in CONDITION_META
    ]
    fig = _render_table(
        rows,
        col_labels,
        "Table 4 \u2014 Full Results (All Conditions)",
        best_col=5,
    )
    return save_figure(fig, "tab4_full_results", tables_dir)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(experiment_dir: str, outputs_dir: str, paper_figures_dir: str) -> None:
    """Run the full figure and table generation pipeline.

    Args:
        experiment_dir: Path to the experiments/experiment_N directory.
        outputs_dir: Path to the Hydra outputs directory.
        paper_figures_dir: Path to the output directory for all paper assets.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    exp_dir = Path(experiment_dir)
    out_dir = Path(outputs_dir)
    fig_dir = Path(paper_figures_dir)
    tabs_dir = fig_dir / "tables"
    tabs_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all results ──────────────────────────────────────────────────
    logger.info("Loading experiment results from %s ...", exp_dir)
    raw_results = load_all_results(exp_dir, out_dir)

    # ── 2. Build DataFrame and write CSV ─────────────────────────────────────
    logger.info("Building summary DataFrame...")
    df = build_dataframe(raw_results)
    csv_path = fig_dir / "all_results.csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv_path)
    logger.info("Summary CSV: %s", csv_path)

    # Reload from CSV — all figures must read from this single source of truth
    df = pl.read_csv(csv_path)

    # ── 3. Configure matplotlib style ────────────────────────────────────────
    setup_style()

    # ── 4. Generate figures and tables ───────────────────────────────────────
    search_dirs = [exp_dir, out_dir, _NAMM_RUNS_SUBDIR]
    generated: list[str] = []
    failed: list[tuple[str, str]] = []

    def _run(name: str, fn, *args) -> None:
        """Execute a figure/table function and record its outcome."""
        try:
            path = fn(*args)
            generated.append(path)
            logger.info("Generated: %s", path)
        except Exception as exc:
            logger.error("ERROR generating %s: %s", name, exc, exc_info=True)
            failed.append((name, str(exc)))

    _run("fig1", fig1_main_results, df, fig_dir)
    _run("fig2", fig2_es_results, df, fig_dir)
    _run("fig3", fig3_rank_ablation, df, fig_dir)
    _run("fig4", fig4_cache_sweep, df, fig_dir)
    _run("fig5", fig5_modularity, df, fig_dir)
    _run("fig6", fig6_frozen_namm, df, fig_dir)
    _run("fig7", fig7_namm_curve, search_dirs, fig_dir)
    _run("fig8", fig8_joint_curve, raw_results, fig_dir)
    _run("tab1", tab1_main_results, df, tabs_dir)
    _run("tab2", tab2_es_results, df, tabs_dir)
    _run("tab3", tab3_rank_ablation, df, tabs_dir)
    _run("tab4", tab4_full_results, df, tabs_dir)

    # ── 5. Print summary ─────────────────────────────────────────────────────
    n_missing = sum(1 for v in raw_results.values() if v is None)
    sep = "=" * 62

    print(f"\n{sep}")
    print("PAPER FIGURE GENERATION SUMMARY")
    print(sep)
    print(f"\nGenerated ({len(generated)}):")
    for path in generated:
        print(f"  {path}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, reason in failed:
            print(f"  {name}: {reason}")
    if n_missing:
        missing_ids = [cid for cid, v in raw_results.items() if v is None]
        print(
            f"\n{n_missing}/{len(raw_results)} result files not yet available "
            f"(placeholders generated):"
        )
        for cid in missing_ids:
            print(f"  {cid}")
    print(f"\nSummary CSV: {csv_path}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and tables for evo-memory"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments/experiment_1",
        help="Path to experiments/experiment_N directory",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Path to Hydra outputs directory (NAMM training logs)",
    )
    parser.add_argument(
        "--paper_figures_dir",
        type=str,
        default="paper_figures",
        help="Output directory for all figures and tables",
    )
    args = parser.parse_args()
    main(args.experiment_dir, args.outputs_dir, args.paper_figures_dir)
