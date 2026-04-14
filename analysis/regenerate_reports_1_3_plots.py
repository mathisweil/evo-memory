#!/usr/bin/env python3
"""Regenerate plots for reports 1, 2, and 3.

Convenience wrapper that runs each report's generate_plots.py script.
Each script is self-contained — this just runs them in sequence.

Run with:
    source activate.sh && PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python \
        analysis/regenerate_reports_1_3_plots.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

SCRIPTS = [
    REPO_ROOT / "analysis" / "report_1" / "scripts" / "generate_plots.py",
    REPO_ROOT / "analysis" / "report_2" / "scripts" / "generate_plots.py",
    REPO_ROOT / "analysis" / "report_3" / "scripts" / "generate_plots.py",
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  Running {script.relative_to(REPO_ROOT)}")
        print(f"{'='*60}")
        result = subprocess.run(
            [PYTHON, str(script)],
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            sys.exit(1)
    print("\nDone. All reports 1-3 plots regenerated.")
