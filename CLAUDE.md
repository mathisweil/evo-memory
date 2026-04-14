# evo-memory

Fine-tuning Llama-3.2-1B-Instruct via LoRA or ES while a NAMM (Cetin et al., ICLR 2025) manages the KV cache. Single-GPU evaluation on a 5-task LongBench QA subset.

## Global constraints

- You MUST run Python through the project's venv (`venv/bin/python` for Python 3.9, `.venv/bin/python` for 3.13) or via `uv run`. You MUST NOT install packages with bare `pip` — use `uv pip install ...`.
- You MUST keep `transformers==4.41.2`, `torch==2.3.1`, `peft==0.11.1`, `numpy<2`. 4.45+ breaks the `DynamicCache` API the codebase relies on.
- You MUST NOT create new top-level directories, new entry-point scripts, or new config files unless the user explicitly asks. Edit existing scripts/configs in place.
- You MUST NOT commit. The user handles all `git commit` / `git push`.
- You MUST NOT modify, rename, or delete anything under `experiments/`, `outputs/`, `exp_local/`, or `experiment_artifacts/` — these are run artifacts, not source.
- You MUST NOT add `print` for diagnostics; use `logger.error` / `logger.info`. Follow the user's global Python guidelines (PEP 8, type hints, polars over pandas, ruff + mypy clean).
- You MUST NOT widen scope: a bug fix touches the bug, not surrounding code; a one-shot script does not get a class hierarchy.
- For any change touching M1/M2/M3/M4 conditions, you MUST preserve FAIR-01 fairness constraints — see `@.claude/rules/training.md`.

## Routing

- Editing `scripts/run_lora.py`, `scripts/run_joint.py`, `scripts/run_es.py`, `grad_lora_finetuning/**`, `es_finetuning/**`, or any `scripts/configs/m{1,3,4}_*.yaml` / `joint_*.yaml` / `es_*.yaml` → read `@.claude/rules/training.md`.
- Editing `scripts/run_namm.py`, `namm/**`, `config/policy/**`, `config/run/namm_*.yaml`, or `config/evolution/**` → read `@.claude/rules/namm.md`.
- Editing `scripts/run_eval.py`, `scripts/eval_namm_splits.py`, `scripts/generate_report.py`, `scripts/generate_paper_figures.py`, `scripts/plot_main_table.py`, or `scripts/configs/eval_default.yaml` → read `@.claude/rules/eval.md`.
- When a coding decision depends on *why* the experiment is structured the way it is (which M-condition a script belongs to, whether a refactor preserves FAIR-01, why eviction is non-differentiable, how to interpret an unexpected eval result, or what the historical `rh_m4_frozen` artefacts in WandB/GCS map to) → load `@.claude/skills/research-context/SKILL.md`.
