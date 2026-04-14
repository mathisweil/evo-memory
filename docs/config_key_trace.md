# Config Key Trace (B1)

**Date:** 2026-04-14

Per-config verification that every YAML key maps to a script parameter that is
actually read. For the exhaustive key-by-key argparse table, see
`docs/config_code_audit.md` — this document records only the deltas and
non-obvious paths that audit missed.

---

## Summary

| Config | Script | Dead keys | Missing keys | Notes |
|---|---|---|---|---|
| `lora_rh_m1_instruct_5t.yaml` | `run_lora.py` | none | none | All keys → argparse (see `config_code_audit.md`). `min/max_conditioning_length` are intentionally NOT in this YAML; they come from the Hydra `run_config` preset. |
| `lora_rh_m4_instruct_5t.yaml` | `run_lora.py` | none | none | Same surface as M1. |
| `joint_lora_m4_5t.yaml` | `run_joint.py` | none | none | `min/max_conditioning_length`, `max_answer_tokens` are read via argparse by `run_joint.py` (it has its own copies; they do NOT come from Hydra for joint). |
| `joint_default.yaml` | `run_joint.py` | none | none | Identical key surface as `joint_lora_m4_5t.yaml`. |
| `eval_default.yaml` | `run_eval.py` | none | none | `batch_size: null` and `cache_size: null` resolve to the Hydra `run_config` defaults. |
| `eval_main_table.yaml` | `run_eval.py` | none | none | All values overridable via CLI for per-condition runs. |

No dead keys were found. No missing argparse flags were found.

---

## Non-obvious paths (worth understanding)

### `min_conditioning_length` / `max_conditioning_length` in `run_lora.py`

The LoRA YAMLs (`lora_rh_m1_instruct_5t.yaml`, `lora_rh_m4_instruct_5t.yaml`)
do **not** contain these keys, and `run_lora.py` does **not** define argparse
flags for them. They are read from the Hydra `cfg` object:

- `scripts/run_lora.py:254-255` — `cfg.get('max_conditioning_length', 6500)` /
  `cfg.get('min_conditioning_length', None)` for the 3-way split.
- `scripts/run_lora.py:320-321` — same values flow into `LoRATrainerConfig`.

The effective values come from `config/run/namm_bam_i1_llama32_1b_5t.yaml`:
- `min_conditioning_length: 4096`
- `max_conditioning_length: 6500`

Adding these keys to `scripts/configs/lora_*.yaml` would NOT reach the script
(argparse would reject unknown keys via `set_defaults`). FAIR-01 compliance is
enforced entirely via the Hydra preset.

### `min_conditioning_length` / `max_conditioning_length` in `run_joint.py`

Unlike `run_lora.py`, `run_joint.py` does define argparse flags for these
(because the joint loop builds its own `LoRATrainerConfig` in `_run_lora_stage`
at `scripts/run_joint.py:765-766`). So `joint_*.yaml` configs DO carry them
explicitly — and must keep them at 4096/6500 for FAIR-01.

### `sft_mode: true` in YAML vs `action="store_true"` in argparse

`--sft_mode` is defined as `action="store_true"` with `default=False`. The YAML
`sft_mode: true` overrides the default via `set_defaults` during config
loading, so the behaviour is correct. But `--sft_mode false` on the CLI would
NOT disable it (store_true does not accept values). All current configs set
`sft_mode: true` and this is never an issue in practice; flagged here only to
note the asymmetry.

### `batch_size_eval: null` in eval configs

Both `eval_default.yaml` and `eval_main_table.yaml` have `batch_size: null`.
This resolves to the `batch_size` declared in the Hydra `run_config` YAML
(`namm_bam_i1_llama32_1b_5t.yaml` sets `batch_size: 4`). This is intentional:
eval batch size is a Hydra-side concern, not a per-run CLI concern.

---

## Keys in LoRA configs that are set explicitly but equal the argparse default

None of these are bugs — they make the YAML the single source of truth — but
they are listed for completeness:

| Config | Key | Value | Script default |
|---|---|---|---|
| `lora_rh_m1_instruct_5t.yaml` | `split_seed` | 42 | 42 |
| `lora_rh_m1_instruct_5t.yaml` | `lora_rank` | 8 | 8 |
| `lora_rh_m4_instruct_5t.yaml` | `split_seed` | 42 | 42 (added by C1 fix) |
| `lora_rh_m4_instruct_5t.yaml` | `lora_rank` | 8 | 8 |

After Part C fixes, every LoRA hyperparameter that matters for reproducibility
is stated explicitly in the YAML.
