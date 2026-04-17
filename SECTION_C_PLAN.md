# Section C — Eviction Mask Drift: Plan

**Goal.** For one frozen NAMM checkpoint, compare the eviction masks and raw
scores it produces under three underlying LLMs on the same prompts:
**B0** (base Llama-3.2-1B-Instruct), **M1** (base + M1-LoRA, trained without
NAMM), **M4** (base + M4-LoRA, trained jointly with NAMM).

All three conditions use the same NAMM checkpoint and the same 70-prompt
test split (FAIR-01: `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`,
`min_cond=4096`, `max_cond=6500`). The extended-test 154-prompt split is
produced as an optional appendix artifact.

---

## 1. Concrete understanding of the eviction pipeline

| What | Where |
|---|---|
| Eval entry point | `scripts/eval_namm_splits.py::main` |
| Task sampler + split | `make_task_sampler(...)` → `apply_train_val_test_split(...)`, chat template applied via `apply_chat_template_to_prompts` |
| Model construction | `namm.run_utils.make_eval_model` → builds a `DeepMP` policy wrapping `MemoryLlamaForCausalLM` |
| LLM forward + eviction chunking | `namm/llms/llama.py::forward` — splits prompt at `memory_policy_fixed_delay=256`, calls `memory_policy.update_cache(...)` after every chunk |
| NAMM per-layer eviction core | `namm/policy/deep.py::DeepMP.update_layer_cache_impl_` |
| Scoring network call producing `token_scores` | `deep.py:450` — shape `(1, n_heads=32, n_kv)`, one fp16 scalar per KV cache position per head for this layer |
| Top-k / threshold selection | `deep.py:573` → returns `retained_idxs` `(1, 32, 1024)` and `new_mask` `(1, 32, 1024)` |
| KV gather (actual eviction) | `deep.py:627-630` |
| Aggregate stats recorder (reference for the hook pattern) | `deep.py::record_dynamic_stats`, `record_mask_dynamic_stats`, `record_deep_stats` |

**n_layers = 16, n_heads = 32** (Llama-3.2-1B architecture). At `cache_size=1024`
NAMM selects exactly 1024 KV positions per (layer, head) once the budget binds.
`+protected_tail_n=5` forces the last 5 positions to be retained unconditionally
at every step, so those positions trivially agree across B0/M1/M4 and must be
excluded from IoU/retention metrics to avoid inflating overlap.

**Eviction step count.** For a 6500-token prompt chunked at 256 tokens,
NAMM fires `ceil(6500/256) ≈ 26` times. The final invocation happens at
the end of prefill, immediately before generation would start. The final
mask is the generation-critical one.

---

## 2. Instrumentation changes

### 2.1 `namm/policy/deep.py` — new dump-tensor hook

Add to `DeepMP` (default off, no behavior change when off):

```python
self._record_dump_tensors = False         # flag
# populated by initialize_dump_buffers():
self._dump_per_step   : list[list[dict]]  # per-layer list of per-step dicts
self._dump_final_scores, _retained_idxs, _new_mask, _position_ids
                                          # per-layer, overwritten each step
```

Inside `update_layer_cache_impl_`, after the scoring + selection block (just
before KV gather), when `self._record_dump_tensors is True`, append a
compact per-step record and overwrite the "final" slot for this layer:

```python
# Per-step (tiny — scalars only; accumulated across whole prompt)
step_record = {
    "n_kv": int(n_kv),
    "retained_count_per_head": new_mask.sum(-1).squeeze(0).to(torch.int32).cpu(),  # (32,)
    "score_mean_per_head": token_scores.float().mean(-1).squeeze(0).cpu(),         # (32,)
    "score_std_per_head":  token_scores.float().std(-1).squeeze(0).cpu(),
}
self._dump_per_step[layer_id].append(step_record)

# Final-step only (heavy — full tensors): overwrite every step; the last
# write stays, which is what we want.
self._dump_final_scores[layer_id]        = token_scores.squeeze(0).to(torch.float16).cpu()   # (32, n_kv)
self._dump_final_retained_idxs[layer_id] = retained_idxs.squeeze(0).to(torch.int32).cpu()    # (32, k ≤ 1024)
self._dump_final_new_mask[layer_id]      = new_mask.squeeze(0).bool().cpu()                   # (32, k)
if position_ids is not None:
    self._dump_final_position_ids[layer_id] = position_ids.squeeze(0)[0].to(torch.int32).cpu()  # (n_kv,)
```

Also add `initialize_dump_buffers(self)` / `pop_dump_buffers(self)` (returns
dicts and resets). `pop_dump_buffers` returns a plain dict of tensors (no
torch refs held across prompts).

**Sizes (per prompt, all layers):**
- `final_scores`: 16 × 32 × 6500 × 2 B (fp16) ≈ 6.4 MB
- `final_retained_idxs`: 16 × 32 × 1024 × 4 B ≈ 2 MB
- `final_new_mask`: 16 × 32 × 1024 × 1 B ≈ 0.5 MB
- `final_position_ids`: 16 × 6500 × 4 B ≈ 0.4 MB
- per-step scalars: 16 × 26 × 32 × (3 × 4 B) ≈ 40 KB
- Total: **~9 MB / prompt**

### 2.2 `scripts/eval_namm_splits.py` — `--dump_namm_state` mode

Add one flag:

```
--dump_namm_state <output_dir>
```

When set (and incompatible with `--plain`, `--truncate_input_to`,
`--use_classic_recency`), the script **replaces** the `task_sampler.evaluate`
loop with a per-prompt dump loop modeled on
`scripts/eviction_representation_analysis.py`:

1. Enable `memory_policy._record_dump_tensors = True`.
2. Register forward hooks on every `model.layers[i].self_attn` (to capture
   the LoRA-LLM attention at each split-processing chunk — see
   `AttentionCapture` in `eviction_representation_analysis.py:55`). We only
   keep attention from the FINAL chunk per prompt.
3. Iterate over `get_split_indices('test')` (and `'extended_test'` if
   requested). For each `(task, orig_idx)`:
   - `initialize_dump_buffers()` on the policy; reset attention capture.
   - Tokenize prompt (chat-template already applied); run
     `memory_model(input_ids, attention_mask, apply_memory_policy=True,
      output_attentions=True, use_cache=True)` under `torch.no_grad()`.
   - `pop_dump_buffers()` on the policy; pull the final-chunk LLM attention
     from the capture.
   - Reduce attention → `attn_mean_per_token: (n_layers, n_kv_final) fp16`
     = `attn.mean(head_axis).mean(query_axis)`.
   - Save as `{output_dir}/{task}__{orig_idx:04d}.pt`.

No F1 generation happens in this mode — the run is pure state capture,
which is ~3× faster than full generation eval.

### 2.3 Interaction with existing code

- Existing behaviour is preserved when `--dump_namm_state` is not passed.
- The `+protected_tail_n=5` override is kept (Section C must reproduce the
  same NAMM configuration as the F1 evals).
- No new Hydra config; all parameters come from the existing eval config.

---

## 3. Dump format

**One `.pt` file per (condition × prompt)** at
`eval_results/section_c_dumps/{B0,M1,M4}/{task}__{orig_idx:04d}.pt`. Each file
contains a dict (loaded by `torch.load(..., weights_only=False)`):

```python
{
    # Per-layer tensors at the FINAL eviction step. Stored as LISTS of
    # per-layer tensors (not stacked) because different layers may see
    # different n_kv / k at their final scoring call — DynamicMemoryPolicy
    # .update_cache lets per-layer buffered attention feed different
    # update_new_tokens values into update_layer_cache_impl_, so we cannot
    # assume uniform shapes across layers.
    "final_scores":         list[Tensor[float16]]  length n_layers=16, each (n_heads=8, n_kv_layer),
    "final_retained_idxs":  list[Tensor[int32]]    length n_layers, each (n_heads, k_layer ≤ 1024),
    "final_new_mask":       list[Tensor[bool]]     length n_layers, each (n_heads, k_layer),
    "final_position_ids":   list[Tensor[int32]] | None,  length n_layers, each (n_kv_layer,)  # head-0 slot→prompt map
    "final_retained_positions": list[Tensor[int32]] | None,  length n_layers, each (n_heads, k_layer)  # per-head retained prompt positions — canonical input to IoU/retention

    # LLM-side attention, reduced to per-KV-token scalar per layer.
    "final_attn_mean_per_token": Tensor[float16] (n_layers, n_kv_final),

    # Per-step aggregates.  n_steps varies with prompt length.
    "per_step_n_kv":                    list[int]               length=n_steps,
    "per_step_retained_count_per_head": Tensor[int32]   (n_steps, n_layers, n_heads),
    "per_step_score_mean_per_head":     Tensor[float32] (n_steps, n_layers, n_heads),
    "per_step_score_std_per_head":      Tensor[float32] (n_steps, n_layers, n_heads),

    # Metadata (all JSON-serializable).
    "prompt_meta": {
        "task": str, "orig_idx": int,
        "prompt_length_tokens": int,
        "n_steps": int,
        "protected_tail_n": int,
    },
    "config": {
        "condition": "B0"|"M1"|"M4",
        "llm_id": str,                 # cfg.pretrained_llm._target_ / model name
        "lora_path": str | None,
        "namm_path": str,
        "cache_size": int,
        "run_config": str,
        "timestamp": str,
    },
}
```

**Rationale for not dumping full per-step score tensors.** A 26-step × 16-layer ×
32-head × 6000-token fp16 tensor per prompt is ~200 MB × 210 prompts ≈ 40 GB,
blowing the 5 GB cap. The final step is what determines what enters generation,
and per-step aggregates (count, mean, std) give enough signal for C2 and for
catching a per-step drift pattern. If reviewers later demand full per-step
scores we can re-run with an opt-in `--dump_all_steps` flag and a subset of
prompts; the instrumentation already supports it (just swap "overwrite final"
for "append").

**Retained indices vs full boolean mask.** We store indices not the materialized
boolean keep-mask because the prompt length varies; a padded bool mask over
`max_seq_len=6500` would be wasteful for shorter prompts and ambiguous in the
padding region. The analysis script re-materializes the dense mask on the fly
from `final_retained_idxs` + `final_position_ids`.

---

## 4. Analysis script `scripts/analyze_mask_drift.py`

Single deterministic script that consumes the dumps and writes both the
numeric summary and all four figures. Pseudocode:

```python
def main():
    dumps = load_all_dumps(eval_results/section_c_dumps/)
    # dumps is a dict[condition -> list of per-prompt dicts] aligned by (task, orig_idx).

    metrics = {
        "pairwise_iou": compute_pairwise_iou(dumps),
        # -> per-layer mean/std/n and overall mean for each of
        #    {B0↔M1, B0↔M4, M1↔M4}. Masks are built in prompt-position space
        #    (not KV-index space) from retained_idxs + position_ids. The
        #    protected_tail_n positions are excluded before IoU.
        "per_layer_retention": compute_per_layer_retention(dumps),
        # -> for each condition and each layer: mean fraction of UNIQUE
        #    prompt positions retained by at least one head at the final step
        #    (this varies by layer even at fixed cache_size because different
        #    layers evict different tokens).
        "ks_distance_scores": compute_ks_distance(dumps),
        # -> Kolmogorov-Smirnov distance between score distributions
        #    pairwise, per layer, on the same prompts.
        "spearman_namm_vs_attn": compute_spearman_per_condition(dumps),
        # -> per-prompt Spearman(NAMM_score, attn_mean) averaged over
        #    layers and heads (should reproduce paper 5.4: M1=−0.115,
        #    M4=−0.168 within noise).
        "reference_cross_prompt_iou_B0": cross_prompt_iou_baseline(dumps["B0"]),
        # -> average pairwise IoU of retained-token SETS across different
        #    prompts under B0 (to quantify NAMM's natural prompt-to-prompt
        #    variability; this is the headline baseline for C1).
        "reference_random_baseline_iou": cache_size / mean(prompt_length_at_final_step),
        # -> analytic IoU of two random masks at the same retention rate.
    }

    orjson.dump(metrics -> eval_results/section_c_metrics.json)
    write_csvs(metrics -> eval_results/section_c/*.csv)

    setup_style()  # import from scripts.generate_paper_figures
    plot_C1_mask_overlap(metrics, figures/section_c/C1_mask_overlap)
    plot_C2_retention_by_layer(metrics, figures/section_c/C2_retention_by_layer)
    plot_C3_score_distributions(dumps, figures/section_c/C3_score_distributions)
    plot_C4_iou_by_layer(metrics, figures/section_c/C4_iou_by_layer)
```

**Determinism.** All randomness (subsampling prompts for KDE in C3, pairwise
sample selection for cross-prompt IoU) uses `numpy.random.default_rng(seed=0)`.

**Figure style.** `setup_style()` from `scripts/generate_paper_figures.py`
is reused verbatim; `save_figure()` is reused for dual PDF+PNG output.
Colour mapping: `B0 = COLORS["baseline"]` (grey, `#888888`), `M1 =
COLORS["lora"]` (blue, `#2166ac`), `M4 = COLORS["joint"]` (red, `#d73027`).
This matches Fig. 2/3/4 of the paper. Figure sizes: C1 heatmap `(5, 4)`;
C2/C4 line plots `(6, 3.5)` matching `drift_ratio.pdf`.

**Figures.**
- **C1 (headline).** 3×3 heatmap of pairwise IoU averaged over prompts,
  heads, layers. Annotated with the cross-prompt IoU baseline (text
  annotation in the figure caption and a dashed reference line in the
  colour bar). We render this as a heatmap because there are only three
  conditions; grouped bars would over-emphasize three numbers.
- **C2.** Line plot, x=layer, y=mean fraction of unique prompt positions
  retained at the final step, one line per condition. Error band = ±1 SD
  across prompts.
- **C3 (appendix).** KDE of `final_scores` values across all (layer, head,
  position), one curve per condition, with a vertical dashed line at the
  NAMM eviction threshold (0 for top-k; the k-th-highest score for per-prompt
  top-k). KS distances listed in the caption and in
  `section_c_metrics.json`.
- **C4 (appendix).** `1 − IoU(B0, M4)` as a function of layer. Compared
  visually to `hidden_states.png` (per-layer hidden-state L2 from §5.3) so
  reviewers can see whether mask drift is also concentrated in the last
  ~3 layers.

---

## 5. Eval runs needed

All six runs below use the same `--namm_checkpoint` and split parameters.
The three **Smoke** runs use `--max_prompts_per_task 1` (or equivalent:
a new CLI flag in the dump loop, not a change to the main eval). The three
**Full** runs dump every prompt in the test split.

Exact commands are in §9 at the bottom.

| Run | LoRA | Purpose |
|---|---|---|
| `B0_smoke`, `B0_full`           | none                       | base LLM + NAMM |
| `M1_smoke`, `M1_full`           | `lora_m1_lr1e4_matched/best_ckpt.pt` | M1 LoRA + NAMM |
| `M4_smoke`, `M4_full`           | `lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt` | M4 LoRA + NAMM |

Optional extended-test runs add `--splits test extended_test` to the Full
commands.

---

## 6. Open questions / risk flags

1. **Per-layer retention flatness under hard top-k.** At `cache_size=1024`
   and prompt length > 1024, top-k retains exactly 1024 positions per
   (layer, head) at the final step — so `retained_count_per_layer_per_head`
   is constant across layers. C2 therefore plots the **union-over-heads**
   retention rate, which DOES vary by layer (different heads evict
   different tokens). I flag this in the plan so the figure caption states
   it's the union rate, not the per-head rate.
2. **Protected tail.** `+protected_tail_n=5` forces the last 5 tokens to
   be retained unconditionally. The analysis script excludes these
   positions before computing IoU, score-distribution KS, and retention
   rate. If it didn't, IoU would be inflated by +5/1024 ≈ 0.5% of trivial
   agreement.
3. **Spearman cross-check precision.** §5.4 reports M1 = −0.115, M4 = −0.168
   computed on some (possibly different) prompt set. Our reproduction uses
   the same 70-prompt test split; if the §5.4 paper number came from a
   superset (e.g., val+test) the values may differ by ~0.01–0.02. If they
   differ by more than 0.03, the analysis script will emit a WARN line
   so the user can check whether §5.4 used a different subset.
4. **M1 full-cache asymmetry.** The paper's M1 full-cache F1 (32.6) is
   higher than M1+NAMM cs1024 (19.0), which is the subtext for Section C
   — NAMM hurts M1 because M1 wasn't trained under eviction. For Section C
   we evaluate M1 WITH NAMM active, not its natural setting, because only
   then does "what does NAMM retain" compare apples-to-apples. This is
   consistent with the spec but worth stating in the caption.
5. **Dump size for extended-test.** 154 prompts × 3 conditions × 9 MB ≈
   4.2 GB. Within the 5 GB cap but tight; if this becomes a problem we
   drop `final_scores` dtype to fp8 (int8 quantize with per-(layer, head)
   scale) and cut 50%.
6. **One-prompt-per-task smoke test.** The smoke test skips prompts
   shorter than `cache_size=1024` because NAMM never evicts those
   (`TopKSelection` returns identity when `num_samples <= cache_size`,
   so the dump would contain no eviction information). The
   `min_conditioning_length=4096` filter already guarantees all test
   prompts exceed 1024, so this is not actually a risk here but the
   dump-loop logs a skip reason if it ever happens.

---

## 7. Reproducing §5.4 Spearman from the dumps

For each prompt and each layer `l`:

```python
# final_scores[l] : (n_heads, n_kv) fp16
# final_attn_mean_per_token[l] : (n_kv,) fp16  — mean over heads & queries of the LoRA-LLM's attention
mean_score_over_heads = final_scores[l].mean(dim=0)   # (n_kv,)
rho_l = spearmanr(mean_score_over_heads, final_attn_mean_per_token[l]).statistic
```

Then `rho_M1 = mean(rho_l)` over (prompts, layers). §5.4 reports −0.115 for
M1 and −0.168 for M4; our reproduction should land in those neighbourhoods.
Reported in `section_c_metrics.json["spearman_namm_vs_attn"]`.

Alternative head-wise variant (`final_scores[l, h]` correlated with
`final_attn_mean_per_token[l, h]`) is also computed and stored; whichever
better matches §5.4 is the one cited in the report.

---

## 8. Smoke test

`scripts/smoke_section_c.sh` runs three 2-prompt dumps and the analysis
end-to-end in one shot, on the laptop (CPU path is too slow for cache_size=
1024 though — see §9 for the cluster command). The smoke test exists to
catch shape bugs in the dump → analyze pipeline, not to produce publishable
numbers.

---

## 9. See `SECTION_C_REPORT.md` for interpretation templates.

---

## 10. Commands to run

All commands assume you are at the repo root. Checkpoints are referenced as
environment variables so the same commands work on the cluster and on the
laptop — set the three `*_CKPT` / `*_LORA` env vars once and re-use.

### Cluster paths (headline — this is what §6.x is computed on)

```bash
export NAMM_CKPT=eval_results/namm_cs1024_maskfix/ckpt.pt
export M1_LORA=checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt
export M4_LORA=checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt
export PY=venv/bin/python
```

### Laptop-local equivalents (only for smoke testing — too slow for the full test split)

```bash
export NAMM_CKPT=local_folder/final_cs1024/namm_cs1024.pt
export M1_LORA=local_folder/final_cs1024/m1_lora_matched.pt
# local_folder/final_cs1024/m3_lora_namm.pt is mis-named — torch.load
# confirms it is the M4 LoRA (best_step=260, best_val_score≈52.06 matches
# the cluster file best_ckpt_step260_val52.06.pt).
export M4_LORA=local_folder/final_cs1024/m3_lora_namm.pt
export PY=venv/bin/python
```

### 10.1 Smoke test (1 prompt per QA task × 3 conditions → 15 tiny dumps)

```bash
bash scripts/smoke_section_c.sh
```

The smoke script dumps into `eval_results/section_c_smoke/{B0,M1,M4}/` and
runs the analyzer against that tiny directory. Exits non-zero on any
failure. Safe to blow away the output directory between runs.

### 10.2 Full dumps (70-prompt FAIR-01 test split, three conditions)

These are the dumps §6.x is computed on. They write to
`eval_results/section_c_dumps/{B0,M1,M4}/` and take ≈ 20 min each on a
single A100.

```bash
mkdir -p eval_results/section_c_dumps/{B0,M1,M4}

# B0 — base Llama + NAMM
"$PY" scripts/eval_namm_splits.py \
    --namm_checkpoint "$NAMM_CKPT" \
    --cache_size 1024 \
    --run_config namm_bam_i1_llama32_1b_5t \
    --splits test \
    --dump_namm_state eval_results/section_c_dumps/B0 \
    --dump_condition_label B0

# M1 — base + M1-LoRA (trained without NAMM in the loop) + NAMM at eval
"$PY" scripts/eval_namm_splits.py \
    --namm_checkpoint "$NAMM_CKPT" \
    --lora_checkpoint "$M1_LORA" \
    --cache_size 1024 \
    --run_config namm_bam_i1_llama32_1b_5t \
    --splits test \
    --dump_namm_state eval_results/section_c_dumps/M1 \
    --dump_condition_label M1

# M4 — base + M4-LoRA (jointly trained with NAMM) + NAMM at eval
"$PY" scripts/eval_namm_splits.py \
    --namm_checkpoint "$NAMM_CKPT" \
    --lora_checkpoint "$M4_LORA" \
    --cache_size 1024 \
    --run_config namm_bam_i1_llama32_1b_5t \
    --splits test \
    --dump_namm_state eval_results/section_c_dumps/M4 \
    --dump_condition_label M4
```

### 10.3 Analysis and figures

```bash
"$PY" scripts/analyze_mask_drift.py \
    --dumps_root eval_results/section_c_dumps \
    --metrics_out eval_results/section_c_metrics.json \
    --figures_out figures/section_c
```

Writes `eval_results/section_c_metrics.json` and the four figures
`figures/section_c/C{1,2,3,4}_*.{pdf,png}`. Spearman §5.4 reproduction
is inside the metrics JSON under `spearman_namm_vs_attn`; a discrepancy
of more than ±0.03 from paper values is logged at WARNING level.

### 10.4 Optional — extended-test appendix (154 prompts)

Dumps the larger `extended_test` split alongside `test` so the appendix
sensitivity analysis can use both. Each condition's run takes ~45 min on
A100 and writes ~2 GB per condition. Only worth running if §6.x's C1 /
C2 numbers look borderline.

```bash
for COND in B0 M1 M4; do
    case "$COND" in
        B0) LORA_ARG="" ;;
        M1) LORA_ARG="--lora_checkpoint $M1_LORA" ;;
        M4) LORA_ARG="--lora_checkpoint $M4_LORA" ;;
    esac
    "$PY" scripts/eval_namm_splits.py \
        --namm_checkpoint "$NAMM_CKPT" $LORA_ARG \
        --cache_size 1024 \
        --run_config namm_bam_i1_llama32_1b_5t \
        --splits test extended_test \
        --extended_max_conditioning_length 10000 \
        --filter_by_length 10000 \
        --dump_namm_state eval_results/section_c_dumps_ext/$COND \
        --dump_condition_label "$COND"
done

"$PY" scripts/analyze_mask_drift.py \
    --dumps_root eval_results/section_c_dumps_ext \
    --metrics_out eval_results/section_c_metrics_ext.json \
    --figures_out figures/section_c_ext
```

