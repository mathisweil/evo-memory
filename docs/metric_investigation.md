# Val vs Test F1 Investigation (B3)

**Date:** 2026-04-14

The M1 and M3 completed runs show large val-to-test drops (val F1 ≈ 45 → test
F1 ≈ 31). Before re-running M3, this document traces whether that gap is a
measurement artefact or real generalisation loss.

---

## Are val and test F1 the same metric?

**Yes.** Both call the same `task_sampler.evaluate(..., train=False, ...)`
function, which in turn calls the per-task LongBench F1 defined in
`namm/evaluation/metrics.py` and registered in
`namm/evaluation/longbench.py:20-44`.

- Val F1 during training: `grad_lora_finetuning/trainer.py:761-792`
  (`_evaluate_f1(split='val')`).
- Test F1 at report time: `scripts/run_eval.py` calls the same
  `task_sampler.evaluate` path.

Both compute a **sample-weighted average** of per-task F1:

```
weighted_sum += lb/<task>_f1 * n_samples_in_task
avg_f1 = weighted_sum / total_samples
```

Normalisation (`normalize_answer`: lowercase, strip articles, strip punctuation,
whitespace-collapse) is identical for both. Tokenization for F1 is
whitespace-split on both paths. There is no micro-vs-macro discrepancy.

## Is temperature the same?

**Yes.** Generation for val eval uses the same `evaluation_model` object as
test eval, and greedy decoding (`temperature=0.0`) is enforced inside
`task_sampler.evaluate(..., train=False, evolved_model=False, pop_reps=1)`.
The run config (`config/run/namm_bam_i1_llama32_1b_5t.yaml`) sets
`temperature: 0.0` and `do_sample` follows from it. Both val and test paths
read the same config; no branch applies sampling to one and greedy to the
other.

## Is NAMM active on both paths?

**Yes, identically.** During training-time val eval, the memory policy is set
with `batch_idxs = np.zeros([1])` (`trainer.py:767-768`). During test eval,
`run_eval.py` does the same thing. M3 val and test both see NAMM eviction with
`cache_size=1024`.

## Split sizes

From FAIR-01: `train=306`, `val=64`, `test=69` after length filtering.

## Conclusion

The val→test drop is **not** a measurement artefact. All three of (metric,
decoding temperature, NAMM state) are identical on both paths. The gap is real
generalisation loss on a small (64) val split:

1. `always_save_checkpoint: true` + `eval_interval: 14` mean training saves the
   single best val F1 across ~50 evaluations. With 64 val samples, the
   best-of-50 max is an upward-biased estimator of generalisation.
2. 64 samples → roughly ±5 F1 noise per eval. The reported val F1 is the peak
   of a noisy series; the corresponding test F1 is a single point measurement
   on a disjoint 69-sample set, both small enough to have similar variance but
   the test set does not get "best-of" selection.

## Implication for the paper

- Report **test F1** as the headline metric (already the convention in
  `run_eval.py` output). Val F1 should not appear in the main results table.
- When reporting val F1 internally (e.g., for checkpoint selection sanity),
  note it as "best-of-N selection estimate" rather than "val performance".
- The val-test gap is not a bug in M1 or M3 code — it is a consequence of the
  small val split plus best-checkpoint selection. Do not "fix" it by changing
  metrics.
