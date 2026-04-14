# Analysis — Giacomo

Operational decomposition of NAMM/LoRA results on the existing checkpoints and eval outputs. All analyses use corrected test-set F1 (chat template + protected tail).

## Reports

- **report_benchmark_difficulty** — Stratification of LongBench samples: 40% universally hard, 20% tail-solvable, 26% genuinely hard (where NAMM dominates by +37 F1). The effective arena where NAMM differentiates is ~37% of the benchmark.
- **report_recency_investigation** — Attribution decomposition: LoRA contributes 58-83% of performance depending on budget. NAMM has not learned recency. Cache non-saturation at 85%. Answer position analysis: truncation wins when answer is in the tail, NAMM wins when it's not.
- **report_error_correlation** — Jaccard overlap of failure/success sets across conditions. M4 and Trunc/lora fail on 73% of the same samples at 2048 budget — functionally interchangeable at loose budgets.
- **report_length_scaling** — Test vs extended_test robustness. M1 improves at longer context (+3.2 F1); M3/M4 degrades (-5 to -8). LoRA overfits to training-length distribution.
- **report_error_analysis** — Per-sample error classification (correct/partial/wrong/hallucination/abstention) across all 14 conditions. B0 fails by abstention, not confusion. LoRA fixes refusal rates (57% to 7%); NAMM does not.
