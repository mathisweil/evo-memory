# Analysis 8 -- Probing for Residual Knowledge of Evicted Content

## Status: Abandoned

This analysis attempted to train linear probes on per-layer hidden states
to detect whether gold-answer tokens had been evicted by NAMM. The probe
label was constructed by tokenising the gold answer string from LongBench,
scanning the input prompt for exact substring matches (full answer and
individual words), and checking whether those token positions survived
eviction.

The approach was flawed for the same reason the "relevant tokens" analysis
was dropped from Report 0: we do not have ground truth for which input
tokens are needed to answer the question. String matching finds verbatim
occurrences of the answer text in the context, but the information required
to produce an answer often appears in paraphrased, indirect, or distributed
form. The labels were unreliable, and the probe results were inconclusive
(both M1 and M3 at or below the 0.60 majority-class baseline).

A meaningful version of this analysis would need either (a) a
ground-truth annotation of which context spans support each answer, or
(b) a reformulation that sidesteps the token-identification problem
entirely — e.g., probing for the answer itself rather than for the
presence of specific tokens.
