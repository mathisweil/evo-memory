# Recency Investigation: Has NAMM Learned Recency?

## The question

Trunc/lora_2048 (keep last 2048 tokens + M1's LoRA) nearly matches M4/cs2048 (NAMM eviction + jointly trained LoRA). On the extended_test split, Trunc/lora_2048 actually BEATS M4. Is this because:
1. NAMM has essentially learned recency (keeping last N tokens)
2. Recency IS a strong prior for these tasks (answers at document end)
3. LoRA does most of the work and eviction strategy barely matters

## Evidence from this investigation

See the individual analysis files for detailed tables and plots.

### Key outputs
- `recency_profiles_*.png/csv` — Mean recency of kept tokens per condition
- `per_layer_recency_*.png` — Per-layer recency profile
- `cache_saturation_*.png/csv` — Does NAMM fill the cache budget?
- `head_to_head_*.md/csv/png` — Per-sample Trunc/lora vs M4 comparison
- `answer_position_*.png/csv` — Is the gold answer in the truncated tail?
- `lora_attribution_*.md/png` — Decomposition: LoRA vs NAMM contribution
- `per_task_decomposition_*.md` — Which tasks explain Trunc/lora ≈ M4?
