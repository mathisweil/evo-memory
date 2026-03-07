#!/bin/bash
# ==============================================================================
# Cross-evaluation: CMA-ES (beta=2.0) agents on ACCEL buffers and vice versa
#
# For each seed of cmaes_vae_beta2.0:
#   - Evaluate on its OWN buffer (self-eval baseline)
#   - Evaluate on each plain_accel seed's buffer
#
# Evaluates at buffer timesteps 30k, 40k, 50k (matching available checkpoints).
# Results averaged across seeds in the summary CSV.
#
# Usage (run on TPU):
#   bash examples/launch_cross_eval.sh
# ==============================================================================
set -e

BUCKET="ucl-ued-project-bucket"
PREFIX="accel"
SEEDS=(0 1 2)
TIMESTEPS=(30k 40k 50k)
NUM_ATTEMPTS=10
OUTPUT_DIR="results/cross_eval"

CMAES_RUN="cmaes_vae_beta2.0"
ACCEL_RUN="plain_accel"

CKPT_BASE="/tmp/cross_eval_checkpoints"
BUFFER_BASE="/tmp/accel_comparison_data"

mkdir -p "$CKPT_BASE" "$BUFFER_BASE" "$OUTPUT_DIR"

# ==============================================================================
# 1. Download checkpoints (agent params) for cmaes_vae_beta2.0
# ==============================================================================
echo "============================================"
echo "  Downloading CMA-ES (beta=2.0) checkpoints"
echo "============================================"
for seed in "${SEEDS[@]}"; do
    dest="$CKPT_BASE/$CMAES_RUN/$seed"
    if [ -d "$dest/models" ]; then
        echo "  [skip] $CMAES_RUN/seed$seed already downloaded"
    else
        echo "  Downloading $CMAES_RUN/seed$seed..."
        mkdir -p "$dest"
        gcloud storage cp "gs://$BUCKET/$PREFIX/checkpoints/$CMAES_RUN/$seed/config.json" "$dest/config.json"
        gcloud storage cp -r "gs://$BUCKET/$PREFIX/checkpoints/$CMAES_RUN/$seed/models/" "$dest/models/"
    fi
done

# ==============================================================================
# 2. Download buffer dumps for both conditions
# ==============================================================================
echo ""
echo "============================================"
echo "  Downloading buffer dumps"
echo "============================================"
for run_name in "$CMAES_RUN" "$ACCEL_RUN"; do
    for seed in "${SEEDS[@]}"; do
        dest="$BUFFER_BASE/$run_name/$seed"
        if [ -d "$dest" ] && ls "$dest"/buffer_dump_*.npz 1>/dev/null 2>&1; then
            echo "  [skip] $run_name/seed$seed buffers already downloaded"
        else
            echo "  Downloading $run_name/seed$seed buffers..."
            mkdir -p "$dest"
            gcloud storage cp "gs://$BUCKET/$PREFIX/buffer_dumps/$run_name/$seed/buffer_dump_*.npz" "$dest/" 2>/dev/null || \
                echo "  [warn] Some buffer dumps missing for $run_name/seed$seed"
        fi
    done
done

# ==============================================================================
# 3. List what we actually have
# ==============================================================================
echo ""
echo "============================================"
echo "  Available data:"
echo "============================================"
echo "Checkpoints:"
for seed in "${SEEDS[@]}"; do
    dir="$CKPT_BASE/$CMAES_RUN/$seed/models"
    if [ -d "$dir" ]; then
        echo "  $CMAES_RUN/seed$seed: $(ls "$dir" | wc -l) checkpoint steps"
    else
        echo "  $CMAES_RUN/seed$seed: MISSING"
    fi
done
echo ""
echo "Buffer dumps:"
for run_name in "$CMAES_RUN" "$ACCEL_RUN"; do
    for seed in "${SEEDS[@]}"; do
        dir="$BUFFER_BASE/$run_name/$seed"
        if [ -d "$dir" ]; then
            echo "  $run_name/seed$seed: $(ls "$dir"/buffer_dump_*.npz 2>/dev/null | wc -l) dumps"
        else
            echo "  $run_name/seed$seed: MISSING"
        fi
    done
done

# ==============================================================================
# 4. Run cross-evaluations
# ==============================================================================
echo ""
echo "============================================"
echo "  Running cross-evaluations"
echo "============================================"

TOTAL=0
for ts in "${TIMESTEPS[@]}"; do
    for agent_seed in "${SEEDS[@]}"; do
        # Self-eval: CMA-ES agent on CMA-ES buffer
        for buf_seed in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))
        done
        # Cross-eval: CMA-ES agent on ACCEL buffer
        for buf_seed in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

RUN_NUM=0
for ts in "${TIMESTEPS[@]}"; do
    # Convert timestep string "30k" -> update count 30000
    UPDATES=$(echo "$ts" | sed 's/k//' | awk '{print $1 * 1000}')

    echo ""
    echo "--- Timestep: ${ts} (agent @ ~${UPDATES} updates) ---"

    for agent_seed in "${SEEDS[@]}"; do
        agent_dir="$CKPT_BASE/$CMAES_RUN/$agent_seed"

        # Self-eval: CMA-ES agent on CMA-ES buffers
        for buf_seed in "${SEEDS[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))
            buf_npz="$BUFFER_BASE/$CMAES_RUN/$buf_seed/buffer_dump_${ts}.npz"
            if [ ! -f "$buf_npz" ]; then
                echo "  [$RUN_NUM/$TOTAL] SKIP (missing $buf_npz)"
                continue
            fi
            echo "  [$RUN_NUM/$TOTAL] ${CMAES_RUN}/s${agent_seed} -> ${CMAES_RUN}/s${buf_seed} @ ${ts}"
            PYTHONUNBUFFERED=1 python3 examples/cross_evaluate.py \
                --agent_checkpoint_dir "$agent_dir" \
                --agent_updates $UPDATES \
                --buffer_npz "$buf_npz" \
                --num_attempts $NUM_ATTEMPTS \
                --output_dir "$OUTPUT_DIR/${ts}"
        done

        # Cross-eval: CMA-ES agent on ACCEL buffers
        for buf_seed in "${SEEDS[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))
            buf_npz="$BUFFER_BASE/$ACCEL_RUN/$buf_seed/buffer_dump_${ts}.npz"
            if [ ! -f "$buf_npz" ]; then
                echo "  [$RUN_NUM/$TOTAL] SKIP (missing $buf_npz)"
                continue
            fi
            echo "  [$RUN_NUM/$TOTAL] ${CMAES_RUN}/s${agent_seed} -> ${ACCEL_RUN}/s${buf_seed} @ ${ts}"
            PYTHONUNBUFFERED=1 python3 examples/cross_evaluate.py \
                --agent_checkpoint_dir "$agent_dir" \
                --agent_updates $UPDATES \
                --buffer_npz "$buf_npz" \
                --num_attempts $NUM_ATTEMPTS \
                --output_dir "$OUTPUT_DIR/${ts}"
        done
    done
done

# ==============================================================================
# 5. Aggregate results
# ==============================================================================
echo ""
echo "============================================"
echo "  Aggregating results"
echo "============================================"

python3 -c "
import os, glob, numpy as np, csv
from collections import defaultdict

output_dir = '$OUTPUT_DIR'
rows = []
for ts_dir in sorted(glob.glob(os.path.join(output_dir, '*k'))):
    ts = os.path.basename(ts_dir)
    for npz_path in sorted(glob.glob(os.path.join(ts_dir, 'cross_eval_*.npz'))):
        data = dict(np.load(npz_path, allow_pickle=True))
        rows.append({
            'timestep': ts,
            'agent_name': str(data.get('agent_name', '?')),
            'agent_seed': str(data.get('agent_seed', '?')),
            'buffer_run_name': str(data.get('buffer_run_name', '?')),
            'buffer_seed': str(data.get('buffer_seed', '?')),
            'buffer_timestep': str(data.get('buffer_timestep', '?')),
            'mean_solve_rate': float(data['solve_rates'].mean()),
            'num_levels': int(data.get('num_levels', len(data['solve_rates']))),
        })

if not rows:
    print('No results found!')
    exit(0)

# Print per-eval results
print(f\"{'Timestep':>8s} {'Agent':>25s} {'Buffer Run':>22s} {'Buf Seed':>8s} {'Solve%':>8s}\")
print('-' * 75)
for r in rows:
    label = f\"{r['agent_name']}/s{r['agent_seed']}\"
    print(f\"{r['timestep']:>8s} {label:>25s} {r['buffer_run_name']:>22s} {'s'+r['buffer_seed']:>8s} {r['mean_solve_rate']:>7.1%}\")

# Aggregate: mean across all seed combos, grouped by (timestep, buffer_run_name)
print()
print('AVERAGED ACROSS ALL AGENT-SEED x BUFFER-SEED COMBOS:')
print(f\"{'Timestep':>8s} {'Buffer Source':>25s} {'Mean Solve%':>12s} {'Std':>8s} {'N evals':>8s}\")
print('-' * 65)

grouped = defaultdict(list)
for r in rows:
    grouped[(r['timestep'], r['buffer_run_name'])].append(r['mean_solve_rate'])

for (ts, buf_run), rates in sorted(grouped.items()):
    rates = np.array(rates)
    print(f\"{ts:>8s} {buf_run:>25s} {rates.mean():>11.1%} {rates.std():>7.1%} {len(rates):>8d}\")

# Save full CSV
csv_path = os.path.join(output_dir, 'cross_eval_all.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'\nSaved: {csv_path}')
"

echo ""
echo "============================================"
echo "  Cross-evaluation complete!"
echo "  Results in: $OUTPUT_DIR/"
echo "============================================"
