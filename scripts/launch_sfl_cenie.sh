#!/bin/bash
# Launch SFL and CENIE training runs:
#   - 2 seeds each WITHOUT VAE (plain ACCEL with SFL/CENIE scoring)
#   - 2 seeds each WITH VAE (CMA-ES + VAE with SFL/CENIE scoring)
# Total: 8 runs, sequential.
#
# Usage: bash scripts/launch_sfl_cenie.sh

set -e

VAE_DIR="vae/runs/beta2.0"
VAE_GCS="gs://ucl-ued-project-bucket/vae/runs/20260227_215731_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta2.0"
VAE_CKPT="${VAE_DIR}/checkpoint_80000.pkl"
VAE_CONFIG="${VAE_DIR}/config.yaml"

PROJECT="JAXUED_VAE_COMPARISON"
NUM_UPDATES=30000
GCS_BUCKET="ucl-ued-project-bucket"
GCS_PREFIX="accel"

COMMON_ARGS="--project $PROJECT --gcs_bucket $GCS_BUCKET --gcs_prefix $GCS_PREFIX --buffer_dump_interval 10000 --num_updates $NUM_UPDATES"

# --- Download VAE if needed ---
mkdir -p "${VAE_DIR}"
if [ ! -f "${VAE_CKPT}" ]; then
    echo "[Setup] Downloading VAE checkpoint..."
    gcloud storage cp "${VAE_GCS}/checkpoints/checkpoint_80000.pkl" "${VAE_CKPT}"
fi
if [ ! -f "${VAE_CONFIG}" ]; then
    echo "[Setup] Downloading VAE config..."
    gcloud storage cp "${VAE_GCS}/config.yaml" "${VAE_CONFIG}"
fi

echo "============================================"
echo "  Launching SFL + CENIE training (8 runs)"
echo "  VAE: ${VAE_CKPT}"
echo "  Updates: ${NUM_UPDATES}"
echo "============================================"

# =============================================
# PART 1: Plain ACCEL (no VAE) with SFL/CENIE
# =============================================

# --- ACCEL + SFL (2 seeds) ---
for SEED in 1 2; do
    echo ""
    echo ">>> ACCEL+SFL seed=${SEED} starting..."
    python examples/maze_plr.py \
        --seed $SEED \
        --run_name accel_sfl \
        --use_accel \
        --score_function sfl \
        --num_sfl_rollouts 10 \
        $COMMON_ARGS
    echo ">>> ACCEL+SFL seed=${SEED} done."
done

# --- ACCEL + CENIE (2 seeds) ---
for SEED in 1 2; do
    echo ""
    echo ">>> ACCEL+CENIE seed=${SEED} starting..."
    python examples/maze_plr.py \
        --seed $SEED \
        --run_name accel_cenie \
        --use_accel \
        --score_function cenie \
        --cenie_alpha 0.5 \
        --cenie_buffer_size 50000 \
        --cenie_num_components 10 \
        --cenie_refit_interval 5 \
        $COMMON_ARGS
    echo ">>> ACCEL+CENIE seed=${SEED} done."
done

# =============================================
# PART 2: CMA-ES + VAE with SFL/CENIE
# =============================================

VAE_ARGS="--use_cmaes --cmaes_sigma_init 1.0 --cmaes_reset_interval 500 --save_cmaes_populations --vae_checkpoint_path $VAE_CKPT --vae_config_path $VAE_CONFIG"

# --- CMA-ES+VAE + SFL (2 seeds) ---
for SEED in 1 2; do
    echo ""
    echo ">>> CMA-ES+VAE+SFL seed=${SEED} starting..."
    python examples/maze_plr.py \
        --seed $SEED \
        --run_name cmaes_vae_beta2.0_sfl \
        --score_function sfl \
        --num_sfl_rollouts 10 \
        $VAE_ARGS \
        $COMMON_ARGS
    echo ">>> CMA-ES+VAE+SFL seed=${SEED} done."
done

# --- CMA-ES+VAE + CENIE (2 seeds) ---
for SEED in 1 2; do
    echo ""
    echo ">>> CMA-ES+VAE+CENIE seed=${SEED} starting..."
    python examples/maze_plr.py \
        --seed $SEED \
        --run_name cmaes_vae_beta2.0_cenie \
        --score_function cenie \
        --cenie_alpha 0.5 \
        --cenie_buffer_size 50000 \
        --cenie_num_components 10 \
        --cenie_refit_interval 5 \
        $VAE_ARGS \
        $COMMON_ARGS
    echo ">>> CMA-ES+VAE+CENIE seed=${SEED} done."
done

echo ""
echo "============================================"
echo "  All 8 runs complete!"
echo "============================================"
