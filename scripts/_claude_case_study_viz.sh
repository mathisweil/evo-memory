#!/bin/bash
set -u
PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

python scripts/case_study_visualization.py \
    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
    --lora_m4 results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --cache_size 1024 \
    --cases "lb/qasper_e:126:extended_test" "lb/2wikimqa:95:extended_test" "lb/qasper_e:180:test" \
    --output_dir analysis_out/case_studies 2>&1
