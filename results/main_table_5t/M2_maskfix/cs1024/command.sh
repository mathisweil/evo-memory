#!/bin/bash
set -e
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

eval_namm_splits.py --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt --cache_size 1024
