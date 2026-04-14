#!/bin/bash
set -e
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

eval_namm_splits.py --lora_checkpoint M4_maskfix --cache_size 8192
