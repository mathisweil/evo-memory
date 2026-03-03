#!/usr/bin/env bash
# Run this on the GPU VM to set everything up.
#
# Auto-detect first GPU:
#   bash setup.sh
#
# Pin to a specific GPU (e.g. GPU 2 on a multi-GPU machine):
#   bash setup.sh --gpu 2
#
# Or one-liner from GitHub (auto-detect GPU):
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/setup.sh | bash
#
# One-liner pinned to GPU 1:
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/setup.sh | bash -s -- --gpu 1
