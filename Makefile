# =============================================================================
# evo-memory — Makefile
#
# Python deps live in pyproject.toml.  This file handles:
#   1. Platform-specific PyTorch installation (different index per hardware)
#   2. Shared project install via uv
#   3. Service logins (HuggingFace, wandb, GCS)
#   4. TPU VM lifecycle
#   5. Activation-script generation
#
# Prerequisites: python 3.10+, uv (https://docs.astral.sh/uv/)
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Quick start:
#   make setup-local          # CPU / local GPU
#   make setup-gpu GPU=2      # pin to a specific CUDA device
#   make setup-tpu            # Google Cloud TPU VM
#   make setup-ucl-gpu        # UCL CSH cluster
#   make help                 # show all targets
# =============================================================================

SHELL       := /bin/bash
.DEFAULT_GOAL := help

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR    := $(CURDIR)
VENV_DIR    := $(REPO_DIR)/.venv
STAMPS      := $(REPO_DIR)/.make
BIN         := $(VENV_DIR)/bin

# ── Tunables ──────────────────────────────────────────────────────────────────
GPU         ?=
GCS_BUCKET  ?= statistical-nlp
GCS_PROJECT ?= statistical-nlp
HF_CACHE    := $(REPO_DIR)/.hf_cache
XLA_CACHE   := $(REPO_DIR)/.xla_cache
PYTHON_VER  ?= 3.10

# ── PyTorch version pins (single source — used by every target) ──────────────
TORCH       := torch==2.3.1
TORCHVISION := torchvision==0.18.1
TORCHAUDIO  := torchaudio==2.3.1

CUDA_INDEX  := https://download.pytorch.org/whl/cu121
PYPI_INDEX  := https://pypi.org/simple/
TPU_LINKS   := https://storage.googleapis.com/libtpu-releases/index.html

# ── Tool checks ──────────────────────────────────────────────────────────────
UV := $(shell command -v uv 2>/dev/null)

.PHONY: _require-uv
_require-uv:
ifndef UV
	$(error uv is not installed. Run: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

# =============================================================================
#  Internal: venv + base deps
# =============================================================================

$(STAMPS):
	@mkdir -p $@

# Create the venv via uv (respects .python-version or PYTHON_VER).
$(STAMPS)/venv: | _require-uv $(STAMPS)
	@echo "Creating virtualenv at $(VENV_DIR) ..."
	uv venv $(VENV_DIR) --python $(PYTHON_VER)
	@touch $@

# Install the project in editable mode (all shared deps from pyproject.toml).
# Torch must already be present — each setup-* target installs it first.
$(STAMPS)/project: $(STAMPS)/venv pyproject.toml | $(STAMPS)
	@echo "Installing project (editable) ..."
	uv pip install --python $(BIN)/python -e "."
	@touch $@

# =============================================================================
#  Public setup targets — one per hardware environment
# =============================================================================

.PHONY: setup-gpu
setup-gpu: $(STAMPS)/venv  ## CUDA GPU (use GPU=N to pin a device)
	@echo "============================================================"
	@echo " evo-memory — GPU setup (CUDA 12.1)"
	@echo "============================================================"
	@echo "Installing PyTorch (cu121) ..."
	uv pip install --python $(BIN)/python \
	    $(TORCH) $(TORCHVISION) $(TORCHAUDIO) \
	    --index-url $(CUDA_INDEX) --extra-index-url $(PYPI_INDEX)
	@$(MAKE) --no-print-directory $(STAMPS)/project
	@if [ -n "$(GPU)" ]; then \
	    echo "CUDA_VISIBLE_DEVICES=$(GPU)" >> $(REPO_DIR)/.env; \
	    echo "Pinned to GPU $(GPU)."; \
	elif command -v nvidia-smi &>/dev/null; then \
	    echo "Detected GPU $$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')."; \
	fi
	@$(MAKE) --no-print-directory _activate-scripts
	@$(MAKE) --no-print-directory _verify
	@$(MAKE) --no-print-directory _done MODE=gpu

.PHONY: setup-tpu
setup-tpu: $(STAMPS)/venv  ## Google Cloud TPU VM (PyTorch + XLA)
	@echo "============================================================"
	@echo " evo-memory — TPU setup"
	@echo "============================================================"
	@# System deps (best-effort)
	@if sudo -n true 2>/dev/null; then \
	    echo "Installing system packages ..."; \
	    sudo apt-get update -qq; \
	    sudo apt-get install -y -qq libopenblas-dev >/dev/null 2>&1; \
	fi
	@echo "Installing PyTorch + XLA ..."
	uv pip install --python $(BIN)/python \
	    $(TORCH) "torch_xla[tpu]" \
	    --find-links $(TPU_LINKS)
	@$(MAKE) --no-print-directory $(STAMPS)/project
	@mkdir -p $(XLA_CACHE)
	@echo "Syncing XLA cache from GCS ..."
	@if command -v gsutil &>/dev/null; then \
	    gsutil -m rsync -r "gs://$(GCS_BUCKET)/xla_cache" $(XLA_CACHE) 2>/dev/null || true; \
	else \
	    echo "  gsutil not found — skipping."; \
	fi
	@$(MAKE) --no-print-directory _activate-scripts
	@$(MAKE) --no-print-directory _verify
	@$(MAKE) --no-print-directory _done MODE=tpu

.PHONY: setup-ucl-gpu
setup-ucl-gpu: $(STAMPS)/venv  ## UCL CSH cluster (GPU, skips GCS/wandb)
	@echo "============================================================"
	@echo " evo-memory — UCL GPU setup"
	@echo "============================================================"
	@echo "Loading CUDA module (if available) ..."
	@bash -c 'source /etc/profile.d/modules.sh 2>/dev/null && module load cuda 2>/dev/null' || true
	@echo "Installing PyTorch (cu121) ..."
	uv pip install --python $(BIN)/python \
	    $(TORCH) $(TORCHVISION) $(TORCHAUDIO) \
	    --index-url $(CUDA_INDEX) --extra-index-url $(PYPI_INDEX)
	@$(MAKE) --no-print-directory $(STAMPS)/project
	@$(MAKE) --no-print-directory _activate-scripts
	@$(MAKE) --no-print-directory _verify
	@echo ""
	@echo "Setup complete. Activate with:"
	@echo "  source activate.sh     # bash/zsh"
	@echo "  source activate.csh    # csh/tcsh"
	@echo ""
	@echo "Then run:  huggingface-cli login"

.PHONY: setup-local
setup-local: $(STAMPS)/venv  ## Local / CPU-only (also works if CUDA is present)
	@echo "============================================================"
	@echo " evo-memory — local setup"
	@echo "============================================================"
	@echo "Installing PyTorch (default index) ..."
	uv pip install --python $(BIN)/python \
	    $(TORCH) $(TORCHVISION) $(TORCHAUDIO)
	@$(MAKE) --no-print-directory $(STAMPS)/project
	@$(MAKE) --no-print-directory _activate-scripts
	@$(MAKE) --no-print-directory _verify
	@$(MAKE) --no-print-directory _done MODE=local

# =============================================================================
#  Service logins — run individually or together via `make logins`
# =============================================================================

.PHONY: hf-login
hf-login:  ## Log in to HuggingFace (required for gated LLaMA 3.2)
	@if $(BIN)/python -c \
	    "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then \
	    echo "Already logged into HuggingFace."; \
	else \
	    echo "Token page: https://huggingface.co/settings/tokens"; \
	    $(BIN)/huggingface-cli login; \
	fi

.PHONY: wandb-login
wandb-login:  ## Log in to Weights & Biases
	@if $(BIN)/python -c "import wandb; wandb.api.api_key" 2>/dev/null; then \
	    echo "Already logged into wandb."; \
	else \
	    echo "API key: https://wandb.ai/authorize"; \
	    $(BIN)/wandb login || true; \
	fi

.PHONY: gcs-auth
gcs-auth:  ## Authenticate with Google Cloud Storage
	@if ! command -v gcloud &>/dev/null; then \
	    echo "gcloud CLI not found — installing ..."; \
	    curl -fsSL https://sdk.cloud.google.com \
	        | bash -s -- --disable-prompts --install-dir="$$HOME/.local" 2>&1 | tail -5; \
	    export PATH="$$HOME/.local/google-cloud-sdk/bin:$$PATH"; \
	fi; \
	if gcloud auth application-default print-access-token &>/dev/null 2>&1; then \
	    echo "Already authenticated with GCS."; \
	else \
	    gcloud auth application-default login \
	        --project $(GCS_PROJECT) --no-launch-browser; \
	fi; \
	if gsutil ls "gs://$(GCS_BUCKET)/" &>/dev/null 2>&1; then \
	    echo "gs://$(GCS_BUCKET)/ accessible."; \
	else \
	    echo "WARNING: cannot access gs://$(GCS_BUCKET)/."; \
	fi

.PHONY: install-claude
install-claude:  ## Install Claude Code CLI
	@if command -v claude &>/dev/null; then \
	    echo "Claude Code already installed ($$(claude --version 2>/dev/null || echo 'unknown'))."; \
	else \
	    if command -v npm &>/dev/null; then \
	        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3; \
	    else \
	        echo "Node.js not found — installing via nvm ..."; \
	        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash; \
	        export NVM_DIR="$$HOME/.nvm"; \
	        . "$$NVM_DIR/nvm.sh"; \
	        nvm install --lts; \
	        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3; \
	    fi; \
	    echo "Done."; \
	fi

.PHONY: logins
logins: hf-login wandb-login gcs-auth  ## Run all three service logins

# =============================================================================
#  TPU VM lifecycle
# =============================================================================

.PHONY: tpu-restart-v6e
tpu-restart-v6e:  ## Restart preempted spot v6e-8 VM (europe-west4-a)
	@$(MAKE) --no-print-directory _tpu-restart \
	    TPU_NAME=hyperscale-v6e ZONE=europe-west4-a \
	    ACCEL=v6e-8 RUNTIME=v2-alpha-tpuv6e \
	    SSH_HOST=gcp-tpu-v6e FLAGS="--spot --preemptible"

.PHONY: tpu-restart-v4
tpu-restart-v4:  ## Restart on-demand v4-8 VM (us-central2-b)
	@$(MAKE) --no-print-directory _tpu-restart \
	    TPU_NAME=hyperscale-v4 ZONE=us-central2-b \
	    ACCEL=v4-8 RUNTIME=tpu-vm-tf-2.17.0-pjrt \
	    SSH_HOST=gcp-tpu-v4 FLAGS=""

.PHONY: _tpu-restart
_tpu-restart:
	@echo "TPU: $(TPU_NAME) / $(ZONE) / $(ACCEL)"
	@STATUS=$$(gcloud compute tpus tpu-vm describe $(TPU_NAME) \
	    --zone=$(ZONE) --format="get(state)" 2>/dev/null || echo "NOT_FOUND"); \
	HEALTH=$$(gcloud compute tpus tpu-vm describe $(TPU_NAME) \
	    --zone=$(ZONE) --format="get(health)" 2>/dev/null || echo "UNKNOWN"); \
	echo "Status: $${STATUS}  Health: $${HEALTH}"; \
	if [ "$${STATUS}" = "READY" ] && [ "$${HEALTH}" != "UNHEALTHY_MAINTENANCE" ]; then \
	    IP=$$(gcloud compute tpus tpu-vm describe $(TPU_NAME) --zone=$(ZONE) \
	        --format="get(networkEndpoints[0].accessConfig.externalIp)"); \
	    echo "Already running at $${IP}"; \
	    sed -i '' "/Host $(SSH_HOST)$$/,/HostName /{s/HostName .*/HostName $${IP}/;}" $$HOME/.ssh/config; \
	    echo "SSH config updated. Run: ssh $(SSH_HOST)"; exit 0; \
	fi; \
	if [ "$${STATUS}" != "NOT_FOUND" ]; then \
	    echo "Deleting (state=$${STATUS}) ..."; \
	    gcloud compute tpus tpu-vm delete $(TPU_NAME) --zone=$(ZONE) --quiet; \
	fi; \
	echo "Creating TPU ..."; \
	gcloud compute tpus tpu-vm create $(TPU_NAME) \
	    --zone=$(ZONE) --accelerator-type=$(ACCEL) --version=$(RUNTIME) $(FLAGS); \
	IP=$$(gcloud compute tpus tpu-vm describe $(TPU_NAME) --zone=$(ZONE) \
	    --format="get(networkEndpoints[0].accessConfig.externalIp)"); \
	echo "New IP: $${IP}"; \
	sed -i '' "/Host $(SSH_HOST)$$/,/HostName /{s/HostName .*/HostName $${IP}/;}" $$HOME/.ssh/config; \
	ssh-keygen -R "$${IP}" 2>/dev/null || true; \
	gcloud compute tpus tpu-vm ssh $(TPU_NAME) --zone=$(ZONE) --command="echo 'SSH OK'"; \
	echo "Done. Run: ssh $(SSH_HOST)"

# =============================================================================
#  Smoke test
# =============================================================================

.PHONY: smoke
smoke:  ## Quick sanity check (ES, no NAMM, 2 iterations)
	$(BIN)/python scripts/run_es.py \
	    --run_name smoke --num_iterations 2 \
	    --population_size 2 --mini_batch_size 2 --no-gcs

# =============================================================================
#  Lock / sync (for CI or reproducible installs)
# =============================================================================

.PHONY: lock
lock: _require-uv  ## Regenerate uv.lock from pyproject.toml
	uv lock

# =============================================================================
#  Cleanup
# =============================================================================

.PHONY: clean
clean:  ## Remove venv, caches, stamps, and generated scripts
	rm -rf $(VENV_DIR) $(STAMPS) $(XLA_CACHE)
	rm -f  $(REPO_DIR)/activate.sh $(REPO_DIR)/activate.csh
	@echo "Cleaned. Re-run your setup-* target to rebuild."

.PHONY: clean-cache
clean-cache:  ## Remove HF + XLA caches only (keeps venv)
	rm -rf $(HF_CACHE) $(XLA_CACHE)
	@echo "Caches removed."

# =============================================================================
#  Internal helpers
# =============================================================================

.PHONY: _activate-scripts
_activate-scripts: $(REPO_DIR)/activate.sh $(REPO_DIR)/activate.csh

# ── activate.sh (bash / zsh) ────────────────────────────────────────────────
$(REPO_DIR)/activate.sh: $(STAMPS)/venv
	@echo "Generating activate.sh ..."
	@{ \
	echo '#!/usr/bin/env bash'; \
	echo '# Auto-generated by Makefile — do not edit.'; \
	echo '# Usage: source activate.sh'; \
	echo ''; \
	echo '_REPO="$$(cd "$$(dirname "$${BASH_SOURCE[0]:-$$0}")" && pwd)"'; \
	echo '_VENV="$${_REPO}/.venv"'; \
	echo ''; \
	echo 'if [ ! -d "$${_VENV}" ]; then'; \
	echo '    echo "ERROR: .venv not found. Run the appropriate make setup-* target first."'; \
	echo '    return 1 2>/dev/null || exit 1'; \
	echo 'fi'; \
	echo ''; \
	echo 'source "$${_VENV}/bin/activate"'; \
	echo ''; \
	echo '[ -f "$${_REPO}/.env" ] && { set -a; source "$${_REPO}/.env"; set +a; }'; \
	echo ''; \
	echo 'export HF_HOME="$${HF_CACHE_DIR:-$${_REPO}/.hf_cache}"'; \
	echo 'export GCS_BUCKET="$${GCS_BUCKET:-statistical-nlp}"'; \
	echo 'export GCS_PROJECT="$${GCS_PROJECT:-statistical-nlp}"'; \
	echo ''; \
	echo 'if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then'; \
	echo '    export PJRT_DEVICE=TPU'; \
	echo '    export VM_ID="$${VM_ID:-$$(hostname)}"'; \
	echo '    export XLA_PERSISTENT_CACHE_PATH="$${_REPO}/.xla_cache"'; \
	echo '    mkdir -p "$${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null'; \
	echo 'else'; \
	echo '    export CUDA_VISIBLE_DEVICES="$${CUDA_VISIBLE_DEVICES:-0}"'; \
	echo 'fi'; \
	echo ''; \
	echo 'cd "$${_REPO}"'; \
	echo 'echo "Activated: $${_VENV}  cwd=$$(pwd)"'; \
	echo 'unset _REPO _VENV'; \
	} > $@
	@chmod +x $@

# ── activate.csh (tcsh — UCL machines) ──────────────────────────────────────
$(REPO_DIR)/activate.csh: $(STAMPS)/venv
	@echo "Generating activate.csh ..."
	@{ \
	echo '#!/bin/tcsh'; \
	echo '# Auto-generated by Makefile — do not edit.'; \
	echo '# Usage: source activate.csh'; \
	echo ''; \
	echo 'set _REPO = "$$cwd"'; \
	echo 'set _VENV = "$${_REPO}/.venv"'; \
	echo ''; \
	echo 'if (! -d "$${_VENV}") then'; \
	echo '    echo "ERROR: .venv not found. Run make setup-ucl-gpu first."'; \
	echo '    exit 1'; \
	echo 'endif'; \
	echo ''; \
	echo 'source "$${_VENV}/bin/activate.csh"'; \
	echo ''; \
	echo 'if (! $$?HF_CACHE_DIR) then'; \
	echo '    setenv HF_HOME "$${_REPO}/.hf_cache"'; \
	echo 'else'; \
	echo '    setenv HF_HOME "$$HF_CACHE_DIR"'; \
	echo 'endif'; \
	echo 'if (! $$?CUDA_VISIBLE_DEVICES) setenv CUDA_VISIBLE_DEVICES 0'; \
	echo 'if (! $$?GCS_BUCKET)           setenv GCS_BUCKET statistical-nlp'; \
	echo 'if (! $$?GCS_PROJECT)          setenv GCS_PROJECT statistical-nlp'; \
	echo ''; \
	echo 'echo "Activated: $${_VENV}  GPU=$${CUDA_VISIBLE_DEVICES}  cwd=`pwd`"'; \
	} > $@

.PHONY: _verify
_verify:
	@echo ""
	@echo "Verifying imports ..."
	@$(BIN)/python -c "from es_finetuning import ESTrainer, ESConfig; print('  imports OK')"
	@echo "  Python: $$($(BIN)/python --version)"
	@echo "  torch:  $$($(BIN)/python -c 'import torch; print(torch.__version__)')"

.PHONY: _done
_done:
	@echo ""
	@echo "============================================================"
	@echo " Setup complete  [$(MODE)]"
	@echo "============================================================"
	@echo ""
	@echo "Activate:  source activate.sh"
	@echo "Test:      make smoke"

# =============================================================================
#  Help
# =============================================================================

.PHONY: help
help:  ## Show this help
	@echo "evo-memory — available targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make setup-local                # CPU or local GPU"
	@echo "  make setup-gpu GPU=2            # pin to CUDA device 2"
	@echo "  make setup-tpu                  # Google Cloud TPU VM"
	@echo "  make setup-ucl-gpu              # UCL CSH cluster"
	@echo "  make logins                     # HF + wandb + GCS in one step"
	@echo "  make smoke                      # quick end-to-end test"
	@echo "  make lock                       # regenerate uv.lock"
