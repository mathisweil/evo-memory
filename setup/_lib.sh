#!/usr/bin/env bash
# Shared helper functions for setup.sh and setup_tpu.sh.
# Source this file; do not execute directly.

# ── HuggingFace login ─────────────────────────────────────────────────────────
check_hf_login() {
    if python -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
        echo '  Already logged into HuggingFace.'
    else
        echo '  Log in to HuggingFace (required for gated Llama 3.2 access).'
        echo '  Token: https://huggingface.co/settings/tokens'
        huggingface-cli login
    fi
}

# ── Weights & Biases login ────────────────────────────────────────────────────
setup_wandb() {
    if python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
        echo '  Already logged into wandb.'
    else
        echo '  Log in to wandb (press Enter to skip).'
        echo '  API key: https://wandb.ai/authorize'
        wandb login || true
    fi
}

# ── Google Cloud Storage ──────────────────────────────────────────────────────
setup_gcs() {
    local bucket="${GCS_BUCKET:-statistical-nlp}"
    local project="${GCS_PROJECT:-statistical-nlp}"

    if ! command -v gcloud &>/dev/null; then
        echo '  gcloud CLI not found — installing...'
        local gcloud_dir="${HOME}/.local/google-cloud-sdk"
        if [ ! -d "${gcloud_dir}" ]; then
            curl -fsSL https://sdk.cloud.google.com \
                | bash -s -- --disable-prompts --install-dir="${HOME}/.local" 2>&1 | tail -5
        fi
        export PATH="${gcloud_dir}/bin:${PATH}"
    fi

    if gcloud auth application-default print-access-token &>/dev/null 2>&1; then
        echo '  Already authenticated with GCS.'
    else
        echo '  Authenticating with Google Cloud...'
        gcloud auth application-default login \
            --project "${project}" --no-launch-browser
    fi

    if gsutil ls "gs://${bucket}/" &>/dev/null 2>&1; then
        echo "  gs://${bucket}/ accessible."
    else
        echo "  WARNING: Cannot access gs://${bucket}/."
        echo "  Run: gcloud auth application-default login"
    fi
}

# ── Claude Code ───────────────────────────────────────────────────────────────
install_claude() {
    if command -v claude &>/dev/null; then
        echo "  Claude Code already installed ($(claude --version 2>/dev/null || echo 'version unknown'))."
        return
    fi

    if command -v npm &>/dev/null; then
        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
    elif command -v node &>/dev/null; then
        echo '  npm not found — installing...'
        curl -fsSL https://npmjs.org/install.sh | sh
        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
    else
        echo '  Node.js not found — installing via nvm...'
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install --lts
        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
    fi
    echo '  Done.'
}
