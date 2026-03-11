#!/bin/bash
# TPU restart script — handles preempted spot/preemptible TPU VMs
# Usage: ./scripts/tpu_restart.sh

set -euo pipefail

# --- Configuration ---
TPU_NAME="hyperscale-v6e"
ZONE="europe-west4-a"
ACCELERATOR="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
SSH_HOST="gcp-tpu-v6e"
SSH_CONFIG="$HOME/.ssh/config"
GCLOUD="/opt/homebrew/bin/gcloud"

# --- Check current status ---
echo "Checking TPU status..."
STATUS=$($GCLOUD compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format="get(state)" 2>/dev/null || echo "NOT_FOUND")

echo "Status: $STATUS"

if [ "$STATUS" = "READY" ]; then
    IP=$($GCLOUD compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
        --format="get(networkEndpoints[0].accessConfig.externalIp)")
    echo "TPU is already running at $IP"
    echo "Updating SSH config..."
    sed -i '' "/Host ${SSH_HOST}$/,/HostName /{s/HostName .*/HostName ${IP}/;}" "$SSH_CONFIG"
    echo "Done. Run: ssh $SSH_HOST"
    exit 0
fi

# --- Delete if exists (PREEMPTED, STOPPED, etc.) ---
if [ "$STATUS" != "NOT_FOUND" ]; then
    echo "Deleting TPU (state: $STATUS)..."
    $GCLOUD compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet
    echo "Deleted."
fi

# --- Recreate ---
echo "Creating TPU..."
$GCLOUD compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$ACCELERATOR" \
    --version="$RUNTIME" \
    --spot --preemptible

# --- Get new IP and update SSH config ---
IP=$($GCLOUD compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
    --format="get(networkEndpoints[0].accessConfig.externalIp)")

echo "New IP: $IP"
sed -i '' "/Host ${SSH_HOST}$/,/HostName /{s/HostName .*/HostName ${IP}/;}" "$SSH_CONFIG"

# --- Push SSH keys and clean known_hosts ---
echo "Pushing SSH keys to new VM..."
ssh-keygen -R "$IP" 2>/dev/null || true
$GCLOUD compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="echo 'SSH key pushed successfully'"

echo "SSH config updated. Run: ssh $SSH_HOST"
