#!/usr/bin/env bash
# TPU VM restart — handles preempted/stopped spot and on-demand TPU VMs.
#
# Usage:
#   bash setup/tpu_restart.sh           # default: v6e
#   bash setup/tpu_restart.sh --v6e     # spot v6e-8 in europe-west4-a
#   bash setup/tpu_restart.sh --v4      # on-demand v4-8 in us-central2-b

set -euo pipefail

GCLOUD="$(command -v gcloud || { echo "ERROR: gcloud not found. Install from https://cloud.google.com/sdk/docs/install"; exit 1; })"
SSH_CONFIG="${HOME}/.ssh/config"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TPU_TYPE="${1:---v6e}"

case "${TPU_TYPE}" in
    --v6e)
        TPU_NAME="hyperscale-v6e"
        ZONE="europe-west4-a"
        ACCELERATOR="v6e-8"
        RUNTIME="v2-alpha-tpuv6e"
        SSH_HOST="gcp-tpu-v6e"
        CREATE_FLAGS="--spot --preemptible"
        ;;
    --v4)
        TPU_NAME="hyperscale-v4"
        ZONE="us-central2-b"
        ACCELERATOR="v4-8"
        RUNTIME="tpu-vm-tf-2.17.0-pjrt"
        SSH_HOST="gcp-tpu-v4"
        CREATE_FLAGS=""
        ;;
    *)
        echo "Usage: $0 [--v6e | --v4]"
        echo "  --v6e  Spot v6e-8 in europe-west4-a (default)"
        echo "  --v4   On-demand v4-8 in us-central2-b"
        exit 1
        ;;
esac

echo "TPU: ${TPU_TYPE} (${TPU_NAME} / ${ZONE} / ${ACCELERATOR})"

# ---------------------------------------------------------------------------
# Check current status
# ---------------------------------------------------------------------------
echo "Checking TPU status..."
STATUS="$("${GCLOUD}" compute tpus tpu-vm describe "${TPU_NAME}" \
    --zone="${ZONE}" --format="get(state)" 2>/dev/null || echo "NOT_FOUND")"
HEALTH="$("${GCLOUD}" compute tpus tpu-vm describe "${TPU_NAME}" \
    --zone="${ZONE}" --format="get(health)" 2>/dev/null || echo "UNKNOWN")"

echo "Status: ${STATUS}  Health: ${HEALTH}"

if [ "${STATUS}" = "READY" ] && [ "${HEALTH}" != "UNHEALTHY_MAINTENANCE" ]; then
    IP="$("${GCLOUD}" compute tpus tpu-vm describe "${TPU_NAME}" \
        --zone="${ZONE}" \
        --format="get(networkEndpoints[0].accessConfig.externalIp)")"
    echo "TPU is already running at ${IP}"
    sed -i '' "/Host ${SSH_HOST}$/,/HostName /{s/HostName .*/HostName ${IP}/;}" "${SSH_CONFIG}"
    echo "SSH config updated. Run: ssh ${SSH_HOST}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Delete if exists (PREEMPTED, STOPPED, UNHEALTHY, etc.)
# ---------------------------------------------------------------------------
if [ "${STATUS}" != "NOT_FOUND" ]; then
    echo "Deleting TPU (state: ${STATUS}, health: ${HEALTH})..."
    "${GCLOUD}" compute tpus tpu-vm delete "${TPU_NAME}" \
        --zone="${ZONE}" --quiet
    echo "Deleted."
fi

# ---------------------------------------------------------------------------
# Recreate
# ---------------------------------------------------------------------------
echo "Creating TPU..."
# shellcheck disable=SC2086
"${GCLOUD}" compute tpus tpu-vm create "${TPU_NAME}" \
    --zone="${ZONE}" \
    --accelerator-type="${ACCELERATOR}" \
    --version="${RUNTIME}" \
    ${CREATE_FLAGS}

# ---------------------------------------------------------------------------
# Get new IP and update SSH config
# ---------------------------------------------------------------------------
IP="$("${GCLOUD}" compute tpus tpu-vm describe "${TPU_NAME}" \
    --zone="${ZONE}" \
    --format="get(networkEndpoints[0].accessConfig.externalIp)")"

echo "New IP: ${IP}"
sed -i '' "/Host ${SSH_HOST}$/,/HostName /{s/HostName .*/HostName ${IP}/;}" "${SSH_CONFIG}"

# Push SSH keys and clean known_hosts
echo "Pushing SSH keys..."
ssh-keygen -R "${IP}" 2>/dev/null || true
"${GCLOUD}" compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" --command="echo 'SSH OK'"

echo "Done. Run: ssh ${SSH_HOST}"
