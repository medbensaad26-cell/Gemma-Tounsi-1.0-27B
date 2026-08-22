#!/usr/bin/env bash
# =============================================================================
# train.sh — THIN wrapper: runs Soup inside Docker
# =============================================================================
#
# This script contains NO training logic. Soup owns the training loop, QLoRA,
# model loading, quantization and checkpointing. This wrapper only:
#   1. runs `soup doctor` to validate the environment, then
#   2. hands soup.yaml to `soup train` inside the container.
#
# It never installs Soup on the host and never duplicates Soup functionality.
#
# Usage:
#   ./scripts/train.sh                       # doctor, then train with soup.yaml
#   ./scripts/train.sh --tensorboard         # extra args go straight to `soup train`
#   CONFIG=configs/releases/v1.0.yaml ./scripts/train.sh
#
# Training uses the GPU-reserving `soup` service. A host without an NVIDIA
# adapter will fail here by design — that is a real missing prerequisite, not
# something to work around: a 27B QLoRA run needs the GPU.
#
# Outputs land under /workspace inside the container, which is this repository
# on the host (see docker-compose.yml), so checkpoints appear directly here.
# =============================================================================

set -euo pipefail

# Always operate from the repository root so `docker compose` finds its files.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Config path as seen INSIDE the container (repo root is mounted at /workspace).
CONFIG="${CONFIG:-soup.yaml}"

# `docker compose` (v2 plugin) is the invocation verified for this project.
COMPOSE=(docker compose)

# GPU service. Overridable only for deliberate, documented exceptions.
SERVICE="${SERVICE:-soup}"

echo "==> [1/2] soup doctor (environment check)"
"${COMPOSE[@]}" run --rm "$SERVICE" soup doctor

echo "==> [2/2] soup train --config /workspace/${CONFIG}"
"${COMPOSE[@]}" run --rm "$SERVICE" soup train --config "/workspace/${CONFIG}" "$@"
