#!/usr/bin/env bash
# =============================================================================
# doctor.sh — environment validation only. NEVER starts training.
# =============================================================================
#
# Answers exactly six questions, all inside the container:
#   1. does the container start, and is Soup available (which version)?
#   2. which Python version? (Soup supports 3.10-3.12)
#   3. what does Soup's own `soup doctor` report?
#   4. is CUDA / the GPU visible to PyTorch?
#   5. is the repository mounted at /workspace?
#   6. is HF_TOKEN present? (the value is NEVER printed)
#
# Usage:
#   ./scripts/doctor.sh                  # uses the GPU service
#   SERVICE=soup-cpu ./scripts/doctor.sh # host without an NVIDIA adapter
# =============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# `soup` reserves the GPU; `soup-cpu` is the same image without the reservation.
SERVICE="${SERVICE:-soup}"

run() { docker compose run --rm "$SERVICE" "$@"; }

echo "==> using compose service: ${SERVICE}"

echo
echo "==> 1/6 container + Soup version"
# NOTE: it is `soup version`, NOT `soup --version` (that option does not exist).
run soup version

echo
echo "==> 2/6 Python version (Soup supports 3.10-3.12)"
run python -c "import sys; print(sys.version)"

echo
echo "==> 3/6 soup doctor (Soup's own environment report)"
run soup doctor

echo
echo "==> 4/6 CUDA / GPU visibility"
run python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo
echo "==> 5/6 mounted repository at /workspace"
run sh -lc "ls -la /workspace"

echo
echo "==> 6/6 Hugging Face token presence (value is never printed)"
run sh -lc 'test -n "$HF_TOKEN" && echo "HF_TOKEN is set" || echo "HF_TOKEN is not set"'

echo
echo "doctor.sh: environment checks finished. No training was started."
