#!/usr/bin/env bash
# =============================================================================
# doctor.sh — environment validation only. NEVER starts training.
# =============================================================================
#
# Answers exactly eight questions. The first runs on the host, the rest inside
# the container:
#   1. are the Docker + Docker Compose runtime assumptions satisfied?
#   2. does the container start, and is Soup available (which version)?
#   3. which Python version? (Soup supports 3.10-3.12)
#   4. what does Soup's own `soup doctor` report?
#   5. is CUDA / the GPU visible to PyTorch?
#   6. is the repository mounted at /workspace?
#   7. is HF_TOKEN present? (the value is NEVER printed)
#   8. does the environment still match soup-env.lock? (ABI drift)
#
# Secrets are NEVER printed: only the PRESENCE of HF_TOKEN is reported.
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
echo "==> 1/8 host Docker runtime"
# The whole stack runs in Docker, so the daemon must be reachable and the
# compose file must resolve before any container check is meaningful.
docker --version
docker compose version
docker info --format 'server: {{.ServerVersion}} | default runtime: {{.DefaultRuntime}}'
docker compose config --quiet
echo "compose configuration resolves"

echo
echo "==> 2/8 container + Soup version"
# NOTE: it is `soup version`, NOT `soup --version` (that option does not exist).
run soup version

echo
echo "==> 3/8 Python version (Soup supports 3.10-3.12)"
run python -c "import sys; print(sys.version)"

echo
echo "==> 4/8 soup doctor (Soup's own environment report)"
run soup doctor

echo
echo "==> 5/8 CUDA / GPU visibility"
run python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo
echo "==> 6/8 mounted repository at /workspace"
run sh -lc "ls -la /workspace"

echo
echo "==> 7/8 Hugging Face token presence (value is never printed)"
run sh -lc 'test -n "$HF_TOKEN" && echo "HF_TOKEN is set" || echo "HF_TOKEN is not set"'

echo
echo "==> 8/8 environment lock drift (soup-env.lock)"
# `soup env check` compares the running env against the committed lock file.
# Regenerate with `soup env lock` when the pinned Soup image changes.
if [ -f soup-env.lock ]; then
  run soup env check
else
  echo "soup-env.lock not found — create it with: docker compose run --rm ${SERVICE} soup env lock"
fi

echo
echo "doctor.sh: environment checks finished. No training was started."
