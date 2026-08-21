# =============================================================================
# Dockerfile — Gemma Tounsi 1.0 (27B)
# =============================================================================
#
# STATUS: PLACEHOLDER. This file contains NO build instructions yet, so
# `docker build` will fail on purpose. The runtime image is defined in step 2,
# after the Soup environment has been validated on real hardware.
#
# Scope of this image (when implemented):
#   - provide a reproducible CUDA + Python environment
#   - install Soup as an EXTERNAL dependency (pinned commit/version)
#   - provide the data-preparation and evaluation tooling of this repository
#
# Out of scope: this image must never bake in datasets, model weights, secrets,
# or Hugging Face caches. Those are mounted at runtime (see docker-compose.yml).
#
# -----------------------------------------------------------------------------
# TODO (step 2)
# -----------------------------------------------------------------------------
# [ ] Choose and pin a base image (CUDA/cuDNN version must match the driver on
#     the training machine). Pin by digest for reproducibility.
# [ ] Pin the Python version.
# [ ] Install system build dependencies required by the training stack.
# [ ] Install Soup from its upstream repository at a pinned commit SHA.
#     Do NOT vendor or copy Soup source into this repository.
# [ ] Pin all Python dependencies with hashes / a lock file.
# [ ] Create a non-root user; do not run training as root.
# [ ] Declare mount points for: dataset dir, output dir, HF cache.
# [ ] Set a safe default entrypoint (no implicit training run).
# [ ] Record the resulting image digest in docs/TRAINING.md for reproducibility.
#
# =============================================================================
