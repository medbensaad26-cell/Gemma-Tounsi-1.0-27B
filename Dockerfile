# =============================================================================
# Dockerfile — Gemma Tounsi 1.0 (27B) training environment
# =============================================================================
#
# A THIN, project-specific layer on top of the OFFICIAL Soup image.
#
# Why not rebuild the CUDA / PyTorch stack here?
#   The upstream image already ships everything the training engine needs:
#     - CUDA 12.1 runtime (base: nvidia/cuda:12.1.0-runtime-ubuntu22.04)
#     - Python 3.10 (Ubuntu 22.04 native)
#     - soup-cli installed with the [train,serve,data,eval] extras
#   Re-deriving that here would only duplicate upstream work and drift from it.
#   Soup stays an EXTERNAL dependency: it is never vendored or forked here.
#
# Soup version is pinned to an EXACT published tag (never `latest`) so this
# environment is reproducible:
#
#   image  : ghcr.io/makazhanalpamys/soup:0.73.3
#   digest : sha256:4536816b4975a4b3abdcc3bd0761e94cb0c5f82cdee675f10f750fa19c4843d1
#   soup   : soup-cli 0.73.3 (current latest on PyPI and on GHCR)
#   python : 3.10 — inside Soup's supported 3.10–3.12 range.
#            Soup requires <3.13, so Python 3.13 is deliberately NOT targeted.
#
# This image contains NO datasets, NO model weights and NO secrets:
#   - the repository is bind-mounted at /workspace  (see docker-compose.yml)
#   - the Hugging Face cache is bind-mounted        (see docker-compose.yml)
#   - HF_TOKEN / WANDB_API_KEY are injected at RUN time from .env (env_file)
#
# =============================================================================

FROM ghcr.io/makazhanalpamys/soup:0.73.3

# The repository is bind-mounted here at run time. Nothing is COPYed into the
# image on purpose: the build context (datasets, caches, checkpoints) must never
# end up in a layer.
WORKDIR /workspace

# Upstream sets `ENTRYPOINT ["soup"]`, which would turn the documented
# `docker compose run --rm soup soup doctor` into `soup soup doctor`.
# Resetting the entrypoint lets the container accept a full command line, so
# `soup ...`, `python ...` and `sh -lc ...` all work through the same service.
ENTRYPOINT []

# Safe default: never start a training run implicitly.
CMD ["soup", "--help"]
