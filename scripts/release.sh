#!/usr/bin/env bash
# =============================================================================
# release.sh — freeze, gate and publish a model release
# =============================================================================
#
# STATUS: PLACEHOLDER. Publishes nothing and exits non-zero on purpose.
#
# Owned by THIS repository (release orchestration is ours, not Soup's).
#
# Purpose: turn an accepted training run into a reproducible, documented
# release. A release is only valid if someone else can reconstruct it from what
# this script records.
#
# Non-negotiables:
#   - A release requires results from ALL THREE evaluation tracks
#     (TounsiBench, Arabizi, retention). No exceptions.
#   - The retention regression gate must pass before publishing.
#   - Model weights/adapters are large artifacts: they are published to a model
#     registry (e.g. the Hugging Face Hub), NEVER committed to Git.
#   - No credentials are ever written into tracked files.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# TODO: implement, in this order.
# -----------------------------------------------------------------------------
# [ ] Parse arguments: --version (e.g. v1.0), --run <training run id/path>.
# [ ] Verify the working tree is clean and record the exact repo commit.
# [ ] Freeze the exact training configuration used, to
#     configs/releases/<version>.yaml (TRACKED — this is the reproducibility
#     record; do not create it before a real run exists).
# [ ] Record the Soup commit SHA, base model revision, and dataset checksums
#     alongside it.
# [ ] Require evaluation reports for all three tracks; fail if any is missing.
# [ ] RETENTION GATE: fail if any guarded capability regressed beyond the
#     agreed threshold versus base Gemma 3 27B.
# [ ] Generate/refresh docs/MODEL_CARD.md with the measured results.
#     Never fabricate numbers.
# [ ] Package the artifact (adapter and/or merged weights) outside Git.
# [ ] Publish to the model registry using credentials from .env; do not echo
#     tokens.
# [ ] Tag the repository with the released version.

echo "release.sh: not implemented yet (repository scaffolding stage)." >&2
echo "See configs/releases/ and docs/MODEL_CARD.md." >&2
exit 1
