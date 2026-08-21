#!/usr/bin/env bash
# =============================================================================
# train.sh — thin wrapper that will invoke Soup
# =============================================================================
#
# STATUS: PLACEHOLDER. Launches nothing and exits non-zero on purpose.
#
# THIS SCRIPT MUST STAY THIN.
#   Soup owns the training loop, the QLoRA implementation, model loading,
#   quantization, checkpointing, and all GPU/training infrastructure. This
#   wrapper only:
#     - checks preconditions
#     - passes soup.yaml and the prepared datasets to Soup
#     - records what was run, for reproducibility
#
#   It must NEVER reimplement, patch, or duplicate Soup functionality, and Soup
#   must NEVER be cloned or vendored into this repository.
#
# Inputs (when implemented):
#   soup.yaml             training configuration
#   data/train.jsonl      built by scripts/prepare_data.sh
#   data/retention.jsonl  replay/rehearsal data
#
# Outputs: adapters/checkpoints/logs, written OUTSIDE Git (see .gitignore).
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# TODO: implement, in this order.
# -----------------------------------------------------------------------------
# [ ] Verify Soup is installed and record its version/commit SHA.
# [ ] Load .env (HF_TOKEN, output dir, HF cache, tracking keys). Never log
#     secret values.
# [ ] Verify soup.yaml exists and is a real configuration, not the placeholder.
# [ ] Verify data/train.jsonl and data/retention.jsonl exist and are non-empty;
#     fail with a pointer to prepare_data.sh otherwise.
# [ ] SAFETY GATE: refuse to start if any evaluation set is reachable from the
#     resolved training inputs. TounsiBench and the retention EVAL set are
#     evaluation-only.
# [ ] Snapshot the run context for reproducibility: git commit of this repo,
#     Soup commit, resolved config, dataset checksums, GPU/driver info.
# [ ] Hand off to Soup with soup.yaml, forwarding any extra CLI arguments.
# [ ] Do not post-process checkpoints here; releasing is scripts/release.sh.

echo "train.sh: not implemented yet (repository scaffolding stage)." >&2
echo "Soup is an external dependency; see docs/TRAINING.md and soup.yaml." >&2
exit 1
