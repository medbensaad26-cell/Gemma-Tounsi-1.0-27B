#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — run the evaluation tracks
# =============================================================================
#
# STATUS: PLACEHOLDER. Runs nothing and exits non-zero on purpose.
#
# Owned by THIS repository (evaluation is ours, not Soup's).
#
# Intended usage, once implemented:
#   scripts/evaluate.sh --track tounsibench|arabizi|retention|all \
#                       --model <base|adapter path|merged path>
#
# Tracks (see eval/README.md):
#   tounsibench  Tunisian Arabic / Derja, Arabic script
#   arabizi      Latin-script Tunisian
#   retention    preservation of Gemma's general capabilities
#
# Non-negotiables:
#   - Evaluation data is EVALUATION ONLY and must never be written back into any
#     training mixture.
#   - Every score is reported next to the same measurement on the unmodified
#     base Gemma 3 27B, at pinned revisions and identical decoding settings.
#   - Predictions and results are generated artifacts: they are gitignored and
#     stored outside Git.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# TODO: implement, in this order.
# -----------------------------------------------------------------------------
# [ ] Parse arguments: --track, --model, --output, --seed, --limit.
# [ ] Load .env (HF_TOKEN, HF cache location). Never log secret values.
# [ ] Resolve the model under test: base model, adapter, or merged weights.
# [ ] Load the track definition from data/manifests/eval.yaml.
# [ ] Assert evaluation-only: fail if the resolved eval data is also referenced
#     by data/manifests/train.yaml or data/manifests/retention.yaml.
# [ ] For the retention track, additionally assert disjointness from the replay
#     data in data/retention.jsonl.
# [ ] Generate predictions with fixed decoding settings; record them.
# [ ] Score with the track's metrics; keep per-task granularity.
# [ ] Emit a report containing: model revision, repo commit, harness version,
#     decoding settings, seed, per-task scores, and the base-model deltas.
# [ ] Write everything to a gitignored output directory.

echo "evaluate.sh: not implemented yet (repository scaffolding stage)." >&2
echo "See eval/README.md and docs/EVALUATION.md for the intended protocol." >&2
exit 1
