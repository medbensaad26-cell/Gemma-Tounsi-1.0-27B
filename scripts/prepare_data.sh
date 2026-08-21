#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — build the training/replay datasets from the manifests
# =============================================================================
#
# STATUS: PLACEHOLDER. Implements nothing yet and exits non-zero on purpose.
#
# Owned by THIS repository (data preparation is ours, not Soup's).
#
# Contract, once implemented:
#   INPUT   data/manifests/train.yaml
#           data/manifests/retention.yaml
#           data/manifests/eval.yaml      (read ONLY for contamination checks)
#   OUTPUT  data/train.jsonl              (gitignored)
#           data/retention.jsonl          (gitignored)
#
# Pipeline stages:
#   manifests -> data/raw/ -> data/processed/ -> data/splits/ -> *.jsonl
#
# Non-negotiables:
#   - Deterministic: same manifests + same pinned sources => identical output.
#   - No undeclared source may enter an output file.
#   - Evaluation data must NEVER reach data/train.jsonl or data/retention.jsonl.
#     If overlap is detected, ABORT — do not silently drop the offending rows.
#   - data/raw/ is treated as immutable once fetched.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# TODO: implement, in this order.
# -----------------------------------------------------------------------------
# [ ] Load configuration from .env if present (HF_TOKEN for gated sources).
#     Never echo secret values.
# [ ] Validate the manifests (required fields, pinned revisions, licenses).
# [ ] Fetch declared sources into data/raw/ (idempotent; skip if present;
#     verify checksums).
# [ ] Normalize/filter into data/processed/ per manifest preprocessing rules.
# [ ] Deduplicate within and across sources.
# [ ] CONTAMINATION GATE: cross-check against every set in
#     data/manifests/eval.yaml; exit non-zero on any overlap.
# [ ] VERIFY retention replay data is disjoint from the retention EVAL set.
# [ ] Build deterministic splits into data/splits/ using the manifest seed.
# [ ] Emit data/train.jsonl and data/retention.jsonl in the schema Soup expects
#     (schema to be confirmed when the Soup environment is validated).
# [ ] Write a provenance/stats report (record counts, per-source shares,
#     checksums) to a gitignored output location.

echo "prepare_data.sh: not implemented yet (repository scaffolding stage)." >&2
echo "See data/README.md and data/manifests/ for the intended contract." >&2
exit 1
