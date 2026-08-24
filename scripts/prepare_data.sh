#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — the single entry point for data preparation
# =============================================================================
#
# STATUS: SYNTHETIC MODE ONLY (Task 4A).
#
# This script currently runs the COMPLETE data-engineering pipeline end to end,
# but exclusively over the synthetic fixtures in data/synthetic/. It exists to
# prove the plumbing works before any real corpus is downloaded.
#
# It NEVER:
#   - downloads an external dataset,
#   - trains or fine-tunes anything,
#   - touches a GPU,
#   - prints a secret.
#
# Pipeline stages (mirrors the project's conceptual data flow):
#
#     synthetic raw data
#         -> canonicalization  (fixtures are authored canonical already)
#         -> validation        (src/data/validate.py)
#         -> statistics        (src/data/stats.py)
#         -> duplicate handling(src/data/dedupe.py  — Soup for near-dups)
#         -> category checks   (technical quota visible in the stats report)
#         -> retention select  (src/data/retention.py)
#         -> holdout split     (reserved inside the selector; written out)
#         -> mixture validation(src/data/mixture.py)
#         -> Soup-compatible JSONL (src/data/export.py)
#
# Usage:
#   ./scripts/prepare_data.sh              # run the synthetic pipeline
#   ./scripts/prepare_data.sh --in-docker  # same, inside the Soup container
#   KEEP_OUTPUT=1 ./scripts/prepare_data.sh
#
# Real datasets are wired in from Task 4B onward, once adapters exist. The
# stage order below will NOT change: only the input sources will.
# =============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- configuration -----------------------------------------------------------
# Resolve a Python interpreter. Inside the Soup container this is simply
# `python`; on a Windows host under Git Bash only `py`/`python3` may exist.
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
else
  PY=""
  for candidate in python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1; then
      # Reject the Windows Store stub, which exits 9009 instead of running.
      if "$candidate" -c "import sys" >/dev/null 2>&1; then
        PY="$candidate"
        break
      fi
    fi
  done
  if [[ -z "$PY" ]]; then
    echo "prepare_data.sh: no working Python interpreter found." >&2
    echo "Set PYTHON=/path/to/python, or run: ./scripts/prepare_data.sh --in-docker" >&2
    exit 1
  fi
fi

SYN_RAW="data/synthetic/raw"

OUT_DIR="${OUT_DIR:-data/processed/synthetic}"
RET_DIR="${OUT_DIR}/retention"
MANIFEST_DIR="${MANIFEST_DIR:-data/manifests/retention}"

# The synthetic pool is tiny, so the real 20,000/2,500 retention targets are
# scaled down. The SAME code path and the SAME constraints are exercised;
# only the magnitude differs. Real runs use SCALE=1.0.
SCALE="${SCALE:-0.01}"

step=0
say() { step=$((step + 1)); printf '\n==> %d/%s %s\n' "$step" "$TOTAL_STEPS" "$1"; }
fail() { printf '\nprepare_data.sh: FAILED at step %d: %s\n' "$step" "$1" >&2; exit 1; }
TOTAL_STEPS=9

# --- optional: run everything inside the Soup container ----------------------
if [[ "${1:-}" == "--in-docker" ]]; then
  echo "==> re-executing inside the Soup container"
  exec docker compose run --rm "${SERVICE:-soup-cpu}" bash scripts/prepare_data.sh
fi

echo "============================================================"
echo " prepare_data.sh — SYNTHETIC MODE (no external data, no training)"
echo " retention target scale: ${SCALE}"
echo "============================================================"

# -----------------------------------------------------------------------------
say "regenerating synthetic fixtures (deterministic)"
# -----------------------------------------------------------------------------
"$PY" data/synthetic/generate.py || fail "could not generate synthetic fixtures"

# -----------------------------------------------------------------------------
say "validating canonical schema on every well-formed slice"
# -----------------------------------------------------------------------------
"$PY" -m src.data.validate \
  "${SYN_RAW}/retention_pool.jsonl" \
  "${SYN_RAW}/arabizi.jsonl" \
  "${SYN_RAW}/arabic_derja.jsonl" \
  "${SYN_RAW}/franco_tunisian.jsonl" \
  "${SYN_RAW}/msa_formal.jsonl" \
  || fail "schema validation rejected a slice that should be valid"

# -----------------------------------------------------------------------------
say "confirming the validator REJECTS the deliberately malformed fixture"
# -----------------------------------------------------------------------------
# A validator that cannot fail is worthless, so assert the negative case too.
if "$PY" -m src.data.validate "${SYN_RAW}/malformed.jsonl" >/dev/null 2>&1; then
  fail "malformed.jsonl passed validation — the validator is not working"
fi
echo "malformed fixture correctly rejected"

# -----------------------------------------------------------------------------
say "dataset statistics (includes the technical-category breakdown)"
# -----------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
"$PY" -m src.data.stats \
  "${SYN_RAW}/retention_pool.jsonl" \
  "${SYN_RAW}/arabizi.jsonl" \
  "${SYN_RAW}/arabic_derja.jsonl" \
  "${SYN_RAW}/franco_tunisian.jsonl" \
  "${SYN_RAW}/msa_formal.jsonl" \
  --out "${OUT_DIR}/stats.json" \
  || fail "statistics failed"

# -----------------------------------------------------------------------------
say "duplicate handling (exact id dedup locally; near-dup via Soup)"
# -----------------------------------------------------------------------------
"$PY" -m src.data.dedupe id "${SYN_RAW}/duplicates.jsonl" \
  --output "${OUT_DIR}/duplicates_deduped.jsonl" \
  || fail "id deduplication failed"

# Near-duplicate removal is Soup's job, not ours. Print the exact command that
# would run so it is auditable; only execute it when Soup is actually present.
echo
echo "--- near-duplicate removal (delegated to Soup 0.73.3) ---"
"$PY" -m src.data.dedupe content "${SYN_RAW}/duplicates.jsonl" \
  --output "${OUT_DIR}/duplicates_near_deduped.jsonl" --dry-run \
  || fail "could not build the Soup dedup command"
if command -v soup >/dev/null 2>&1; then
  echo "soup found — running the real Soup deduplication"
  "$PY" -m src.data.dedupe content "${SYN_RAW}/duplicates.jsonl" \
    --output "${OUT_DIR}/duplicates_near_deduped.jsonl" \
    || fail "soup data dedup failed"
else
  echo "soup not on PATH (host run) — skipped; use --in-docker to execute it"
fi

# -----------------------------------------------------------------------------
say "retention selection + holdout reservation (deterministic, per-category)"
# -----------------------------------------------------------------------------
"$PY" -m src.data.retention \
  --candidates "${SYN_RAW}/retention_pool.jsonl" \
  --scale "$SCALE" \
  --train-out "${RET_DIR}/train.jsonl" \
  --holdout-out "${RET_DIR}/holdout.jsonl" \
  --manifest-out "${MANIFEST_DIR}/synthetic_selection.json" \
  || fail "retention selection failed"

# -----------------------------------------------------------------------------
say "verifying train/holdout separation (zero contamination)"
# -----------------------------------------------------------------------------
"$PY" - "$RET_DIR" <<'PYTHON' || fail "train/holdout contamination detected"
import json, sys
from pathlib import Path

directory = Path(sys.argv[1])

def ids(name):
    path = directory / name
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}

train, holdout = ids("train.jsonl"), ids("holdout.jsonl")
overlap = train & holdout
if overlap:
    print(f"CONTAMINATION: {len(overlap)} shared id(s): {sorted(overlap)[:5]}")
    raise SystemExit(1)
print(f"train={len(train)} holdout={len(holdout)} overlap=0")
PYTHON

# -----------------------------------------------------------------------------
say "mixture validation (shares, slices, technical quotas, leakage)"
# -----------------------------------------------------------------------------
"$PY" -m src.data.mixture \
  --slice "arabizi=${SYN_RAW}/arabizi.jsonl" \
  --slice "arabic_derja=${SYN_RAW}/arabic_derja.jsonl" \
  --slice "franco_tunisian=${SYN_RAW}/franco_tunisian.jsonl" \
  --slice "msa_formal=${SYN_RAW}/msa_formal.jsonl" \
  --slice "retention=${RET_DIR}/train.jsonl" \
  --holdout "${RET_DIR}/holdout.jsonl" \
  || fail "mixture validation failed"

# The quota gate must also be able to FAIL. Prove it with the low-technical
# Arabizi fixture, which sits deliberately below the 20% minimum.
echo
echo "--- confirming the technical quota gate rejects a bad mixture ---"
if "$PY" -m src.data.mixture \
  --slice "arabizi=${SYN_RAW}/arabizi_low_technical.jsonl" \
  --slice "arabic_derja=${SYN_RAW}/arabic_derja.jsonl" \
  --slice "franco_tunisian=${SYN_RAW}/franco_tunisian.jsonl" \
  --slice "msa_formal=${SYN_RAW}/msa_formal.jsonl" \
  --slice "retention=${RET_DIR}/train.jsonl" >/dev/null 2>&1; then
  fail "a below-quota Arabizi mixture passed validation — the gate is broken"
fi
echo "below-quota mixture correctly rejected"

# -----------------------------------------------------------------------------
say "exporting the final Soup-compatible JSONL"
# -----------------------------------------------------------------------------
"$PY" -m src.data.export \
  --in "${RET_DIR}/train.jsonl" \
  --out "${OUT_DIR}/final_sharegpt.jsonl" \
  || fail "export failed"

# Let Soup itself confirm the file it will consume is well-formed.
if command -v soup >/dev/null 2>&1; then
  echo
  echo "--- validating the exported file with Soup ---"
  soup data validate "${OUT_DIR}/final_sharegpt.jsonl" --format sharegpt \
    || fail "soup data validate rejected the exported file"
else
  echo "soup not on PATH (host run) — skipping 'soup data validate'"
fi

echo
echo "============================================================"
echo " prepare_data.sh: synthetic pipeline completed successfully"
echo "============================================================"
echo "  retention train : ${RET_DIR}/train.jsonl"
echo "  retention holdout: ${RET_DIR}/holdout.jsonl"
echo "  selection manifest: ${MANIFEST_DIR}/synthetic_selection.json"
echo "  statistics      : ${OUT_DIR}/stats.json"
echo "  final export    : ${OUT_DIR}/final_sharegpt.jsonl"
echo
echo "No external dataset was downloaded. No training was started."
