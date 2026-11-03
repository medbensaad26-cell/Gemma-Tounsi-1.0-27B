"""Token-budget selection for Tunisian adaptation slices (Arabizi, Derja, Franco Tunisian).

Mirrors src/data/retention.py:select_retention_token_budget() — the same method
used for retention selection:

  1. Validate + deduplicate candidates
  2. Tokenize every candidate (Gemma tokenizer via ApproximateTokenCounter fallback)
  3. Drop over-length examples (>15,500 formatted tokens, whole only)
  4. Reserve holdout FIRST (token-proportional across categories)
  5. Fill train to multiplier × T_anchor (whole examples, ±2% tolerance)
  6. Deterministic: seed 42, stable sort by id

Usage:
    python scripts/select_tunisian_token_budget.py --slice arabizi
    python scripts/select_tunisian_token_budget.py --slice arabic_derja
    python scripts/select_tunisian_token_budget.py --slice franco_tunisian
    python scripts/select_tunisian_token_budget.py --slice all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data import schema, split  # noqa: E402
from src.data.retention import (
    ApproximateTokenCounter,
    SelectionError,
    TokenCount,
    _budget_value,
    _fill_token_budget,
    _pick_with_source_diversity,
    _remove_ids,
    _record_key,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifests"


@dataclass(frozen=True)
class TokenCount:
    total: int
    assistant: int


@dataclass
class TunisianSelection:
    train: List[Dict[str, Any]] = field(default_factory=list)
    holdout: List[Dict[str, Any]] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    seed: int = 42
    anchor_tokens: int = 0
    token_metric: str = "assistant_tokens"
    train_token_targets: Dict[str, float] = field(default_factory=dict)
    train_total_tokens: Dict[str, int] = field(default_factory=dict)
    train_assistant_tokens: Dict[str, int] = field(default_factory=dict)
    holdout_token_totals: Dict[str, int] = field(default_factory=dict)

    @property
    def train_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.train:
            k = str(r.get("category"))
            counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def holdout_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.holdout:
            k = str(r.get("category"))
            counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items()))

    def assert_no_contamination(self) -> None:
        train_ids = {r["id"] for r in self.train if isinstance(r.get("id"), str)}
        holdout_ids = {r["id"] for r in self.holdout if isinstance(r.get("id"), str)}
        overlap = train_ids & holdout_ids
        if overlap:
            raise SelectionError(
                f"train/holdout contamination: {len(overlap)} shared id(s), "
                f"e.g. {sorted(overlap)[:5]}"
            )

    def manifest(self, *, candidates: int, sources: Sequence[str], slice_name: str) -> Dict[str, Any]:
        return {
            "slice": slice_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "candidates_considered": candidates,
            "train": {
                "total": len(self.train),
                "targets": self.train_token_targets,
                "counts": self.train_counts,
            },
            "holdout": {
                "total": len(self.holdout),
                "targets": self.holdout_token_totals,
                "counts": self.holdout_counts,
            },
            "rejected": self.rejected,
            "sources": sorted(set(sources)),
            "deterministic": True,
            "token_budget": {
                "metric": self.token_metric,
                "anchor_tokens": self.anchor_tokens,
                "train_token_targets": self.train_token_targets,
                "train_total_tokens": self.train_total_tokens,
                "train_assistant_tokens": self.train_assistant_tokens,
                "holdout_token_totals": self.holdout_token_totals,
            },
        }


def load_slice_config(slice_name: str) -> Dict[str, Any]:
    path = ROOT / "configs" / "data" / f"{slice_name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_pool_records(slice_name: str) -> List[Dict[str, Any]]:
    """Load combined train + holdout records as the candidate pool."""
    records: List[Dict[str, Any]] = []
    for fname in ("train.jsonl", "holdout.jsonl"):
        path = PROC / slice_name / fname
        if path.exists():
            data = schema.load_jsonl(path)
            records.extend(data.records)
    return records


def validate_records(
    records: List[Dict[str, Any]], slice_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    S = schema.load_schema()
    valid: List[Dict[str, Any]] = []
    rejections: Dict[str, int] = {}
    quality = slice_config.get("quality", {})
    allowed_languages = quality.get("language_allowed", [])
    allowed_scripts = quality.get("script_allowed", [])
    exclude_flagged = quality.get("exclude_flagged", True)
    require_unique_ids = quality.get("require_unique_ids", True)

    seen_ids: set = set()
    for idx, record in enumerate(records):
        errors = schema.validate_record(record, S, line=idx + 1)
        if errors:
            for e in errors:
                rejections[e.code] = rejections.get(e.code, 0) + 1
            continue
        if exclude_flagged and isinstance(record.get("quality"), dict):
            if record["quality"].get("flagged") is True:
                rejections["quality_flagged"] = rejections.get("quality_flagged", 0) + 1
                continue
        if allowed_languages and record.get("language") not in allowed_languages:
            rejections["language_not_allowed"] = rejections.get("language_not_allowed", 0) + 1
            continue
        if allowed_scripts and record.get("script") not in allowed_scripts:
            rejections["script_not_allowed"] = rejections.get("script_not_allowed", 0) + 1
            continue
        if require_unique_ids:
            rid = record.get("id")
            if isinstance(rid, str) and rid in seen_ids:
                rejections["duplicate_id"] = rejections.get("duplicate_id", 0) + 1
                continue
            if isinstance(rid, str):
                seen_ids.add(rid)
        valid.append(record)
    return valid, rejections


def dedupe_by_id(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    removed = 0
    for record in records:
        rid = record.get("id")
        if isinstance(rid, str) and rid in seen:
            removed += 1
            continue
        if isinstance(rid, str):
            seen.add(rid)
        unique.append(record)
    return unique, removed


def tokenize_records(
    records: List[Dict[str, Any]],
    max_tokens: int,
    counter: Optional[ApproximateTokenCounter] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, TokenCount], Dict[str, int]]:
    """Tokenize all records, drop over-length. Returns (within_length, counts, rejections)."""
    counter = counter or ApproximateTokenCounter()
    counts: Dict[str, TokenCount] = {}
    rejections: Dict[str, int] = {}
    within_length: List[Dict[str, Any]] = []

    for record in records:
        key = _record_key(record)
        tc = counter.count_messages(record.get("messages", []))
        counts[key] = tc
        if tc.total > max_tokens:
            rejections["over_length"] = rejections.get("over_length", 0) + 1
        else:
            within_length.append(record)

    return within_length, counts, rejections


def select_token_budget(
    records: List[Dict[str, Any]],
    slice_config: Dict[str, Any],
    slice_name: str,
    t_code: Optional[int] = None,
    counter: Optional[ApproximateTokenCounter] = None,
) -> TunisianSelection:
    """Token-budget selection mirroring retention.select_retention_token_budget.

    Holdout is reserved FIRST from the full pool (token-proportional across
    categories), then train receives the remaining records. This matches the
    Mohamed docs: "Reserve the holdout BEFORE anything else."
    """
    spec = slice_config
    tb = spec["token_budget"]
    metric = tb["metric"]
    tolerance = tb["tolerance"]
    max_len = tb["max_formatted_tokens_per_example"]
    category_multipliers = tb["category_multipliers"]
    anchor_cfg = tb["anchor"]
    anchor_category = anchor_cfg["category"]

    counter = counter or ApproximateTokenCounter()
    rejected: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    # --- validate + dedupe ---
    valid, val_rej = validate_records(records, spec)
    for k, v in val_rej.items():
        rejected[k] = rejected.get(k, 0) + v

    unique, dedup_removed = dedupe_by_id(valid)
    rejected["dedup_removed"] = dedup_removed

    # --- tokenize all candidates ---
    within_length, counts, over_len_rej = tokenize_records(unique, max_len, counter)
    for k, v in over_len_rej.items():
        rejected[k] = rejected.get(k, 0) + v

    if not within_length:
        raise SelectionError("no candidates within the formatted token limit")

    # --- determine T_anchor ---
    if anchor_category in category_multipliers:
        anchor_pool = [
            r for r in within_length if str(r.get("category")) == anchor_category
        ]
    else:
        anchor_pool = list(within_length)

    if not anchor_pool:
        raise SelectionError(f"no anchor candidates for category '{anchor_category}'")

    if t_code is not None:
        # Franco Tunisian: T_franco = 2.4 × T_code (from retention)
        T_anchor = t_code
    else:
        T_anchor = sum(_budget_value(r, counts, metric) for r in anchor_pool)

    # --- holdout FIRST, token-proportional across categories ---
    holdout_cfg = tb["holdout"]
    holdout_shares = holdout_cfg["category_shares"]
    # Holdout target = 5% of T_anchor (matches retention's DEFAULT_HOLDOUT_RATIO)
    holdout_target_tokens = T_anchor * 0.05

    # Group all within_length records by category for holdout selection
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for r in within_length:
        by_category.setdefault(str(r.get("category")), []).append(r)

    holdout_records: List[Dict[str, Any]] = []
    holdout_token_totals: Dict[str, int] = {}
    holdout_ids: set = set()

    for category in sorted(holdout_shares):
        if category not in by_category:
            continue
        share = holdout_shares[category]
        category_target = holdout_target_tokens * share
        category_pool = by_category[category]
        selected, total = _fill_token_budget(
            category_pool, category_target, tolerance=tolerance,
            metric=metric, counts=counts, seed=spec["selection"]["seed"],
        )
        holdout_records.extend(selected)
        holdout_token_totals[category] = total
        for r in selected:
            rid = r.get("id")
            if isinstance(rid, str):
                holdout_ids.add(rid)

    # --- train: fill category by category to multiplier × T_anchor (±2%) ---
    train_records: List[Dict[str, Any]] = []
    train_token_targets: Dict[str, float] = {}
    train_total_tokens: Dict[str, int] = {}
    train_assistant_tokens: Dict[str, int] = {}

    # Build remaining pool by category (after holdout removal)
    remaining_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for r in _remove_ids(within_length, holdout_ids):
        remaining_by_category.setdefault(str(r.get("category")), []).append(r)

    for category in sorted(category_multipliers):
        mult = category_multipliers[category]
        if mult <= 0:
            continue
        target_tokens = mult * T_anchor
        train_token_targets[category] = target_tokens
        category_pool = remaining_by_category.get(category, [])
        selected, total = _fill_token_budget(
            category_pool, target_tokens, tolerance=tolerance,
            metric=metric, counts=counts, seed=spec["selection"]["seed"],
        )
        train_records.extend(selected)
        train_total_tokens[category] = sum(
            counts[_record_key(r)].total for r in selected
        )
        train_assistant_tokens[category] = sum(
            counts[_record_key(r)].assistant for r in selected
        )

    # Anchor token total (for reporting - this is T_anchor, the target)
    # anchor_tokens already set to T_anchor above; do NOT overwrite.

    # Sort deterministically
    train_records.sort(key=lambda r: str(r.get("id", "")))
    holdout_records.sort(key=lambda r: str(r.get("id", "")))

    selection = TunisianSelection(
        train=train_records,
        holdout=holdout_records,
        rejected=dict(sorted(rejected.items())),
        seed=spec["selection"]["seed"],
        anchor_tokens=T_anchor,
        token_metric=metric,
        train_token_targets=train_token_targets,
        train_total_tokens=train_total_tokens,
        train_assistant_tokens=train_assistant_tokens,
        holdout_token_totals=holdout_token_totals,
    )
    selection.assert_no_contamination()
    return selection
def write_outputs(
    slice_name: str,
    selection: TunisianSelection,
    sources: Sequence[str],
    candidates: int,
) -> Path:
    """Write train.jsonl, holdout.jsonl, and selection manifest."""
    out_dir = PROC / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = schema.write_jsonl(out_dir / "train.jsonl", selection.train)
    n_holdout = schema.write_jsonl(out_dir / "holdout.jsonl", selection.holdout)

    manifest_dir = MANIFEST / slice_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "selection.json"
    manifest = selection.manifest(
        candidates=candidates, sources=sources, slice_name=slice_name
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return manifest_path


def get_t_code() -> Optional[int]:
    """Read T_code (coding anchor tokens) from retention selection manifest."""
    path = MANIFEST / "retention" / "selection.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    tb = data.get("token_budget", {})
    return tb.get("anchor_tokens")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Token-budget selection for Tunisian adaptation slices."
    )
    parser.add_argument(
        "--slice",
        choices=["arabizi", "arabic_derja", "franco_tunisian", "all"],
        default="all",
        help="slice to process (default: all)",
    )
    args = parser.parse_args()

    slices = ["arabizi", "arabic_derja", "franco_tunisian"] if args.slice == "all" else [args.slice]

    # Get T_code from retention for franco_tunisian anchoring
    t_code = get_t_code()
    if t_code:
        print(f"Retention T_code (anchor_tokens): {t_code}")
    else:
        print("Warning: retention T_code not found; franco_tunisian will use self-anchoring")

    counter = ApproximateTokenCounter()

    for slice_name in slices:
        print(f"\n=== Processing {slice_name} ===")

        spec = load_slice_config(slice_name)
        tb = spec["token_budget"]

        # Determine T_anchor source
        t_anchor_override = None
        if slice_name == "franco_tunisian" and t_code:
            # T_franco = 2.4 × T_code
            t_anchor_override = int(2.4 * t_code)
            print(f"T_franco target = 2.4 × T_code = {t_anchor_override}")

        # Load candidate pool (combine existing train + holdout)
        records = load_pool_records(slice_name)
        print(f"Loaded {len(records)} candidate records")

        sources = sorted({str(r.get("source", "")) for r in records})
        print(f"Sources: {sources}")

        # Run token-budget selection
        selection = select_token_budget(
            records, spec, slice_name,
            t_code=t_anchor_override,
            counter=counter,
        )

        print(f"Rejected: {selection.rejected}")
        print(f"Train: {len(selection.train)} records")
        print(f"Holdout: {len(selection.holdout)} records")
        print(f"T_anchor: {selection.anchor_tokens} ({selection.token_metric})")
        print(f"Train assistant tokens: {selection.train_assistant_tokens}")
        print(f"Holdout token totals: {selection.holdout_token_totals}")

        # Write outputs
        manifest_path = write_outputs(slice_name, selection, sources, len(records))
        print(f"Wrote {manifest_path}")
        print(f"Wrote {PROC / slice_name / 'train.jsonl'} ({len(selection.train)} records)")
        print(f"Wrote {PROC / slice_name / 'holdout.jsonl'} ({len(selection.holdout)} records)")

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
