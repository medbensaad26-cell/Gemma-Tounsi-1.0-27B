"""Retention slice selection.

The retention slice preserves Gemma 3 27B's ORIGINAL general capabilities after
Tunisian adaptation. It is primarily ENGLISH and is NOT a Tunisian-language
dataset; it is also NOT the same thing as the ``msa_formal`` register slice.

Given a pool of canonical candidate records plus ``configs/data/retention.yaml``,
this module produces:

  * a retention TRAIN set that hits every per-category target exactly,
  * a stratified HOLDOUT reserved BEFORE selection (never trained on),
  * a selection MANIFEST recording provenance and reproducibility inputs.

Guarantees:
  * category targets are HARD — never unconstrained random sampling;
  * deterministic — same candidates + same seed => byte-identical output;
  * source diversity — an optional per-source cap inside each category;
  * fails loudly when a category cannot be filled (no silent under-filling).

No external dataset is referenced. Candidate pools arrive as canonical JSONL.

CLI
---
    python -m src.data.retention --candidates data/synthetic/raw/retention_pool.jsonl
    python -m src.data.retention --candidates POOL.jsonl --scale 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .schema import (
    REPO_ROOT,
    Schema,
    load_jsonl,
    load_schema,
    validate_record,
    write_jsonl,
)
from .split import stable_key, stratified_split

__all__ = [
    "RetentionSpec",
    "RetentionSelection",
    "SelectionError",
    "load_retention_spec",
    "select_retention",
    "main",
]

#: Default retention specification.
RETENTION_CONFIG_PATH: Path = REPO_ROOT / "configs" / "data" / "retention.yaml"


class SelectionError(Exception):
    """Retention selection could not satisfy the specification."""


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetentionSpec:
    """Typed view over ``configs/data/retention.yaml``."""

    target_examples: int
    categories: Dict[str, int]
    category_aliases: Dict[str, str]
    allowed_languages: Tuple[str, ...]
    allow_other_languages: bool
    holdout_examples: int
    holdout_stratified: bool
    seed: int
    stratify_by: str
    max_share_per_source: float
    exclude_flagged: bool
    on_insufficient_candidates: str
    train_output: str
    holdout_output: str
    manifest_output: str
    purpose: str
    is_tunisian_adaptation: bool

    def canonical_targets(self) -> Dict[str, int]:
        """Return targets keyed by CANONICAL category name.

        ``instruction_following`` in the config maps to the canonical
        ``general_instruction`` category from ``configs/data/schema.yaml``.
        """
        resolved: Dict[str, int] = {}
        for name, count in self.categories.items():
            resolved[self.category_aliases.get(name, name)] = count
        return resolved

    def validate(self, schema: Optional[Schema] = None) -> None:
        """Check internal consistency of the spec.

        Raises:
            SelectionError: targets do not sum to ``target_examples``, a category
                is unknown to the canonical schema, or the purpose was corrupted.
        """
        schema = schema or load_schema()

        total = sum(self.categories.values())
        if total != self.target_examples:
            raise SelectionError(
                f"retention category targets sum to {total} but target_examples "
                f"is {self.target_examples}"
            )

        unknown = [
            name for name in self.canonical_targets() if name not in schema.categories
        ]
        if unknown:
            raise SelectionError(
                f"unknown retention categories (not in the canonical schema): "
                f"{', '.join(sorted(unknown))}; allowed: {', '.join(schema.categories)}"
            )

        if self.is_tunisian_adaptation:
            raise SelectionError(
                "retention.yaml declares is_tunisian_adaptation: true — retention "
                "is English capability preservation, NOT Tunisian adaptation"
            )

        if self.holdout_examples < 0:
            raise SelectionError(
                f"holdout target_examples must be >= 0, got {self.holdout_examples}"
            )


def load_retention_spec(path: Optional[Path | str] = None) -> RetentionSpec:
    """Load and validate the retention specification."""
    config_path = Path(path) if path else RETENTION_CONFIG_PATH
    if not config_path.is_file():
        raise SelectionError(f"retention config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SelectionError(f"{config_path} must contain a YAML mapping")

    categories = raw.get("categories") or {}
    if not categories:
        raise SelectionError(f"{config_path} defines no categories")

    language = raw.get("language") or {}
    holdout = raw.get("holdout") or {}
    selection = raw.get("selection") or {}
    output = raw.get("output") or {}

    spec = RetentionSpec(
        target_examples=int(raw.get("target_examples", 0)),
        categories={str(k): int(v) for k, v in categories.items()},
        category_aliases={
            str(k): str(v) for k, v in (raw.get("category_aliases") or {}).items()
        },
        allowed_languages=tuple(language.get("allowed", ("en",))),
        allow_other_languages=bool(language.get("allow_other_languages", False)),
        holdout_examples=int(holdout.get("target_examples", 0)),
        holdout_stratified=bool(holdout.get("stratified", True)),
        seed=int(selection.get("seed", 42)),
        stratify_by=str(selection.get("stratify_by", "category")),
        max_share_per_source=float(selection.get("max_share_per_source", 1.0)),
        exclude_flagged=bool(selection.get("exclude_flagged", True)),
        on_insufficient_candidates=str(
            selection.get("on_insufficient_candidates", "error")
        ),
        train_output=str(output.get("train", "data/processed/retention/train.jsonl")),
        holdout_output=str(
            output.get("holdout", "data/processed/retention/holdout.jsonl")
        ),
        manifest_output=str(
            output.get("manifest", "data/manifests/retention/selection.json")
        ),
        purpose=str(raw.get("purpose", "english_capability_preservation")),
        is_tunisian_adaptation=bool(raw.get("is_tunisian_adaptation", False)),
    )
    spec.validate()
    return spec


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@dataclass
class RetentionSelection:
    """Result of a retention selection run."""

    train: List[Dict[str, Any]] = field(default_factory=list)
    holdout: List[Dict[str, Any]] = field(default_factory=list)
    train_targets: Dict[str, int] = field(default_factory=dict)
    holdout_targets: Dict[str, int] = field(default_factory=dict)
    rejected: Dict[str, int] = field(default_factory=dict)
    seed: int = 42
    scale: float = 1.0

    @property
    def train_counts(self) -> Dict[str, int]:
        """Per-category counts in the training set."""
        return _count_by_category(self.train)

    @property
    def holdout_counts(self) -> Dict[str, int]:
        """Per-category counts in the holdout set."""
        return _count_by_category(self.holdout)

    def assert_no_contamination(self) -> None:
        """Raise ``SelectionError`` if any id appears in both partitions."""
        train_ids = {r["id"] for r in self.train if isinstance(r.get("id"), str)}
        holdout_ids = {r["id"] for r in self.holdout if isinstance(r.get("id"), str)}
        overlap = train_ids & holdout_ids
        if overlap:
            raise SelectionError(
                f"retention train/holdout contamination: {len(overlap)} shared id(s), "
                f"e.g. {sorted(overlap)[:5]}"
            )

    def manifest(self, *, candidates: int, sources: Sequence[str]) -> Dict[str, Any]:
        """Build the reproducibility manifest for this selection."""
        return {
            "slice": "retention",
            "purpose": "english_capability_preservation",
            "note": (
                "Retention preserves the base model's original (primarily English) "
                "capabilities. It is NOT Tunisian adaptation data and is NOT the "
                "msa_formal register slice."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "scale": self.scale,
            "candidates_considered": candidates,
            "train": {
                "total": len(self.train),
                "targets": self.train_targets,
                "counts": self.train_counts,
            },
            "holdout": {
                "total": len(self.holdout),
                "targets": self.holdout_targets,
                "counts": self.holdout_counts,
            },
            "rejected": self.rejected,
            "sources": sorted(set(sources)),
            "deterministic": True,
        }


def _count_by_category(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count records per canonical ``category``."""
    counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get("category"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _scale_targets(targets: Dict[str, int], scale: float) -> Dict[str, int]:
    """Scale per-category targets, keeping at least 1 per non-zero category."""
    if scale == 1.0:
        return dict(targets)
    if scale <= 0:
        raise SelectionError(f"scale must be > 0, got {scale}")
    return {
        name: max(1, int(round(count * scale))) if count else 0
        for name, count in targets.items()
    }


def _pick_with_source_diversity(
    pool: Sequence[Dict[str, Any]],
    need: int,
    *,
    seed: int,
    max_share_per_source: float,
) -> List[Dict[str, Any]]:
    """Pick ``need`` records deterministically, spreading across sources.

    Records are visited in deterministic hash order. A per-source cap is applied
    first; if the cap makes the target unreachable, the remainder is filled from
    the leftovers (still deterministically), because hitting the category target
    matters more than a soft diversity preference.
    """
    ordered = sorted(pool, key=lambda record: stable_key(record, seed))
    if need >= len(ordered):
        return list(ordered)

    cap = (
        max(1, int(need * max_share_per_source))
        if 0 < max_share_per_source < 1.0
        else need
    )

    picked: List[Dict[str, Any]] = []
    used: Dict[str, int] = {}
    leftovers: List[Dict[str, Any]] = []

    for record in ordered:
        if len(picked) >= need:
            leftovers.append(record)
            continue
        source = str(record.get("source", ""))
        if used.get(source, 0) >= cap:
            leftovers.append(record)
            continue
        picked.append(record)
        used[source] = used.get(source, 0) + 1

    for record in leftovers:
        if len(picked) >= need:
            break
        picked.append(record)

    return picked


def select_retention(
    candidates: Sequence[Dict[str, Any]],
    spec: Optional[RetentionSpec] = None,
    *,
    schema: Optional[Schema] = None,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> RetentionSelection:
    """Select the retention train set and reserve a stratified holdout.

    Pipeline: validate -> filter (language / flags / category) -> reserve
    holdout (stratified) -> fill per-category train targets with source
    diversity.

    Args:
        candidates: canonical candidate records.
        spec: retention specification; loaded from config when omitted.
        schema: canonical schema; loaded from config when omitted.
        scale: scale factor for targets. ``1.0`` means the real 20,000/2,500
            specification; smaller values let the synthetic pipeline exercise the
            same code path on a tiny fixture.
        seed: override the spec's deterministic seed.

    Raises:
        SelectionError: a category cannot be filled (when the spec says
            ``on_insufficient_candidates: error``), or the pool is unusable.
    """
    spec = spec or load_retention_spec()
    schema = schema or load_schema()
    spec.validate(schema)

    effective_seed = spec.seed if seed is None else seed
    targets = _scale_targets(spec.canonical_targets(), scale)
    total_target = sum(targets.values())
    holdout_target = (
        spec.holdout_examples
        if scale == 1.0
        else max(0, int(round(spec.holdout_examples * scale)))
    )

    rejected: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    # --- filter ------------------------------------------------------------
    eligible: List[Dict[str, Any]] = []
    for index, record in enumerate(candidates):
        if validate_record(record, schema, line=index + 1):
            reject("schema_invalid")
            continue
        if spec.exclude_flagged and isinstance(record.get("quality"), dict):
            if record["quality"].get("flagged") is True:
                reject("quality_flagged")
                continue
        if not spec.allow_other_languages:
            if record.get("language") not in spec.allowed_languages:
                reject("language_not_allowed")
                continue
        if str(record.get("category")) not in targets:
            reject("category_not_requested")
            continue
        eligible.append(record)

    if not eligible:
        raise SelectionError(
            "no eligible retention candidates after filtering "
            f"(rejections: {rejected or 'none'}); retention requires "
            f"language in {list(spec.allowed_languages)} and categories "
            f"{sorted(targets)}"
        )

    # --- deduplicate by id (first occurrence wins, deterministic) ----------
    unique: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in eligible:
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id in seen_ids:
            reject("duplicate_id")
            continue
        if isinstance(record_id, str):
            seen_ids.add(record_id)
        unique.append(record)

    # --- feasibility -------------------------------------------------------
    available = _count_by_category(unique)
    needed_total = {
        name: targets[name]
        + _holdout_share(targets, holdout_target).get(name, 0)
        for name in targets
    }
    shortfalls = {
        name: (available.get(name, 0), need)
        for name, need in needed_total.items()
        if available.get(name, 0) < need
    }
    if shortfalls and spec.on_insufficient_candidates == "error":
        detail = ", ".join(
            f"{name}: have {have}, need {need}" for name, (have, need) in sorted(shortfalls.items())
        )
        raise SelectionError(
            "insufficient retention candidates per category "
            f"(train+holdout): {detail}. Add candidates or lower --scale; "
            "the selector never under-fills a category silently."
        )

    # --- holdout first (stratified, reserved before training selection) ----
    if holdout_target > 0:
        holdout_target = min(holdout_target, len(unique))
        split = stratified_split(
            unique,
            holdout_target,
            stratify_by=spec.stratify_by,
            seed=effective_seed,
        )
        holdout_records = split.holdout
        remaining = split.train
    else:
        holdout_records = []
        remaining = list(unique)

    # --- train selection: hard per-category targets ------------------------
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for record in remaining:
        by_category.setdefault(str(record.get("category")), []).append(record)

    train_records: List[Dict[str, Any]] = []
    for name in sorted(targets):
        need = targets[name]
        if need <= 0:
            continue
        pool = by_category.get(name, [])
        if len(pool) < need and spec.on_insufficient_candidates == "error":
            raise SelectionError(
                f"category '{name}': need {need} training examples but only "
                f"{len(pool)} candidate(s) remain after holdout reservation"
            )
        train_records.extend(
            _pick_with_source_diversity(
                pool,
                min(need, len(pool)),
                seed=effective_seed,
                max_share_per_source=spec.max_share_per_source,
            )
        )

    train_records.sort(key=lambda record: str(record.get("id", "")))
    holdout_records = sorted(holdout_records, key=lambda record: str(record.get("id", "")))

    selection = RetentionSelection(
        train=train_records,
        holdout=holdout_records,
        train_targets=targets,
        holdout_targets=_holdout_share(targets, holdout_target),
        rejected=dict(sorted(rejected.items())),
        seed=effective_seed,
        scale=scale,
    )
    selection.assert_no_contamination()

    if len(selection.train) != total_target and spec.on_insufficient_candidates == "error":
        raise SelectionError(
            f"selected {len(selection.train)} training examples but the target is "
            f"{total_target}; per-category counts: {selection.train_counts}"
        )

    return selection


def _holdout_share(targets: Dict[str, int], holdout_total: int) -> Dict[str, int]:
    """Apportion the holdout across categories in proportion to the targets."""
    if holdout_total <= 0:
        return {name: 0 for name in targets}
    weight_sum = sum(targets.values())
    if weight_sum == 0:
        return {name: 0 for name in targets}
    exact = {name: holdout_total * count / weight_sum for name, count in targets.items()}
    floored = {name: int(value) for name, value in exact.items()}
    remainder = holdout_total - sum(floored.values())
    order = sorted(targets, key=lambda name: (exact[name] - floored[name], name), reverse=True)
    for index in range(remainder):
        floored[order[index % len(order)]] += 1
    return floored


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.retention",
        description=(
            "Select the retention train set + holdout from a canonical candidate "
            "pool. Retention = English capability preservation (NOT Tunisian "
            "adaptation, NOT msa_formal)."
        ),
    )
    parser.add_argument(
        "--candidates", required=True, help="canonical candidate JSONL pool"
    )
    parser.add_argument("--config", default=None, help="override retention.yaml path")
    parser.add_argument("--train-out", default=None, help="override train output path")
    parser.add_argument("--holdout-out", default=None, help="override holdout output path")
    parser.add_argument("--manifest-out", default=None, help="override manifest path")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "scale the 20000/2500 targets (default 1.0). Use a small value to "
            "exercise the pipeline on synthetic fixtures."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    args = parser.parse_args(argv)

    try:
        spec = load_retention_spec(args.config)
        loaded = load_jsonl(args.candidates)
        selection = select_retention(
            loaded.records, spec, scale=args.scale, seed=args.seed
        )
    except (SelectionError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    train_out = Path(args.train_out or spec.train_output)
    holdout_out = Path(args.holdout_out or spec.holdout_output)
    manifest_out = Path(args.manifest_out or spec.manifest_output)

    write_jsonl(train_out, selection.train)
    write_jsonl(holdout_out, selection.holdout)

    manifest = selection.manifest(
        candidates=len(loaded.records),
        sources=[str(r.get("source", "")) for r in selection.train],
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"retention selection (scale={args.scale}, seed={selection.seed})")
    print(f"  train   : {len(selection.train):>6} -> {train_out}")
    print(f"            counts {selection.train_counts}")
    print(f"  holdout : {len(selection.holdout):>6} -> {holdout_out}")
    print(f"            counts {selection.holdout_counts}")
    if selection.rejected:
        print(f"  rejected: {selection.rejected}")
    print(f"  manifest: {manifest_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
