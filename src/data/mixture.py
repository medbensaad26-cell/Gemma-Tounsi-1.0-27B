"""Project-wide mixture validation.

Checks the final Gemma Tounsi 1.0 training cocktail against
``configs/data/mixture.yaml``. On any violation it returns a clear error and
STOPS. It never silently rebalances, drops, or "fixes" a slice: an invalid
mixture is a defect in the mixture definition or in the data, not something to
paper over at run time.

Constraints enforced (mirrors the config's ``validation`` block):

1. slice shares sum to 1.0
2. every required slice exists
3. requested per-slice counts are feasible against the available data
4. arabizi technical ratio  >= arabizi_technical_min
5. arabic_derja technical ratio >= derja_technical_min
6. retention is treated as its own slice
7. msa_formal is NOT counted as retention
8. no holdout/eval record leaks into any training slice

CLI
---
    python -m src.data.mixture \\
        --slice arabizi=data/processed/arabizi.jsonl \\
        --slice arabic_derja=data/processed/derja.jsonl \\
        --slice franco_tunisian=data/processed/franco.jsonl \\
        --slice msa_formal=data/processed/msa.jsonl \\
        --slice retention=data/processed/retention/train.jsonl \\
        --holdout data/processed/retention/holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .schema import REPO_ROOT, Schema, load_jsonl, load_schema

__all__ = [
    "MixtureSpec",
    "MixtureReport",
    "MixtureError",
    "load_mixture_spec",
    "validate_mixture",
    "main",
]

#: Default mixture specification.
MIXTURE_CONFIG_PATH: Path = REPO_ROOT / "configs" / "data" / "mixture.yaml"


class MixtureError(Exception):
    """The mixture specification or the data violates a hard constraint."""


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MixtureSpec:
    """Typed view over ``configs/data/mixture.yaml``."""

    slices: Dict[str, float]
    arabizi_technical_min: float
    derja_technical_min: float
    technical_categories: Tuple[str, ...]
    technical_quota_slices: Tuple[str, ...]
    slice_semantics: Dict[str, str]
    required_slices: Tuple[str, ...]
    require_total: float
    total_tolerance: float
    enforce_technical_minimums: bool
    forbid_holdout_in_training: bool

    def technical_minimum(self, slice_name: str) -> Optional[float]:
        """Return the technical minimum for a slice, or ``None`` if it has none."""
        if slice_name == "arabizi":
            return self.arabizi_technical_min
        if slice_name == "arabic_derja":
            return self.derja_technical_min
        return None

    def validate(self) -> List[str]:
        """Validate the SPEC itself (no data). Returns a list of errors."""
        errors: List[str] = []

        # 1. total must equal 1.0
        total = sum(self.slices.values())
        if abs(total - self.require_total) > self.total_tolerance:
            errors.append(
                f"mixture shares sum to {total:.6f}, expected {self.require_total} "
                f"(tolerance {self.total_tolerance}); fix configs/data/mixture.yaml "
                f"— the mixture is never silently rebalanced"
            )

        # 2. every required slice must exist
        for name in self.required_slices:
            if name not in self.slices:
                errors.append(f"required slice '{name}' is missing from the mixture")

        for name, share in self.slices.items():
            if not 0.0 <= share <= 1.0:
                errors.append(f"slice '{name}' has share {share}, expected 0..1")

        # 6 & 7. retention and msa_formal are distinct slices with distinct roles
        if "retention" in self.slices and "msa_formal" in self.slices:
            retention_role = self.slice_semantics.get("retention")
            msa_role = self.slice_semantics.get("msa_formal")
            if retention_role != "english_capability_preservation":
                errors.append(
                    f"retention must be 'english_capability_preservation', "
                    f"got '{retention_role}'"
                )
            if msa_role == "english_capability_preservation":
                errors.append(
                    "msa_formal is declared as retention; MSA/formal Arabic is "
                    "register coverage and is NEVER counted as retention"
                )

        return errors


def load_mixture_spec(path: Optional[Path | str] = None) -> MixtureSpec:
    """Load the mixture specification.

    Raises:
        MixtureError: file missing/unparsable, or the spec itself is invalid.
    """
    config_path = Path(path) if path else MIXTURE_CONFIG_PATH
    if not config_path.is_file():
        raise MixtureError(f"mixture config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MixtureError(f"{config_path} must contain a YAML mapping")

    slices = raw.get("slices") or {}
    if not slices:
        raise MixtureError(f"{config_path} defines no slices")

    cross = raw.get("cross_cutting") or {}
    validation = raw.get("validation") or {}

    spec = MixtureSpec(
        slices={str(k): float(v) for k, v in slices.items()},
        arabizi_technical_min=float(cross.get("arabizi_technical_min", 0.20)),
        derja_technical_min=float(cross.get("derja_technical_min", 0.20)),
        technical_categories=tuple(
            raw.get("technical_categories", ("mathematics", "reasoning", "coding"))
        ),
        technical_quota_slices=tuple(
            raw.get("technical_quota_slices", ("arabizi", "arabic_derja"))
        ),
        slice_semantics={
            str(k): str(v) for k, v in (raw.get("slice_semantics") or {}).items()
        },
        required_slices=tuple(validation.get("required_slices", tuple(slices))),
        require_total=float(validation.get("require_total", 1.0)),
        total_tolerance=float(validation.get("total_tolerance", 1e-6)),
        enforce_technical_minimums=bool(
            validation.get("enforce_technical_minimums", True)
        ),
        forbid_holdout_in_training=bool(
            validation.get("forbid_holdout_in_training", True)
        ),
    )

    spec_errors = spec.validate()
    if spec_errors:
        raise MixtureError(
            "invalid mixture specification:\n  - " + "\n  - ".join(spec_errors)
        )
    return spec


# --------------------------------------------------------------------------- #
# Data-level validation
# --------------------------------------------------------------------------- #


@dataclass
class SliceStats:
    """Per-slice counts used by the mixture checks."""

    name: str
    total: int
    technical: int
    ids: set = field(default_factory=set)

    @property
    def technical_ratio(self) -> float:
        """Fraction of the slice tagged as a technical category."""
        return self.technical / self.total if self.total else 0.0


@dataclass
class MixtureReport:
    """Outcome of mixture validation."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    slices: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_share: float = 0.0
    total_examples: int = 0

    def raise_if_invalid(self) -> None:
        """Raise ``MixtureError`` listing every violation."""
        if not self.ok:
            raise MixtureError(
                "mixture validation failed:\n  - " + "\n  - ".join(self.errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "total_share": self.total_share,
            "total_examples": self.total_examples,
            "slices": self.slices,
            "errors": self.errors,
        }

    def render(self) -> str:
        """Return a human-readable report."""
        lines = [f"mixture: {'OK' if self.ok else 'FAILED'}"]
        lines.append(f"  total share   : {self.total_share:.4f}")
        lines.append(f"  total examples: {self.total_examples}")
        for name in sorted(self.slices):
            info = self.slices[name]
            line = (
                f"  {name:<16} share={info['share']:.2f} "
                f"examples={info['examples']}"
            )
            if info.get("technical_min") is not None:
                line += (
                    f" technical={info['technical']}"
                    f" ({info['technical_ratio'] * 100:.1f}%"
                    f" >= {info['technical_min'] * 100:.0f}%)"
                )
            lines.append(line)
        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        return "\n".join(lines)


def _slice_stats(
    name: str, records: Sequence[Dict[str, Any]], spec: MixtureSpec
) -> SliceStats:
    """Compute counts for one slice."""
    technical = sum(
        1 for r in records if str(r.get("category")) in spec.technical_categories
    )
    ids = {r["id"] for r in records if isinstance(r.get("id"), str)}
    return SliceStats(name=name, total=len(records), technical=technical, ids=ids)


def validate_mixture(
    slice_records: Dict[str, Sequence[Dict[str, Any]]],
    spec: Optional[MixtureSpec] = None,
    *,
    schema: Optional[Schema] = None,
    holdout_ids: Optional[Sequence[str]] = None,
    requested_counts: Optional[Dict[str, int]] = None,
) -> MixtureReport:
    """Validate the assembled mixture against the specification.

    Args:
        slice_records: canonical records per slice name.
        spec: mixture specification; loaded from config when omitted.
        schema: canonical schema; loaded from config when omitted.
        holdout_ids: ids reserved for holdout/evaluation. Any of these appearing
            in a training slice is a hard failure (train/test contamination).
        requested_counts: optional per-slice counts the caller intends to draw;
            checked for feasibility against what is actually available.

    Returns:
        A :class:`MixtureReport`. Inspect ``ok``/``errors``, or call
        ``raise_if_invalid()`` to turn violations into an exception.
    """
    spec = spec or load_mixture_spec()
    schema = schema or load_schema()

    errors: List[str] = []
    report_slices: Dict[str, Dict[str, Any]] = {}

    # 1. total shares
    total_share = sum(spec.slices.values())
    if abs(total_share - spec.require_total) > spec.total_tolerance:
        errors.append(
            f"mixture shares sum to {total_share:.6f}, expected {spec.require_total}"
        )

    # 2. every required slice must be present in the DATA as well
    for name in spec.required_slices:
        if name not in slice_records:
            errors.append(
                f"required slice '{name}' has no dataset; expected one of the "
                f"provided slices: {sorted(slice_records) or 'none'}"
            )

    stats: Dict[str, SliceStats] = {}
    for name, records in slice_records.items():
        if name not in spec.slices:
            errors.append(
                f"slice '{name}' is not declared in the mixture "
                f"(declared: {', '.join(sorted(spec.slices))})"
            )
            continue
        stats[name] = _slice_stats(name, records, spec)

    total_examples = sum(item.total for item in stats.values())

    for name, item in stats.items():
        minimum = spec.technical_minimum(name)
        info: Dict[str, Any] = {
            "share": spec.slices.get(name, 0.0),
            "examples": item.total,
            "technical": item.technical,
            "technical_ratio": round(item.technical_ratio, 4),
            "technical_min": minimum,
            "semantics": spec.slice_semantics.get(name),
        }
        report_slices[name] = info

        # 3. feasibility of requested counts
        if requested_counts and name in requested_counts:
            wanted = requested_counts[name]
            if wanted > item.total:
                errors.append(
                    f"slice '{name}': requested {wanted} examples but only "
                    f"{item.total} are available (infeasible)"
                )

        # empty slices cannot satisfy anything downstream
        if item.total == 0:
            errors.append(f"slice '{name}' is empty")

        # 4 & 5. cross-cutting technical quota
        if spec.enforce_technical_minimums and minimum is not None and item.total:
            if item.technical_ratio < minimum:
                errors.append(
                    f"slice '{name}': technical ratio "
                    f"{item.technical_ratio * 100:.1f}% is below the required "
                    f"{minimum * 100:.0f}% "
                    f"({item.technical}/{item.total} in "
                    f"{', '.join(spec.technical_categories)})"
                )

    # 6 & 7. retention / msa_formal must stay distinct
    if "retention" in stats and "msa_formal" in stats:
        if spec.slice_semantics.get("msa_formal") == spec.slice_semantics.get("retention"):
            errors.append(
                "msa_formal and retention share the same declared purpose; MSA is "
                "register coverage and must never be counted as retention"
            )
    # Retention must be primarily English, never Tunisian adaptation data.
    if "retention" in slice_records:
        non_english = [
            r.get("id")
            for r in slice_records["retention"]
            if r.get("language") != "en"
        ]
        if non_english:
            errors.append(
                f"retention slice contains {len(non_english)} non-English record(s) "
                f"(e.g. {non_english[:3]}); retention preserves the base model's "
                f"English capabilities and is not Tunisian adaptation data"
            )

    # 8. no holdout/eval leakage into training slices
    if spec.forbid_holdout_in_training and holdout_ids:
        reserved = set(holdout_ids)
        for name, item in stats.items():
            leaked = item.ids & reserved
            if leaked:
                errors.append(
                    f"slice '{name}' contains {len(leaked)} holdout/eval record(s) "
                    f"(e.g. {sorted(leaked)[:3]}); holdout data must never be trained on"
                )

    return MixtureReport(
        ok=not errors,
        errors=errors,
        slices=report_slices,
        total_share=total_share,
        total_examples=total_examples,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_slice_argument(value: str) -> Tuple[str, str]:
    """Parse a ``name=path`` CLI pair."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--slice expects NAME=PATH, got {value!r}"
        )
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--slice expects NAME=PATH, got {value!r}")
    return name, path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.mixture",
        description="Validate the Gemma Tounsi training mixture. Never rebalances.",
    )
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        dest="slices",
        type=_parse_slice_argument,
        metavar="NAME=PATH",
        help="a processed slice, e.g. arabizi=data/processed/arabizi.jsonl",
    )
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        help="holdout/eval JSONL whose ids must NOT appear in any training slice",
    )
    parser.add_argument("--config", default=None, help="override mixture.yaml path")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit machine-readable JSON"
    )
    args = parser.parse_args(argv)

    if not args.slices:
        print("error: at least one --slice NAME=PATH is required", file=sys.stderr)
        return 2

    try:
        spec = load_mixture_spec(args.config)
    except MixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    slice_records: Dict[str, Sequence[Dict[str, Any]]] = {}
    try:
        for name, path in args.slices:
            slice_records[name] = load_jsonl(path).records
        holdout_ids: List[str] = []
        for path in args.holdout:
            holdout_ids.extend(
                r["id"] for r in load_jsonl(path).records if isinstance(r.get("id"), str)
            )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = validate_mixture(slice_records, spec, holdout_ids=holdout_ids)

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.render())

    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
