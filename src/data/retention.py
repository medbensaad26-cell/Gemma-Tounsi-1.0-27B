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
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

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
    "TokenCounter",
    "ApproximateTokenCounter",
    "TokenCount",
    "load_retention_spec",
    "select_retention",
    "select_retention_token_budget",
    "main",
]

#: Default retention specification.
RETENTION_CONFIG_PATH: Path = REPO_ROOT / "configs" / "data" / "retention.yaml"

#: Dependency-free chars-per-token approximation (matches src/data/stats.py).
CHARS_PER_TOKEN: float = 4.0

#: Holdout ratio of the total retention token budget when ``holdout.target_tokens``
#: is not set (matches the legacy 2,500 / 20,000 = 12.5% example ratio).
DEFAULT_HOLDOUT_RATIO: float = 0.125


@dataclass(frozen=True)
class TokenCount:
    """Token counts for one formatted example.

    Attributes:
        total: every token the model reads (chat template included).
        assistant: tokens in assistant turns only — the actual loss signal.
    """

    total: int
    assistant: int


@runtime_checkable
class TokenCounter(Protocol):
    """Count tokens in text or formatted messages.

    Implementations swap between the dependency-free approximation and
    Gemma's own tokenizer once it is available (gated model, not installed
    locally).  Selection logic never depends on a concrete counter.
    """

    def count_text(self, text: str) -> int:
        """Count tokens in raw text."""

    def count_messages(
        self, messages: Sequence[Dict[str, Any]]
    ) -> TokenCount:
        """Count tokens in a formatted conversation (total + assistant-only)."""


class ApproximateTokenCounter:
    """Dependency-free token counter: ``chars / CHARS_PER_TOKEN``.

    Used until Gemma's tokenizer is available.  Swappable for a
    :class:`GemmaTokenCounter` (using ``transformers.AutoTokenizer`` +
    ``apply_chat_template``) without changing any selection logic.
    """

    def __init__(self, chars_per_token: float = CHARS_PER_TOKEN) -> None:
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def count_messages(
        self, messages: Sequence[Dict[str, Any]]
    ) -> TokenCount:
        total = 0
        assistant = 0
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content", ""))
            n = self.count_text(content)
            total += n
            total += 1  # role-marker overhead per turn
            if turn.get("role") == "assistant":
                assistant += n
        return TokenCount(total=total, assistant=assistant)


class SelectionError(Exception):
    """Retention selection could not satisfy the specification."""


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetentionSpec:
    """Typed view over ``configs/data/retention.yaml``.

    Supports two selection modes:

    * ``legacy`` — exact per-category example counts (synthetic smoke tests).
    * ``token_budget`` — fill each category to a measured token budget
      anchored to the coding slice (production).
    """

    # --- legacy example-count fields ---------------------------------------
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

    # --- token-budget fields ----------------------------------------------
    token_budget_enabled: bool
    category_multipliers: Dict[str, float]
    anchor_category: str
    anchor_target_examples: int
    token_tolerance: float
    holdout_reserve_first: bool
    holdout_unit: str
    holdout_target_tokens: Optional[float]
    holdout_category_shares: Dict[str, float]
    context_window_tokens: int
    max_formatted_tokens_per_example: int
    tokenizer: str
    include_chat_template: bool
    token_count_metrics: Tuple[str, ...]
    token_budget_metric: str

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

        # --- token-budget consistency -------------------------------------
        if self.token_budget_enabled:
            anchor = self.anchor_category
            if anchor not in self.category_multipliers:
                raise SelectionError(
                    f"token_budget anchor category '{anchor}' is not in "
                    f"category_multipliers: {sorted(self.category_multipliers)}"
                )
            unknown_mult = [
                name
                for name in self.category_multipliers
                if name not in schema.categories
            ]
            if unknown_mult:
                raise SelectionError(
                    f"unknown token-budget categories (not in the canonical "
                    f"schema): {', '.join(sorted(unknown_mult))}"
                )
            holdout_share_sum = sum(self.holdout_category_shares.values())
            if abs(holdout_share_sum - 1.0) > 1e-6:
                raise SelectionError(
                    f"holdout category_shares sum to {holdout_share_sum:.6f}, "
                    f"expected 1.0"
                )
            if self.token_budget_metric not in ("total_tokens", "assistant_tokens"):
                raise SelectionError(
                    f"token_budget.metric must be 'total_tokens' or "
                    f"'assistant_tokens', got {self.token_budget_metric!r}"
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
    token_budget = raw.get("token_budget") or {}

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
        # --- token-budget fields ------------------------------------------
        token_budget_enabled=bool(token_budget.get("unit") == "tokens"),
        category_multipliers={
            str(k): float(v)
            for k, v in (token_budget.get("category_multipliers") or {}).items()
        },
        anchor_category=str(
            (token_budget.get("anchor") or {}).get("category", "coding")
        ),
        anchor_target_examples=int(
            (token_budget.get("anchor") or {}).get("target_examples", 5000)
        ),
        token_tolerance=float(token_budget.get("tolerance", 0.02)),
        holdout_reserve_first=bool(
            (token_budget.get("holdout") or {}).get("reserve_first", True)
        ),
        holdout_unit=str((token_budget.get("holdout") or {}).get("unit", "tokens")),
        holdout_target_tokens=(
            (token_budget.get("holdout") or {}).get("target_tokens")
        ),
        holdout_category_shares={
            str(k): float(v)
            for k, v in (
                (token_budget.get("holdout") or {}).get("category_shares") or {}
            ).items()
        },
        context_window_tokens=int(
            token_budget.get("context_window_tokens", 16384)
        ),
        max_formatted_tokens_per_example=int(
            token_budget.get("max_formatted_tokens_per_example", 15500)
        ),
        tokenizer=str(token_budget.get("tokenizer", "gemma")),
        include_chat_template=bool(
            token_budget.get("include_chat_template", True)
        ),
        token_count_metrics=tuple(
            token_budget.get("count", ["total_tokens", "assistant_tokens"])
        ),
        token_budget_metric=str(
            token_budget.get("metric", "assistant_tokens")
        ),
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

    # --- token-budget mode -------------------------------------------------
    anchor_tokens: int = 0
    token_metric: str = "assistant_tokens"
    train_token_targets: Dict[str, float] = field(default_factory=dict)
    train_total_tokens: Dict[str, int] = field(default_factory=dict)
    train_assistant_tokens: Dict[str, int] = field(default_factory=dict)
    holdout_token_totals: Dict[str, int] = field(default_factory=dict)

    # --- tagged-source hygiene (Task 9, retention_haithem.md) ----------
    tag_rule_version: str = ""
    tagging_audit_results: List[Dict[str, Any]] = field(default_factory=list)
    template_cap_counts: Dict[str, Any] = field(default_factory=dict)
    translation_task_decision: str = ""
    cross_contamination_check: Dict[str, Any] = field(default_factory=dict)

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
        manifest = {
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
        if self.token_metric:
            manifest["token_budget"] = {
                "metric": self.token_metric,
                "anchor_tokens": self.anchor_tokens,
                "train_token_targets": self.train_token_targets,
                "train_total_tokens": self.train_total_tokens,
                "train_assistant_tokens": self.train_assistant_tokens,
                "holdout_token_totals": self.holdout_token_totals,
            }
        if self.tag_rule_version:
            manifest["retention_haithem"] = {
                "tag_rule_version": self.tag_rule_version,
                "tagging_audit_results": self.tagging_audit_results,
                "template_cap_counts": self.template_cap_counts,
                "translation_task_decision": self.translation_task_decision,
                "cross_contamination_check": self.cross_contamination_check,
            }
        return manifest


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


def _record_key(record: Dict[str, Any]) -> str:
    """Stable lookup key for a record's pre-computed token counts."""
    record_id = record.get("id")
    return record_id if isinstance(record_id, str) else repr(sorted(record.items()))


def _budget_value(record: Dict[str, Any], counts: Dict[str, TokenCount], metric: str) -> int:
    """Return the configured budget metric (total or assistant tokens) for a record."""
    token_count = counts[_record_key(record)]
    return token_count.assistant if metric == "assistant_tokens" else token_count.total


def _fill_token_budget(
    pool: Sequence[Dict[str, Any]],
    target_tokens: float,
    *,
    tolerance: float,
    metric: str,
    counts: Dict[str, TokenCount],
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Greedily fill a token budget with whole examples in deterministic order.

    Visits records in seed-stable order and keeps an example only while the
    running total stays within the upper tolerance bound. Stops as soon as the
    total enters the ``±tolerance`` band around ``target_tokens``. Whole
    examples only — nothing is ever truncated.
    """
    if target_tokens <= 0:
        return [], 0

    ordered = sorted(pool, key=lambda record: stable_key(record, seed))
    lower = target_tokens * (1.0 - tolerance)
    upper = target_tokens * (1.0 + tolerance)

    selected: List[Dict[str, Any]] = []
    total = 0
    for record in ordered:
        if total >= lower:
            break
        value = _budget_value(record, counts, metric)
        if value <= 0:
            continue
        if total + value <= upper:
            selected.append(record)
            total += value

    return selected, total


def _remove_ids(records: Sequence[Dict[str, Any]], ids: set[str]) -> List[Dict[str, Any]]:
    """Return records whose string id is not in ``ids``."""
    return [
        record
        for record in records
        if not (isinstance(record.get("id"), str) and record["id"] in ids)
    ]


def _system_prompt_text(record: Dict[str, Any]) -> str:
    """Return the system turn content, or empty string if none."""
    for turn in record.get("messages", []):
        if isinstance(turn, dict) and turn.get("role") == "system":
            return str(turn.get("content", ""))
    return ""


def _template_key(record: Dict[str, Any]) -> str:
    """Stable hash of the system prompt (FLAN task template identifier)."""
    return hashlib.sha256(_system_prompt_text(record).encode("utf-8")).hexdigest()[:16]


def _cap_templates(
    pool: Sequence[Dict[str, Any]],
    cap_per_template: int,
    source_to_cap: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Cap records of ``source_to_cap`` at ``cap_per_template`` per unique system prompt.

    Records of other sources pass through unchanged. Within each template,
    records are kept in deterministic id order. When ``source_to_cap`` is empty,
    no capping is applied.

    Returns (capped_pool, per_template_counts) where per_template_counts maps
    template key to {"kept": n, "dropped": n}.
    """
    if not source_to_cap:
        return list(pool), {}

    capped_source: Dict[str, List[Dict[str, Any]]] = {}
    others: List[Dict[str, Any]] = []
    for record in pool:
        if str(record.get("source", "")) == source_to_cap:
            capped_source.setdefault(_template_key(record), []).append(record)
        else:
            others.append(record)

    kept: List[Dict[str, Any]] = list(others)
    counts: Dict[str, Dict[str, int]] = {}
    for template_id in sorted(capped_source):
        group = capped_source[template_id]
        group.sort(key=lambda r: str(r.get("id", "")))
        take = min(cap_per_template, len(group))
        kept.extend(group[:take])
        counts[template_id] = {
            "kept": take,
            "dropped": len(group) - take,
            "system_prompt_preview": _system_prompt_text(group[0])[:80] if group else "",
        }

    kept.sort(key=lambda record: str(record.get("id", "")))
    return kept, counts


def _check_cross_contamination(
    reasoning_picks: Sequence[Dict[str, Any]],
    math_picks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check reasoning picks against math picks for exact duplicates.

    Compares user-turn content across both sets. Exact matches indicate
    cross-dataset contamination (the same example selected for two
    different retention categories).
    """
    reasoning_texts = {
        _user_content_text(r)
        for r in reasoning_picks
        if str(r.get("category")) == "reasoning"
    }
    math_texts = {
        _user_content_text(r)
        for r in math_picks
        if str(r.get("category")) == "mathematics"
    }
    exact_matches = reasoning_texts & math_texts

    return {
        "reasoning_count": len(reasoning_texts),
        "math_count": len(math_texts),
        "exact_matches": len(exact_matches),
        "examples": sorted(exact_matches)[:5],
        "clean": len(exact_matches) == 0,
    }


def _user_content_text(record: Dict[str, Any]) -> str:
    """Return the last user turn content from a record."""
    user_turns = [m.get("content", "") for m in record.get("messages", []) if m.get("role") == "user"]
    return user_turns[-1] if user_turns else ""


def select_retention_token_budget(
    candidates: Sequence[Dict[str, Any]],
    spec: Optional[RetentionSpec] = None,
    *,
    schema: Optional[Schema] = None,
    scale: float = 1.0,
    seed: Optional[int] = None,
    counter: Optional[TokenCounter] = None,
    # --- tagged-source hygiene (Task 9, retention_haithem.md) ----------
    tag_rule_version: str = "",
    tagging_audit_results: Optional[Sequence[Dict[str, Any]]] = None,
    template_cap_per_template: int = 200,
    template_cap_source: str = "",
    translation_task_decision: str = "",
) -> RetentionSelection:
    """Select the retention train set + holdout by token budget (production).

    Implements docs/data/retention_mohamed.md Section 3 and
    docs/data/retention_haithem.md Section 2:

    * filter + dedupe + tokenize every candidate;
    * drop whole examples over ``max_formatted_tokens_per_example``;
    * select the coding anchor (``anchor_target_examples`` × scale) with source
      diversity and measure its token mass ``T_code``;
    * reserve a token-proportional holdout FIRST (never trained on);
    * fill every other category to ``multiplier × T_code`` whole examples,
      stopping inside the ``±tolerance`` band;
    * apply tagged-source template caps (≤ ``template_cap_per_template`` per unique
      system prompt) before budget filling;
    * fail loudly when a category cannot be filled.
    * fill every other category to ``multiplier × T_code`` whole examples,
      stopping inside the ``±tolerance`` band;
    * fail loudly when a category cannot be filled.

    The anchor category is ``coding`` (multiplier 1.0), so its train token mass
    is exactly ``T_code``. Other categories use ``category_multipliers``
    (reasoning 0.8, general_instruction 0.6, knowledge_qa 0.6).
    """
    spec = spec or load_retention_spec()
    schema = schema or load_schema()
    if not spec.token_budget_enabled:
        raise SelectionError(
            "token_budget mode requires retention.yaml token_budget.unit == 'tokens'"
        )
    spec.validate(schema)

    effective_seed = spec.seed if seed is None else seed
    counter = counter or ApproximateTokenCounter()
    metric = spec.token_budget_metric
    tolerance = spec.token_tolerance
    max_len = spec.max_formatted_tokens_per_example

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
        if str(record.get("category")) not in spec.category_multipliers:
            reject("category_not_requested")
            continue
        eligible.append(record)

    if not eligible:
        raise SelectionError(
            "no eligible retention candidates after filtering "
            f"(rejections: {rejected or 'none'}); categories requested: "
            f"{sorted(spec.category_multipliers)}"
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

    # --- tokenize every candidate ------------------------------------------
    counts: Dict[str, TokenCount] = {}
    for record in unique:
        counts[_record_key(record)] = counter.count_messages(
            record.get("messages", [])
        )

    # --- drop over-length examples (whole examples only, never truncated) --
    within_length: List[Dict[str, Any]] = []
    for record in unique:
        if counts[_record_key(record)].total <= max_len:
            within_length.append(record)
        else:
            reject("over_length")

    if not within_length:
        raise SelectionError(
            "no retention candidates within the formatted token limit "
            f"({max_len} tokens)"
        )

    # --- anchor: coding (count-based, source-diverse) -> T_code ------------
    anchor_category = spec.anchor_category
    anchor_pool = [
        record for record in within_length if str(record.get("category")) == anchor_category
    ]
    anchor_need = max(1, int(round(spec.anchor_target_examples * scale)))
    if len(anchor_pool) < anchor_need and spec.on_insufficient_candidates == "error":
        raise SelectionError(
            f"anchor category '{anchor_category}': need {anchor_need} examples "
            f"but only {len(anchor_pool)} candidate(s) available"
        )
    anchor_selected = _pick_with_source_diversity(
        anchor_pool,
        min(anchor_need, len(anchor_pool)),
        seed=effective_seed,
        max_share_per_source=spec.max_share_per_source,
    )
    anchor_ids = {
        record["id"] for record in anchor_selected if isinstance(record.get("id"), str)
    }
    T_code = sum(_budget_value(record, counts, metric) for record in anchor_selected)

    remaining = _remove_ids(within_length, anchor_ids)

    # --- holdout first, token-proportional across categories ---------------
    holdout_target_tokens = spec.holdout_target_tokens
    if holdout_target_tokens is None:
        holdout_target_tokens = DEFAULT_HOLDOUT_RATIO * T_code
    holdout_target_tokens = float(holdout_target_tokens)

    holdout_records: List[Dict[str, Any]] = []
    holdout_token_totals: Dict[str, int] = {}
    # Accumulate tagged-source template-cap stats across holdout + train (Task 9).
    all_template_counts: Dict[str, Any] = {}
    for category in sorted(spec.holdout_category_shares):
        share = spec.holdout_category_shares[category]
        category_target = holdout_target_tokens * share
        category_pool = [
            record
            for record in remaining
            if str(record.get("category")) == category
        ]
        # --- template cap per source (Task 9: retention_haithem.md Sec 2.3) ---
        category_pool, template_counts = _cap_templates(
            category_pool, template_cap_per_template, template_cap_source
        )
        all_template_counts.update(template_counts)
        selected, total = _fill_token_budget(
            category_pool,
            category_target,
            tolerance=tolerance,
            metric=metric,
            counts=counts,
            seed=effective_seed,
        )
        holdout_records.extend(selected)
        holdout_token_totals[category] = total
        if total < category_target * (1.0 - tolerance) and spec.on_insufficient_candidates == "error":
            raise SelectionError(
                f"holdout category '{category}': need {category_target:.0f} {metric} "
                f"but could only reserve {total}"
            )
        selected_ids = {
            record["id"] for record in selected if isinstance(record.get("id"), str)
        }
        remaining = _remove_ids(remaining, selected_ids)

    # --- train selection: fill each category to multiplier × T_code --------
    train_records: List[Dict[str, Any]] = list(anchor_selected)
    train_token_targets: Dict[str, float] = {anchor_category: float(T_code)}
    train_total_tokens: Dict[str, int] = {}
    train_assistant_tokens: Dict[str, int] = {}

    anchor_total = sum(counts[_record_key(r)].total for r in anchor_selected)
    anchor_assistant = sum(counts[_record_key(r)].assistant for r in anchor_selected)
    train_total_tokens[anchor_category] = anchor_total
    train_assistant_tokens[anchor_category] = anchor_assistant

    for category in sorted(spec.category_multipliers):
        if category == anchor_category:
            continue
        target_tokens = spec.category_multipliers[category] * T_code
        category_pool = [
            record
            for record in remaining
            if str(record.get("category")) == category
        ]
        # --- template cap per source (Task 9: retention_haithem.md Sec 2.3) ---
        category_pool, template_counts = _cap_templates(
            category_pool, template_cap_per_template, template_cap_source
        )
        all_template_counts.update(template_counts)
        selected, total = _fill_token_budget(
            category_pool,
            target_tokens,
            tolerance=tolerance,
            metric=metric,
            counts=counts,
            seed=effective_seed,
        )
        train_records.extend(selected)
        train_token_targets[category] = target_tokens
        train_total_tokens[category] = sum(
            counts[_record_key(r)].total for r in selected
        )
        train_assistant_tokens[category] = sum(
            counts[_record_key(r)].assistant for r in selected
        )
        if total < target_tokens * (1.0 - tolerance) and spec.on_insufficient_candidates == "error":
            raise SelectionError(
                f"category '{category}': need {target_tokens:.0f} {metric} "
                f"but could only select {total}"
            )
        selected_ids = {
            record["id"] for record in selected if isinstance(record.get("id"), str)
        }
        remaining = _remove_ids(remaining, selected_ids)

    train_records.sort(key=lambda record: str(record.get("id", "")))
    holdout_records.sort(key=lambda record: str(record.get("id", "")))

    # --- tagged-source cross-contamination vs math picks (Task 9) ---
    reasoning_picks = [
        r for r in train_records if str(r.get("category")) == "reasoning"
    ]
    math_picks = [
        r for r in train_records if str(r.get("category")) == "mathematics"
    ]
    cross_contamination = _check_cross_contamination(
        reasoning_picks, math_picks
    )

    selection = RetentionSelection(
        train=train_records,
        holdout=holdout_records,
        train_targets={name: int(round(v)) for name, v in train_token_targets.items()},
        holdout_targets={name: int(round(v)) for name, v in holdout_token_totals.items()},
        rejected=dict(sorted(rejected.items())),
        seed=effective_seed,
        scale=scale,
        anchor_tokens=T_code,
        token_metric=metric,
        train_token_targets=train_token_targets,
        train_total_tokens=train_total_tokens,
        train_assistant_tokens=train_assistant_tokens,
        holdout_token_totals=holdout_token_totals,
        tag_rule_version=tag_rule_version,
        tagging_audit_results=list(tagging_audit_results or []),
        template_cap_counts=all_template_counts,
        translation_task_decision=translation_task_decision,
        cross_contamination_check=cross_contamination,
    )
    selection.assert_no_contamination()

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
    parser.add_argument(
        "--mode",
        choices=["legacy", "token_budget"],
        default="legacy",
        help=(
            "selection mode: 'legacy' fills exact per-category example counts "
            "(synthetic smoke tests); 'token_budget' fills measured token budgets "
            "anchored to the coding slice (production, docs/data/retention_mohamed.md "
            "Section 3)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    parser.add_argument(
        "--tag-rule-version",
        default="",
        help="tagged-source tag-rule version (retention_haithem.md Sec 2.2)",
    )
    parser.add_argument(
        "--tagging-audit",
        default=None,
        help="path to tagged-source tag audit JSON (retention_haithem.md Sec 2.2)",
    )
    parser.add_argument(
        "--template-cap",
        type=int,
        default=200,
        help="max tagged-source examples per unique system prompt template (default 200)",
    )
    parser.add_argument(
        "--translation-decision",
        default="keep",
        help="translation-task decision for the manifest",
    )
    parser.add_argument(
        "--template-cap-source",
        default="",
        help="source whose records are template-capped (Task 9 hygiene)",
    )
    args = parser.parse_args(argv)

    try:
        spec = load_retention_spec(args.config)
        loaded = load_jsonl(args.candidates)
        tagging_audit_results = None
        if args.tagging_audit:
            import json as _json
            tagging_audit_results = _json.loads(
                Path(args.tagging_audit).read_text(encoding="utf-8")
            ).get("audit_results")
        if args.mode == "token_budget":
            selection = select_retention_token_budget(
                loaded.records,
                spec,
                scale=args.scale,
                seed=args.seed,
                tag_rule_version=args.tag_rule_version,
                tagging_audit_results=tagging_audit_results,
                template_cap_per_template=args.template_cap,
                template_cap_source=args.template_cap_source,
                translation_task_decision=args.translation_decision,
            )
        else:
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

    print(f"retention selection (mode={args.mode}, scale={args.scale}, seed={selection.seed})")
    print(f"  train   : {len(selection.train):>6} -> {train_out}")
    print(f"            counts {selection.train_counts}")
    print(f"  holdout : {len(selection.holdout):>6} -> {holdout_out}")
    print(f"            counts {selection.holdout_counts}")
    if selection.rejected:
        print(f"  rejected: {selection.rejected}")
    if getattr(selection, "token_metric", None):
        print(
            f"  token_budget: metric={selection.token_metric} "
            f"anchor_tokens={selection.anchor_tokens}"
        )
    print(f"  manifest: {manifest_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
