"""Deterministic train / holdout splitting.

Generic, dataset-independent, and reproducible: the same input records plus the
same seed always produce the same split, and the train and holdout sets are
guaranteed disjoint at the ``id`` level (no train/test contamination through
duplicated examples).

Why not just ``soup data split``?
    Soup 0.73.3 *does* provide ``soup data split`` (train/val/test, ``--seed``,
    ``--stratify FIELD``), and the pipeline uses it for the final Soup-format
    artifacts. This module exists because the retention selector needs an
    in-memory, id-safe, stratified reservation of a holdout *before* selection,
    returning Python objects (not files) so counts can be asserted in tests. It
    complements Soup rather than replacing it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["SplitResult", "stable_key", "split_records", "stratified_split"]


@dataclass
class SplitResult:
    """A disjoint train/holdout partition.

    Attributes:
        train: records assigned to training.
        holdout: records reserved for holdout (never trained on).
    """

    train: List[Dict[str, Any]]
    holdout: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        self.assert_disjoint()

    @property
    def train_ids(self) -> List[str]:
        """Ids present in the training set (records with a string id)."""
        return [r["id"] for r in self.train if isinstance(r.get("id"), str)]

    @property
    def holdout_ids(self) -> List[str]:
        """Ids present in the holdout set (records with a string id)."""
        return [r["id"] for r in self.holdout if isinstance(r.get("id"), str)]

    def assert_disjoint(self) -> None:
        """Raise ``ValueError`` if any id appears in both partitions."""
        overlap = set(self.train_ids) & set(self.holdout_ids)
        if overlap:
            example = sorted(overlap)[:5]
            raise ValueError(
                f"train/holdout contamination: {len(overlap)} shared id(s), "
                f"e.g. {example}"
            )


def stable_key(record: Dict[str, Any], seed: int) -> str:
    """Return a deterministic sort key for a record under ``seed``.

    Uses a SHA-256 hash of ``seed`` and the record ``id`` (falling back to the
    JSON-ish repr when no id is present). This gives a stable, uniformly spread
    ordering that does not depend on Python's hash randomization, so splits are
    reproducible across processes and platforms.
    """
    identity = record.get("id")
    if not isinstance(identity, str) or not identity:
        identity = repr(sorted(record.items()))
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
    return digest


def _ordered(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Return records in deterministic, seed-dependent order."""
    return sorted(records, key=lambda record: stable_key(record, seed))


def split_records(
    records: Sequence[Dict[str, Any]],
    holdout_size: int,
    *,
    seed: int = 42,
) -> SplitResult:
    """Split ``records`` into train and a holdout of exactly ``holdout_size``.

    The holdout is taken as a deterministic pseudo-random subset (by
    :func:`stable_key`). Everything else becomes train.

    Raises:
        ValueError: ``holdout_size`` is negative or exceeds ``len(records)``.
    """
    if holdout_size < 0:
        raise ValueError(f"holdout_size must be >= 0, got {holdout_size}")
    if holdout_size > len(records):
        raise ValueError(
            f"holdout_size {holdout_size} exceeds available records {len(records)}"
        )

    ordered = _ordered(records, seed)
    holdout = ordered[:holdout_size]
    train = ordered[holdout_size:]
    return SplitResult(train=train, holdout=holdout)


def _largest_remainder(
    total: int, weights: Dict[str, int]
) -> Dict[str, int]:
    """Apportion ``total`` across keys proportionally to ``weights``.

    Uses the largest-remainder method so the parts sum to exactly ``total``
    (no rounding drift). Deterministic tie-breaking by key name.
    """
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return {key: 0 for key in weights}

    exact = {key: total * value / weight_sum for key, value in weights.items()}
    floored = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(floored.values())

    order = sorted(
        weights,
        key=lambda key: (exact[key] - floored[key], key),
        reverse=True,
    )
    for index in range(remainder):
        floored[order[index % len(order)]] += 1
    return floored


def stratified_split(
    records: Sequence[Dict[str, Any]],
    holdout_size: int,
    *,
    stratify_by: str = "category",
    seed: int = 42,
    key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> SplitResult:
    """Split with the holdout stratified across a field's values.

    The holdout is apportioned across each distinct value of ``stratify_by`` in
    proportion to that stratum's size, so the holdout mirrors the overall
    distribution instead of over-sampling one category.

    Args:
        records: records to split.
        holdout_size: total holdout size across all strata.
        stratify_by: field whose values define the strata (default ``category``).
        seed: deterministic seed.
        key_fn: optional custom stratum key extractor; defaults to reading
            ``stratify_by`` (missing/other -> ``"__unknown__"``).

    Raises:
        ValueError: ``holdout_size`` is negative or exceeds ``len(records)``.
    """
    if holdout_size < 0:
        raise ValueError(f"holdout_size must be >= 0, got {holdout_size}")
    if holdout_size > len(records):
        raise ValueError(
            f"holdout_size {holdout_size} exceeds available records {len(records)}"
        )

    def default_key(record: Dict[str, Any]) -> str:
        value = record.get(stratify_by)
        return value if isinstance(value, str) and value else "__unknown__"

    extract = key_fn or default_key

    strata: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        strata.setdefault(extract(record), []).append(record)

    sizes = {key: len(items) for key, items in strata.items()}
    per_stratum_holdout = _largest_remainder(holdout_size, sizes)

    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    for key in sorted(strata):
        ordered = _ordered(strata[key], seed)
        take = min(per_stratum_holdout.get(key, 0), len(ordered))
        holdout.extend(ordered[:take])
        train.extend(ordered[take:])

    return SplitResult(train=train, holdout=holdout)
