"""Canonical data engineering for Gemma Tounsi 1.0.

Modules
-------
schema     canonical record schema + per-record validation primitives
validate   JSONL file validation (CLI: ``python -m src.data.validate``)
stats      generic dataset statistics (CLI: ``python -m src.data.stats``)
dedupe     deduplication interface — delegates to ``soup data dedup``
retention  retention slice selection + holdout reservation
split      deterministic train/holdout splitting
mixture    project-wide mixture validation
export     canonical -> Soup-compatible JSONL

Nothing here downloads a dataset, and nothing here trains a model.
"""

from __future__ import annotations

__all__ = [
    "schema",
    "validate",
    "stats",
    "dedupe",
    "retention",
    "split",
    "mixture",
    "export",
]
