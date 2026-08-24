"""Generic dataset statistics for canonical Gemma-Tounsi datasets.

Reports counts and distributions over the canonical schema's own fields
(``category``, ``subcategory``, ``source``, ``language``, ``script``) plus the
project-specific technical ratio used by the cross-cutting Arabizi/Derja quota.

Relationship to Soup
--------------------
``soup data stats`` (verified in 0.73.3) gives length distribution, token counts
and language detection for Soup's wire formats, and ``soup data inspect`` shows
sample rows. Neither knows about our canonical fields or the technical quota, so
this module covers only that gap. Token counts here are a dependency-free
*approximation*: no tokenizer is installed just to count tokens.

CLI
---
    python -m src.data.stats data/synthetic/raw/arabizi.jsonl
    python -m src.data.stats --json data/processed/retention/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import (
    Schema,
    find_duplicate_ids,
    load_jsonl,
    load_schema,
    validate_record,
)

__all__ = ["DatasetStats", "compute_stats", "analyze_file", "main"]

#: Rough characters-per-token ratio used for the dependency-free estimate.
#: Deliberately coarse: real token counts come from ``soup data stats``.
CHARS_PER_TOKEN: float = 4.0


def _count(values: Sequence[Any]) -> Dict[str, int]:
    """Count occurrences, sorted by descending count then key."""
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@dataclass
class DatasetStats:
    """Statistics for one canonical dataset."""

    path: str
    total_examples: int = 0
    malformed_records: int = 0
    duplicate_ids: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_subcategory: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    by_language: Dict[str, int] = field(default_factory=dict)
    by_script: Dict[str, int] = field(default_factory=dict)
    avg_messages: float = 0.0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    approx_avg_tokens: float = 0.0
    approx_total_tokens: int = 0
    technical_examples: int = 0
    technical_ratio: float = 0.0
    code_switching_examples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "path": self.path,
            "total_examples": self.total_examples,
            "malformed_records": self.malformed_records,
            "duplicate_ids": self.duplicate_ids,
            "by_category": self.by_category,
            "by_subcategory": self.by_subcategory,
            "by_source": self.by_source,
            "by_language": self.by_language,
            "by_script": self.by_script,
            "avg_messages": round(self.avg_messages, 3),
            "avg_chars": round(self.avg_chars, 1),
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "approx_avg_tokens": round(self.approx_avg_tokens, 1),
            "approx_total_tokens": self.approx_total_tokens,
            "technical_examples": self.technical_examples,
            "technical_ratio": round(self.technical_ratio, 4),
            "code_switching_examples": self.code_switching_examples,
        }

    def render(self) -> str:
        """Return a human-readable multi-line report."""
        lines = [
            f"dataset: {self.path}",
            f"  total examples      : {self.total_examples}",
            f"  malformed records   : {self.malformed_records}",
            f"  duplicate ids       : {self.duplicate_ids}",
            f"  avg messages/example: {self.avg_messages:.2f}",
            f"  avg chars/example   : {self.avg_chars:.1f} (min {self.min_chars}, max {self.max_chars})",
            f"  approx tokens       : {self.approx_total_tokens} total, "
            f"{self.approx_avg_tokens:.1f} avg (~{CHARS_PER_TOKEN} chars/token)",
            f"  technical examples  : {self.technical_examples} "
            f"({self.technical_ratio * 100:.1f}%)",
            f"  code-switching      : {self.code_switching_examples}",
        ]
        for title, mapping in (
            ("by category", self.by_category),
            ("by subcategory", self.by_subcategory),
            ("by source", self.by_source),
            ("by language", self.by_language),
            ("by script", self.by_script),
        ):
            if mapping:
                lines.append(f"  {title}:")
                for key, value in mapping.items():
                    lines.append(f"      {key}: {value}")
        return "\n".join(lines)


def _record_chars(record: Dict[str, Any]) -> int:
    """Total characters across all message contents of a record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        len(turn["content"])
        for turn in messages
        if isinstance(turn, dict) and isinstance(turn.get("content"), str)
    )


def compute_stats(
    records: Sequence[Dict[str, Any]],
    schema: Optional[Schema] = None,
    *,
    path: str = "<memory>",
    malformed: int = 0,
) -> DatasetStats:
    """Compute statistics over parsed canonical records.

    Records that fail schema validation are counted as malformed and excluded
    from the distributions, so a broken example never skews the numbers.
    """
    schema = schema or load_schema()
    stats = DatasetStats(path=path, malformed_records=malformed)

    good: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if validate_record(record, schema, line=index + 1):
            stats.malformed_records += 1
        else:
            good.append(record)

    stats.duplicate_ids = len(find_duplicate_ids(records))
    stats.total_examples = len(good)
    if not good:
        return stats

    stats.by_category = _count([r.get("category") for r in good])
    stats.by_subcategory = _count(
        [r["subcategory"] for r in good if isinstance(r.get("subcategory"), str)]
    )
    stats.by_source = _count([r.get("source") for r in good])
    stats.by_language = _count([r.get("language") for r in good])
    stats.by_script = _count([r["script"] for r in good if isinstance(r.get("script"), str)])

    message_counts = [len(r["messages"]) for r in good if isinstance(r.get("messages"), list)]
    stats.avg_messages = sum(message_counts) / len(message_counts) if message_counts else 0.0

    char_counts = [_record_chars(r) for r in good]
    total_chars = sum(char_counts)
    stats.avg_chars = total_chars / len(char_counts)
    stats.min_chars = min(char_counts)
    stats.max_chars = max(char_counts)
    stats.approx_total_tokens = int(total_chars / CHARS_PER_TOKEN)
    stats.approx_avg_tokens = stats.approx_total_tokens / len(good)

    stats.technical_examples = sum(
        1 for r in good if schema.is_technical(str(r.get("category")))
    )
    stats.technical_ratio = stats.technical_examples / len(good)
    stats.code_switching_examples = sum(1 for r in good if r.get("code_switching") is True)

    return stats


def analyze_file(path: Path | str, schema: Optional[Schema] = None) -> DatasetStats:
    """Load a canonical JSONL file and compute its statistics."""
    path = Path(path)
    loaded = load_jsonl(path)
    return compute_stats(
        loaded.records, schema, path=str(path), malformed=len(loaded.errors)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.stats",
        description="Report statistics for canonical Gemma-Tounsi datasets.",
    )
    parser.add_argument("paths", nargs="+", help="canonical JSONL file(s) to analyze")
    parser.add_argument("--schema", default=None, help="override the schema YAML path")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--out", default=None, help="also write the JSON report to this path"
    )
    args = parser.parse_args(argv)

    try:
        schema = load_schema(args.schema)
        reports = [analyze_file(path, schema) for path in args.paths]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = {"datasets": [report.to_dict() for report in reports]}

    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(report.render())
            print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
