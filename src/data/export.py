"""Canonical records -> Soup-compatible JSONL.

The canonical schema carries project metadata (``category``, ``language``,
``script``, ``quality``, ...) that Soup does not consume. The final training file
must therefore be emitted in a shape Soup accepts.

Soup 0.73.3 supports the ``sharegpt`` conversation format, whose rows look like::

    {"conversations": [{"from": "human", "value": "..."},
                       {"from": "gpt",   "value": "..."}]}

Our canonical ``messages`` map onto it directly:
``system -> system``, ``user -> human``, ``assistant -> gpt``.

Role mapping and format names are taken from what ``soup data convert --to`` and
``soup data validate --format`` accept in 0.73.3 (``alpaca``, ``sharegpt``,
``chatml``). Nothing is invented; ``sharegpt`` is chosen because, unlike
``alpaca``, it represents multi-turn conversations without loss.

CLI
---
    python -m src.data.export --in data/processed/mix.jsonl --out data/train.jsonl
    python -m src.data.export --in IN.jsonl --out OUT.jsonl --keep-metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import Schema, load_jsonl, load_schema, validate_record, write_jsonl

__all__ = ["SHAREGPT_ROLE_MAP", "to_sharegpt", "export_records", "main"]

#: Canonical role -> ShareGPT ``from`` value.
SHAREGPT_ROLE_MAP: Dict[str, str] = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}

#: Canonical metadata preserved when ``keep_metadata`` is requested. Soup ignores
#: unknown columns, so keeping these aids traceability without breaking training.
TRACEABILITY_FIELDS = ("id", "category", "source", "language")


def to_sharegpt(record: Dict[str, Any], *, keep_metadata: bool = False) -> Dict[str, Any]:
    """Convert one canonical record into a ShareGPT row.

    Args:
        record: a canonical record (assumed already validated).
        keep_metadata: also copy ``id``/``category``/``source``/``language``
            through, for provenance during debugging.

    Raises:
        ValueError: the record has no usable ``messages`` list, or a role is not
            mappable to ShareGPT.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {record.get('id')!r} has no usable 'messages'")

    conversations: List[Dict[str, str]] = []
    for turn in messages:
        role = turn.get("role")
        if role not in SHAREGPT_ROLE_MAP:
            raise ValueError(
                f"record {record.get('id')!r}: role {role!r} cannot be mapped to "
                f"ShareGPT; expected one of {', '.join(SHAREGPT_ROLE_MAP)}"
            )
        conversations.append(
            {"from": SHAREGPT_ROLE_MAP[role], "value": turn.get("content", "")}
        )

    row: Dict[str, Any] = {"conversations": conversations}
    if keep_metadata:
        for name in TRACEABILITY_FIELDS:
            if name in record:
                row[name] = record[name]
    return row


def export_records(
    input_path: Path | str,
    output_path: Path | str,
    *,
    schema: Optional[Schema] = None,
    keep_metadata: bool = False,
    validate: bool = True,
) -> int:
    """Export a canonical JSONL file to Soup-compatible ShareGPT JSONL.

    Args:
        input_path: canonical JSONL input.
        output_path: ShareGPT JSONL output.
        schema: canonical schema; loaded from config when omitted.
        keep_metadata: preserve traceability fields in the output.
        validate: refuse to export if any record fails schema validation.

    Returns:
        Number of rows written.

    Raises:
        ValueError: validation is enabled and the input contains invalid records.
    """
    schema = schema or load_schema()
    loaded = load_jsonl(input_path)

    if validate:
        problems: List[str] = [str(error) for error in loaded.errors]
        for index, record in enumerate(loaded.records):
            line = (
                loaded.line_numbers[index]
                if index < len(loaded.line_numbers)
                else index + 1
            )
            problems.extend(str(e) for e in validate_record(record, schema, line=line))
        if problems:
            preview = "\n  - ".join(problems[:10])
            raise ValueError(
                f"refusing to export {input_path}: {len(problems)} validation "
                f"error(s):\n  - {preview}"
            )

    rows = [to_sharegpt(record, keep_metadata=keep_metadata) for record in loaded.records]
    return write_jsonl(output_path, rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.export",
        description=(
            "Convert canonical JSONL into Soup-compatible ShareGPT JSONL "
            "(format verified against Soup 0.73.3)."
        ),
    )
    parser.add_argument("--in", dest="input", required=True, help="canonical JSONL input")
    parser.add_argument("--out", dest="output", required=True, help="ShareGPT JSONL output")
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="preserve id/category/source/language for traceability",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip validation (NOT recommended)",
    )
    args = parser.parse_args(argv)

    try:
        written = export_records(
            args.input,
            args.output,
            keep_metadata=args.keep_metadata,
            validate=not args.no_validate,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"exported {written} row(s) -> {args.output} (sharegpt)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
