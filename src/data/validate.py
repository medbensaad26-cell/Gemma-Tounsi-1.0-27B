"""Canonical JSONL dataset validator.

Validates a canonical-schema JSONL file and reports every problem it finds. It
never silently drops a bad example: each defect becomes a ``RecordError`` with a
line number, an id, a stable code and a human-readable message.

This complements ``soup data validate``, which checks Soup's *wire* formats
(alpaca / sharegpt / chatml / dpo / kto / plaintext). Soup has no knowledge of
Gemma-Tounsi's project-specific fields (``category``, ``language``, ``script``,
slices, technical tagging), so those checks live here.

CLI
---
    python -m src.data.validate data/synthetic/raw/*.jsonl
    python -m src.data.validate --strict data/processed/retention/train.jsonl
    python -m src.data.validate --json data/synthetic/raw/arabizi.jsonl

Exit codes: 0 = valid, 1 = validation errors found, 2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import (
    LoadResult,
    RecordError,
    Schema,
    ValidationError,
    find_duplicate_ids,
    load_jsonl,
    load_schema,
    validate_record,
)

__all__ = [
    "ValidationReport",
    "validate_records",
    "validate_file",
    "validate_files",
    "main",
]


@dataclass
class ValidationReport:
    """Aggregated validation outcome for one file.

    Attributes:
        path: file that was validated (``"<memory>"`` for in-memory records).
        total_records: parsed records (excludes unparsable lines).
        errors: every defect found, in file order.
        valid_records: records that passed every check.
    """

    path: str
    total_records: int = 0
    errors: List[RecordError] = field(default_factory=list)
    valid_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no error was recorded."""
        return not self.errors

    @property
    def error_count(self) -> int:
        """Number of defects found."""
        return len(self.errors)

    @property
    def invalid_count(self) -> int:
        """Number of records that failed at least one check."""
        return self.total_records - len(self.valid_records)

    def codes(self) -> Dict[str, int]:
        """Return ``{error_code: occurrences}``, sorted by code."""
        counts: Dict[str, int] = {}
        for error in self.errors:
            counts[error.code] = counts.get(error.code, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return {
            "path": self.path,
            "ok": self.ok,
            "total_records": self.total_records,
            "valid_records": len(self.valid_records),
            "invalid_records": self.invalid_count,
            "error_count": self.error_count,
            "error_codes": self.codes(),
            "errors": [error.to_dict() for error in self.errors],
        }

    def raise_if_invalid(self) -> None:
        """Raise ``ValidationError`` if anything failed."""
        if not self.ok:
            raise ValidationError(
                f"{self.path}: {self.error_count} validation error(s) "
                f"across {self.invalid_count} record(s)",
                self.errors,
            )

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        state = "OK" if self.ok else "FAILED"
        return (
            f"{state}  {self.path}  records={self.total_records} "
            f"valid={len(self.valid_records)} invalid={self.invalid_count} "
            f"errors={self.error_count}"
        )


def validate_records(
    records: Sequence[Dict[str, Any]],
    schema: Optional[Schema] = None,
    *,
    line_numbers: Optional[Sequence[int]] = None,
    path: str = "<memory>",
    check_duplicates: bool = True,
) -> ValidationReport:
    """Validate already-parsed records against the canonical schema.

    Args:
        records: parsed JSON objects to check.
        schema: canonical schema; loaded from config when omitted.
        line_numbers: source line for each record, for precise error reporting.
        path: label used in the report.
        check_duplicates: also detect duplicate ``id`` values.
    """
    schema = schema or load_schema()
    report = ValidationReport(path=path, total_records=len(records))

    bad_indices: set[int] = set()
    for index, record in enumerate(records):
        line = (
            line_numbers[index] if line_numbers and index < len(line_numbers) else index + 1
        )
        record_errors = validate_record(record, schema, line=line)
        if record_errors:
            bad_indices.add(index)
            report.errors.extend(record_errors)

    if check_duplicates:
        duplicate_errors = find_duplicate_ids(records, line_numbers)
        report.errors.extend(duplicate_errors)
        duplicate_ids = {error.record_id for error in duplicate_errors}
        for index, record in enumerate(records):
            if record.get("id") in duplicate_ids:
                bad_indices.add(index)

    report.valid_records = [
        record for index, record in enumerate(records) if index not in bad_indices
    ]
    report.errors.sort(key=lambda error: (error.line, error.code))
    return report


def validate_file(
    path: Path | str,
    schema: Optional[Schema] = None,
    *,
    check_duplicates: bool = True,
) -> ValidationReport:
    """Load and validate a canonical JSONL file.

    Raises:
        FileNotFoundError: the file does not exist.
    """
    path = Path(path)
    loaded: LoadResult = load_jsonl(path)
    report = validate_records(
        loaded.records,
        schema,
        line_numbers=loaded.line_numbers,
        path=str(path),
        check_duplicates=check_duplicates,
    )
    # Unparsable lines are defects too, and are not part of `records`.
    report.errors.extend(loaded.errors)
    report.total_records = len(loaded.records) + len(loaded.errors)
    report.errors.sort(key=lambda error: (error.line, error.code))
    return report


def validate_files(
    paths: Sequence[Path | str],
    schema: Optional[Schema] = None,
    *,
    check_duplicates: bool = True,
) -> List[ValidationReport]:
    """Validate several files, returning one report per file."""
    schema = schema or load_schema()
    return [
        validate_file(path, schema, check_duplicates=check_duplicates) for path in paths
    ]


def _print_report(report: ValidationReport, *, max_errors: int) -> None:
    """Print a report in human-readable form."""
    print(report.summary())
    if report.ok:
        return
    shown = report.errors[:max_errors] if max_errors > 0 else report.errors
    for error in shown:
        print(f"  - {error}")
    remaining = report.error_count - len(shown)
    if remaining > 0:
        print(f"  ... and {remaining} more error(s)")
    print(f"  error codes: {report.codes()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.validate",
        description="Validate canonical Gemma-Tounsi JSONL datasets.",
    )
    parser.add_argument("paths", nargs="+", help="canonical JSONL file(s) to validate")
    parser.add_argument(
        "--schema",
        default=None,
        help="override the schema YAML path (default: configs/data/schema.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="max errors to print per file (0 = all); default: 20",
    )
    parser.add_argument(
        "--allow-duplicate-ids",
        action="store_true",
        help="do not treat duplicate ids as errors (NOT recommended)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any file contains zero records",
    )
    args = parser.parse_args(argv)

    try:
        schema = load_schema(args.schema)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reports: List[ValidationReport] = []
    for raw_path in args.paths:
        try:
            reports.append(
                validate_file(
                    raw_path,
                    schema,
                    check_duplicates=not args.allow_duplicate_ids,
                )
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    failed = any(not report.ok for report in reports)
    if args.strict:
        for report in reports:
            if report.total_records == 0:
                print(f"error: {report.path} contains no records (--strict)", file=sys.stderr)
                failed = True

    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "files": [report.to_dict() for report in reports],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for report in reports:
            _print_report(report, max_errors=args.max_errors)
        total_errors = sum(report.error_count for report in reports)
        total_records = sum(report.total_records for report in reports)
        print(
            f"\nvalidated {len(reports)} file(s), {total_records} record(s), "
            f"{total_errors} error(s)"
        )

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
