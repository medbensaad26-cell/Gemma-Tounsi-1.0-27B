"""Canonical training-example schema for Gemma Tounsi 1.0.

The schema itself is declared in ``configs/data/schema.yaml``; this module loads
that file and turns it into typed, deterministic validation primitives. Keeping
one source of truth means the accepted values in the config and the behaviour of
the code can never drift apart.

Design rules:
  * dataset-INDEPENDENT — no external dataset is referenced anywhere;
  * errors are RETURNED (as ``RecordError``), never silently dropped;
  * deterministic — no randomness, no ordering surprises;
  * stdlib + PyYAML only, Python 3.10 compatible (the container runs 3.10.12).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (``src/data/schema.py`` -> ``src/data`` -> ``src`` -> root).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Declarative schema consumed by this module.
SCHEMA_PATH: Path = REPO_ROOT / "configs" / "data" / "schema.yaml"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class SchemaError(Exception):
    """The schema configuration itself is missing or malformed."""


class ValidationError(Exception):
    """A dataset failed validation. Carries the individual record errors."""

    def __init__(self, message: str, errors: Optional[Sequence["RecordError"]] = None) -> None:
        super().__init__(message)
        self.errors: List[RecordError] = list(errors or [])


# --------------------------------------------------------------------------- #
# Error reporting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecordError:
    """A single, actionable validation failure.

    Attributes:
        line: 1-based line number in the source JSONL file (0 if not applicable).
        record_id: the record's ``id`` when it could be read, else ``None``.
        code: stable machine-readable error code (e.g. ``missing_field``).
        message: human-readable explanation of what is wrong.
        field: the offending field name, when the error is field-scoped.
    """

    line: int
    record_id: Optional[str]
    code: str
    message: str
    field: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        where = f"line {self.line}" if self.line else "record"
        who = f" [id={self.record_id}]" if self.record_id else ""
        what = f" ({self.field})" if self.field else ""
        return f"{where}{who} {self.code}{what}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this error."""
        return {
            "line": self.line,
            "record_id": self.record_id,
            "code": self.code,
            "message": self.message,
            "field": self.field,
        }


# --------------------------------------------------------------------------- #
# Schema model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Schema:
    """Typed view over ``configs/data/schema.yaml``."""

    schema_version: str
    required_fields: Tuple[str, ...]
    optional_fields: Tuple[str, ...]
    categories: Tuple[str, ...]
    technical_categories: Tuple[str, ...]
    languages: Tuple[str, ...]
    scripts: Tuple[str, ...]
    difficulties: Tuple[str, ...]
    quality_flags: Tuple[str, ...]
    roles: Tuple[str, ...]
    slices: Tuple[str, ...]
    min_turns: int
    id_pattern: str

    # -- derived helpers ---------------------------------------------------- #

    @property
    def known_fields(self) -> Tuple[str, ...]:
        """Every field name the schema recognises (required + optional)."""
        return self.required_fields + self.optional_fields

    def is_technical(self, category: str) -> bool:
        """Return True if ``category`` counts toward the technical quota."""
        return category in self.technical_categories


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    """Fetch ``key`` from ``mapping`` or raise a descriptive ``SchemaError``."""
    if key not in mapping:
        raise SchemaError(f"missing key '{key}' in {where} of {SCHEMA_PATH.name}")
    return mapping[key]


@lru_cache(maxsize=None)
def load_schema(path: Optional[str] = None) -> Schema:
    """Load and cache the canonical schema.

    Args:
        path: optional override for the schema YAML location.

    Raises:
        SchemaError: the file is missing, unparsable, or incomplete.
    """
    schema_path = Path(path) if path else SCHEMA_PATH
    if not schema_path.is_file():
        raise SchemaError(f"schema file not found: {schema_path}")

    try:
        raw = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on a broken file
        raise SchemaError(f"could not parse {schema_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SchemaError(f"{schema_path} must contain a YAML mapping")

    record = _require(raw, "record", "top level")
    messages = _require(raw, "messages", "top level")
    enums = _require(raw, "enums", "top level")

    return Schema(
        schema_version=str(raw.get("schema_version", "unknown")),
        required_fields=tuple(_require(record, "required_fields", "record")),
        optional_fields=tuple(record.get("optional_fields", ())),
        categories=tuple(_require(enums, "category", "enums")),
        technical_categories=tuple(_require(enums, "technical_categories", "enums")),
        languages=tuple(_require(enums, "language", "enums")),
        scripts=tuple(enums.get("script", ())),
        difficulties=tuple(enums.get("difficulty", ())),
        quality_flags=tuple(enums.get("quality", ())),
        roles=tuple(_require(_require(messages, "roles", "messages"), "allowed", "messages.roles")),
        slices=tuple(raw.get("slices", ())),
        min_turns=int(messages.get("min_turns", 2)),
        id_pattern=str(raw.get("id", {}).get("pattern", r"^[A-Za-z0-9._:-]+$")),
    )


# --------------------------------------------------------------------------- #
# Record-level validation
# --------------------------------------------------------------------------- #


def validate_messages(
    messages: Any,
    schema: Schema,
    *,
    line: int = 0,
    record_id: Optional[str] = None,
) -> List[RecordError]:
    """Validate the ``messages`` block of one record.

    Checks non-emptiness, allowed roles, strict user/assistant alternation with
    an optional leading ``system`` turn, a final ``assistant`` turn, and
    non-empty string content on every turn.
    """
    errors: List[RecordError] = []

    def err(code: str, message: str) -> None:
        errors.append(RecordError(line, record_id, code, message, "messages"))

    if not isinstance(messages, list):
        err("invalid_type", f"'messages' must be a list, got {type(messages).__name__}")
        return errors

    if not messages:
        err("empty_messages", "'messages' must not be empty")
        return errors

    if len(messages) < schema.min_turns:
        err(
            "too_few_turns",
            f"expected at least {schema.min_turns} turns, got {len(messages)}",
        )

    for index, turn in enumerate(messages):
        if not isinstance(turn, dict):
            err("invalid_type", f"turn {index} must be an object, got {type(turn).__name__}")
            return errors
        if "role" not in turn:
            err("missing_field", f"turn {index} is missing 'role'")
            return errors
        if "content" not in turn:
            err("missing_field", f"turn {index} is missing 'content'")
            return errors

        role = turn["role"]
        content = turn["content"]

        if role not in schema.roles:
            err(
                "invalid_role",
                f"turn {index} has role '{role}'; allowed: {', '.join(schema.roles)}",
            )
        if not isinstance(content, str):
            err(
                "invalid_type",
                f"turn {index} content must be a string, got {type(content).__name__}",
            )
        elif not content.strip():
            err("empty_content", f"turn {index} has empty content")

    roles = [turn.get("role") for turn in messages if isinstance(turn, dict)]

    # `system` may appear at most once, and only as the very first turn.
    system_positions = [i for i, role in enumerate(roles) if role == "system"]
    if len(system_positions) > 1:
        err("invalid_ordering", "at most one 'system' turn is allowed")
    if system_positions and system_positions[0] != 0:
        err("invalid_ordering", "'system' is only allowed as the first turn")

    body = roles[1:] if roles and roles[0] == "system" else roles
    if not body:
        err("empty_messages", "conversation has no user/assistant turns")
        return errors

    if body[0] != "user":
        err("invalid_ordering", f"first non-system turn must be 'user', got '{body[0]}'")

    expected = ["user", "assistant"]
    for offset, role in enumerate(body):
        want = expected[offset % 2]
        if role != want:
            err(
                "invalid_ordering",
                f"turn {offset} of the conversation must be '{want}', got '{role}'",
            )
            break

    if body[-1] != "assistant":
        err("invalid_ordering", f"conversation must end on 'assistant', got '{body[-1]}'")

    return errors


def validate_record(
    record: Any,
    schema: Optional[Schema] = None,
    *,
    line: int = 0,
) -> List[RecordError]:
    """Validate a single canonical record.

    Returns every problem found rather than raising on the first one, so a
    caller can report all defects in one pass.
    """
    schema = schema or load_schema()
    errors: List[RecordError] = []

    if not isinstance(record, dict):
        return [
            RecordError(
                line, None, "invalid_type", f"record must be an object, got {type(record).__name__}"
            )
        ]

    raw_id = record.get("id")
    record_id = raw_id if isinstance(raw_id, str) else None

    def err(code: str, message: str, field_name: Optional[str] = None) -> None:
        errors.append(RecordError(line, record_id, code, message, field_name))

    # --- required fields ---------------------------------------------------
    for name in schema.required_fields:
        if name not in record:
            err("missing_field", f"required field '{name}' is missing", name)

    # --- id ----------------------------------------------------------------
    if "id" in record:
        if not isinstance(raw_id, str):
            err("invalid_type", f"'id' must be a string, got {type(raw_id).__name__}", "id")
        elif not raw_id.strip():
            err("empty_id", "'id' must not be empty", "id")
        elif not re.match(schema.id_pattern, raw_id):
            err(
                "invalid_id",
                f"'id' {raw_id!r} does not match {schema.id_pattern}",
                "id",
            )

    # --- messages ----------------------------------------------------------
    if "messages" in record:
        errors.extend(
            validate_messages(record["messages"], schema, line=line, record_id=record_id)
        )

    # --- category ----------------------------------------------------------
    if "category" in record:
        category = record["category"]
        if not isinstance(category, str):
            err("invalid_type", f"'category' must be a string, got {type(category).__name__}", "category")
        elif category not in schema.categories:
            err(
                "invalid_category",
                f"'{category}' is not a valid category; allowed: {', '.join(schema.categories)}",
                "category",
            )

    # --- language ----------------------------------------------------------
    if "language" in record:
        language = record["language"]
        if not isinstance(language, str):
            err("invalid_type", f"'language' must be a string, got {type(language).__name__}", "language")
        elif language not in schema.languages:
            err(
                "invalid_language",
                f"'{language}' is not a valid language; allowed: {', '.join(schema.languages)}",
                "language",
            )

    # --- source ------------------------------------------------------------
    if "source" in record:
        source = record["source"]
        if not isinstance(source, str):
            err("invalid_type", f"'source' must be a string, got {type(source).__name__}", "source")
        elif not source.strip():
            err("empty_source", "'source' must not be empty", "source")

    # --- optional metadata -------------------------------------------------
    if "script" in record and record["script"] not in schema.scripts:
        err(
            "invalid_script",
            f"'{record['script']}' is not a valid script; allowed: {', '.join(schema.scripts)}",
            "script",
        )

    if "difficulty" in record and record["difficulty"] not in schema.difficulties:
        err(
            "invalid_difficulty",
            f"'{record['difficulty']}' is not a valid difficulty; "
            f"allowed: {', '.join(schema.difficulties)}",
            "difficulty",
        )

    if "code_switching" in record and not isinstance(record["code_switching"], bool):
        err(
            "invalid_type",
            f"'code_switching' must be a boolean, got {type(record['code_switching']).__name__}",
            "code_switching",
        )

    if "subcategory" in record and not isinstance(record["subcategory"], str):
        err(
            "invalid_type",
            f"'subcategory' must be a string, got {type(record['subcategory']).__name__}",
            "subcategory",
        )

    if "variation_group" in record and not isinstance(record["variation_group"], str):
        err(
            "invalid_type",
            f"'variation_group' must be a string, got {type(record['variation_group']).__name__}",
            "variation_group",
        )

    if "quality" in record:
        quality = record["quality"]
        if not isinstance(quality, dict):
            err("invalid_type", f"'quality' must be an object, got {type(quality).__name__}", "quality")
        else:
            for key, value in quality.items():
                if key not in schema.quality_flags:
                    err(
                        "invalid_quality_flag",
                        f"'{key}' is not a known quality flag; "
                        f"allowed: {', '.join(schema.quality_flags)}",
                        "quality",
                    )
                elif not isinstance(value, bool):
                    err(
                        "invalid_type",
                        f"quality flag '{key}' must be a boolean, got {type(value).__name__}",
                        "quality",
                    )

    # --- unknown fields ----------------------------------------------------
    for name in record:
        if name not in schema.known_fields:
            err(
                "unknown_field",
                f"'{name}' is not part of the canonical schema; "
                f"known fields: {', '.join(schema.known_fields)}",
                name,
            )

    return errors


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #


@dataclass
class LoadResult:
    """Outcome of reading a JSONL file.

    Attributes:
        records: successfully parsed records, in file order.
        errors: parse failures (malformed JSON, non-object rows).
        line_numbers: 1-based source line for each entry in ``records``.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[RecordError] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_number, raw_line)`` for every non-blank line in ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if raw.strip():
                yield line_number, raw


def load_jsonl(path: Path | str) -> LoadResult:
    """Read a JSONL file, collecting parse errors instead of raising.

    Blank lines are skipped. Every other line must be a JSON object.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    result = LoadResult()
    for line_number, raw in iter_jsonl(path):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            result.errors.append(
                RecordError(line_number, None, "invalid_json", f"line is not valid JSON: {exc.msg}")
            )
            continue
        if not isinstance(parsed, dict):
            result.errors.append(
                RecordError(
                    line_number,
                    None,
                    "invalid_type",
                    f"line must be a JSON object, got {type(parsed).__name__}",
                )
            )
            continue
        result.records.append(parsed)
        result.line_numbers.append(line_number)
    return result


def write_jsonl(path: Path | str, records: Iterable[Dict[str, Any]]) -> int:
    """Write ``records`` as UTF-8 JSONL deterministically; return the count.

    Keys are sorted and non-ASCII characters are preserved verbatim, so the
    same input always produces a byte-identical file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def find_duplicate_ids(
    records: Sequence[Dict[str, Any]],
    line_numbers: Optional[Sequence[int]] = None,
) -> List[RecordError]:
    """Report every record whose ``id`` was already seen earlier in the file."""
    seen: Dict[str, int] = {}
    errors: List[RecordError] = []
    for index, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            continue
        line = line_numbers[index] if line_numbers and index < len(line_numbers) else index + 1
        if record_id in seen:
            errors.append(
                RecordError(
                    line,
                    record_id,
                    "duplicate_id",
                    f"id '{record_id}' already used on line {seen[record_id]}",
                    "id",
                )
            )
        else:
            seen[record_id] = line
    return errors
