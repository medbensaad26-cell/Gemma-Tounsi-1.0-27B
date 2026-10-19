"""Normalize retention candidate pools into the canonical schema.

Reads the three acquired retention sources declared in
``data/manifests/retention.yaml`` and emits canonical JSONL records
(``id, messages, category, source, language`` + optional fields) that
``src/data/retention.py`` can select from.

Usage:
    python scripts/normalize_retention_candidates.py

Outputs (all under ``data/processed/retention/candidates/``):
    metamathqa.jsonl      -> category: mathematics
    code-feedback.jsonl   -> category: coding
    slimorca.jsonl        -> category: reasoning / knowledge_qa / general_instruction
    retention_pool.jsonl  -> combined pool of all three

Adapters are dataset-specific by design; retention.yaml names no dataset.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import load_schema, validate_record
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed" / "retention" / "candidates"

SCHEMA = load_schema()

# Role mapping for ShareGPT-style ``from`` fields.
SHAREGPY_ROLE = {"system": "system", "human": "user", "gpt": "assistant"}

# SlimOrca has no explicit category field, so we tag deterministically from
# the user-turn content. These are heuristics to be audited (see
# docs/data/retention_haithem.md Section 2 / retention_mohamed.md Section 3).
_REASONING_HINTS = (
    "solve", "prove", "calculate", "theorem", "equation", "step by step",
    "logic", "puzzle", "derive", "proof", "math", "geometry", "algebra",
    "integral", "derivative", "matrix", "algorithm",
)
_KNOWLEDGE_HINTS = (
    "who", "what", "when", "where", "why", "capital", "president",
    "fact", "definition", "biography", "born", "invented", "discovered",
    "author of", "discovered by", "located in",
)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _messages_valid(messages: List[Dict[str, str]]) -> bool:
    """Quick structural pre-check before full schema validation."""
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if not roles:
        return False
    # A leading system turn is allowed; the conversation body must be
    # user/assistant alternating and end on assistant.
    body = roles[1:] if roles and roles[0] == "system" else roles
    if not body or body[0] != "user" or body[-1] != "assistant":
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if not _clean_text(m.get("content", "")):
            return False
        if m.get("role") not in ("system", "user", "assistant"):
            return False
    return True


def _tag_slimorca(content: str) -> str:
    lower = content.lower()
    if any(hint in lower for hint in _REASONING_HINTS):
        return "reasoning"
    if any(hint in lower for hint in _KNOWLEDGE_HINTS):
        return "knowledge_qa"
    return "general_instruction"


def _build_record(
    record_id: str,
    messages: List[Dict[str, str]],
    category: str,
    source: str,
    subcategory: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record: Dict[str, Any] = {
        "id": record_id,
        "messages": messages,
        "category": category,
        "source": source,
        "language": "en",
        "script": "latin",
        "quality": {"synthetic": False},
    }
    if subcategory:
        record["subcategory"] = subcategory
    return record


def _valid_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = validate_record(record, SCHEMA)
    return (len(errors) == 0), [e.code for e in errors]


# --------------------------------------------------------------------------- #
# MetaMathQA -> mathematics
# --------------------------------------------------------------------------- #


def normalize_metamathqa() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    path = RAW_ROOT / "meta-math__MetaMathQA" / "MetaMathQA-395K.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"MetaMathQA expected a JSON list, got {type(raw).__name__}")

    records: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}
    for index, row in enumerate(raw):
        query = _clean_text(row.get("query", ""))
        response = _clean_text(row.get("response", ""))
        if not query or not response:
            rejected["empty_content"] = rejected.get("empty_content", 0) + 1
            continue
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        record = _build_record(
            record_id=f"metamathqa-{index}",
            messages=messages,
            category="mathematics",
            source="metamathqa",
            subcategory=str(row.get("type", "unknown")),
        )
        ok, codes = _valid_record(record)
        if ok:
            records.append(record)
        else:
            for code in codes:
                rejected[code] = rejected.get(code, 0) + 1
    return records, rejected


# --------------------------------------------------------------------------- #
# Code-Feedback -> coding  (native ChatML, messages already [user, assistant])
# --------------------------------------------------------------------------- #


def normalize_code_feedback() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    path = RAW_ROOT / "m-a-p__Code-Feedback" / "Code-Feedback.jsonl"
    records: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected["invalid_json"] = rejected.get("invalid_json", 0) + 1
                continue
            if not isinstance(row, dict):
                rejected["invalid_type"] = rejected.get("invalid_type", 0) + 1
                continue
            messages = row.get("messages")
            if not _messages_valid(messages):
                rejected["invalid_messages"] = rejected.get("invalid_messages", 0) + 1
                continue
            record_id = str(row.get("id", f"code-feedback-{line_number}"))
            record = _build_record(
                record_id=f"code-feedback-{record_id}",
                messages=messages,
                category="coding",
                source="code-feedback",
            )
            ok, codes = _valid_record(record)
            if ok:
                records.append(record)
            else:
                for code in codes:
                    rejected[code] = rejected.get(code, 0) + 1
    return records, rejected


# --------------------------------------------------------------------------- #
# SlimOrca -> reasoning / knowledge_qa / general_instruction (tagged)
# --------------------------------------------------------------------------- #


def normalize_slimorca() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    path = (
        RAW_ROOT
        / "Open-Orca__SlimOrca"
        / "oo-labeled_correct.gpt4.sharegpt.jsonl"
    )
    records: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected["invalid_json"] = rejected.get("invalid_json", 0) + 1
                continue
            if not isinstance(row, dict):
                rejected["invalid_type"] = rejected.get("invalid_type", 0) + 1
                continue
            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                rejected["invalid_messages"] = rejected.get("invalid_messages", 0) + 1
                continue
            messages: List[Dict[str, str]] = []
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = SHAREGPY_ROLE.get(turn.get("from"))
                if role is None:
                    continue
                content = _clean_text(turn.get("value", ""))
                messages.append({"role": role, "content": content})
            if not _messages_valid(messages):
                rejected["invalid_messages"] = rejected.get("invalid_messages", 0) + 1
                continue
            # Tag from the last user turn (deterministic heuristic).
            user_turns = [m["content"] for m in messages if m["role"] == "user"]
            category = _tag_slimorca(user_turns[-1] if user_turns else "")
            tag_counts[category] = tag_counts.get(category, 0) + 1
            record = _build_record(
                record_id=f"slimorca-{index}",
                messages=messages,
                category=category,
                source="slimorca",
            )
            ok, codes = _valid_record(record)
            if ok:
                records.append(record)
            else:
                for code in codes:
                    rejected[code] = rejected.get(code, 0) + 1
    return records, rejected, tag_counts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mm_records, mm_rej = normalize_metamathqa()
    _write_jsonl(OUT_DIR / "metamathqa.jsonl", mm_records)
    print(f"metamathqa: {len(mm_records)} valid, rejected={mm_rej}")

    cf_records, cf_rej = normalize_code_feedback()
    _write_jsonl(OUT_DIR / "code-feedback.jsonl", cf_records)
    print(f"code-feedback: {len(cf_records)} valid, rejected={cf_rej}")

    so_records, so_rej, so_tags = normalize_slimorca()
    _write_jsonl(OUT_DIR / "slimorca.jsonl", so_records)
    print(f"slimorca: {len(so_records)} valid, rejected={so_rej}, tags={so_tags}")

    pool = mm_records + cf_records + so_records
    _write_jsonl(OUT_DIR / "retention_pool.jsonl", pool)
    print(f"combined pool: {len(pool)} records -> {OUT_DIR / 'retention_pool.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
