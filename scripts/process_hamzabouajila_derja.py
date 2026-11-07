"""Process hamzabouajila Tunisian Derja corpus into canonical JSONL.

Reads both parquet shards from data/raw/hamzabouajila__tunisian-derja-unified-raw-corpus/,
converts each text entry to a canonical record, validates against the schema,
deduplicates, splits train/holdout, and appends to the arabic_derja slice.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import schema, split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SCHEMA = schema.load_schema()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]",
    flags=re.UNICODE,
)
HAS_ARABIC = re.compile(r"[\u0600-\u06FF]")


def _clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_short(text: str, min_chars: int = 10) -> bool:
    return len(text.strip()) < min_chars


def _has_arabic(text: str) -> bool:
    return bool(HAS_ARABIC.search(text))


def _canonical_record(
    *,
    id: str,
    user_text: str,
    assistant_text: str,
    category: str,
    source: str,
    language: str,
    script: str,
    code_switching: bool,
    quality: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    record: Dict[str, Any] = {
        "id": id,
        "messages": messages,
        "category": category,
        "source": source,
        "language": language,
        "script": script,
        "code_switching": code_switching,
    }
    if quality:
        record["quality"] = quality
    return record


def build_hamzabouajila() -> List[Dict[str, Any]]:
    """Read both hamzabouajila parquet shards and convert to canonical records."""
    records: List[Dict[str, Any]] = []
    data_dir = RAW / "hamzabouajila__tunisian-derja-unified-raw-corpus" / "data"
    counter = 0
    skipped_short = 0
    skipped_no_arabic = 0

    for shard in sorted(data_dir.glob("train-*-of-*.parquet")):
        table = pq.read_table(str(shard))
        col_text = table["text"]
        for i in range(table.num_rows):
            text = str(col_text[i])
            text = _clean_text(text)
            if not text or _is_short(text):
                skipped_short += 1
                continue
            if not _has_arabic(text):
                skipped_no_arabic += 1
                continue
            counter += 1
            rid = f"hamzabouajila-{counter}"
            records.append(
                _canonical_record(
                    id=rid,
                    user_text="Read this Tunisian Derja text:",
                    assistant_text=text,
                    category="general_instruction",
                    source="hamzabouajila_tunisian_derja_unified",
                    language="ar",
                    script="arabic",
                    code_switching=False,
                    quality={"reviewed": False},
                )
            )

    print(f"  Built {len(records)} records")
    print(f"  Skipped short: {skipped_short}")
    print(f"  Skipped no-Arabic: {skipped_no_arabic}")
    return records


def validate_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    valid: List[Dict[str, Any]] = []
    rejections: Dict[str, int] = {}
    for idx, record in enumerate(records):
        errors = schema.validate_record(record, SCHEMA, line=idx + 1)
        if errors:
            for e in errors:
                rejections[e.code] = rejections.get(e.code, 0) + 1
        else:
            valid.append(record)
    return valid, rejections


def dedupe_by_id(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    removed = 0
    for record in records:
        rid = record.get("id")
        if isinstance(rid, str) and rid in seen:
            removed += 1
            continue
        if isinstance(rid, str):
            seen.add(rid)
        unique.append(record)
    return unique, removed


def split_train_holdout(records, holdout_size: int, seed=42):
    return split.stratified_split(records, holdout_size, stratify_by="source", seed=seed)


def write_manifest(slice_name, sources, train_records, holdout_records, rejections, dedup_removed, existing_manifest=None):
    manifest_dir = ROOT / "data" / "manifests" / slice_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "slice": slice_name,
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "train": {"count": len(train_records)},
        "holdout": {"count": len(holdout_records)},
        "rejected": rejections,
        "dedup_removed": dedup_removed,
        "deterministic": True,
    }
    if existing_manifest:
        manifest["previous"] = existing_manifest
    path = manifest_dir / "selection.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> int:
    """Append records to an existing JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def load_existing_ids(path: Path) -> set:
    """Load all existing IDs from a JSONL file for deduplication."""
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rid = record.get("id")
                if isinstance(rid, str):
                    ids.add(rid)
            except json.JSONDecodeError:
                continue
    return ids


def main() -> int:
    # --- Build hamzabouajila records ---------------------------------------
    print("==> Building hamzabouajila Derja records...")
    records = build_hamzabouajila()

    # --- Deduplicate within new data ---------------------------------------
    unique, dedup_removed = dedupe_by_id(records)
    print(f"  After dedup: {len(unique)} records (removed {dedup_removed})")

    # --- Filter out IDs already in existing arabic_derja train/holdout -----
    existing_train = PROC / "arabic_derja" / "train.jsonl"
    existing_holdout = PROC / "arabic_derja" / "holdout.jsonl"
    existing_ids = load_existing_ids(existing_train) | load_existing_ids(existing_holdout)

    filtered = [r for r in unique if r.get("id") not in existing_ids]
    print(f"  After removing already-present IDs: {len(filtered)} records")

    # --- Validate -----------------------------------------------------------
    print("==> Validating...")
    valid, rejections = validate_records(filtered)
    print(f"  valid={len(valid)} rejections={rejections}")

    # --- Split train/holdout ------------------------------------------------
    print("==> Splitting train/holdout...")
    holdout_size = max(1, int(len(valid) * 0.05))
    result = split_train_holdout(valid, holdout_size=holdout_size, seed=42)
    print(f"  train={len(result.train)} holdout={len(result.holdout)}")

    # --- Append to existing slice files -------------------------------------
    derja_dir = PROC / "arabic_derja"
    derja_dir.mkdir(parents=True, exist_ok=True)

    n_train = append_jsonl(derja_dir / "train.jsonl", result.train)
    n_holdout = append_jsonl(derja_dir / "holdout.jsonl", result.holdout)
    print(f"  Appended {n_train} train records to arabic_derja/train.jsonl")
    print(f"  Appended {n_holdout} holdout records to arabic_derja/holdout.jsonl")

    # --- Write manifest ------------------------------------------------------
    existing_manifest = None
    manifest_path = ROOT / "data" / "manifests" / "arabic_derja" / "selection.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    m = write_manifest(
        "arabic_derja",
        ["hamzabouajila_tunisian_derja_unified"],
        result.train,
        result.holdout,
        rejections,
        dedup_removed,
        existing_manifest=existing_manifest,
    )
    print(f"  Wrote manifest: {m}")

    # --- Regenerate combined data/train.jsonl --------------------------------
    print("==> Regenerating combined data/train.jsonl...")
    all_records: List[Dict[str, Any]] = []

    for name, p in [
        ("MSA", PROC / "msa" / "train.jsonl"),
        ("Retention", PROC / "retention" / "train.jsonl"),
        ("Arabizi", PROC / "arabizi" / "train.jsonl"),
        ("Derja", PROC / "arabic_derja" / "train.jsonl"),
    ]:
        if p.exists():
            data = schema.load_jsonl(p)
            all_records.extend(data.records)
            print(f"    {name} train: {len(data.records)} records")

    # Add franco_tunisian if it exists
    franco_path = PROC / "franco_tunisian" / "train.jsonl"
    if franco_path.exists():
        data = schema.load_jsonl(franco_path)
        all_records.extend(data.records)
        print(f"    Franco_Tunisian train: {len(data.records)} records")

    train_path = ROOT / "data" / "train.jsonl"
    n_total = schema.write_jsonl(train_path, all_records)
    print(f"  Wrote data/train.jsonl: {n_total} total records")

    print("==> DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
