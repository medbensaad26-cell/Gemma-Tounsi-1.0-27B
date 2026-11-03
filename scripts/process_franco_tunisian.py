"""Process franco_tunisian (French/Tunisian code-switching) raw data into canonical JSONL.

Processes pools incrementally to avoid memory issues with large sources.
"""
from __future__ import annotations

import hashlib
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
MANIFEST = ROOT / "data" / "manifests"

SCHEMA = schema.load_schema()

FRENCH_WORDS = re.compile(
    r"\b("
    r"le|la|les|un|une|des|et|ou|a|de|du|en|je|tu|il|elle|nous|vous|ils|elles|"
    r"ce|cette|ces|mon|ton|son|ma|ta|sa|mes|tes|ses|est|sont|etre|avoir|faire|"
    r"aller|dire|voir|savoir|pouvoir|vouloir|devoir|falloir|qui|que|quoi|dont|"
    r"plus|moins|tres|bien|mal|ici|maintenant|aujourdhui|demain|hier|"
    r"soir|matin|jour|nuit|an|ans|fois|pour|sur|avec|sans|sous|dans|chez|"
    r"par|pour|quoi|comment|quand|si|car|mais|donc|or|ni|que|l|d|j|qu|n|m|t|s|c|jusqu|presque|depuis|pendant|avant|apres|"
    r"entre|vers|contre|selon|malgre|via"
    r")\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]",
    flags=re.UNICODE,
)


def _clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_french(text: str) -> bool:
    return bool(FRENCH_WORDS.search(text))


def _is_short(text: str, min_chars: int = 10) -> bool:
    return len(text.strip()) < min_chars


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


CAT_MAP = {
    "mathematics": "mathematics", "reasoning": "reasoning", "coding": "coding",
    "general_instruction": "general_instruction", "knowledge_qa": "knowledge_qa",
}


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


def build_farah21() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = RAW / "Farah21__reddit-tunisia-qa" / "data" / "train-00000-of-00001.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    cs_dialects = {"mixed", "french", "tunisian_arabizi"}
    for row in rows:
        dialect = (row.get("dialect") or "").strip()
        if dialect not in cs_dialects:
            continue
        instruction = _clean_text(row.get("instruction") or "")
        input_text = _clean_text(row.get("input") or "")
        output = _clean_text(row.get("output") or "")
        if not instruction and not input_text and not output:
            continue
        if _is_short(instruction) and _is_short(input_text) and _is_short(output):
            continue
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n{input_text}" if instruction else input_text
        if not user_text:
            user_text = output
        if dialect == "tunisian_arabizi":
            language, script = "arabizi", "latin"
        elif dialect == "french":
            language, script = "fr", "latin"
        else:
            language, script = "mixed", "mixed"
        rid = str(row.get("id", "")) or hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:16]
        records.append(
            _canonical_record(
                id=f"farah21-{rid}",
                user_text=user_text,
                assistant_text=output or user_text,
                category="knowledge_qa",
                source="farah21_reddit_tunisia_qa",
                language=language,
                script=script,
                code_switching=True,
                quality={"reviewed": False},
            )
        )
    return records


def build_syrinesmati() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = RAW / "Syrinesmati__tunisian-question-response-dataset" / "data" / "train-00000-of-00001.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for idx, row in enumerate(rows):
        instruction = _clean_text(row.get("instruction") or "")
        response = _clean_text(row.get("response") or "")
        category = (row.get("category") or "general_instruction").strip()
        if not instruction or not response:
            continue
        if _is_short(instruction) and _is_short(response):
            continue
        if not _has_french(f"{instruction} {response}"):
            continue
        has_arabic = _has_arabic(f"{instruction} {response}")
        script = "arabic" if has_arabic else "latin"
        language = "ar" if has_arabic else "mixed"
        cat = CAT_MAP.get(category, "general_instruction")
        rid = f"syrinesmati-{idx}"
        records.append(
            _canonical_record(
                id=rid,
                user_text=instruction,
                assistant_text=response,
                category=cat,
                source="syrinesmati_tunisian_question_response",
                language=language,
                script=script,
                code_switching=True,
                quality={"reviewed": True},
            )
        )
    return records


def build_linagora_tsc() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = RAW / "linagora__Tunisian_Derja_Dataset" / "TunSwitchCodeSwitching" / "train-00000-of-00001.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for idx, row in enumerate(rows):
        text = _clean_text(row.get("text") or "")
        if not text or _is_short(text):
            continue
        has_arabic = _has_arabic(text)
        script = "arabic" if has_arabic else "latin"
        language = "ar" if has_arabic else "mixed"
        rid = f"linagora_tsc-{idx}"
        records.append(
            _canonical_record(
                id=rid,
                user_text="Read this Tunisian French code-switching text:",
                assistant_text=text,
                category="general_instruction",
                source="linagora_tunisian_derja_dataset_tsc",
                language=language,
                script=script,
                code_switching=True,
                quality={"reviewed": False},
            )
        )
    return records


def build_hamzabouajila() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    data_dir = RAW / "hamzabouajila__tunisian-derja-unified-raw-corpus" / "data"
    counter = 0
    for shard in sorted(data_dir.glob("train-*-of-*.parquet")):
        table = pq.read_table(str(shard))
        rows = table.to_pylist()
        for row in rows:
            text = _clean_text(row.get("text") or "")
            if not text or _is_short(text):
                continue
            if not _has_french(text):
                continue
            has_arabic = _has_arabic(text)
            script = "arabic" if has_arabic else "latin"
            language = "ar" if has_arabic else "mixed"
            counter += 1
            rid = f"hamzabouajila-{counter}"
            records.append(
                _canonical_record(
                    id=rid,
                    user_text="Read this Tunisian French code-switching text:",
                    assistant_text=text,
                    category="general_instruction",
                    source="hamzabouajila_tunisian_derja_unified",
                    language=language,
                    script=script,
                    code_switching=True,
                    quality={"reviewed": False},
                )
            )
    return records


def build_linto_tsc() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    data_dir = RAW / "linagora__linto-dataset-text-ar-tn" / "data" / "TunSwitchCodeSwitching"
    if not data_dir.is_dir():
        return records
    for shard in sorted(data_dir.glob("train-*-of-*.parquet")):
        table = pq.read_table(str(shard))
        rows = table.to_pylist()
        for row in rows:
            text = _clean_text(row.get("text") or row.get("text_raw") or "")
            if not text or _is_short(text):
                continue
            rid_raw = row.get("ids") or ""
            has_arabic = _has_arabic(text)
            script = "arabic" if has_arabic else "latin"
            language = "ar" if has_arabic else "mixed"
            rid = f"linto_tsc-{rid_raw or hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
            records.append(
                _canonical_record(
                    id=rid,
                    user_text="Read this Tunisian French code-switching text:",
                    assistant_text=text,
                    category="general_instruction",
                    source="linto_text_ar_tn_tsc",
                    language=language,
                    script=script,
                    code_switching=True,
                    quality={"reviewed": False},
                )
            )
    return records


def validate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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


def dedupe_by_id(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
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


def split_train_holdout(records, holdout_ratio=0.05, seed=42):
    n_holdout = max(1, int(len(records) * holdout_ratio))
    result = split.stratified_split(records, n_holdout, stratify_by="source", seed=seed)
    return result.train, result.holdout


def write_manifest(slice_name, sources, train_records, holdout_records, rejections, dedup_removed):
    manifest_dir = MANIFEST / slice_name
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
    path = manifest_dir / "selection.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


import json


def main() -> int:
    sources = []

    print("==> Building Farah21 reddit-tunisia-qa records...", flush=True)
    farah21 = build_farah21()
    sources.append("farah21_reddit_tunisia_qa")
    print(f"    {len(farah21)} records", flush=True)

    print("==> Building Syrinesmati records...", flush=True)
    syrinesmati = build_syrinesmati()
    sources.append("syrinesmati_tunisian_question_response")
    print(f"    {len(syrinesmati)} records", flush=True)

    print("==> Building linagora TunSwitchCodeSwitching records...", flush=True)
    linagora_tsc = build_linagora_tsc()
    sources.append("linagora_tunisian_derja_dataset_tsc")
    print(f"    {len(linagora_tsc)} records", flush=True)

    print("==> Building hamzabouajila records...", flush=True)
    hamzabouajila = build_hamzabouajila()
    sources.append("hamzabouajila_tunisian_derja_unified")
    print(f"    {len(hamzabouajila)} records", flush=True)

    print("==> Building linto TunSwitchCodeSwitching records...", flush=True)
    linto_tsc = build_linto_tsc()
    sources.append("linto_text_ar_tn_tsc")
    print(f"    {len(linto_tsc)} records", flush=True)

    all_raw = farah21 + syrinesmati + linagora_tsc + hamzabouajila + linto_tsc
    print(f"==> Total raw franco_tunisian records: {len(all_raw)}", flush=True)

    print("==> Validating...", flush=True)
    valid, rejections = validate_records(all_raw)
    print(f"    valid={len(valid)} rejections={rejections}", flush=True)

    print("==> Deduplicating...", flush=True)
    unique, dedup_removed = dedupe_by_id(valid)
    print(f"    unique={len(unique)} removed={dedup_removed}", flush=True)

    print("==> Splitting train/holdout...", flush=True)
    train, holdout = split_train_holdout(unique)
    print(f"    train={len(train)} holdout={len(holdout)}", flush=True)

    franco_dir = PROC / "franco_tunisian"
    franco_dir.mkdir(parents=True, exist_ok=True)

    n1 = schema.write_jsonl(franco_dir / "train.jsonl", train)
    n2 = schema.write_jsonl(franco_dir / "holdout.jsonl", holdout)
    print(f"    wrote franco_tunisian/train.jsonl: {n1}", flush=True)
    print(f"    wrote franco_tunisian/holdout.jsonl: {n2}", flush=True)

    m = write_manifest("franco_tunisian", sources, train, holdout, rejections, dedup_removed)
    print(f"    wrote {m}", flush=True)

    print("==> Combining all slices into data/train.jsonl...", flush=True)
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
            print(f"    {name} train: {len(data.records)} records", flush=True)

    all_records.extend(train)
    print(f"    Franco Tunisian train: {len(train)} records", flush=True)

    train_path = ROOT / "data" / "train.jsonl"
    n_total = schema.write_jsonl(train_path, all_records)
    print(f"    wrote data/train.jsonl: {n_total} total records", flush=True)

    holdout_ids = {r["id"] for r in holdout}
    train_ids = {r["id"] for r in train}
    overlap = train_ids & holdout_ids
    if overlap:
        print(f"ERROR: {len(overlap)} train/holdout overlap!", flush=True)
        return 1
    print("    No train/holdout contamination: OK", flush=True)
    print("==> DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
