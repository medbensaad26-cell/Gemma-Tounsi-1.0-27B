"""Process Arabizi and Derja raw data into canonical training JSONL.

Mirrors the MSA/retention processing pipeline: raw/ -> canonical schema ->
validate -> deduplicate -> train/holdout split -> data/processed/.

Sources:
  - TUNIZI v2 (Arabizi sentiment): data/raw/iCompass__TUNIZI_v2/tunizi_train
  - TArC (Arabizi + CODA): data/raw/arbml__TArC/data/*.parquet
  - linagora/Tunisian_Derja_Dataset (Arabic script Derja sub-corpora):
      data/raw/linagora__Tunisian_Derja_Dataset/{Derja_tunsi,HkayetErwi,...
      MADAR_TunisianDialect,QADI_TunisianDialect,Sentiment_Derja,TA_Segmentation,
      TSAC,TuDiCOI,TunBERT,TunisianSentimentAnalysis,Tweet_TN}/*.parquet

Excluded from Derja slice (code-switched/English):
  - TunSwitchCodeSwitching, TunSwitchTunisiaOnly, Tunisian_Dialectic_English_Derja

Output:
  - data/processed/arabizi/train.jsonl   (language: arabizi, script: latin)
  - data/processed/arabizi/holdout.jsonl
  - data/processed/arabic_derja/train.jsonl   (language: ar, script: arabic)
  - data/processed/arabic_derja/holdout.jsonl
  - data/manifests/arabizi/selection.json
  - data/manifests/arabic_derja/selection.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dedupe, mixture, retention, schema, split, validate  # noqa: E402


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifests"

SCHEMA = schema.load_schema()

# --- Derja Arabic-script sub-corpora (from regular_derja_mohamed.md) ---
DERJA_SUBCORPORA = [
    "Derja_tunsi", "HkayetErwi", "MADAR_TunisianDialect", "QADI_TunisianDialect",
    "Sentiment_Derja", "TA_Segmentation", "TSAC", "TuDiCOI",
    "TunBERT", "TunisianSentimentAnalysis", "Tweet_TN",
]

# --- Derja code-switched/English sub-corpora (EXCLUDED from this slice) ---
EXCLUDED_DERJA = {
    "TunSwitchCodeSwitching", "TunSwitchTunisiaOnly", "Tunisian_Dialectic_English_Derja",
}

LABEL_MAP = {"1": "positive", "0": "neutral", "-1": "negative"}


# --------------------------------------------------------------------------- #
# Canonical record builders
# --------------------------------------------------------------------------- #

def _canonical_record(
    *,
    id: str,
    user_text: str,
    assistant_text: str,
    category: str,
    source: str,
    language: str,
    script: str,
    system_text: str | None = None,
    quality: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})

    record: Dict[str, Any] = {
        "id": id,
        "messages": messages,
        "category": category,
        "source": source,
        "language": language,
    }
    if script:
        record["script"] = script
    if quality:
        record["quality"] = quality
    return record


def build_tunizi_records() -> List[Dict[str, Any]]:
    """TUNIZI v2 sentiment classification -> Arabizi knowledge_qa."""
    records: List[Dict[str, Any]] = []
    path = RAW / "iCompass__TUNIZI_v2" / "tunizi_train"
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid = row["ID"].strip()
            text = row["text"].strip()
            label = LABEL_MAP.get(row["label"].strip(), "neutral")
            if not text:
                continue
            records.append(
                _canonical_record(
                    id=f"tunizi-{rid}",
                    user_text=f"Classify the sentiment of this Tunisian Arabizi text: {text}",
                    assistant_text=label,
                    category="knowledge_qa",
                    source="tunizi_v2",
                    language="arabizi",
                    script="latin",
                    quality={"reviewed": True, "native_authored": True},
                )
            )
    return records


def build_tarc_records() -> List[Dict[str, Any]]:
    """TArC Arabizi + CODA transliteration -> Arabizi general_instruction."""
    records: List[Dict[str, Any]] = []
    data_dir = RAW / "arbml__TArC" / "data"
    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table()
    rows = table.to_pylist()
    for idx, row in enumerate(rows):
        arabish = (row.get("arabish") or "").strip()
        coda = (row.get("CODA") or "").strip()
        if not arabish:
            continue
        # Use CODA as the assistant response (Arabic script transliteration),
        # falling back to echoing the Arabizi if CODA is empty.
        assistant = coda if coda else arabish
        records.append(
            _canonical_record(
                id=f"tarc-{idx}",
                user_text=f"Transliterate this Tunisian Arabizi text to Arabic script: {arabish}",
                assistant_text=assistant,
                category="general_instruction",
                source="tarc",
                language="arabizi",
                script="latin",
                quality={"reviewed": False},
            )
        )
    return records


def build_farah21_reddit_arabizi_records() -> List[Dict[str, Any]]:
    """Farah21/reddit-tunisia-qa rows labeled dialect==tunisian_arabizi -> Arabizi knowledge_qa (QA pairs)."""
    records: List[Dict[str, Any]] = []
    path = RAW / "Farah21__reddit-tunisia-qa" / "data" / "train-00000-of-00001.parquet"
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    rows = table.to_pylist()
    for idx, row in enumerate(rows):
        dialect = (row.get("dialect") or "").strip()
        if dialect != "tunisian_arabizi":
            continue
        instruction = _clean_text(row.get("instruction") or "")
        input_text = _clean_text(row.get("input") or "")
        output = _clean_text(row.get("output") or "")
        if not instruction and not input_text and not output:
            continue
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n{input_text}" if instruction else input_text
        if not user_text:
            user_text = output
        rid = str(row.get("id", "")) or f"farah21_reddit_arabizi-{idx}"
        records.append(
            _canonical_record(
                id=f"farah21_reddit_arabizi-{rid}",
                user_text=user_text,
                assistant_text=output or user_text,
                category="knowledge_qa",
                source="farah21_reddit_tunisia_qa",
                language="arabizi",
                script="latin",
                quality={"reviewed": False},
            )
        )
    return records


def build_ctab_records() -> List[Dict[str, Any]]:
    """CTAB (Corpus of Tunisian Arabizi) raw messages -> Arabizi general_instruction."""
    records: List[Dict[str, Any]] = []
    ctab_dir = RAW / "zenodo__ctab-4781769"
    if not ctab_dir.is_dir():
        return records
    counter = 0
    for txt_file in sorted(ctab_dir.glob("CTAB-SAMPLE*.txt")):
        raw = txt_file.read_bytes()
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            text_content = raw.decode("latin-1")
        for line in text_content.splitlines():
                text = _clean_text(line)
                if not text:
                    continue
                counter += 1
                rid = f"ctab-{txt_file.stem}-{counter}"
                records.append(
                    _canonical_record(
                        id=rid,
                        user_text=f"Transliterate this Tunisian Arabizi text to Arabic script: {text}",
                        assistant_text=text,
                        category="general_instruction",
                        source="ctab",
                        language="arabizi",
                        script="latin",
                        quality={"reviewed": False},
                    )
                )
    return records


def build_farah21_tunisia_tech_records() -> List[Dict[str, Any]]:
    """Farah21/Tunisia_Tech rows labeled dialect==tunisian_arabizi -> Arabizi general_instruction."""
    records: List[Dict[str, Any]] = []
    data_dir = RAW / "Farah21__Tunisia_Tech" / "data"
    if not data_dir.is_dir():
        return records
    import pyarrow.parquet as pq
    counter = 0
    for shard in sorted(data_dir.glob("train-*.parquet")):
        table = pq.read_table(str(shard))
        rows = table.to_pylist()
        for row in rows:
            dialect = (row.get("dialect") or "").strip()
            if dialect != "tunisian_arabizi":
                continue
            instruction = _clean_text(row.get("instruction") or "")
            input_text = _clean_text(row.get("input") or "")
            output = _clean_text(row.get("output") or "")
            if not instruction and not input_text and not output:
                continue
            user_text = instruction
            if input_text:
                user_text = f"{instruction}\n{input_text}" if instruction else input_text
            if not user_text:
                user_text = output
            counter += 1
            rid = str(row.get("id", "")) or f"farah21_tunisia_tech-{counter}"
            records.append(
                _canonical_record(
                    id=f"farah21_tunisia_tech-{rid}",
                    user_text=user_text,
                    assistant_text=output or user_text,
                    category="general_instruction",
                    source="farah21_tunisia_tech",
                    language="arabizi",
                    script="latin",
                    quality={"reviewed": False},
                )
            )
    return records


def build_tunizi_v1_records() -> List[Dict[str, Any]]:
    """TUNIZI v1 (chaymafourati/tunizi) sentiment sentences -> Arabizi knowledge_qa."""
    records: List[Dict[str, Any]] = []
    path = RAW / "chaymafourati__tunizi" / "TUNIZI-Dataset.txt"
    if not path.exists():
        return records
    LABEL_MAP = {"1": "positive", "0": "neutral", "-1": "negative"}
    counter = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Format: label;text  (some lines have % prefix)
            line = line.lstrip("%")
            if ";" not in line:
                continue
            label, text = line.split(";", 1)
            label = label.strip()
            text = _clean_text(text)
            if not text:
                continue
            sentiment = LABEL_MAP.get(label, "neutral")
            counter += 1
            rid = f"tunizi_v1-{counter}"
            records.append(
                _canonical_record(
                    id=rid,
                    user_text=f"Classify the sentiment of this Tunisian Arabizi text: {text}",
                    assistant_text=sentiment,
                    category="knowledge_qa",
                    source="tunizi_v1",
                    language="arabizi",
                    script="latin",
                    quality={"reviewed": True, "native_authored": True},
                )
            )
    return records


def build_farah21_mixed_arabizi_records() -> List[Dict[str, Any]]:
    """Farah21/reddit-tunisia-qa mixed-dialect rows containing Arabizi digits -> Arabizi general_instruction."""
    records: List[Dict[str, Any]] = []
    path = RAW / "Farah21__reddit-tunisia-qa" / "data" / "train-00000-of-00001.parquet"
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    rows = table.to_pylist()
    arabizi_re = re.compile(r"\b\w*[379]\w*\b")
    counter = 0
    for row in rows:
        dialect = (row.get("dialect") or "").strip()
        if dialect != "mixed":
            continue
        instruction = _clean_text(row.get("instruction") or "")
        input_text = _clean_text(row.get("input") or "")
        output = _clean_text(row.get("output") or "")
        combined = f"{instruction} {input_text} {output}"
        if not arabizi_re.search(combined):
            continue
        if not instruction and not input_text and not output:
            continue
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n{input_text}" if instruction else input_text
        if not user_text:
            user_text = output
        counter += 1
        rid = str(row.get("id", "")) or f"farah21_mixed_arabizi-{counter}"
        records.append(
            _canonical_record(
                id=f"farah21_mixed_arabizi-{rid}",
                user_text=user_text,
                assistant_text=output or user_text,
                category="general_instruction",
                source="farah21_reddit_tunisia_qa_mixed",
                language="arabizi",
                script="latin",
                quality={"reviewed": False},
            )
        )
    return records


def build_derja_records() -> List[Dict[str, Any]]:
    """linagora Arabic-script Derja sub-corpora -> Derja general_instruction."""
    records: List[Dict[str, Any]] = []
    base = RAW / "linagora__Tunisian_Derja_Dataset"
    counter = 0
    for sub in DERJA_SUBCORPORA:
        subdir = base / sub
        if not subdir.is_dir():
            continue
        dataset = ds.dataset(str(subdir), format="parquet")
        table = dataset.to_table()
        rows = table.to_pylist()
        for row in rows:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            counter += 1
            records.append(
                _canonical_record(
                    id=f"linagora-{sub}-{counter}",
                    user_text="Read this Tunisian Derja text:",
                    assistant_text=text,
                    category="general_instruction",
                    source=f"linagora_{sub}",
                    language="ar",
                    script="arabic",
                    quality={"reviewed": False},
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Validation + dedup
# --------------------------------------------------------------------------- #

def validate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Validate every record against the canonical schema. Return (valid, rejections)."""
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
    """Keep first occurrence of each id; return (unique, removed_count)."""
    seen: set[str] = set()
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


# --------------------------------------------------------------------------- #
# Train / holdout split
# --------------------------------------------------------------------------- #

def split_train_holdout(
    records: List[Dict[str, Any]], holdout_ratio: float = 0.05, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train/holdout split. Holdout reserved first."""
    n_holdout = max(1, int(len(records) * holdout_ratio))
    result = split.stratified_split(records, n_holdout, stratify_by="source", seed=seed)
    return result.train, result.holdout


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def write_manifest(
    slice_name: str,
    source: str,
    train_records: List[Dict[str, Any]],
    holdout_records: List[Dict[str, Any]],
    rejections: Dict[str, int],
    dedup_removed: int,
) -> Path:
    manifest_dir = MANIFEST / slice_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "slice": slice_name,
        "source": source,
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    # --- Build raw candidate pools ---
    print("==> Building TUNIZI v2 (Arabizi) records...")
    tunizi = build_tunizi_records()
    print(f"    {len(tunizi)} records")

    print("==> Building TArC (Arabizi) records...")
    tarc = build_tarc_records()
    print(f"    {len(tarc)} records")

    print("==> Building linagora Derja (Arabic script) records...")
    derja = build_derja_records()
    print(f"    {len(derja)} records")

    print("==> Building Farah21 reddit-tunisia-qa Arabizi records...")
    farah21_arabi = build_farah21_reddit_arabizi_records()
    print(f"    {len(farah21_arabi)} records")

    print("==> Building CTAB Arabizi records...")
    ctab = build_ctab_records()
    print(f"    {len(ctab)} records")

    print("==> Building Farah21/Tunisia_Tech Arabizi records...")
    farah21_tech = build_farah21_tunisia_tech_records()
    print(f"    {len(farah21_tech)} records")

    print("==> Building TUNIZI v1 (chaymafourati/tunizi) Arabizi records...")
    tunizi_v1 = build_tunizi_v1_records()
    print(f"    {len(tunizi_v1)} records")

    print("==> Building Farah21 mixed-dialect Arabizi records...")
    farah21_mixed = build_farah21_mixed_arabizi_records()
    print(f"    {len(farah21_mixed)} records")

    # --- Combine by slice ---
    arabizi_raw = tunizi + tarc + farah21_arabi + ctab + farah21_tech + tunizi_v1 + farah21_mixed
    derja_raw = derja

    # --- Validate ---
    print("==> Validating Arabizi records...")
    arabizi_valid, arabizi_rej = validate_records(arabizi_raw)
    print(f"    valid={len(arabizi_valid)} rejections={arabizi_rej}")

    print("==> Validating Derja records...")
    derja_valid, derja_rej = validate_records(derja_raw)
    print(f"    valid={len(derja_valid)} rejections={derja_rej}")

    # --- Deduplicate ---
    print("==> Deduplicating Arabizi by id...")
    arabizi_unique, arabizi_dedup = dedupe_by_id(arabizi_valid)
    print(f"    unique={len(arabizi_unique)} removed={arabizi_dedup}")

    print("==> Deduplicating Derja by id...")
    derja_unique, derja_dedup = dedupe_by_id(derja_valid)
    print(f"    unique={len(derja_unique)} removed={derja_dedup}")

    # --- Train / holdout split ---
    print("==> Splitting Arabizi into train/holdout...")
    arabizi_train, arabizi_holdout = split_train_holdout(arabizi_unique)
    print(f"    train={len(arabizi_train)} holdout={len(arabizi_holdout)}")

    print("==> Splitting Derja into train/holdout...")
    derja_train, derja_holdout = split_train_holdout(derja_unique)
    print(f"    train={len(derja_train)} holdout={len(derja_holdout)}")

    # --- Write processed files ---
    arabizi_dir = PROC / "arabizi"
    derja_dir = PROC / "arabic_derja"
    arabizi_dir.mkdir(parents=True, exist_ok=True)
    derja_dir.mkdir(parents=True, exist_ok=True)

    n1 = schema.write_jsonl(arabizi_dir / "train.jsonl", arabizi_train)
    n2 = schema.write_jsonl(arabizi_dir / "holdout.jsonl", arabizi_holdout)
    n3 = schema.write_jsonl(derja_dir / "train.jsonl", derja_train)
    n4 = schema.write_jsonl(derja_dir / "holdout.jsonl", derja_holdout)
    print(f"    wrote arabizi/train.jsonl: {n1}")
    print(f"    wrote arabizi/holdout.jsonl: {n2}")
    print(f"    wrote arabic_derja/train.jsonl: {n3}")
    print(f"    wrote arabic_derja/holdout.jsonl: {n4}")

    # --- Write manifests ---
    m1 = write_manifest("arabizi", "tunizi_v2+tunizi_v1+tarc+farah21_reddit_tunisia_qa+farah21_reddit_tunisia_qa_mixed+ctab+farah21_tunisia_tech", arabizi_train, arabizi_holdout, arabizi_rej, arabizi_dedup)
    m2 = write_manifest("arabic_derja", "linagora_Tunisian_Derja_Dataset", derja_train, derja_holdout, derja_rej, derja_dedup)
    print(f"    wrote {m1}")
    print(f"    wrote {m2}")

    # --- Combine into data/train.jsonl (msa + retention + arabizi + derja) ---
    print("==> Combining all slices into data/train.jsonl...")
    all_records: List[Dict[str, Any]] = []

    # MSA
    msa_train = PROC / "msa" / "train.jsonl"
    if msa_train.exists():
        msa_data = schema.load_jsonl(msa_train)
        all_records.extend(msa_data.records)
        print(f"    MSA train: {len(msa_data.records)} records")

    # Retention
    ret_train = PROC / "retention" / "train.jsonl"
    if ret_train.exists():
        ret_data = schema.load_jsonl(ret_train)
        all_records.extend(ret_data.records)
        print(f"    Retention train: {len(ret_data.records)} records")

    # Arabizi
    all_records.extend(arabizi_train)
    print(f"    Arabizi train: {len(arabizi_train)} records")

    # Derja
    all_records.extend(derja_train)
    print(f"    Derja train: {len(derja_train)} records")

    train_path = ROOT / "data" / "train.jsonl"
    n_total = schema.write_jsonl(train_path, all_records)
    print(f"    wrote data/train.jsonl: {n_total} total records")

    # --- Verify no train/holdout contamination ---
    holdout_ids = {r["id"] for r in arabizi_holdout} | {r["id"] for r in derja_holdout}
    train_ids = {r["id"] for r in all_records}
    overlap = train_ids & holdout_ids
    if overlap:
        print(f"ERROR: {len(overlap)} train/holdout overlap!")
        return 1
    print("    No train/holdout contamination: OK")

    print("==> DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
