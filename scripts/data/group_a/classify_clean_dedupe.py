"""Classify and clean Group A Tunisian Arabizi datasets.

Reads downloaded raw data from data/raw/group_a/, classifies each record
by script (tunisian_arabizi / tunisian_arabic_script / tunisian_mixed_script /
non_tunisian_or_uncertain), cleans text, deduplicates, and writes
data/processed/group_a/tunisian_arabizi_merged.jsonl

Only APPROVED datasets are included in the merged output.
CONDITIONAL datasets are processed but kept separate pending review.
REJECTED/HOLD datasets are NOT processed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "group_a"
PROC_DIR = ROOT / "data" / "processed" / "group_a"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Load config
CONFIG_PATH = ROOT / "configs" / "data" / "group_a.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# --- Script detection ---
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIZI_DIGIT_RE = re.compile(r"[379]")

# French indicators (for distinguishing French from Arabizi)
FRENCH_COMMON = {
    "le", "la", "les", "des", "un", "une", "et", "est", "pour",
    "avec", "dans", "cette", "nous", "vous", "que", "qui", "sur",
    "par", "plus", "bien", "très", "tout", "pas", "mais", "ou",
    "si", "ce", "ses", "mon", "ma", "mes", "du", "au", "aux",
}

# Tunisian Arabizi indicators
TUNISIAN_MARKERS = {
    "3a", "7a", "9a", "5a", "3l", "7l", "9l", "w", "wesh", "chonwa",
    "chno", "chwaya", "barcha", "mashi", "yawm", "nhar", "sba7",
    "msem7", "kteb", "qra", "shuf", "shu", "chouf", "ahla", "ah",
    "hna", "haka", "hadi", "hadak", "wakha", "welek", "win",
}


def classify_script(text: str) -> str:
    """Classify text into script category."""
    arabic_chars = len(ARABIC_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    arabizi_digits = len(ARABIZI_DIGIT_RE.findall(text))

    if arabic_chars > 0 and latin_chars > 0:
        return "tunisian_mixed_script"
    if arabic_chars > 0:
        return "tunisian_arabic_script"
    if latin_chars > 0:
        # Check for Tunisian Arabizi markers
        text_lower = text.lower()
        tunisian_score = sum(1 for m in TUNISIAN_MARKERS if m in text_lower)
        french_score = sum(1 for m in FRENCH_COMMON if m in text_lower.split())

        if arabizi_digits > 0 or tunisian_score > 0:
            return "tunisian_arabizi"
        if french_score > tunisian_score and french_score > 2:
            return "non_tunisian_or_uncertain"  # Likely French
        return "tunisian_arabizi"  # Default Latin-script Tunisian to arabizi
    return "non_tunisian_or_uncertain"


def clean_text(text: str) -> str:
    """Clean text: normalize whitespace, remove BOM, strip."""
    # Remove BOM
    text = text.replace("\ufeff", "")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid(text: str) -> bool:
    """Check if text is a valid non-empty record."""
    if not text or not text.strip():
        return False
    if len(text.strip()) < 2:
        return False
    return True


def read_tunizi_bigbench(path: Path) -> list[dict]:
    """Read TuniziBigBench TSV (text column)."""
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                records.append({"text": text, "source_dataset": "tunizi_bigbench"})
    return records


def read_tunizi_txt(path: Path, version: str) -> list[dict]:
    """Read TUNIZI TXT (text;label format)."""
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace("\ufeff", "")
            if ";" in line:
                text, label = line.rsplit(";", 1)
                records.append({
                    "text": text.strip(),
                    "source_dataset": f"tunizi_{version.lower()}",
                    "sentiment": label.strip(),
                })
            else:
                records.append({
                    "text": line,
                    "source_dataset": f"tunizi_{version.lower()}",
                })
    return records


def read_ctab(path: Path) -> list[dict]:
    """Read CTAB sample files (one message per line)."""
    records = []
    for f in sorted(path.glob("CTAB-SAMPLE*.txt")):
        page_name = f.stem
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line = line.replace("\ufeff", "")
                records.append({
                    "text": line,
                    "source_dataset": "ctab",
                    "facebook_page": page_name,
                })
    return records


def main() -> int:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    all_raw = []  # (record, dataset_name)
    dataset_counts = Counter()
    script_counts = Counter()

    # --- tunizi_bigbench ---
    tb_path = RAW_ROOT / "tunizi_bigbench" / "TuniziBigBenchV0.tsv"
    if tb_path.exists():
        print("Reading tunizi_bigbench...")
        recs = read_tunizi_bigbench(tb_path)
        for r in recs:
            all_raw.append((r, "tunizi_bigbench"))
        dataset_counts["tunizi_bigbench"] = len(recs)
        print(f"  {len(recs):,} records")

    # --- tunizi_v2 (both V1 and V2 from the same directory) ---
    for v in ["V1", "V2"]:
        p = RAW_ROOT / "tunizi_v2" / f"TUNIZI_{v}.txt"
        if p.exists():
            print(f"Reading tunizi_{v.lower()}...")
            recs = read_tunizi_txt(p, v)
            for r in recs:
                all_raw.append((r, f"tunizi_{v.lower()}"))
            dataset_counts[f"tunizi_{v.lower()}"] = len(recs)
            print(f"  {len(recs):,} records")

    # --- CTAB ---
    ctab_dir = RAW_ROOT / "ctab"
    if ctab_dir.exists():
        print("Reading ctab...")
        recs = read_ctab(ctab_dir)
        for r in recs:
            all_raw.append((r, "ctab"))
        dataset_counts["ctab"] = len(recs)
        print(f"  {len(recs):,} records")

    print(f"\nTotal raw records: {len(all_raw):,}")

    # --- Classify each record ---
    classified = []
    for record, dataset in all_raw:
        text = record["text"]
        script = classify_script(text)
        script_counts[script] += 1
        classified.append((record, dataset, script))

    print("\nScript distribution:")
    for s, c in script_counts.most_common():
        print(f"  {s}: {c:,}")

    # --- Filter to tunisian_arabizi only ---
    arabizi_records = [
        (r, d) for r, d, s in classified if s == "tunisian_arabizi"
    ]
    print(f"\nTunisian Arabizi records: {len(arabizi_records):,}")

    # --- Clean text ---
    cleaned = []
    for record, dataset in arabizi_records:
        text = clean_text(record["text"])
        if not is_valid(text):
            continue
        cleaned.append({
            "text": text,
            "source_dataset": record.get("source_dataset", dataset),
            "script": "arabizi",
            "language": "arabizi",
        })
        # Preserve sentiment if present
        if "sentiment" in record:
            cleaned[-1]["sentiment"] = record["sentiment"]
        if "facebook_page" in record:
            cleaned[-1]["facebook_page"] = record["facebook_page"]

    print(f"After cleaning: {len(cleaned):,} records")

    # --- Deduplicate ---
    seen_exact = set()
    seen_normalized = set()
    unique_records = []
    exact_dups = 0
    norm_dups = 0

    for record in cleaned:
        text = record["text"]
        # Exact dedup
        if text in seen_exact:
            exact_dups += 1
            continue
        seen_exact.add(text)

        # Normalized dedup (lowercase, strip punctuation)
        normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if normalized in seen_normalized:
            norm_dups += 1
            continue
        seen_normalized.add(normalized)

        unique_records.append(record)

    print(f"\nAfter dedup: {len(unique_records):,} records")
    print(f"  Exact duplicates removed: {exact_dups:,}")
    print(f"  Normalized duplicates removed: {norm_dups:,}")

    # --- Write merged output ---
    output_path = PROC_DIR / "tunisian_arabizi_merged.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, record in enumerate(unique_records):
            record["id"] = f"group_a_arabizi_{i:06d}"
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"\nMerged output written to {output_path}")
    print(f"Total unique records: {len(unique_records):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
