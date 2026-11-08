"""Inspect Group A Tunisian Arabizi datasets.

Inspects all downloaded datasets in data/raw/group_a/ and produces
data/reports/group_a/dataset_inspection.json with per-dataset structure,
record counts, script classification, and observations.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "group_a"
REPORT_PATH = ROOT / "data" / "reports" / "group_a" / "dataset_inspection.json"

# Script detection
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIZI_DIGIT_RE = re.compile(r"[379]")
FRENCH_MARKERS = [
    "les", "des", "une", "est", "pour", "avec", "dans", "cette",
    "nous", "vous", "etre", "avoir", "faire", "tres", "plus", "aussi",
    "mais", "alors", "chez", "est", "une", "des", "le", "la", "un",
]


def detect_script(text: str) -> str:
    """Classify text script: arabizi, arabic_script, mixed_script, uncertain."""
    arabic_chars = len(ARABIC_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    arabizi_digits = len(ARABIZI_DIGIT_RE.findall(text))

    if arabic_chars > 0 and latin_chars > 0:
        return "mixed_script"
    if arabic_chars > 0:
        return "arabic_script"
    if latin_chars > 0:
        # Check if it looks like Arabizi (Tunisian dialect written in Latin)
        if arabizi_digits > 0 or latin_chars > arabic_chars:
            return "arabizi"
        return "uncertain"  # Could be French or English
    return "uncertain"


def inspect_tsv(path: Path) -> dict:
    """Inspect a TSV file with 'text' column."""
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                records.append(text)
    return records


def inspect_txt_semicolon(path: Path) -> dict:
    """Inspect a TXT file with 'text;label' format."""
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                text, label = line.rsplit(";", 1)
                records.append((text.strip(), label.strip()))
            else:
                records.append((line, "unknown"))
    return records


def inspect_ctab(path: Path) -> dict:
    """Inspect CTAB sample text files (one message per line)."""
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Remove BOM
            if line.startswith("\ufeff"):
                line = line[1:]
            if line:
                records.append(line)
    return records


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- tunizi_bigbench (TSV) ---
    tb_path = RAW_ROOT / "tunizi_bigbench" / "TuniziBigBenchV0.tsv"
    if tb_path.exists():
        print("Inspecting tunizi_bigbench...")
        records = inspect_tsv(tb_path)
        script_counts = Counter(detect_script(r) for r in records)
        results["tunizi_bigbench"] = {
            "file": str(tb_path.relative_to(ROOT)),
            "format": "TSV",
            "text_column": "text",
            "total_records": len(records),
            "script_distribution": dict(script_counts),
            "sample": records[:5],
        }
        print(f"  {len(records):,} records")
        for s, c in script_counts.most_common():
            print(f"    {s}: {c:,}")

    # --- tunizi_v2 (TUNIZI_V1.txt and TUNIZI_V2.txt) ---
    for v in ["V1", "V2"]:
        p = RAW_ROOT / "tunizi_v2" / f"TUNIZI_{v}.txt"
        if p.exists():
            print(f"\nInspecting tunizi_{v.lower()}...")
            records = inspect_txt_semicolon(p)
            script_counts = Counter(detect_script(r[0]) for r in records)
            results[f"tunizi_{v.lower()}"] = {
                "file": str(p.relative_to(ROOT)),
                "format": "TXT (text;label)",
                "total_records": len(records),
                "script_distribution": dict(script_counts),
                "sample": records[:5],
            }
            print(f"  {len(records):,} records")
            for s, c in script_counts.most_common():
                print(f"    {s}: {c:,}")

    # --- CTAB samples ---
    ctab_dir = RAW_ROOT / "ctab"
    all_ctab = []
    for f in sorted(ctab_dir.glob("CTAB-SAMPLE*.txt")):
        records = inspect_ctab(f)
        all_ctab.extend(records)
    if all_ctab:
        print(f"\nInspecting ctab ({len(all_ctab)} records across 13 samples)...")
        script_counts = Counter(detect_script(r) for r in all_ctab)
        results["ctab"] = {
            "file": str(ctab_dir.relative_to(ROOT)),
            "format": "TXT (one message per line, grouped by Facebook page)",
            "total_records": len(all_ctab),
            "script_distribution": dict(script_counts),
            "sample": all_ctab[:5],
        }
        print(f"  {len(all_ctab):,} records")
        for s, c in script_counts.most_common():
            print(f"    {s}: {c:,}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)
    total_records = sum(r.get("total_records", 0) for r in results.values())
    print(f"Total records across all datasets: {total_records:,}")
    for name, info in results.items():
        print(f"  {name}: {info['total_records']:,} records")

    REPORT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nInspection report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
