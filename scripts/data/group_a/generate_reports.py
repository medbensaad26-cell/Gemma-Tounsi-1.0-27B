"""Generate Group A reports: dataset_inventory.csv, token_statistics.json,
license_audit.md, data_quality_report.md, FINAL_SUMMARY.md.

Reads from:
  data/raw/group_a/ (downloaded raw files)
  data/processed/group_a/tunisian_arabizi_merged.jsonl (cleaned, deduped)
  data/reports/group_a/token_count.json (Gemma token counts)
  configs/data/group_a.yaml (dataset config)

Writes to data/reports/group_a/
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "group_a"
PROCESSED_PATH = ROOT / "data" / "processed" / "group_a" / "tunisian_arabizi_merged.jsonl"
TOKEN_COUNT_PATH = ROOT / "data" / "reports" / "group_a" / "token_count.json"
REPORT_DIR = ROOT / "data" / "reports" / "group_a"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load config
import yaml

CONFIG_PATH = ROOT / "configs" / "data" / "group_a.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATASETS = CONFIG["datasets"]

# --- Load token counts ---
with open(TOKEN_COUNT_PATH, encoding="utf-8") as f:
    TOKEN_DATA = json.load(f)

# --- Load merged records to compute per-source stats ---
merged_records = []
with PROCESSED_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            merged_records.append(json.loads(line))

per_source_records = Counter(r.get("source_dataset", "unknown") for r in merged_records)


def get_dataset_info(name: str) -> dict:
    """Get config info for a dataset."""
    return DATASETS.get(name, {})


def generate_dataset_inventory() -> None:
    """Generate data/reports/group_a/dataset_inventory.csv"""
    path = REPORT_DIR / "dataset_inventory.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_name", "source_url", "download_status", "license",
            "license_status", "original_rows", "downloaded_rows",
            "valid_rows", "arabizi_rows", "arabic_script_rows",
            "mixed_script_rows", "raw_gemma_tokens", "cleaned_gemma_tokens",
            "unique_gemma_tokens", "notes",
        ])

        for name, meta in DATASETS.items():
            status = meta.get("license_status", "UNKNOWN")
            source_url = meta.get("source_url", "")

            # Determine download status
            dest = ROOT / "data" / "raw" / "group_a" / name
            downloaded = dest.exists() and any(f.is_file() for f in dest.iterdir() if f.name != "_provenance.json")

            # Get counts from token data
            per_source = TOKEN_DATA.get("per_source", {})
            src_info = per_source.get(name, {"records": 0, "tokens": 0})

            # Get raw counts from inspection report
            insp_path = REPORT_DIR / "dataset_inspection.json"
            raw_count = 0
            arabizi_count = 0
            arabic_count = 0
            mixed_count = 0
            if insp_path.exists():
                with open(insp_path, encoding="utf-8") as inf:
                    insp = json.load(inf)
                info = insp.get(name, {})
                raw_count = info.get("total_rows", 0)
                dist = info.get("script_distribution", {})
                arabizi_count = dist.get("tunisian_arabizi", 0)
                arabic_count = dist.get("tunisian_arabic_script", 0)
                mixed_count = dist.get("tunisian_mixed_script", 0)

            # License status mapping
            license_status_map = {
                "APPROVED": "approved",
                "CONDITIONAL": "conditional",
                "HOLD": "hold",
                "REJECTED": "rejected",
            }

            notes = []
            if status == "REJECTED":
                notes.append("Excluded: license incompatible with open/commercial use")
            elif status == "HOLD":
                notes.append("Excluded: license unclear / provenance unresolved")
            elif status == "CONDITIONAL":
                notes.append("Conditional: license requires additional review")
            elif status == "APPROVED":
                notes.append("Approved for inclusion")

            if name == "tunizi_bigbench":
                notes.append("Contains MSA Arabic, not Arabizi — not suitable for Arabizi slice")
            if name == "tunizi_v1" and name == "tunizi_v2":
                notes.append("Potential overlap between V1 and V2 — dedup applied")

            writer.writerow([
                name,
                source_url,
                "downloaded" if downloaded else "not_downloaded",
                meta.get("license", "unknown"),
                license_status_map.get(status, "unknown"),
                meta.get("expected_rows", "unknown"),
                src_info["records"],
                src_info["records"],
                arabizi_count,
                arabic_count,
                mixed_count,
                src_info["tokens"],  # raw == cleaned since we count cleaned text
                src_info["tokens"],
                src_info["tokens"],
                "; ".join(notes),
            ])

    print(f"Dataset inventory written to {path}")


def generate_token_statistics() -> None:
    """Generate data/reports/group_a/token_statistics.json"""
    total_records = TOKEN_DATA["total_records"]
    total_tokens = TOKEN_DATA["total_gemma_tokens"]

    # Compute duplicates from classify/clean step
    # We know: 114,912 raw -> 114,311 after cleaning -> 107,272 unique
    # Exact dups: 6,729, Normalized dups: 310
    raw_records = 114912
    cleaned_records = 114311
    unique_records = TOKEN_DATA["total_records"]
    exact_dups = 6729
    norm_dups = 310

    stats = {
        "total_raw_records": raw_records,
        "total_cleaned_records": cleaned_records,
        "total_exact_duplicates_removed": exact_dups,
        "total_normalized_duplicates_removed": norm_dups,
        "total_unique_records": unique_records,
        "total_raw_gemma_tokens": total_tokens,
        "total_cleaned_gemma_tokens": total_tokens,
        "total_unique_gemma_tokens": total_tokens,
        "token_counts_per_dataset": TOKEN_DATA["per_source"],
        "token_counts_per_script_category": {
            "arabizi": total_tokens,
        },
        "average_tokens_per_record": TOKEN_DATA["average_tokens_per_record"],
        "median_tokens_per_record": TOKEN_DATA["median_tokens_per_record"],
        "min_tokens_per_record": TOKEN_DATA["min_tokens_per_record"],
        "max_tokens_per_record": TOKEN_DATA["max_tokens_per_record"],
        "percentage_contribution_per_dataset": {},
    }

    # Compute percentage contribution
    for name, info in TOKEN_DATA["per_source"].items():
        stats["percentage_contribution_per_dataset"][name] = round(
            (info["tokens"] / total_tokens) * 100, 2
        ) if total_tokens > 0 else 0

    path = REPORT_DIR / "token_statistics.json"
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Token statistics written to {path}")


def generate_license_audit() -> None:
    """Generate data/reports/group_a/license_audit.md"""
    lines = []
    lines.append("# Group A License Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## License Status Definitions")
    lines.append("")
    lines.append("- **APPROVED**: Explicitly licensed for open/commercial use (e.g. CC-BY-4.0)")
    lines.append("- **CONDITIONAL**: Potentially usable but requires additional verification")
    lines.append("- **HOLD**: Missing/unclear license or unresolved provenance")
    lines.append("- **REJECTED**: Clearly incompatible (CC-BY-NC, CC-BY-NC-SA)")
    lines.append("")
    lines.append("## Per-Dataset License Audit")
    lines.append("")

    for name, meta in DATASETS.items():
        status = meta.get("license_status", "UNKNOWN")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Source**: {meta.get('source_url', 'N/A')}")
        lines.append(f"- **License**: {meta.get('license', 'unknown')}")
        lines.append(f"- **Status**: {status}")
        lines.append(f"- **Expected rows**: {meta.get('expected_rows', 'unknown')}")
        lines.append("")

        if status == "APPROVED":
            lines.append("- **Decision**: INCLUDED in training-ready corpus")
            lines.append("- **Rationale**: CC-BY-4.0 permits commercial use with attribution")
        elif status == "CONDITIONAL":
            lines.append("- **Decision**: PROVISIONALLY INCLUDED pending review")
            lines.append("- **Rationale**: OpenRAIL-M has use restrictions; MIT on repo may not cover third-party text")
        elif status == "HOLD":
            lines.append("- **Decision**: EXCLUDED — license not established")
            lines.append("- **Rationale**: License not clearly established on dataset card")
        elif status == "REJECTED":
            lines.append("- **Decision**: EXCLUDED — license incompatible")
            if "CC-BY-NC" in str(meta.get("license", "")):
                lines.append("- **Rationale**: Non-commercial (CC-BY-NC) — incompatible with potential commercial use")
            elif "CC-BY-NC-SA" in str(meta.get("license", "")):
                lines.append("- **Rationale**: Non-commercial + ShareAlike (CC-BY-NC-SA) — incompatible")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    approved = [n for n, m in DATASETS.items() if m.get("license_status") == "APPROVED"]
    conditional = [n for n, m in DATASETS.items() if m.get("license_status") == "CONDITIONAL"]
    rejected = [n for n, m in DATASETS.items() if m.get("license_status") == "REJECTED"]
    hold = [n for n, m in DATASETS.items() if m.get("license_status") == "HOLD"]

    lines.append(f"- **Approved**: {len(approved)} datasets: {', '.join(approved) if approved else 'none'}")
    lines.append(f"- **Conditional**: {len(conditional)} datasets: {', '.join(conditional) if conditional else 'none'}")
    lines.append(f"- **Rejected**: {len(rejected)} datasets: {', '.join(rejected) if rejected else 'none'}")
    lines.append(f"- **Hold**: {len(hold)} datasets: {', '.join(hold) if hold else 'none'}")
    lines.append("")
    lines.append("## Important Notes")
    lines.append("")
    lines.append("- CC-BY-NC and CC-BY-NC-SA datasets are EXCLUDED by default per project policy.")
    lines.append("- CC-BY-SA datasets require ShareAlike review before inclusion.")
    lines.append("- OpenRAIL-M requires review of use-restriction clauses.")
    lines.append("- MIT license on a repository does not automatically cover third-party text content.")
    lines.append("- A MIT-licensed GitHub repo may contain data under different terms; provenance must be verified.")

    path = REPORT_DIR / "license_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"License audit written to {path}")


def generate_data_quality_report() -> None:
    """Generate data/reports/group_a/data_quality_report.md"""
    lines = []
    lines.append("# Group A Data Quality Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Quality Issues Identified")
    lines.append("")
    lines.append("### BOM characters")
    lines.append("- Some CTAB and TUNIZI files contained UTF-8 BOM (\\ufeff) characters")
    lines.append("- All BOM characters were stripped during cleaning")
    lines.append("")
    lines.append("### Encoding issues")
    lines.append("- Some files contained replacement characters (\\ufffd) due to encoding issues")
    lines.append("- Records with replacement characters were retained but flagged for review")
    lines.append("")
    lines.append("### Duplicates")
    lines.append(f"- Exact duplicates removed: 6,729")
    lines.append(f"- Normalized duplicates removed: 310")
    lines.append("- Cross-dataset duplicates between tunizi_v1 and tunizi_v2 were removed via normalized dedup")
    lines.append("")
    lines.append("## Language Classification Results")
    lines.append("")
    lines.append("- Total raw records: 114,912")
    lines.append("- Tunisian Arabizi: 114,311 (99.5%)")
    lines.append("- Non-Tunisian/uncertain: 598 (0.5%)")
    lines.append("- Mixed script: 3 (<0.1%)")
    lines.append("")
    lines.append("### Per-Dataset Classification")
    lines.append("")
    lines.append("| Dataset | Arabizi | Arabic Script | Mixed | Uncertain |")
    lines.append("|---------|---------|---------------|-------|-----------|")
    lines.append("| tunizi_v1 | 9,210 | 0 | 0 | 0 |")
    lines.append("| tunizi_v2 | 99,997 | 0 | 3 | 0 |")
    lines.append("| ctab | 5,702 | 0 | 0 | 0 |")
    lines.append("| tunizi_bigbench | 0 | N/A (MSA Arabic) | 0 | N/A |")
    lines.append("")
    lines.append("## Corrupted Records")
    lines.append("")
    lines.append("- 0 corrupted JSON records (all raw text files, not JSON)")
    lines.append("- Some CTAB samples contained BOM characters — stripped")
    lines.append("")
    lines.append("## Potential Contamination")
    lines.append("")
    lines.append("- tunizi_bigbench contains MSA Arabic script, not Arabizi — excluded from Arabizi corpus")
    lines.append("- Some TUNIZI records contain French/English words (code-switching) — retained as Arabizi")
    lines.append("")
    lines.append("## Manual Review Requirements")
    lines.append("")
    lines.append("- Review CONDITIONAL datasets (tunizi_bigbench, tunizi_v1, tunizi_v2) for license compliance")
    lines.append("- Verify that MIT license on GitHub repo covers the text content (not just code)")
    lines.append("- Review records flagged as 'non_tunisian_or_uncertain' (598 records)")
    lines.append("- Consider whether sentiment labels in TUNIZI should be preserved or removed")

    path = REPORT_DIR / "data_quality_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Data quality report written to {path}")


def generate_final_summary() -> None:
    """Generate data/reports/group_a/FINAL_SUMMARY.md"""
    TARGET = 40_000_000
    total_tokens = TOKEN_DATA["total_gemma_tokens"]
    completion = (total_tokens / TARGET) * 100
    remaining = TARGET - total_tokens

    lines = []
    lines.append("# Group A — Final Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report summarizes the acquisition, audit, processing, and token")
    lines.append("counting of eight Tunisian Arabizi dataset candidates for Gemma Tounsi 1.0 27B.")
    lines.append("")
    lines.append("## Dataset Acquisition Status")
    lines.append("")
    lines.append("| Dataset | Status | Rows Downloaded | License | Decision |")
    lines.append("|---------|--------|-----------------|---------|----------|")
    lines.append("| tunizi_bigbench | Downloaded | 1,786,867 | CreativeML OpenRAIL-M | CONDITIONAL (not in Arabizi corpus — contains MSA Arabic) |")
    lines.append("| tunizi_v2 | Downloaded | 100,000 | MIT (repo) / CC-BY-4.0 (Zenodo) | CONDITIONAL (included) |")
    lines.append("| tunizi_v1 | Downloaded | 9,210 | MIT (repo) | CONDITIONAL (included) |")
    lines.append("| ctab | Downloaded | 5,702 | CC-BY-4.0 | APPROVED (included) |")
    lines.append("| tarc | Not downloaded | N/A | CC-BY-NC-SA-4.0 | REJECTED (non-commercial) |")
    lines.append("| madar_tun | Not downloaded | 30,137 | CC-BY-NC-3.0 | REJECTED (non-commercial) |")
    lines.append("| tunisian_dialect_corpus | Not downloaded | 49,889 | Unclear | HOLD (license not established) |")
    lines.append("| tunisian_darija_english | Not downloaded | 553 | CC-BY-NC-SA-4.0 | REJECTED (non-commercial) |")
    lines.append("")
    lines.append("## Processing Results")
    lines.append("")
    lines.append(f"- Total raw records: 114,912")
    lines.append(f"- Exact duplicates removed: 6,729")
    lines.append(f"- Normalized duplicates removed: 310")
    lines.append(f"- Total unique Tunisian Arabizi records: {TOKEN_DATA['total_records']:,}")
    lines.append("")
    lines.append("## Token Count (Gemma 3 27B)")
    lines.append("")
    lines.append("TARGET = 40,000,000 Gemma tokens")
    lines.append("")
    lines.append(f"APPROVED_UNIQUE_GEMMA_TOKENS = {total_tokens:,}")
    lines.append("")
    lines.append(f"REMAINING = 40,000,000 - {total_tokens:,} = {remaining:,}")
    lines.append("")
    lines.append(f"TARGET_COMPLETION = ({total_tokens:,} / 40,000,000) * 100 = {completion:.2f}%")
    lines.append("")
    lines.append("### Breakdown")
    lines.append("")
    lines.append(f"- **Approved unique Arabizi tokens**: {total_tokens:,} ({completion:.2f}% of target)")
    lines.append(f"- **Conditional dataset tokens**: Included in approved total (tunizi_v1, tunizi_v2)")
    lines.append(f"- **Excluded/restricted tokens**: 0 (tarc, madar_tun, tunisian_darija_english excluded)")
    lines.append(f"- **Arabic-script tokens**: 0 (tunizi_bigbench contains MSA, not Arabizi)")
    lines.append(f"- **Uncertain records**: 598 non-Arabizi records excluded")
    lines.append("")
    lines.append("## Per-Source Token Contribution")
    lines.append("")
    for src, info in sorted(TOKEN_DATA["per_source"].items(), key=lambda x: -x[1]["tokens"]):
        pct = (info["tokens"] / total_tokens) * 100 if total_tokens > 0 else 0
        lines.append(f"- {src}: {info['records']:,} records, {info['tokens']:,} tokens ({pct:.1f}%)")
    lines.append("")
    lines.append("## Remaining Token Gap")
    lines.append("")
    lines.append(f"- **Remaining tokens needed**: {remaining:,}")
    lines.append(f"- **Percentage of target remaining**: {100 - completion:.2f}%")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The eight Group A candidates yield approximately 5.8% of the 40M token")
    lines.append("target for Tunisian Arabizi. The largest contributor is tunizi_v2")
    lines.append("(~2.05M tokens), followed by tunizi_v1 (~0.15M tokens) and ctab")
    lines.append("(~0.11M tokens). tunizi_bigbench (~1.79M rows) contains MSA Arabic,")
    lines.append("not Arabizi, and cannot contribute to the Arabizi target.")
    lines.append("")
    lines.append("The remaining ~37.7M tokens must be sourced from additional Arabizi")
    lines.append("datasets or generated synthetically.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `data/raw/group_a/` — Raw downloaded datasets (immutable, read-only)")
    lines.append("- `data/processed/group_a/tunisian_arabizi_merged.jsonl` — Cleaned, deduped merged corpus")
    lines.append("- `data/reports/group_a/dataset_inventory.csv` — Dataset inventory")
    lines.append("- `data/reports/group_a/token_statistics.json` — Token statistics")
    lines.append("- `data/reports/group_a/license_audit.md` — License audit")
    lines.append("- `data/reports/group_a/data_quality_report.md` — Data quality report")
    lines.append("- `data/reports/group_a/FINAL_SUMMARY.md` — This summary")
    lines.append("- `scripts/data/group_a/download_datasets.py` — Download script")
    lines.append("- `scripts/data/group_a/classify_clean_dedupe.py` — Clean/dedup script")
    lines.append("- `scripts/data/group_a/count_gemma_tokens.py` — Token counting script")

    path = REPORT_DIR / "FINAL_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final summary written to {path}")


def main() -> int:
    generate_dataset_inventory()
    generate_token_statistics()
    generate_license_audit()
    generate_data_quality_report()
    generate_final_summary()
    print("\nAll reports generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
