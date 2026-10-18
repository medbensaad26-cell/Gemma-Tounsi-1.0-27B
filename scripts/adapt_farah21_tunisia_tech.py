"""Adapter: Farah21/Tunisia_Tech (train split) -> canonical schema (Arabizi rows only).

Reads the original, read-only download in data/raw/ and writes canonical JSONL
to data/processed/derja_arabizi/. Only rows labelled dialect == "tunisian_arabizi"
enter the arabizi slice (see data/manifests/train.yaml, source
farah21-tunisia-tech-arabizi). The test split is never read.

Nothing is dropped silently: every skipped row is counted by reason and printed.
Output must pass:  python -m src.data.validate <output file>
PII / toxicity filtering is a separate preprocessing step and is NOT done here.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("data/raw/Farah21__Tunisia_Tech/data")
OUT_DIR = Path("data/processed/derja_arabizi")
OUT = OUT_DIR / "farah21_tunisia_tech_arabizi.jsonl"
ID_PREFIX = "farah21-tunisia-tech-arabizi"
SOURCE = "Farah21/Tunisia_Tech@be5ca6c5"
KEEP_DIALECT = "tunisian_arabizi"


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def adapt_row(row: dict):
    """Return (record, None) or (None, skip_reason)."""
    if row.get("dialect") != KEEP_DIALECT:
        return None, "dialect_not_arabizi"
    instruction, extra, output = _text(row.get("instruction")), _text(row.get("input")), _text(row.get("output"))
    if not instruction:
        return None, "empty_instruction"
    if not output:
        return None, "empty_output"
    user = f"{instruction}\n\n{extra}" if extra else instruction
    safe_id = re.sub(r"[^A-Za-z0-9._:-]", "_", str(row.get("id")))
    return {
        "id": f"{ID_PREFIX}-{safe_id}",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ],
        "category": "general_instruction",
        "subcategory": "reddit_qa",
        "source": SOURCE,
        "language": "arabizi",
        "script": "latin",
    }, None


def main() -> int:
    files = sorted(RAW.glob("train-*.parquet"))
    if not files:
        print(f"ERROR: no train parquet under {RAW} (run the acquisition script first)")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped, kept = Counter(), 0
    with open(OUT, "w", encoding="utf-8") as out:
        for f in files:
            for row in pq.read_table(f).to_pylist():
                rec, reason = adapt_row(row)
                if rec is None:
                    skipped[reason] += 1
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept={kept} skipped={dict(skipped)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())