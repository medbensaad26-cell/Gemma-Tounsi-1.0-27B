"""Count Gemma 3 tokens for Group A Tunisian Arabizi merged dataset.

Uses google/gemma-3-27b-it tokenizer via transformers.
Counts raw text tokens (no chat template, no special tokens).

Usage:
    python scripts/data/group_a/count_gemma_tokens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data" / "processed" / "group_a" / "tunisian_arabizi_merged.jsonl"
OUTPUT_PATH = ROOT / "data" / "reports" / "group_a" / "token_count.json"

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def count_tokens(text: str, tok) -> int:
    """Count Gemma 3 tokens for a single text string."""
    return len(tok.encode(text, add_special_tokens=False))


def main() -> int:
    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed. Cannot count Gemma tokens.")
        return 1

    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        return 1

    print("Loading Gemma 3 27B tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-it",
            trust_remote_code=False,
        )
        print(f"  Tokenizer loaded: {tok.__class__.__name__}")
        print(f"  Vocab size: {tok.vocab_size}")
    except Exception as exc:
        print(f"ERROR: Could not load Gemma 3 tokenizer: {exc}")
        print("  Make sure you have HF authentication configured if the model is gated.")
        return 1

    # Read all records
    print(f"Reading records from {INPUT_PATH}...")
    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_records = len(records)
    print(f"Total records: {total_records:,}")

    # Count tokens per record
    print("Counting Gemma 3 tokens...")
    token_counts = []
    total_tokens = 0
    errors = 0

    for i, record in enumerate(records):
        text = record.get("text", "")
        try:
            n = count_tokens(text, tok)
            token_counts.append(n)
            total_tokens += n
        except Exception:
            errors += 1
            token_counts.append(0)

        if (i + 1) % 20000 == 0:
            print(f"  Processed {i + 1:,}/{total_records:,} records...")

    # Statistics
    token_counts_sorted = sorted(token_counts)
    n = len(token_counts_sorted)
    avg_tokens = total_tokens / n if n > 0 else 0
    median_tokens = token_counts_sorted[n // 2] if n > 0 else 0
    min_tokens = min(token_counts_sorted) if n > 0 else 0
    max_tokens = max(token_counts_sorted) if n > 0 else 0

    # Per-source stats
    source_stats = {}
    for record, tc in zip(records, token_counts):
        src = record.get("source_dataset", "unknown")
        if src not in source_stats:
            source_stats[src] = {"records": 0, "tokens": 0}
        source_stats[src]["records"] += 1
        source_stats[src]["tokens"] += tc

    # Target comparison
    TARGET = 40_000_000
    completion_pct = (total_tokens / TARGET) * 100
    remaining = TARGET - total_tokens

    result = {
        "tokenizer_model": "google/gemma-3-27b-it",
        "metric": "raw_text_tokens",
        "total_records": total_records,
        "total_gemma_tokens": total_tokens,
        "average_tokens_per_record": round(avg_tokens, 2),
        "median_tokens_per_record": median_tokens,
        "min_tokens_per_record": min_tokens,
        "max_tokens_per_record": max_tokens,
        "errors": errors,
        "target_gemma_tokens": TARGET,
        "target_completion_percent": round(completion_pct, 2),
        "remaining_tokens": remaining,
        "per_source": source_stats,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("GEMMA 3 TOKEN COUNT RESULTS")
    print(f"{'=' * 60}")
    print(f"Total records: {total_records:,}")
    print(f"Total Gemma 3 tokens: {total_tokens:,}")
    print(f"Average tokens/record: {avg_tokens:.2f}")
    print(f"Median tokens/record: {median_tokens}")
    print(f"Min tokens/record: {min_tokens}")
    print(f"Max tokens/record: {max_tokens}")
    print(f"Errors: {errors}")
    print(f"\nTarget (40M): {TARGET:,}")
    print(f"Completion: {completion_pct:.2f}%")
    print(f"Remaining: {remaining:,}")
    print(f"\nPer source:")
    for src, stats in sorted(source_stats.items()):
        print(f"  {src}: {stats['records']:,} records, {stats['tokens']:,} tokens")
    print(f"\nResults written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
