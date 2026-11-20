#!/usr/bin/env python3
"""
create_custom_75_25_english.py — Gemma Tounsi 1.0 (27B)

Creates a custom downsampled variant from the EXISTING 75% downsampled dataset
as the ONLY source, preserving 100% of arabizi / ar / mixed examples and
retaining approximately 25% of the English examples via deterministic seeded
sampling (seed 42, matching the project's established seed).

Source (only):
    data/experiments/downsampled/75_percent/train.jsonl

Outputs:
    data/experiments/downsampled/custom_75_25_english/train.jsonl
    data/experiments/downsampled/custom_75_25_english/manifest.json
    data/experiments/downsampled/custom_75_25_english/report.md

Original datasets (75%, 50%, 25%, train/validation/test) are never modified.
Token counts use the local Gemma 3 27B tokenizer (google/gemma-3-27b-it) via
apply_chat_template(..., add_special_tokens=False), identical to the original
training pipeline. Only the selected English examples are tokenized; the three
preserved categories reuse the per-category token totals already recorded in
the downsampled manifest (same tokenizer, same method).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml  # noqa: F401  (kept for repo convention parity)
except Exception:  # pragma: no cover - yaml optional at runtime
    yaml = None

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    REPO_ROOT / "data" / "experiments" / "downsampled" / "75_percent" / "train.jsonl"
)
DOWNMANIFEST = REPO_ROOT / "data" / "experiments" / "downsampled" / "manifest.json"

OUT_DIR = (
    REPO_ROOT / "data" / "experiments" / "downsampled" / "custom_75_25_english"
)
OUT_FILE = OUT_DIR / "train.jsonl"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_REPORT = OUT_DIR / "report.md"

ORIGINAL_FILES = {
    "train": REPO_ROOT / "data" / "splits" / "train.jsonl",
    "validation": REPO_ROOT / "data" / "splits" / "validation.jsonl",
    "test": REPO_ROOT / "data" / "splits" / "test.jsonl",
    "soup_trainval": REPO_ROOT / "data" / "splits" / "soup_trainval.jsonl",
    "source_75": SOURCE_FILE,
}

# Tokenizer (cached locally — no network, no HF_TOKEN needed for local files)
TOKENIZER_DIR = (
    REPO_ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--google--gemma-3-27b-it"
    / "snapshots"
    / "005ad3404e59d6023443cb575daa05336842228a"
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 42  # Established project seed (see data/experiments/downsampled/manifest.json)
EN_RATIO = 0.25

# Expected source counts (from the existing dataset comparison report).
# The script verifies these against the actual source file and aborts on mismatch.
EXPECTED_SOURCE_COUNTS = {
    "arabizi": 183423,
    "ar": 161837,
    "mixed": 11905,
    "en": 109626,
}
EXPECTED_SOURCE_TOTAL = 466791

PRESERVE_FULL_CATEGORIES = {"arabizi", "ar", "mixed"}

# English target example count (25% of 109,626 = 27,406.5 -> 27,407).
EN_TARGET = int(EN_RATIO * EXPECTED_SOURCE_COUNTS["en"] + 0.5)

CATEGORY_ORDER = ["arabizi", "ar", "mixed", "en"]

_tokenizer = None

# Populated in main() before the report is written.
pre_checksums: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Checksums / IO
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Tokenizer (lazy, local-only)
# --------------------------------------------------------------------------- #


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR),
            local_files_only=True,
            trust_remote_code=False,
        )
    return _tokenizer


def count_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens for a chat-formatted example (Gemma 3 chat template).

    Total formatted sequence (system + user + assistant turns), matching the
    original pipeline: apply_chat_template(chatml/gemma3, add_special_tokens=False).
    """
    tok = get_tokenizer()
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    if hasattr(ids, "input_ids"):
        return len(ids["input_ids"])
    return len(ids)


def stable_order_key(record: Dict[str, Any], seed: int) -> str:
    """Deterministic sort key based on record id + seed."""
    identity = record.get("id")
    if not isinstance(identity, str) or not identity:
        identity = repr(sorted(record.items()))
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
    return digest


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def largest_remainder(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    """Apportion `total` across keys proportionally to `weights`."""
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return {k: 0 for k in weights}
    exact = {k: total * v / weight_sum for k, v in weights.items()}
    floored = {k: int(v) for k, v in exact.items()}
    remainder = total - sum(floored.values())
    order = sorted(weights, key=lambda k: (exact[k] - floored[k], k), reverse=True)
    for i in range(remainder):
        floored[order[i % len(order)]] += 1
    return floored


def select_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the custom variant; returns (selected_records, info)."""
    # Attach original index so we can restore source file order.
    indexed = list(enumerate(records))

    by_lang: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, r in indexed:
        by_lang[r.get("language", "unknown")].append((idx, r))

    source_counts = {lang: len(recs) for lang, recs in by_lang.items()}

    # ---- Verify source counts against expected --------------------------- #
    discrepancies: List[str] = []
    for lang, expected in EXPECTED_SOURCE_COUNTS.items():
        actual = source_counts.get(lang, 0)
        if actual != expected:
            discrepancies.append(f"  {lang}: expected {expected:,}, actual {actual:,}")
    actual_total = sum(source_counts.values())
    if actual_total != EXPECTED_SOURCE_TOTAL:
        discrepancies.append(
            f"  TOTAL: expected {EXPECTED_SOURCE_TOTAL:,}, actual {actual_total:,}"
        )
    if discrepancies:
        msg = [
            "SOURCE COUNT MISMATCH — aborting before creating the new dataset.",
            "The actual source file does not match the expected counts:",
        ]
        msg.extend(discrepancies)
        msg.append(
            "Do not proceed until the source is confirmed. "
            "No output files were created."
        )
        raise RuntimeError("\n".join(msg))

    # ---- Select: preserve full categories -------------------------------- #
    selected: List[Tuple[int, Dict[str, Any]]] = []
    for lang in PRESERVE_FULL_CATEGORIES:
        selected.extend(by_lang.get(lang, []))

    # ---- Select English: stratified deterministic sample (~25%) ---------- #
    en_recs = by_lang.get("en", [])
    en_by_category: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, r in en_recs:
        en_by_category[r.get("category", "unknown")].append((idx, r))

    en_weights = {cat: len(recs) for cat, recs in en_by_category.items()}
    en_allocation = largest_remainder(EN_TARGET, en_weights)

    en_selected: List[Tuple[int, Dict[str, Any]]] = []
    for cat in sorted(en_by_category):
        cat_recs = list(en_by_category[cat])
        # Deterministic shuffle (seeded), stable across runs.
        cat_recs.sort(key=lambda item: stable_order_key(item[1], SEED))
        take = en_allocation.get(cat, 0)
        en_selected.extend(cat_recs[:take])

    # Restore source file order within English selection.
    en_selected.sort(key=lambda item: item[0])
    selected.extend(en_selected)

    # Restore overall source file order.
    selected.sort(key=lambda item: item[0])
    selected_records = [r for _, r in selected]

    info = {
        "seed": SEED,
        "source_counts": source_counts,
        "source_total": actual_total,
        "en_target": EN_TARGET,
        "en_selected": len(en_selected),
        "en_allocation": en_allocation,
        "selected_by_lang": {
            lang: sum(1 for r in selected_records if r.get("language") == lang)
            for lang in CATEGORY_ORDER
        },
    }
    return selected_records, info


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_lang = Counter(r.get("language", "unknown") for r in records)
    tokens_by_lang: Dict[str, int] = defaultdict(int)
    for r in records:
        tokens_by_lang[r.get("language", "unknown")] += int(r.get("_tokens", 0))

    by_cat = Counter(r.get("category", "unknown") for r in records)
    by_src = Counter(r.get("source", "unknown") for r in records)
    total_tok = sum(tokens_by_lang.values())
    n = len(records)

    return {
        "total_examples": n,
        "total_tokens": total_tok,
        "by_language": dict(by_lang),
        "tokens_by_language": dict(tokens_by_lang),
        "by_category": dict(by_cat),
        "by_source": dict(by_src),
        "pct_examples_by_language": {k: v / n * 100 for k, v in by_lang.items()},
        "pct_tokens_by_language": {k: v / total_tok * 100 for k, v in tokens_by_lang.items()},
        "pct_examples_by_category": {k: v / n * 100 for k, v in by_cat.items()},
        "pct_tokens_by_category": {
            k: v / total_tok * 100 for k, v in
            {c: sum(int(r.get("_tokens", 0)) for r in records if r.get("category") == c)
             for c in by_cat}.items()
        },
        "avg_tokens_per_example": total_tok / n if n else 0,
    }


# --------------------------------------------------------------------------- #
# Manifest (downsampled) — reused for preserved-category token totals
# --------------------------------------------------------------------------- #


def load_downsampled_manifest() -> Dict[str, Any]:
    with DOWNMANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def verify_integrity(
    source_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    info: Dict[str, Any],
    new_stats: Dict[str, Any],
) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    source_by_id: Dict[str, Dict[str, Any]] = {}
    for r in source_records:
        rid = r.get("id")
        if rid:
            source_by_id[rid] = r

    selected_by_id: Dict[str, Dict[str, Any]] = {}
    dup_selected: Set[str] = set()
    for r in selected_records:
        rid = r.get("id")
        if rid:
            if rid in selected_by_id:
                dup_selected.add(rid)
            else:
                selected_by_id[rid] = r

    # 1. All selected ids exist in source.
    missing_in_source = sorted(set(selected_by_id) - set(source_by_id))
    checks["every_output_example_exists_in_source"] = not missing_in_source
    details["examples_missing_in_source"] = missing_in_source[:10]

    # 2. No duplicates introduced.
    checks["no_duplicates_in_output"] = not dup_selected
    details["duplicate_ids_in_output"] = sorted(dup_selected)[:10]

    # 3. Preserved categories present in full.
    source_by_lang = Counter(r.get("language", "unknown") for r in source_records)
    sel_by_lang = Counter(r.get("language", "unknown") for r in selected_records)
    full_preserved = True
    preservation: Dict[str, Dict[str, Any]] = {}
    for lang in PRESERVE_FULL_CATEGORIES:
        s = source_by_lang.get(lang, 0)
        k = sel_by_lang.get(lang, 0)
        preservation[lang] = {"source": s, "kept": k, "complete": s == k}
        if s != k:
            full_preserved = False
    checks["preserved_categories_complete"] = full_preserved
    details["preservation_detail"] = preservation

    # 4. English count approximately 25%.
    en_src = source_by_lang.get("en", 0)
    en_sel = sel_by_lang.get("en", 0)
    en_ratio = en_sel / en_src if en_src else 0
    en_ok = abs(en_ratio - EN_RATIO) <= 0.005
    checks["english_approximately_25_percent"] = en_ok
    details["english_ratio"] = en_ratio
    details["english_source"] = en_src
    details["english_selected"] = en_sel

    # 5. No validation / test leakage.
    val_ids: Set[str] = set()
    test_ids: Set[str] = set()
    for label, path in [("validation", ORIGINAL_FILES["validation"]), ("test", ORIGINAL_FILES["test"])]:
        if path.exists():
            for r in load_records(path):
                rid = r.get("id")
                if rid:
                    (val_ids if label == "validation" else test_ids).add(rid)
    leaked_val = sorted(set(selected_by_id) & val_ids)
    leaked_test = sorted(set(selected_by_id) & test_ids)
    checks["no_validation_test_leakage"] = not leaked_val and not leaked_test
    details["leaked_validation_ids"] = leaked_val[:10]
    details["leaked_test_ids"] = leaked_test[:10]

    # 6. Source datasets unchanged (checksums vs manifest-recorded values).
    manifest = load_downsampled_manifest()
    orig = manifest.get("original_source", {})
    unchanged: Dict[str, bool] = {}
    all_unchanged = True
    for label, path in ORIGINAL_FILES.items():
        if not path.exists():
            unchanged[label] = False
            all_unchanged = False
            continue
        current = sha256_file(path)
        recorded = None
        key = label
        if key in orig:
            recorded = orig[key].get("sha256")
        elif key == "source_75":
            recorded = manifest.get("variants", {}).get("75_percent", {}).get("sha256")
        ok = recorded is None or current == recorded
        unchanged[label] = ok
        if not ok:
            all_unchanged = False
    checks["all_source_datasets_unchanged"] = all_unchanged
    details["checksums_verified"] = unchanged

    passed = all(checks.values())
    return {"checks": checks, "details": details, "passed": passed}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    # ---- 0. Checksums of all source/original files before processing ----- #
    print("[Phase 0] Recording original file checksums (before any processing)...")
    for label, path in ORIGINAL_FILES.items():
        pre_checksums[label] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        print(f"  {label}: {pre_checksums[label]['sha256']}  ({path.stat().st_size:,} bytes)")

    # ---- Phase 1: Load source ------------------------------------------- #
    print(f"\n[Phase 1] Loading source: {SOURCE_FILE.relative_to(REPO_ROOT)}")
    source_records = load_records(SOURCE_FILE)
    print(f"  Loaded {len(source_records)} records")

    # ---- Phase 2: Select ------------------------------------------------- #
    print("\n[Phase 2] Selecting records (seed=42, ~25% of English)...")
    try:
        selected_records, info = select_records(source_records)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1
    print(f"  Selected {len(selected_records)} examples")
    print(f"  Selected by language: {info['selected_by_lang']}")
    print(f"  English selection: {info['en_selected']} / {info['source_counts']['en']} "
          f"({info['en_selected'] / info['source_counts']['en'] * 100:.4f}%)")

    # ---- Phase 3: Tokenize selected English only -------------------------- #
    print("\n[Phase 3] Tokenizing selected examples (Gemma 3 27B, chat template)...")
    # Preserved categories reuse token totals already recorded in the manifest
    # (identical tokenizer + method — no need to re-tokenize them).
    manifest = load_downsampled_manifest()
    v75 = manifest["variants"]["75_percent"]["stats"]
    src_tokens_by_lang = v75["tokens_by_language"]  # source == 75% variant

    tok = get_tokenizer()
    en_tokens_total = 0
    en_token_count = 0
    for r in selected_records:
        if r.get("language") == "en":
            ids = tok.apply_chat_template(
                r.get("messages", []),
                tokenize=True,
                add_generation_prompt=False,
                add_special_tokens=False,
            )
            n = len(ids["input_ids"]) if hasattr(ids, "input_ids") else len(ids)
            r["_tokens"] = n
            en_tokens_total += n
            en_token_count += 1

    # Assemble token totals for all selected records.
    tokens_by_lang_selected: Dict[str, int] = {}
    for lang in PRESERVE_FULL_CATEGORIES:
        tokens_by_lang_selected[lang] = int(src_tokens_by_lang.get(lang, 0))
    tokens_by_lang_selected["en"] = en_tokens_total

    total_tokens_new = sum(tokens_by_lang_selected.values())
    for r in selected_records:
        r.pop("_tokens", None)

    print(f"  Tokenized {en_token_count} English examples = {en_tokens_total:,} tokens")
    print(f"  New dataset total tokens: {total_tokens_new:,}")

    # ---- Phase 4: Stats -------------------------------------------------- #
    new_stats = {
        "total_examples": len(selected_records),
        "total_tokens": total_tokens_new,
        "by_language": info["selected_by_lang"],
        "tokens_by_language": tokens_by_lang_selected,
        "by_category": Counter(r.get("category", "unknown") for r in selected_records),
        "by_source": Counter(r.get("source", "unknown") for r in selected_records),
    }
    n = new_stats["total_examples"]
    new_stats["pct_examples_by_language"] = {
        k: v / n * 100 for k, v in new_stats["by_language"].items()
    }
    new_stats["pct_tokens_by_language"] = {
        k: v / total_tokens_new * 100 for k, v in tokens_by_lang_selected.items()
    }
    new_stats["avg_tokens_per_example"] = total_tokens_new / n if n else 0

    # Source stats (75% variant) — reused from manifest, no re-tokenization.
    src_stats = {
        "total_examples": v75["total_examples"],
        "total_tokens": v75["total_tokens"],
        "by_language": v75["by_language"],
        "tokens_by_language": v75["tokens_by_language"],
        "disk_size_bytes": pre_checksums["source_75"]["size_bytes"],
    }

    # ---- Phase 5: Write outputs ------------------------------------------ #
    print("\n[Phase 4] Writing outputs...")
    write_jsonl(selected_records, OUT_FILE)
    out_size = OUT_FILE.stat().st_size
    out_sha = sha256_file(OUT_FILE)
    print(f"  Wrote {OUT_FILE.relative_to(REPO_ROOT)} ({out_size:,} bytes, sha256={out_sha})")

    # ---- Phase 6: Integrity ---------------------------------------------- #
    print("\n[Phase 5] Running integrity verification...")
    integrity = verify_integrity(source_records, selected_records, info, new_stats)
    for name, ok in integrity["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name.replace('_', ' ').title()}")
    for key, val in integrity["details"].items():
        if isinstance(val, list) and val:
            print(f"  DETAIL {key}: {val}")
    if not integrity["passed"]:
        print("  INTEGRITY CHECKS FAILED — see details above.")
        return 1
    print("  All integrity checks passed.")

    # Confirm source files unchanged after processing.
    print("\n[Phase 6] Confirming source/original datasets unchanged after processing...")
    still_unchanged = True
    for label, path in ORIGINAL_FILES.items():
        current = sha256_file(path)
        ok = current == pre_checksums[label]["sha256"]
        print(f"  {label}: {'UNCHANGED' if ok else 'MODIFIED!'} ({current})")
        if not ok:
            still_unchanged = False
    if not still_unchanged:
        print("ERROR: Source datasets were modified!")
        return 1

    # ---- Phase 7: Manifest ---------------------------------------------- #
    print("\n[Phase 7] Writing manifest...")
    en_ratio_pct = (new_stats["by_language"].get("en", 0) / info["source_counts"]["en"]) * 100
    token_reduction_pct = (1 - total_tokens_new / src_stats["total_tokens"]) * 100
    example_reduction_pct = (1 - n / src_stats["total_examples"]) * 100

    out_manifest = {
        "project": "Gemma Tounsi 1.0 27B",
        "base_model": "google/gemma-3-27b-it",
        "framework": "Soup (QLoRA SFT)",
        "variant": "custom_75_25_english",
        "seed": SEED,
        "description": (
            "Custom variant derived strictly from data/experiments/downsampled/"
            "75_percent/train.jsonl. Preserves 100% of arabizi / ar / mixed "
            "examples and retains ~25% of English examples (deterministic, "
            f"seed {SEED}, stratified by task category)."
        ),
        "tokenizer": {
            "name": "google/gemma-3-27b-it",
            "local_path": str(TOKENIZER_DIR.relative_to(REPO_ROOT)),
            "method": "apply_chat_template(chatml/gemma3, add_special_tokens=False)",
            "note": "Total formatted sequence tokens (user + assistant). "
                    "Preserved-category token totals reused from downsampled "
                    "manifest (identical tokenizer/method); only selected "
                    "English examples were tokenized.",
        },
        "source": {
            "file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
            "size_bytes": pre_checksums["source_75"]["size_bytes"],
            "sha256": pre_checksums["source_75"]["sha256"],
        },
        "original_files": {
            label: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": pre_checksums[label]["sha256"],
            }
            for label, path in ORIGINAL_FILES.items()
        },
        "selection": {
            "en_target_examples": EN_TARGET,
            "en_selected_examples": new_stats["by_language"].get("en", 0),
            "en_ratio_percent": en_ratio_pct,
            "en_allocation_by_category": info["en_allocation"],
            "source_counts": info["source_counts"],
        },
        "output": {
            "file": str(OUT_FILE.relative_to(REPO_ROOT)),
            "size_bytes": out_size,
            "sha256": out_sha,
        },
        "stats": {
            "total_examples": new_stats["total_examples"],
            "total_tokens": new_stats["total_tokens"],
            "disk_size_bytes": out_size,
            "by_language": new_stats["by_language"],
            "tokens_by_language": new_stats["tokens_by_language"],
            "by_category": dict(new_stats["by_category"]),
            "by_source": dict(new_stats["by_source"]),
            "pct_examples_by_language": new_stats["pct_examples_by_language"],
            "pct_tokens_by_language": new_stats["pct_tokens_by_language"],
            "avg_tokens_per_example": new_stats["avg_tokens_per_example"],
        },
        "source_stats": src_stats,
        "reduction_vs_source": {
            "examples": src_stats["total_examples"] - n,
            "examples_percent": example_reduction_pct,
            "tokens": src_stats["total_tokens"] - total_tokens_new,
            "tokens_percent": token_reduction_pct,
            "disk_size_bytes": src_stats["disk_size_bytes"] - out_size,
        },
        "integrity": dict(integrity["checks"]),
        "integrity_detail": integrity["details"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest written: {OUT_MANIFEST.relative_to(REPO_ROOT)}")

    # ---- Phase 8: Report ------------------------------------------------- #
    print("\n[Phase 8] Writing report...")
    write_report(
        source_records, src_stats, selected_records, new_stats,
        info, integrity, out_manifest, token_reduction_pct, example_reduction_pct,
    )
    print(f"  Report written: {OUT_REPORT.relative_to(REPO_ROOT)}")
    print(f"  report.md SHA-256: {sha256_file(OUT_REPORT)}")

    print("\nDone. Custom variant created and validated.")
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(
    source_records: List[Dict[str, Any]],
    src_stats: Dict[str, Any],
    selected_records: List[Dict[str, Any]],
    new_stats: Dict[str, Any],
    info: Dict[str, Any],
    integrity: Dict[str, Any],
    manifest: Dict[str, Any],
    token_reduction_pct: float,
    example_reduction_pct: float,
) -> None:
    src_by_lang = Counter(r.get("language", "unknown") for r in source_records)
    src_cat = Counter(r.get("category", "unknown") for r in source_records)
    src_tokens_by_lang = manifest["source_stats"]["tokens_by_language"]

    lines: List[str] = []
    lines.append("# Custom Dataset Report — Gemma Tounsi 1.0 27B")
    lines.append("")
    lines.append("## Variant: `custom_75_25_english`")
    lines.append("")
    lines.append(
        "- Source (only): `data/experiments/downsampled/75_percent/train.jsonl`"
    )
    lines.append(f"- Output: `{OUT_FILE.relative_to(REPO_ROOT)}`")
    lines.append(f"- Seed: {SEED} (established project seed)")
    lines.append(
        f"- English retention target: ~{EN_RATIO*100:.0f}% of English example count "
        f"({EN_TARGET:,} of {info['source_counts']['en']:,})"
    )
    lines.append("")
    lines.append("> The 25% English target refers to EXAMPLE COUNT, not token count.")
    lines.append("")

    # 1. Source dataset statistics
    lines.append("## 1. Source dataset statistics (75% variant)")
    lines.append("")
    lines.append(f"- Total examples: {src_stats['total_examples']:,}")
    lines.append(f"- Total tokens: {src_stats['total_tokens']:,}")
    lines.append(f"- Disk size: {src_stats['disk_size_bytes']:,} bytes "
                 f"({src_stats['disk_size_bytes'] / (1024 * 1024):.1f} MB)")
    lines.append(f"- SHA-256: {manifest['source']['sha256']}")
    lines.append("")
    lines.append("| Category | Examples | Tokens | % Examples | % Tokens |")
    lines.append("|----------|----------|--------|------------|----------|")
    src_total_tok = src_stats["total_tokens"]
    src_total_ex = src_stats["total_examples"]
    for lang in CATEGORY_ORDER:
        if lang in src_by_lang:
            ex = src_by_lang[lang]
            tk = src_tokens_by_lang.get(lang, 0)
            lines.append(
                f"| {lang} | {ex:,} | {tk:,} | {ex/src_total_ex*100:.2f}% | "
                f"{tk/src_total_tok*100:.2f}% |"
            )
    lines.append("")

    # 2. New dataset statistics
    lines.append("## 2. New dataset statistics")
    lines.append("")
    lines.append(f"- Total examples: {new_stats['total_examples']:,}")
    lines.append(f"- Total tokens: {new_stats['total_tokens']:,}")
    lines.append(f"- Disk size: {manifest['output']['size_bytes']:,} bytes "
                 f"({manifest['output']['size_bytes'] / (1024 * 1024):.1f} MB)")
    lines.append(f"- SHA-256: {manifest['output']['sha256']}")
    lines.append(f"- Average tokens per example: {new_stats['avg_tokens_per_example']:.2f}")
    lines.append("")

    # 3. Exact examples retained per category
    lines.append("## 3. Exact examples retained per category")
    lines.append("")
    lines.append("| Category | Source examples | Target examples | Retained | % Retained |")
    lines.append("|----------|-----------------|-----------------|----------|------------|")
    targets = {
        "arabizi": info["source_counts"]["arabizi"],
        "ar": info["source_counts"]["ar"],
        "mixed": info["source_counts"]["mixed"],
        "en": EN_TARGET,
    }
    for lang in CATEGORY_ORDER:
        src_c = src_by_lang.get(lang, 0)
        tgt = targets.get(lang, 0)
        kept = new_stats["by_language"].get(lang, 0)
        lines.append(
            f"| {lang} | {src_c:,} | {tgt:,} | {kept:,} | "
            f"{kept/src_c*100:.2f}% |" if src_c else f"| {lang} | {src_c:,} | {tgt:,} | {kept:,} | n/a |"
        )
    lines.append(
        f"| TOTAL | {src_total_ex:,} | {sum(targets.values()):,} | {new_stats['total_examples']:,} | "
        f"{new_stats['total_examples']/src_total_ex*100:.2f}% |"
    )
    lines.append("")

    # 4. Exact tokens retained per category
    lines.append("## 4. Exact tokens retained per category")
    lines.append("")
    lines.append(
        "> Tokens computed with the Gemma 3 27B tokenizer via apply_chat_template "
        "(add_special_tokens=False). Preserved-category token totals are reused "
        "from the downsampled manifest (identical tokenizer/method); only the "
        "selected English examples were tokenized."
    )
    lines.append("")
    lines.append("| Category | Source tokens | Retained tokens | % Retained |")
    lines.append("|----------|---------------|-----------------|------------|")
    for lang in CATEGORY_ORDER:
        src_c = src_by_lang.get(lang, 0)
        src_tk = src_tokens_by_lang.get(lang, 0)
        kept_tk = new_stats["tokens_by_language"].get(lang, 0)
        lines.append(
            f"| {lang} | {src_tk:,} | {kept_tk:,} | "
            f"{kept_tk/src_tk*100:.2f}% |" if src_tk else f"| {lang} | {src_tk:,} | {kept_tk:,} | n/a |"
        )
    lines.append(
        f"| TOTAL | {src_total_tok:,} | {new_stats['total_tokens']:,} | "
        f"{new_stats['total_tokens']/src_total_tok*100:.2f}% |"
    )
    lines.append("")

    # 5. New category distribution by examples
    lines.append("## 5. New category distribution by examples")
    lines.append("")
    lines.append("| Category | Examples | % Examples |")
    lines.append("|----------|----------|------------|")
    n = new_stats["total_examples"]
    for lang in CATEGORY_ORDER:
        ex = new_stats["by_language"].get(lang, 0)
        lines.append(f"| {lang} | {ex:,} | {new_stats['pct_examples_by_language'].get(lang, 0):.2f}% |")
    lines.append(f"| TOTAL | {n:,} | 100.00% |")
    lines.append("")

    # 6. New category distribution by tokens
    lines.append("## 6. New category distribution by tokens")
    lines.append("")
    lines.append("| Category | Tokens | % Tokens |")
    lines.append("|----------|--------|----------|")
    for lang in CATEGORY_ORDER:
        tk = new_stats["tokens_by_language"].get(lang, 0)
        lines.append(f"| {lang} | {tk:,} | {new_stats['pct_tokens_by_language'].get(lang, 0):.2f}% |")
    lines.append(f"| TOTAL | {new_stats['total_tokens']:,} | 100.00% |")
    lines.append("")

    # 7. Total token reduction
    lines.append("## 7. Total token reduction (vs 75% variant)")
    lines.append("")
    lines.append(f"- Source examples: {src_stats['total_examples']:,}")
    lines.append(f"- New examples: {n:,}")
    lines.append(f"- Example reduction: {src_stats['total_examples'] - n:,} "
                 f"({example_reduction_pct:.2f}%)")
    lines.append(f"- Source tokens: {src_stats['total_tokens']:,}")
    lines.append(f"- New tokens: {new_stats['total_tokens']:,}")
    lines.append(f"- Token reduction: {src_stats['total_tokens'] - new_stats['total_tokens']:,} "
                 f"({token_reduction_pct:.2f}%)")
    lines.append(
        f"- Disk size reduction: {src_stats['disk_size_bytes'] - manifest['output']['size_bytes']:,} bytes"
    )
    lines.append(
        "> Note: 25% of English examples does NOT equal 25% of English tokens; "
        "English examples carry a higher-than-average token load, so English "
        "token retention differs from its example retention."
    )
    lines.append("")

    # 8. Integrity verification results
    lines.append("## 8. Integrity verification results")
    lines.append("")
    checks = integrity["checks"]
    for name in [
        "every_output_example_exists_in_source",
        "no_duplicates_in_output",
        "preserved_categories_complete",
        "english_approximately_25_percent",
        "no_validation_test_leakage",
        "all_source_datasets_unchanged",
    ]:
        label = name.replace("_", " ").title()
        ok = checks.get(name, False)
        lines.append(f"- {label}: {'PASS' if ok else 'FAIL'}")
    lines.append(f"- English ratio: {integrity['details'].get('english_ratio', 0):.4f} "
                 f"({integrity['details'].get('english_selected', 0)}/{integrity['details'].get('english_source', 0)})")
    lines.append(
        f"- Preserved categories: {integrity['details']['preservation_detail']}"
    )
    lines.append("")

    # 9. SHA-256 checksums
    lines.append("## 9. SHA-256 checksums")
    lines.append("")
    lines.append("### Output files")
    lines.append(f"- `train.jsonl`: {manifest['output']['sha256']}  "
                 f"({manifest['output']['size_bytes']:,} bytes)")
    lines.append(f"- `manifest.json`: {sha256_file(OUT_MANIFEST)}  "
                 f"({OUT_MANIFEST.stat().st_size:,} bytes)")
    lines.append(
        "- `report.md`: SHA-256 printed at the end of this run (self-checksum "
        "omitted to keep the report self-consistent)."
    )
    lines.append("")
    lines.append("### Source / original files (verified unchanged)")
    for label, path in ORIGINAL_FILES.items():
        lines.append(
            f"- `{path.relative_to(REPO_ROOT)}`: "
            f"{pre_checksums_for_report(label)}  ({pre_checksums_for_report_size(label):,} bytes)"
        )
    lines.append("")

    # 10. Limitations / deviations
    lines.append("## 10. Limitations and deviations")
    lines.append("")
    lines.append("- The dataset is a strict subset of the 75% variant; no examples were "
                 "generated, translated, augmented, or duplicated.")
    lines.append("- No content or metadata of any selected example was modified.")
    lines.append("- The 25% English target is approximate at the example-count level "
                 f"(actual: {integrity['details'].get('english_ratio', 0)*100:.2f}% of English examples).")
    lines.append(
        "- English tokens retained differ from 25% of English tokens because English "
        "examples are longer on average; this is expected and documented."
    )
    lines.append(
        "- The `language` field is coarse: `ar` combines Arabic-script Tunisian Derja "
        "and MSA formal; `en` maps to capability retention but may include general "
        "English content. No finer-grained breakdown was invented."
    )
    lines.append(
        "- English sampling is stratified by task `category` to preserve topic/task "
        "diversity, but is otherwise a deterministic random sample (seed 42)."
    )
    lines.append("- Token counts for preserved categories (arabizi, ar, mixed) are reused "
                 "from the existing downsampled manifest rather than re-tokenized, "
                 "since the tokenizer and method are identical.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Path |")
    lines.append("|------|------|")
    lines.append(f"| train.jsonl | `{OUT_FILE.relative_to(REPO_ROOT)}` |")
    lines.append(f"| manifest.json | `{OUT_MANIFEST.relative_to(REPO_ROOT)}` |")
    lines.append(f"| report.md | `{OUT_REPORT.relative_to(REPO_ROOT)}` |")
    lines.append("")
    lines.append("## Using this dataset with Soup")
    lines.append("")
    lines.append("Do NOT modify the production training configuration (`soup.yaml`).")
    lines.append("Instead, create a variant config that points `data.train` at the new file:")
    lines.append("")
    lines.append("```yaml")
    lines.append("# soup.custom_75_25_english.yaml")
    lines.append("base: google/gemma-3-27b-it")
    lines.append("task: sft")
    lines.append("backend: transformers")
    lines.append("modality: text")
    lines.append("data:")
    lines.append("  train: ./data/experiments/downsampled/custom_75_25_english/train.jsonl")
    lines.append("  format: chatml")
    lines.append("  val_split: 0.05263068143061777")
    lines.append("  max_length: 8192")
    lines.append("  train_on_responses_only: true")
    lines.append("  chat_template: gemma3")
    lines.append("training:")
    lines.append("  epochs: 2")
    lines.append("  lr: 1.0e-5")
    lines.append("  batch_size: 1")
    lines.append("  gradient_accumulation_steps: 8")
    lines.append("  quantization: 4bit")
    lines.append("  lora:")
    lines.append("    r: 16")
    lines.append("    alpha: 32")
    lines.append("    dropout: 0.05")
    lines.append("    target_modules: auto")
    lines.append("  gradient_checkpointing: true")
    lines.append("  train_on_eot: true")
    lines.append("  seed: 42")
    lines.append("  logging_steps: 10")
    lines.append("  save_steps: 100")
    lines.append("  stream_layers: false")
    lines.append("output: ./output/gemma-tounsi-1.0-27b-custom-75-25-english")
    lines.append("```")
    lines.append("")
    lines.append("Run it with:")
    lines.append("")
    lines.append("```bash")
    lines.append("docker compose run --rm soup soup train --config soup.custom_75_25_english.yaml")
    lines.append("```")
    lines.append("")
    lines.append(
        "The variant config only changes `data.train` (and the output directory); "
        "all other settings mirror `soup.yaml`. Validation is carved internally "
        "from the train file via `val_split`, consistent with Soup 0.73.3."
    )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def pre_checksums_for_report(label: str) -> str:
    return pre_checksums[label]["sha256"]


def pre_checksums_for_report_size(label: str) -> int:
    return pre_checksums[label]["size_bytes"]


if __name__ == "__main__":
    sys.exit(main())
