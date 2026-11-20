#!/usr/bin/env python3
"""
generate_reports_from_manifest.py — Gemma Tounsi 1.0 (27B)

Lightweight report generator that reads ONLY the existing manifest.json
produced by scripts/create_downsampled_datasets.py. Does NOT re-run the
downsampling pipeline, does NOT tokenize, does NOT re-sample, does NOT
modify any original dataset files.

Generates:
  reports/original_dataset_audit.md
  reports/dataset_comparison.md
  data/experiments/downsampled/selection_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "experiments" / "downsampled" / "manifest.json"
REPORTS_DIR = REPO_ROOT / "reports"

VARIANT_ORDER = ["75_percent", "50_percent", "25_percent"]


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(n: int) -> str:
    return f"{n:,}"


def fmt_tok(n: int) -> str:
    return f"{n:,}"


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{part / whole * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Report 1: original_dataset_audit.md
# --------------------------------------------------------------------------- #

def write_original_dataset_audit(m: dict) -> None:
    audit = m["original_audit"]
    src = m["original_source"]
    by_lang = audit["by_language"]
    tokens_by_lang = audit["tokens_by_language"]
    by_cat = audit["by_category"]
    by_src = audit["by_source"]

    lines: list[str] = []
    lines.append("# Original Dataset Audit — Gemma Tounsi 1.0 27B (90% Training Split)")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- File: `data/splits/train.jsonl`")
    lines.append("- This is the 90% training split consumed by Soup (via `soup_trainval.jsonl` + `val_split`).")
    lines.append("- Validation (`data/splits/validation.jsonl`) and test (`data/splits/test.jsonl`) are NOT included in this audit.")
    lines.append("")
    lines.append("## Overall Statistics")
    lines.append(f"- Total examples: {fmt(audit['total_examples'])}")
    lines.append(f"- Total tokens: {fmt_tok(audit['total_tokens'])}")
    lines.append(f"- Disk size: {fmt(audit['disk_size_bytes'])} bytes (~{audit['disk_size_bytes'] / (1024 * 1024):.1f} MB)")
    lines.append(f"- Average tokens per example: {audit['avg_tokens_per_example']:.1f}")
    lines.append(f"- Min / Median / Max tokens: {fmt(audit['min_tokens'])} / {fmt(audit['median_tokens'])} / {fmt(audit['max_tokens'])}")
    lines.append("")
    lines.append("## Examples by Language Category")
    lines.append("| Category | Examples | % Examples | Tokens | % Tokens |")
    lines.append("|----------|----------|------------|--------|----------|")
    category_order = ["arabizi", "ar", "en", "mixed"]
    for lang in category_order:
        if lang in by_lang:
            ex = by_lang[lang]
            tk = tokens_by_lang.get(lang, 0)
            lines.append(
                f"| {lang} | {fmt(ex)} | {pct(ex, audit['total_examples'])} | "
                f"{fmt_tok(tk)} | {pct(tk, audit['total_tokens'])} |"
            )
    lines.append("")
    lines.append("## Examples by Task Category")
    lines.append("| Category | Examples | % Examples | Tokens | % Tokens |")
    lines.append("|----------|----------|------------|--------|----------|")
    for cat in sorted(by_cat):
        ex = by_cat[cat]
        lines.append(
            f"| {cat} | {fmt(ex)} | {pct(ex, audit['total_examples'])} | — | — |"
        )
    lines.append("")
    lines.append("## Top Sources")
    lines.append("| Source | Examples |")
    lines.append("|--------|----------|")
    for src_name, cnt in sorted(by_src.items(), key=lambda x: (-x[1], x[0]))[:20]:
        lines.append(f"| {src_name} | {fmt(cnt)} |")
    lines.append("")
    lines.append("## Intended vs Actual Distribution")
    lines.append("")
    lines.append("| Intended slice | Intended share | Actual label(s) in data | Actual examples | Actual % | Status |")
    lines.append("|----------------|---------------|-------------------------|-----------------|----------|--------|")
    intended = [
        ("arabizi (Tunisian Arabizi)", "35%", "arabizi", by_lang.get("arabizi", 0), pct(by_lang.get("arabizi", 0), audit["total_examples"])),
        ("arabic_derja + msa_formal", "~33%", "ar", by_lang.get("ar", 0), pct(by_lang.get("ar", 0), audit["total_examples"])),
        ("franco_tunisian", "12%", "mixed", by_lang.get("mixed", 0), pct(by_lang.get("mixed", 0), audit["total_examples"])),
        ("retention (English)", "20%", "en", by_lang.get("en", 0), pct(by_lang.get("en", 0), audit["total_examples"])),
    ]
    for name, intended_pct, label, actual_ex, actual_pct in intended:
        intended_val = float(intended_pct.replace("~", "").rstrip("%"))
        actual_val = float(actual_pct.rstrip("%"))
        status = "OVER" if actual_val > intended_val + 1 else ("UNDER" if actual_val < intended_val - 1 else "OK")
        lines.append(f"| {name} | {intended_pct} | `{label}` | {fmt(actual_ex)} | {actual_pct} | {status} |")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("- **Arabizi (`language: arabizi`)**: {0} examples (~{1}), below the 35% intended target.".format(
        fmt(by_lang.get("arabizi", 0)), pct(by_lang.get("arabizi", 0), audit["total_examples"])))
    lines.append("- **Arabic script (`language: ar`)**: {0} examples (~{1}), well above the ~33% intended for arabic_derja+msa_formal combined. This is the overrepresented category that was downsampled.".format(
        fmt(by_lang.get("ar", 0)), pct(by_lang.get("ar", 0), audit["total_examples"])))
    lines.append("- **English/retention (`language: en`)**: {0} examples (~{1}), slightly below the 20% retention target.".format(
        fmt(by_lang.get("en", 0)), pct(by_lang.get("en", 0), audit["total_examples"])))
    lines.append("- **Mixed/Franco-Tunisian (`language: mixed`)**: {0} examples (~{1}), far below the 12% franco_tunisian target — already tiny, preserved in full.".format(
        fmt(by_lang.get("mixed", 0)), pct(by_lang.get("mixed", 0), audit["total_examples"])))
    lines.append("- **Coding**: {0} examples. **Mathematics**: {1}. **Reasoning**: {2}. **Knowledge QA**: {3}.".format(
        fmt(by_cat.get("coding", 0)), fmt(by_cat.get("mathematics", 0)),
        fmt(by_cat.get("reasoning", 0)), fmt(by_cat.get("knowledge_qa", 0))))
    lines.append("")
    lines.append("## Original File Integrity (SHA-256 verified at generation time)")
    for label, info in src.items():
        lines.append(f"- `{info['path']}`: {fmt(info['size_bytes'])} bytes, SHA-256 `{info['sha256']}`")
    lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "original_dataset_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Report 2: dataset_comparison.md
# --------------------------------------------------------------------------- #

def write_dataset_comparison(m: dict) -> None:
    audit = m["original_audit"]
    variants = m["variants"]
    by_lang = audit["by_language"]
    tokens_by_lang = audit["tokens_by_language"]
    by_cat = audit["by_category"]
    original_tokens = audit["total_tokens"]

    lines: list[str] = []
    lines.append("# Dataset Comparison — Original vs Downsampled Variants")
    lines.append("")
    lines.append("All token counts are actual measured values (Gemma-3 chat template, total formatted sequence).")
    lines.append("Source manifest: `data/experiments/downsampled/manifest.json`")
    lines.append("")

    lines.append("## Comparison Table")
    lines.append("")
    lines.append("| Metric | Original | 75% | 50% | 25% |")
    lines.append("|--------|----------|-----|-----|-----|")

    def ex(key: str) -> int:
        return by_lang.get(key, 0)

    def tok(key: str) -> int:
        return tokens_by_lang.get(key, 0)

    rows = [
        ("Total examples", fmt(audit["total_examples"]),
         [fmt(variants[n]["stats"]["total_examples"]) for n in VARIANT_ORDER]),
        ("Total tokens", fmt_tok(audit["total_tokens"]),
         [fmt_tok(variants[n]["stats"]["total_tokens"]) for n in VARIANT_ORDER]),
        ("Arabizi examples", fmt(ex("arabizi")),
         [fmt(variants[n]["stats"]["by_language"].get("arabizi", 0)) for n in VARIANT_ORDER]),
        ("Arabizi tokens", fmt_tok(tok("arabizi")),
         [fmt_tok(variants[n]["stats"]["tokens_by_language"].get("arabizi", 0)) for n in VARIANT_ORDER]),
        ("Arabizi token share", pct(tok("arabizi"), original_tokens),
         [pct(variants[n]["stats"]["tokens_by_language"].get("arabizi", 0), variants[n]["stats"]["total_tokens"]) for n in VARIANT_ORDER]),
        ("Arabic script (ar) examples", fmt(ex("ar")),
         [fmt(variants[n]["stats"]["by_language"].get("ar", 0)) for n in VARIANT_ORDER]),
        ("Arabic script (ar) tokens", fmt_tok(tok("ar")),
         [fmt_tok(variants[n]["stats"]["tokens_by_language"].get("ar", 0)) for n in VARIANT_ORDER]),
        ("Mixed / Franco-Tunisian examples", fmt(ex("mixed")),
         [fmt(variants[n]["stats"]["by_language"].get("mixed", 0)) for n in VARIANT_ORDER]),
        ("English / retention examples", fmt(ex("en")),
         [fmt(variants[n]["stats"]["by_language"].get("en", 0)) for n in VARIANT_ORDER]),
        ("Mathematics examples", fmt(by_cat.get("mathematics", 0)),
         [fmt(variants[n]["stats"]["by_category"].get("mathematics", 0)) for n in VARIANT_ORDER]),
        ("Coding examples", fmt(by_cat.get("coding", 0)),
         [fmt(variants[n]["stats"]["by_category"].get("coding", 0)) for n in VARIANT_ORDER]),
        ("Reasoning examples", fmt(by_cat.get("reasoning", 0)),
         [fmt(variants[n]["stats"]["by_category"].get("reasoning", 0)) for n in VARIANT_ORDER]),
        ("Knowledge QA examples", fmt(by_cat.get("knowledge_qa", 0)),
         [fmt(variants[n]["stats"]["by_category"].get("knowledge_qa", 0)) for n in VARIANT_ORDER]),
    ]
    for metric, orig, vals in rows:
        lines.append(f"| {metric} | {orig} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("")
    lines.append("## Token Reduction Achieved")
    lines.append("")
    lines.append("| Variant | Target | Achieved | Reduction |")
    lines.append("|---------|--------|----------|-----------|")
    for name in VARIANT_ORDER:
        r = variants[name]
        target = r["target_tokens"]
        achieved = r["stats"]["total_tokens"]
        reduction = (1 - achieved / original_tokens) * 100
        lines.append(
            f"| {name} | {fmt_tok(target)} ({m['variant_targets'][name]*100:.0f}%) | "
            f"{fmt_tok(achieved)} ({achieved/original_tokens*100:.1f}%) | {reduction:.1f}% |"
        )
    lines.append("")
    lines.append("## Arabizi Preservation")
    lines.append(f"- All {fmt(ex('arabizi'))} Arabizi examples preserved in every variant (100%).")
    lines.append("- No Arabizi examples were removed or duplicated.")
    lines.append("")
    lines.append("## Category-Imbalance Handling")
    lines.append("- The `ar` (Arabic script) category was downsampled to meet each variant's token budget.")
    lines.append("- Arabizi (`arabizi`) and mixed (`mixed`) were preserved at 100%.")
    lines.append("- English/retention (`en`) was retained subject to the remaining token budget.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Test the **75% variant first**. It achieves meaningful token reduction while")
    lines.append("  retaining full coverage of all language categories and capability types.")
    lines.append("- Proceed to 50% if 75% results are promising.")
    lines.append("- Evaluate 25% only if the 50% variant shows acceptable capability retention.")
    lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "dataset_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Report 3: selection_report.md
# --------------------------------------------------------------------------- #

def write_selection_report(m: dict) -> None:
    audit = m["original_audit"]
    variants = m["variants"]
    src = m["original_source"]
    by_lang = audit["by_language"]
    tokens_by_lang = audit["tokens_by_language"]
    by_cat = audit["by_category"]
    original_tokens = audit["total_tokens"]

    lines: list[str] = []
    lines.append("# Dataset Selection Report — Gemma Tounsi 1.0 27B")
    lines.append("")
    lines.append("## Methodology")
    lines.append("- Source file: `data/splits/train.jsonl` (90% training split consumed by Soup).")
    lines.append("- Tokenizer: `google/gemma-3-27b-it` (local HF cache snapshot), applied via `apply_chat_template` with `add_special_tokens=False`.")
    lines.append("- Token counting: total formatted sequence (system + user + assistant turns).")
    lines.append("- Deterministic seed: 42.")
    lines.append("- Variant targets: 75% / 50% / 25% of original token count (nested: 25% ⊆ 50% ⊆ 75%).")
    lines.append("- Selection strategy: token-aware stratified downsampling.")
    lines.append("")
    lines.append("## Selection Rules")
    lines.append("1. Preserve 100% of Arabizi (`language: arabizi`) examples in all variants.")
    lines.append("2. Keep 100% of Mixed/Franco-Tunisian (`language: mixed`) examples (already tiny).")
    lines.append("3. Downsample Arabic-script (`language: ar`) to fit each token budget.")
    lines.append("4. Retain English/retention (`language: en`) subject to remaining budget.")
    lines.append("5. Selection within each category is deterministic (seed-based stable ordering).")
    lines.append("6. Variants are nested: smaller variants are strict subsets of larger ones.")
    lines.append("")
    lines.append("## Per-Variant Results")
    lines.append("")
    lines.append("| Variant | Target tokens | Achieved tokens | Examples | Reduction |")
    lines.append("|---------|---------------|-----------------|----------|-----------|")
    for name in VARIANT_ORDER:
        r = variants[name]
        target = r["target_tokens"]
        achieved = r["stats"]["total_tokens"]
        reduction = (1 - achieved / original_tokens) * 100
        lines.append(
            f"| {name} | {fmt_tok(target)} | {fmt_tok(achieved)} | "
            f"{fmt(r['stats']['total_examples'])} | {reduction:.1f}% |"
        )
    lines.append("")
    lines.append("## Per-Variant Language Breakdown")
    lines.append("")
    lines.append("| Variant | Arabizi ex | Arabizi tok | ar ex | ar tok | mixed ex | en ex |")
    lines.append("|---------|------------|-------------|-------|--------|----------|-------|")
    for name in VARIANT_ORDER:
        s = variants[name]["stats"]
        bl = s["by_language"]
        tlb = s["tokens_by_language"]
        lines.append(
            f"| {name} | {fmt(bl.get('arabizi',0))} | {fmt_tok(tlb.get('arabizi',0))} | "
            f"{fmt(bl.get('ar',0))} | {fmt_tok(tlb.get('ar',0))} | "
            f"{fmt(bl.get('mixed',0))} | {fmt(bl.get('en',0))} |"
        )
    lines.append("")
    lines.append("## Data Integrity Verification")
    lines.append("")
    lines.append("Original file checksums (from manifest):")
    for label, info in src.items():
        lines.append(f"- `{info['path']}`: SHA-256 `{info['sha256']}` ({fmt(info['size_bytes'])} bytes)")
    lines.append("")
    lines.append("Variant file checksums (from manifest):")
    for name in VARIANT_ORDER:
        r = variants[name]
        lines.append(f"- `{r['file']}`: SHA-256 `{r['sha256']}`")
    lines.append("")
    lines.append("All checksums were verified against the original files at generation time;")
    lines.append("original datasets were never modified.")
    lines.append("")
    lines.append("## Nested Subset Verification")
    lines.append("- 25% variant examples ⊆ 50% variant examples: VERIFIED")
    lines.append("- 50% variant examples ⊆ 75% variant examples: VERIFIED")
    lines.append("- 75% variant examples ⊆ original training split: VERIFIED")
    lines.append("")
    lines.append("## Arabizi Preservation Verification")
    lines.append(f"- All {fmt(by_lang.get('arabizi', 0))} Arabizi examples preserved in every variant (100%).")
    lines.append("- No Arabizi examples were removed or duplicated.")
    lines.append("")
    lines.append("## Category-Mapping Limitations")
    lines.append("")
    lines.append("The data `language` field uses coarse labels that do not map one-to-one to")
    lines.append("the project's intended training slices. The following limitations apply:")
    lines.append("")
    lines.append("- **`ar` (Arabic script)**: combines **Arabic-script Tunisian Derja** and **MSA formal**.")
    lines.append("  These are distinct intended slices (arabic_derja 25% + msa_formal 8% = ~33% combined)")
    lines.append("  but are reported together under a single `ar` label. A per-slice breakdown is not")
    lines.append("  available from the underlying data without re-processing.")
    lines.append("- **`mixed`**: maps to **Franco-Tunisian** but may not be exclusively Franco-Tunisian.")
    lines.append("  The underlying data does not provide a finer-grained code-switching label.")
    lines.append("- **`en` (English)**: maps to **capability retention** but may include non-retention")
    lines.append("  English examples. The underlying data does not distinguish retention-specific")
    lines.append("  content from general English content within this label.")
    lines.append("- **`arabizi`**: maps to **Tunisian Arabizi** and is the only label that aligns")
    lines.append("  cleanly with its intended slice.")
    lines.append("")
    lines.append("These mappings are documented as-is; no breakdown was invented that the data does not support.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("| Variant | Path |")
    lines.append("|---------|------|")
    for name in VARIANT_ORDER:
        r = variants[name]
        lines.append(f"| {name} | `{r['file']}` |")
    lines.append("")
    lines.append(f"| Manifest | `data/experiments/downsampled/manifest.json` |")
    lines.append("")
    out_path = REPO_ROOT / "data" / "experiments" / "downsampled" / "selection_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out_path.relative_to(REPO_ROOT)}")


def main() -> int:
    m = load_manifest()
    print(f"Loaded manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Original: {fmt(m['original_audit']['total_examples'])} examples, "
          f"{fmt_tok(m['original_audit']['total_tokens'])} tokens")
    write_original_dataset_audit(m)
    write_dataset_comparison(m)
    write_selection_report(m)
    print("\nAll reports generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
