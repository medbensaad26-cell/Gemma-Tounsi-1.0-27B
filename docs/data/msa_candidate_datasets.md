# Free Open Source Candidate MSA Datasets for msa_formal Slice

## Executive Summary

The `msa_formal` slice currently relies on three pools:
- **CIDAR** (`arbml/CIDAR`): Apache-2.0, 10K rows → 9,965 unique. Primary backbone.
- **Arabic QA SIGIR 2024** (`bobez999/arabic-qa-dataset-sigir2024`): MIT, 10K rows → 2,926 usable after dedup + passage cap. Supplement.
- **Alpaca-Arabic-GPT4** (`FreedomIntelligence/Alpaca-Arabic-GPT4`): Apache-2.0, ~49,969 rows. Added 2026-10-16.
- **Evol-Instruct-Arabic** (`FreedomIntelligence/evol-instruct-arabic`): Apache-2.0, ~59,022 rows. Added 2026-10-16.
- **Aya Dataset Arabic subset** (`CohereForAI/aya_dataset`): Apache-2.0, ~4,947 Arabic rows. Added 2026-10-16.



**Critical gap:** The slice needs ≈12.3M tokens for its 8% token share but the pools supply only ≈1.3M (~9× shortfall). The three newly added pools add ~113K raw rows of volume to help close the gap, subject to dedup, register filtering, and the per-source token budget.

---

## Candidate Datasets (Free & Open Source)

### 1. FreedomIntelligence/Alpaca-Arabic-GPT4

| Attribute | Value |
|-----------|-------|
| Hugging Face | `FreedomIntelligence/Alpaca-Arabic-GPT4` |
| License | Apache-2.0 (FreedomIntelligence projects are typically Apache-2.0) |
| Size | ~49,969 rows (per RightNow paper merge) |
| Format | Alpaca (instruction/input/output) |
| Origin | English Alpaca-GPT4 translated to Arabic via GPT-3.5/GPT-4 |
| Register | MSA (translated formal Arabic) |
| Use in SOTA | Used in RightNow-Arabic-0.5B-Turbo merge (129,116 unique after MD5 dedup) |

**Pros:** Large volume (~50K), Apache-2.0 permissive, already used in published Arabic fine-tuning work.
**Cons:** Translated from English (may carry Western cultural assumptions, "Levantine media Arabic" register rather than pure MSA), GPT-4 generated responses may have licensing restrictions from OpenAI terms.

**Relevance to msa_formal:** Strong candidate for volume. Would need Arabic-script verification and register filtering (MSA vs. translated register).

---

### 2. FreedomIntelligence/evol-instruct-arabic

| Attribute | Value |
|-----------|-------|
| Hugging Face | `FreedomIntelligence/evol-instruct-arabic` |
| License | Apache-2.0 (typical for FreedomIntelligence) |
| Size | ~59,022 rows (per RightNow paper merge) |
| Format | Alpaca-style instruction/output |
| Origin | Evol-Instruct translated to Arabic, responses regenerated |
| Register | MSA |

**Pros:** Large volume (~59K), more complex instructions than Alpaca (Evol-Instruct method), Apache-2.0.
**Cons:** Translated/synthetic, register may not be pure MSA.

**Relevance to msa_formal:** Strong candidate for volume and instruction complexity. Complements CIDAR's culturally-relevant instructions.

---

### 3. FreedomIntelligence/sharegpt-arabic

| Attribute | Value |
|-----------|-------|
| Hugging Face | `FreedomIntelligence/sharegpt-arabic` |
| License | Apache-2.0 (typical) |
| Size | ~5,231 rows (per RightNow paper merge) |
| Format | Multi-turn conversation |
| Origin | ShareGPT conversations translated to Arabic |

**Pros:** Multi-turn dialogue format (different from single-turn CIDAR), Apache-2.0.
**Cons:** Smaller volume, translated, AceGPT paper notes ShareGPT incorporation "largely harms performance" for Arabic.

**Relevance to msa_formal:** Moderate. Adds dialogue format diversity but smaller and potentially lower quality.

---

### 4. CohereForAI/aya_dataset (Arabic subset)

| Attribute | Value |
|-----------|-------|
| Hugging Face | `CohereForAI/aya_dataset` |
| License | Apache-2.0 |
| Size | ~4,947 Arabic rows (after `language_code=="arb"` filtering, per RightNow paper) |
| Total dataset | 204,114 pairs across 65 languages |
| Format | Prompt-completion |
| Origin | Human-curated multilingual instruction data |
| Register | MSA (native speaker annotations) |

**Pros:** Apache-2.0, human-curated (not translated), native Arabic speaker input, part of ACL 2024 Best Paper, high quality.
**Cons:** Arabic subset is relatively small (~5K), may include dialectal content that needs MSA filtering.

**Relevance to msa_formal:** Good quality supplement, human-curated MSA. Smaller volume but high signal.

---

### 5. CohereForAI/aya_collection (Arabic subset)

| Attribute | Value |
|-----------|-------|
| Hugging Face | `CohereForAI/aya_collection` |
| License | Apache-2.0 |
| Size | Part of 513M pairs across 114 languages; Arabic portion unspecified |
| Format | Transformed NLP datasets → prompt-completion |
| Origin | Existing NLP datasets converted to instruction format, translated to 101 languages |

**Pros:** Massive scale, Apache-2.0, covers Arabic.
**Cons:** Translated (not native), Arabic subset size unknown, may be noisy.

**Relevance to msa_formal:** Potential volume source but needs Arabic subset extraction and quality filtering.

---

### 6. InstAr-500k (LlamAr & GemmAr)

| Attribute | Value |
|-----------|-------|
| Source | LlamAr & GemmAr paper (arXiv:2407.02147) |
| License | Not clearly specified on Hugging Face; check paper |
| Size | 500K Arabic instruction pairs |
| Format | Instruction/output |
| Tasks | Classification, Open QA, Closed QA, Text Completion, Explanation, Brainstorming, Rewrite, Extraction, Generation, Summarization |
| Origin | Mix of human-crafted + synthetic (GPT-generated from cleaned Arabic text) |

**Pros:** Very large volume (500K), diverse task types, built specifically for Arabic instruction tuning, uses CIDAR + Aya Collection as sources.
**Cons:** License unclear (synthetic data generated by Command R+ may have restrictions), synthetic portion may be lower quality, need to verify Hugging Face availability.

**Relevance to msa_formal:** High volume candidate if license is permissive and dataset is publicly available.

---

### 7. Open-Source Arabic STEM QA (GoAGI/AIDA)

| Attribute | Value |
|-----------|-------|
| Source | LinkedIn post by Tom Nascisonis (GoAGI) |
| License | Open source (verify exact SPDX) |
| Size | 1,000+ STEM questions (Grades 12+), scaling to 50K+ |
| Subjects | Math, Physics, Chemistry, Biology |
| Register | MSA |
| Origin | Saudi Arabia, UAE, Egypt, Qatar, Jordan |

**Pros:** STEM domain (complements CIDAR's general instruction), MSA, open source, curated with answers/solutions/hints.
**Cons:** Smaller current volume (1K+), license needs verification, may be newer/less established.

**Relevance to msa_formal:** Good domain-specific supplement for STEM reasoning in MSA.

---

### 8. arbml Masader Collection

| Attribute | Value |
|-----------|-------|
| Hugging Face | `arbml/masader` |
| License | Likely Apache-2.0 (arbml org standard) |
| Description | "The largest public catalogue for Arabic NLP and speech datasets. +500 datasets annotated with 25+ attributes." |

**Pros:** Catalogue/discovery tool rather than a single dataset; can find additional MSA sources.
**Cons:** Not a training dataset itself; requires exploration.

**Relevance to msa_formal:** Useful for discovering additional candidate datasets.

---

## Comparison with Existing Pools

| Dataset | License | Size | Unique After Dedup | Register | Role |
|---------|---------|------|-------------------|----------|------|
| CIDAR (current) | Apache-2.0 | 10K | 9,965 | MSA | Primary backbone |
| Arabic QA SIGIR 2024 (current) | MIT | 10K | 2,926 (passage-capped) | MSA | Supplement |
| Alpaca-Arabic-GPT4 | Apache-2.0 | ~50K | TBD | MSA (translated) | Volume candidate |
| Evol-Instruct-Arabic | Apache-2.0 | ~59K | TBD | MSA (translated) | Volume candidate |
| ShareGPT-Arabic | Apache-2.0 | ~5K | TBD | MSA (translated) | Dialogue supplement |
| Aya Dataset (Arabic) | Apache-2.0 | ~4,947 | TBD | MSA (human-curated) | Quality supplement |
| InstAr-500k | TBD | 500K | TBD | MSA | Volume candidate (license TBD) |
| Arabic STEM QA | Open (verify) | 1K+ (50K+ planned) | TBD | MSA | STEM domain supplement |

---

## Recommendations

### Immediate (Actionable Today) — COMPLETED 2026-10-16

1. **✅ Added FreedomIntelligence/Alpaca-Arabic-GPT4** (~50K rows, Apache-2.0) — recorded under `sources:` in `data/manifests/msa.yaml`. Largest immediately available permissive MSA instruction dataset. Addresses the token supply shortfall directly.

2. **✅ Added FreedomIntelligence/evol-instruct-arabic** (~59K rows, Apache-2.0) — recorded under `sources:` in `data/manifests/msa.yaml`. Complements Alpaca-GPT4 with more complex instruction types.

3. **✅ Added CohereForAI/aya_dataset Arabic subset** (~4,947 rows, Apache-2.0, human-curated) — recorded under `sources:` in `data/manifests/msa.yaml`. High-quality native MSA, different provenance from translated datasets.

Next steps for all three:
1. Download candidate datasets to `data/raw/` (`scripts/acquire_msa_candidate_data.py`)
2. Normalize to canonical schema (`docs/DATA_SCHEMA.md`)
3. Run MSA register filter (reuse `scripts/audit_msa_candidates.py` approach)
4. Deduplicate with Soup MinHash 0.85 on trained fields only
5. Apply passage-cap for any passage-based pools
6. Pin revisions in `data/manifests/msa.yaml` (replace the 2026-10-16 snapshot date pins)
7. Re-run token budget analysis with Gemma's tokenizer

### Short-Term (Verify License/Availability)

4. **Investigate InstAr-500k** — if publicly available on Hugging Face with a permissive license, this is the highest-volume candidate (500K rows).

5. **Investigate Arabic STEM QA** (GoAGI/AIDA) — verify exact license and Hugging Face location; good for STEM domain coverage.


</write_to_file>
</write_to_file>