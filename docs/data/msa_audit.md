# MSA Candidate Data Audit: CIDAR + Arabic QA Dataset – SIGIR 2024

**Author:** Mohamed
**Date:** September 5, 2026
**Status:** Complete (both candidates inspected, validated and deduplicated)
**Scope:** `msa_formal` slice candidate pools only — the English retention pools (MetaMathQA, Code-Feedback, SlimOrca) were NOT touched, re-downloaded, or re-analyzed.

---

## Executive Summary

Both candidate pools were audited for the 8% `msa_formal` slice using the Soup 0.73.3 dataset tooling (`soup data validate`, `stats`, `dedup`, `langdetect`) plus a deterministic Arabic-register classifier (`scripts/audit_msa_candidates.py`).

**Bottom line:**

- **CIDAR is the primary MSA candidate.** It is 97%+ MSA, structurally clean (0 malformed, 35 near-duplicates), culturally relevant, and instruction-diverse. Its ~9965 unique rows comfortably cover the 8,000-example MSA target **on their own**.
- **Arabic QA Dataset – SIGIR 2024 is a usable but structurally weaker supplement.** It is 99.9% MSA and clean (1 malformed row), but **21.3% of its rows are near-duplicates** — 10,000 questions are built on only **1,500 unique passages**, and the pool is heavily knowledge-QA weighted with very short answers (median 33 chars). Its usable unique content is roughly **1,500 passages / ~7,872 deduplicated rows**.
- **The critical MSA-vs-dialect assumption held**: neither dataset is a dialect dataset. Tunisian Derja markers are essentially absent (0 unambiguous hits in both pools after false-positive correction). Latin/code-switched content is a CIDAR-only artifact (217 rows, mostly code answers — expected and acceptable for the technical quota).
- **Recommendation:** source the MSA slice primarily from CIDAR; use Arabic QA (deduplicated, passage-aware) as a knowledge-QA/topical diversity supplement; cap its share so passage repetition does not skew the slice.

---

## Tooling and Method

| Step | Tool | Command |
|---|---|---|
| Format validation | Soup 0.73.3 (pinned Docker image) | `soup data validate <file>` |
| Length/token statistics | Soup 0.73.3 | `soup data stats <file>` |
| Near-duplicate removal | Soup 0.73.3 (MinHash Jaccard) | `soup data dedup --threshold 0.85 -o <out> <file>` |
| Language tagging | Soup 0.73.3 | `soup data langdetect -i <in> -o <out>` |
| Register / task / formality / structure | Local deterministic script | `python scripts/audit_msa_candidates.py` |

Notes on tool behavior (documented honestly):

- `soup data langdetect` is a Latin-script-centric heuristic: it tags Arabic-script rows as `unknown`/`_language: "unknown"`. It therefore **cannot confirm Arabic**; the Arabic-script ratio was measured locally instead (script-shape counting), and the QA pool's own `language: "ar"` field was verified (10,000/10,000).
- The local register classifier is deliberately conservative: **whole-word** Tunisian marker matching. Substring matching produced false positives during the audit (e.g. "ما تشاء" and "تشمل" flagged as Tunisian negation; the proper name "ياسر" / Yasser; MSA participles like "ماشيا"). All were corrected before the final numbers below.

---

## 1. MSA vs. Dialect Content

Deterministic whole-word Tunisian-Derja marker scan over instruction + input + output:

| Dataset | Rows | MSA | Dialect | Mixed | Latin/code-switched | Other |
|---|---:|---:|---:|---:|---:|---:|
| CIDAR | 10,000 | 9,723 (97.2%) | 0 | 1 | 217 (2.2%) | 59 |
| Arabic QA – SIGIR 2024 | 10,000 | 9,993 (99.9%) | 0 | 0 | 7 (0.07%) | 0 |

Findings:

- **Zero unambiguous Tunisian Derja rows in either pool.** CIDAR is MSA by construction (Alpagasus subset machine-translated into formal Arabic + 891 reviewed Arabic-grammar instructions from Al Jazeera's "Ask the teacher"); Arabic QA is formal encyclopedic Arabic.
- The 217 CIDAR "Latin/code-switched" rows are almost entirely **code answers to Arabic instructions** (Python, SQL). This is *technical content in MSA instructions*, not Franco-Tunisian code-switching — no Arabizi (`3`/`7`/`9`) orthography patterns or French markers were detected in them. They are usable, and in fact serve the MSA technical quota.
- CIDAR's dialect-style content, where present at all, is limited to a handful of *poetic/lyrical outputs* (song lyrics), which is a register/genre issue, not a dialect issue.

**Verdict: both pools pass the "not secretly a dialect dataset" test.**

## 2. Formal-Language Quality

| Dataset | Formal markers present | Colloquial penalty > 0 | Median instruction / output length |
|---|---|---:|---|
| CIDAR | Yes (relative pronouns, formal modals, discourse connectives) | 0 / 10,000 | 49 / 247 chars |
| Arabic QA | Yes (encyclopedic register) | 0 / 10,000 | 40 / 33 chars |

- CIDAR's outputs average ~306 chars (p90: 596; max: 10,108) — full explanatory paragraphs in formal Arabic.
- Arabic QA's outputs are **very short** (mean 49, median 33 chars; 409 rows have answers ≤ 2 characters, e.g. `"15"`, `"88٪"`). Short extractive answers are fine for knowledge QA, but they contribute little *language modeling signal per example* for teaching formal-register generation. This is the pool's main quality limitation.

## 3. Task Types

Instruction-only classification (passages excluded to avoid contamination of the counts):

**CIDAR** (10,000):

| Task | Rows | Share |
|---|---:|---:|
| general_instruction | 3,992 | 39.9% |
| other (creative: poems, stories, lyrics, tables) | 3,509 | 35.1% |
| knowledge_qa | 1,142 | 11.4% |
| reasoning | 721 | 7.2% |
| mathematics | 398 | 4.0% |
| coding | 238 | 2.4% |

**Arabic QA – SIGIR 2024** (10,000):

| Task | Rows | Share |
|---|---:|---:|
| knowledge_qa | 6,692 | 66.9% |
| other (untagged question forms) | 2,588 | 25.9% |
| reasoning | 432 | 4.3% |
| mathematics | 191 | 1.9% |
| coding | 50 | 0.5% |
| general_instruction | 47 | 0.5% |

Findings:

- The two pools are **complementary**: CIDAR is instruction/creative-heavy; Arabic QA is knowledge-QA-heavy (66.9%). This matches the MSA slice's designed category profile (`configs/data/msa.yaml`: knowledge_qa 3,000; instruction_following 2,500; mathematics 1,000; reasoning 1,000; coding 500).
- Neither pool alone fills the MSA mathematics (1,000) target: CIDAR ~398 + QA ~191 ≈ **589 genuine math examples**. **A dedicated Arabic-math source (or synthetic/authored math in MSA) is still needed** for the MSA technical quota.
- Arabic QA's "mathematics" questions are mostly **counting questions about facts** ("How many provinces does Estonia have?") — they are knowledge QA, not mathematics. Do not count them toward the math target.

## 4. Duplicates

Soup 0.73.3 MinHash dedup at threshold 0.85 (same setting as the retention reports):

| Dataset | Exact duplicate rows (Soup validate) | Near-duplicates removed (Soup dedup) | Unique rows remaining |
|---|---:|---:|---:|
| CIDAR | 29 reported (33 by strict (instruction, output) pair; consistent) | 35 | **9,965** |
| Arabic QA – SIGIR 2024 | 1 | 2,128 | **7,872** |

**CIDAR is essentially duplicate-free (0.35%).**

**Arabic QA's 21.3% near-duplicate rate is structural, not accidental:** the pool is **passage-based** —

- 10,000 questions are built on only **1,500 unique passages** (mean ≈ 6.7 questions/passage, max 67).
- 99.84% of rows share their passage with at least one other row.
- 45 instructions repeat verbatim (63 extra rows); 3 exact (instruction, output) duplicate pairs (4 extra rows).

Consequence for selection: the near-duplicates flagged by MinHash are mostly **distinct questions over identical passages** (the shared passage dominates the concatenated text that MinHash hashes). They are not true content copies — but training on many questions over the same passage still over-weights that passage. **Selection must therefore be passage-aware, not just row-aware** (see recommendation).

## 5. Malformed Examples

| Check | CIDAR | Arabic QA |
|---|---|---:|
| Soup format validation (alpaca) | 10,000/10,000 valid | 10,000/10,000 valid |
| Non-JSON / parse errors | 0 | 0 |
| Empty instruction or output | 0 | **1** |
| Missing `language` tag | n/a (no field) | 0 (all `ar`) |
| Output ≤ 2 characters | — | 409 (informational; valid extractive answers) |

Both pools are structurally sound. The single empty-output QA row must be dropped during adapter normalization.

## 6. Source Distribution

| Dataset | Source field / provenance |
|---|---|
| CIDAR | Single corpus: ~9,109 Alpagasus-derived samples (ChatGPT-translated to Arabic, human-reviewed by ~12 reviewers) + ~891 Arabic grammar instructions (Al Jazeera "Ask the teacher"). No per-row source field; `index` column only. |
| Arabic QA – SIGIR 2024 | Single value for all 10,000 rows: `source: "ArabicaQA-SIGIR2024"`. Underlying provenance: ArabicaQA (SIGIR 2024), passage-based QA over Wikipedia-style Arabic articles. |

Both pools are therefore **single-source**. Neither provides fine-grained per-row provenance beyond these labels. For CIDAR, the two sub-origins (translated Alpagasus vs. grammar instructions) are not separable from the data alone — treat it as one source in diversity caps.

## 7. Suitability for the 8% MSA Slice

The `msa_formal` slice target is **8,000 training examples** (+1,000 holdout) with category targets knowledge_qa 3,000 / instruction_following 2,500 / mathematics 1,000 / reasoning 1,000 / coding 500 (`configs/data/msa.yaml`).

| Criterion | CIDAR | Arabic QA – SIGIR 2024 |
|---|---|---|
| Register (MSA) | ✅ 97.2% | ✅ 99.9% |
| Duplicates | ✅ 9,965 unique (0.35% dup) | ⚠️ 7,872 unique rows, but only ~1,500 unique passages |
| Formality | ✅ full explanatory outputs | ⚠️ very short answers (median 33 chars) |
| Category coverage | ✅ instruction + creative + QA + reasoning + some math/code | ⚠️ knowledge-QA dominant |
| Coverage of 8,000 target | ✅ alone sufficient (9,965) | ❌ alone insufficient after passage-aware selection |
| Arabic script purity | ✅ (Latin rows are code answers) | ✅ |
| Verified language tag | n/a | ✅ `language: "ar"` 10,000/10,000 |

**Supply check (post-dedup, MSA-only):**

- CIDAR MSA-flagged unique rows ≈ 9,723 − ~4 duplicates ≈ **~9,700**
- Arabic QA usable unique rows ≈ 7,872 (but ~1,500 passages ⇒ passage-aware effective diversity ≈ 1,500 topics)
- Combined unique supply ≈ **~17,500 rows** vs. a 9,000-row requirement (8,000 train + 1,000 holdout) → supply is adequate **only if Arabic QA is counted passage-aware**; on passages alone it would be tight.

### Verdict and recommended split

1. **Primary source: CIDAR** — take the bulk of the MSA slice from its ~9,700 unique MSA rows (instruction, knowledge, reasoning, creative/general categories).
2. **Supplement: Arabic QA (deduplicated, passage-capped)** — use it to top up `knowledge_qa` toward its 3,000 target, with a **cap of ≤ 2 questions per passage** (diversity rule, analogous to the retention strategy's template caps). At 2 questions/passage this yields ~3,000 rows from ~1,500 passages — enough to fill the knowledge_qa category without passage monoculture.
3. **Mathematics gap:** neither pool fills the 1,000-example MSA math target (combined ≈ 589). Acquire a dedicated Arabic-math corpus, or author/synthesize MSA math examples, before finalizing the slice.
4. **Drop before selection:** the 1 empty-output QA row; treat the 409 ultra-short answers as low-value for register learning (keep only if needed to hit knowledge_qa quotas).

### Selection rules carried forward (mirroring the retention strategy)

- Dedup with Soup MinHash 0.85 first (done — artifacts in `data/processed/msa/*_deduped.jsonl`).
- Reserve the 1,000-example stratified holdout **before** selection.
- Deterministic selection: fixed order, stable sort by id, seed 42.
- Record in the selection manifest: per-source counts, passage-cap counts, category counts, dedup thresholds.
- No mixture-weight decisions are made in this audit; per-source weights remain deliberately undecided in `data/manifests/msa.yaml`.

---

## Artifacts

| Artifact | Path |
|---|---|
| Audit script (register/task/quality) | `scripts/audit_msa_candidates.py` |
| Soup dedup output — CIDAR | `data/processed/msa/cidar_deduped.jsonl` (9,965 rows) |
| Soup dedup output — Arabic QA | `data/processed/msa/arabic_qa_deduped.jsonl` (7,872 rows) |
| Soup langdetect output | `data/processed/msa/cidar_langdetect.jsonl`, `data/processed/msa/arabic_qa_langdetect.jsonl` |
| Full audit results (JSON) | `data/processed/msa/audit_results.json` |
| Source manifests | `data/manifests/msa.yaml` (revisions, licenses, provenance) |
| Acquisition script | `scripts/acquire_msa_candidate_data.py` |

Raw sources (immutable, read-only): `data/raw/arbml__CIDAR/`, `data/raw/bobez999__arabic-qa-dataset-sigir2024/`.

---

## Conclusion & Next Steps

Both candidates are genuine MSA corpora — the central audit question ("is an Arabic dataset automatically an MSA dataset?") is answered **yes for these two, with evidence**. CIDAR is clean, diverse, and sufficient alone for the 8,000-example slice; Arabic QA adds knowledge-QA depth but is passage-constrained and answer-short.

Next steps:

1. Write the CIDAR and Arabic QA adapters (normalize to canonical schema, per `docs/DATA_SCHEMA.md`), dropping the empty-output row and keeping the passage in `messages` for QA rows.
2. Apply passage-aware selection for the QA pool (≤ 2 questions/passage).
3. Close the MSA mathematics gap (dedicated source or authored content).
4. Reserve the stratified 1,000-example holdout before any selection.
5. Record the final per-source mixture weights and selection manifest, then update `docs/DATA.md`.