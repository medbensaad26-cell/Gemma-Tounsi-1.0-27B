# Arabizi Data Gap: Training Corpus Status

## Current State of `data/train.jsonl`

The combined training file contains **755,094 records** across five slices, with estimated Gemma 3 assistant-token counts as follows:

| Slice | Records | Est. Assistant Tokens | Target Share | Target Tokens | Gap | Status |
|---|---|---|---|---|---|---|
| arabic_derja | 209,619 | 28,831,207 | 0.25 | 28,935,935 | -104,728 (-0.4%) | OK |
| arabizi | 203,803 | 2,683,352 | 0.35 | 40,510,309 | -37,826,957 (-93.4%) | SEVERELY UNDER |
| franco_tunisian | 154,958 | 14,449,059 | 0.12 | 13,889,249 | +559,810 (+4.0%) | OVER |
| msa_formal | 63,889 | 9,209,880 | 0.08 | 9,259,499 | -49,619 (-0.5%) | OK |
| retention (en) | 122,825 | 22,167,702 | 0.20 | 23,148,748 | -981,046 (-4.2%) | UNDER |
| **TOTAL** | **755,094** | **~77.3M** | **1.00** | **115,743,740** | — | — |

Token counts measured with `google/gemma-3-27b-it` tokenizer (`add_special_tokens=False`), 5,000-record stratified sample per slice.

## The Arabizi Problem

### Root Cause

The `arabizi` slice (target: 0.35 of total training tokens, ~40.5M assistant tokens) currently holds only **~2.7M assistant tokens** — about **6.6% of its target**. This gap (~37.8M tokens) cannot be closed with currently available public Tunisian Arabizi data.

### Why the gap exists

The previous data strategy allocated a portion of the Arabizi slice to specialized technical domains (mathematics, programming, technical problem-solving). That requirement was permanently dropped in favor of ordinary, natural Tunisian Arabizi content. However, the available public Arabizi corpora are extremely sparse:

- **TUNIZI v1**: ~9,210 short social-media comments (~151K assistant tokens)
- **TUNIZI v2**: ~100,000 short social-media comments (~2.05M assistant tokens)
- **CTAB**: ~5,702 Facebook comments (~113K assistant tokens)
- **chaymafourati**: ~1,500 short lines (~minimal contribution)
- **Nehdi/TuniziBigBench**: 1.79M rows but contains **MSA Arabic script, not Arabizi** — 0 usable Arabizi tokens

Most Arabizi records are very short (2–5 assistant tokens on average), dominated by transliteration tasks and social-media snippets. Even combining every publicly available Arabizi dataset yields at most ~5–10M assistant tokens, far short of the 40.5M target.

### Datasets investigated (Group A)

See `data/reports/group_a/FINAL_SUMMARY.md` for the full license audit and acquisition report. Eight candidates were reviewed:

- **APPROVED**: ctab (CC-BY-4.0)
- **CONDITIONAL**: tunizi_v1, tunizi_v2 (MIT on GitHub — provenance of third-party text unverified), tunizi_bigbench (OpenRAIL-M — contains MSA Arabic, not Arabizi)
- **REJECTED**: tarc (CC-BY-NC-SA), madar_tun (CC-BY-NC-3.0), tunisian_darija_english (CC-BY-NC-SA)
- **HOLD**: tunisian_dialect_corpus (license unclear)

## What was changed during the migration

### Migration summary (Tunisian math/coding strategy → ordinary Tunisian content)

- Removed the cross-cutting 20% technical quota from `configs/data/mixture.yaml`
- Removed Math-and-Coding-Arabizi (15% of cocktail) and Math-and-Coding-Derja (10% of cocktail) sub-allocations
- Both `arabizi` (0.35) and `arabic_derja` (0.25) slices are now 100% ordinary Tunisian content
- Removed `technical_categories` enum from `configs/data/schema.yaml`
- Removed technical quota enforcement from `src/data/mixture.py`, `src/data/schema.py`, `src/data/stats.py`
- Updated `README.md`, `docs/DATA_SCHEMA.md`, `docs/data/derja_and_arabizi_strategy.md`
- Marked `docs/data/regular_arabizi_mohamed.md` and `docs/data/regular_derja_mohamed.md` as SUPERSEDED
- Removed technical quota tests from `tests/test_data_pipeline.py`
- Removed obsolete fixture `data/synthetic/raw/arabizi_low_technical.jsonl`

### Data added during Group A work

- Downloaded and processed TUNIZI v1 (9,210 records), TUNIZI v2 (100,000 records), CTAB (5,702 records), chaymafourati (1,498 records)
- Merged 103,222 new unique Arabizi records into `data/processed/arabizi/train.jsonl` (4,050 duplicates with existing data removed)
- `arabizi/train.jsonl` now has 203,803 records (was 100,581)

## Recommended next steps

1. **Collect new conversational Arabizi data** — native Tunisian dialogues, Q&A, storytelling, explanations in Latin script, targeting ~38M assistant tokens
2. **OR reduce the Arabizi share** in `configs/data/mixture.yaml` from 0.35 to a level achievable with available data
3. **OR accept under-fill** — the Arabizi slice will train at ~6.6% of its target share, which may limit the model's ability to communicate naturally in Arabizi

## Files affected

- `data/train.jsonl` — combined training file (755,094 records)
- `data/processed/arabizi/train.jsonl` — Arabizi slice (203,803 records)
- `data/processed/arabic_derja/train.jsonl` — Derja slice (209,619 records)
- `data/processed/franco_tunisian/train.jsonl` — Franco-Tunisian slice (154,958 records)
- `data/processed/msa/train.jsonl` — MSA slice (63,889 records)
- `data/processed/retention/train.jsonl` — Retention slice (122,825 records)
- `configs/data/mixture.yaml` — slice shares (unchanged: arabizi 0.35, derja 0.25, franco 0.12, msa 0.08, retention 0.20)
- `docs/data/derja_and_arabizi_strategy.md` — current strategy (100% ordinary content)

## Training Decision

Training will proceed on this dataset as-is (`data/train.jsonl`, 755,094 records).

The Arabizi slice will train at approximately 6.6% of its target share (~2.7M of ~40.5M assistant tokens). This under-fill is a known limitation documented here. The model will still be trained on all other slices at or near their targets:

- **arabic_derja**: OK (-0.4%)
- **franco_tunisian**: OK (+4.0%)
- **msa_formal**: OK (-0.5%)
- **retention**: OK (-4.2%)

No further data acquisition is planned for the Arabizi slice at this time. If Arabizi fluency proves insufficient after training, revisit the options in the "Remaining Manual Work" section above.
