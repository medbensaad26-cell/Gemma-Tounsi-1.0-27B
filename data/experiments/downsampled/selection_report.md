# Dataset Selection Report — Gemma Tounsi 1.0 27B

## Methodology
- Source file: `data/splits/train.jsonl` (90% training split consumed by Soup).
- Tokenizer: `google/gemma-3-27b-it` (local HF cache snapshot), applied via `apply_chat_template` with `add_special_tokens=False`.
- Token counting: total formatted sequence (system + user + assistant turns).
- Deterministic seed: 42.
- Variant targets: 75% / 50% / 25% of original token count (nested: 25% ⊆ 50% ⊆ 75%).
- Selection strategy: token-aware stratified downsampling.

## Selection Rules
1. Preserve 100% of Arabizi (`language: arabizi`) examples in all variants.
2. Keep 100% of Mixed/Franco-Tunisian (`language: mixed`) examples (already tiny).
3. Downsample Arabic-script (`language: ar`) to fit each token budget.
4. Retain English/retention (`language: en`) subject to remaining budget.
5. Selection within each category is deterministic (seed-based stable ordering).
6. Variants are nested: smaller variants are strict subsets of larger ones.

## Per-Variant Results

| Variant | Target tokens | Achieved tokens | Examples | Reduction |
|---------|---------------|-----------------|----------|-----------|
| 75_percent | 79,092,866 | 79,092,862 | 466,791 | 25.0% |
| 50_percent | 52,728,577 | 52,728,575 | 345,949 | 50.0% |
| 25_percent | 26,364,288 | 26,364,284 | 242,817 | 75.0% |

## Per-Variant Language Breakdown

| Variant | Arabizi ex | Arabizi tok | ar ex | ar tok | mixed ex | en ex |
|---------|------------|-------------|-------|--------|----------|-------|
| 75_percent | 183,423 | 8,940,111 | 161,837 | 25,475,341 | 11,905 | 109,626 |
| 50_percent | 183,423 | 8,940,111 | 67,893 | 10,365,899 | 11,905 | 82,728 |
| 25_percent | 183,423 | 8,940,111 | 11,000 | 1,486,162 | 11,905 | 36,489 |

## Data Integrity Verification

Original file checksums (from manifest):
- `data\splits\train.jsonl`: SHA-256 `86fd39254eda303c7e2f8dc4adaa11bfb5fa7e6a542fbf4fa0675548765b3a14` (626,458,876 bytes)
- `data\splits\validation.jsonl`: SHA-256 `b57b3bb87999c6f58c656d1b25db747267ef26913b32be0f6610f716451060d4` (35,166,425 bytes)
- `data\splits\test.jsonl`: SHA-256 `95f43a8fba2f5e0f7f309a00d8a52b705952e4bfa62158aff029a1e0ac79e53a` (34,850,410 bytes)
- `data\splits\soup_trainval.jsonl`: SHA-256 `c6299fac164f6772175342b986856927a3c1c5a7496cd273f937206361f5cfe7` (661,625,301 bytes)

Variant file checksums (from manifest):
- `data\experiments\downsampled\75_percent\train.jsonl`: SHA-256 `33bddcd738e1ff72cccac0c72026356d45cd45575523678018bd9db8dea7de73`
- `data\experiments\downsampled\50_percent\train.jsonl`: SHA-256 `7ae7dbb4147dde2f47bd6a8e46b34d4b1d7bd696d57762b994c0c6bd2b0384ee`
- `data\experiments\downsampled\25_percent\train.jsonl`: SHA-256 `3085f2d930e84f9b0d8159dbe9844e734950d049482a1dcde237676588108ddc`

All checksums were verified against the original files at generation time;
original datasets were never modified.

## Nested Subset Verification
- 25% variant examples ⊆ 50% variant examples: VERIFIED
- 50% variant examples ⊆ 75% variant examples: VERIFIED
- 75% variant examples ⊆ original training split: VERIFIED

## Arabizi Preservation Verification
- All 183,423 Arabizi examples preserved in every variant (100%).
- No Arabizi examples were removed or duplicated.

## Category-Mapping Limitations

The data `language` field uses coarse labels that do not map one-to-one to
the project's intended training slices. The following limitations apply:

- **`ar` (Arabic script)**: combines **Arabic-script Tunisian Derja** and **MSA formal**.
  These are distinct intended slices (arabic_derja 25% + msa_formal 8% = ~33% combined)
  but are reported together under a single `ar` label. A per-slice breakdown is not
  available from the underlying data without re-processing.
- **`mixed`**: maps to **Franco-Tunisian** but may not be exclusively Franco-Tunisian.
  The underlying data does not provide a finer-grained code-switching label.
- **`en` (English)**: maps to **capability retention** but may include non-retention
  English examples. The underlying data does not distinguish retention-specific
  content from general English content within this label.
- **`arabizi`**: maps to **Tunisian Arabizi** and is the only label that aligns
  cleanly with its intended slice.

These mappings are documented as-is; no breakdown was invented that the data does not support.

## Output Files

| Variant | Path |
|---------|------|
| 75_percent | `data\experiments\downsampled\75_percent\train.jsonl` |
| 50_percent | `data\experiments\downsampled\50_percent\train.jsonl` |
| 25_percent | `data\experiments\downsampled\25_percent\train.jsonl` |

| Manifest | `data/experiments/downsampled/manifest.json` |
