# French/Tunisian Code-Switching Data Analysis (franco_tunisian slice)

**Author:** Mohamed Bensaad
**Date:** October 16, 2026  
**Status:** Candidate sources acquired into `data/raw/`; normalization + selection NOT yet performed (see `data/manifests/franco_tunisian.yaml` TODO list)

## Executive Summary

This document details the structural and qualitative analysis of the five candidate pools for the French/Tunisian code-switching slice — `franco_tunisian` (12% of the training cocktail per `configs/data/mixture.yaml`). All five pools (Farah21/reddit-tunisia-qa, Syrinesmati/tunisian-question-response-dataset, linagora/Tunisian_Derja_Dataset TunSwitchCodeSwitching config, hamzabouajila/tunisian-derja-unified-raw-corpus, linagora/linto-dataset-text-ar-tn) have been acquired and pinned in `data/manifests/franco_tunisian.yaml`; none have yet been normalized into the canonical schema or selected.

`franco_tunisian` is a **tunisian_adaptation** slice (not retention, not msa_formal) and is measured in **tokens**, not example counts. Its budget is derived by the anchoring rule from `T_code` (the measured token total of the retention coding slice): `T_franco = (0.12 ÷ 0.05) × T_code = 2.4 × T_code ≈ 18.5M tokens` (assuming `T_code ≈ 7.7M`).

Two of the five pools carry CC-BY-SA-4.0 (share-alike) licenses that may conflict with releasing adapted weights under Apache-2.0 — legal review is required before those pools enter production training. The remaining three are CC-BY-4.0 or TBD and are safer candidates.

The slice has ample raw volume (~860K+ rows across the five pools after counting is finalized) but is **selection-constrained, not supply-constrained**: the real work is normalizing to the canonical schema, filtering for genuine French↔Tunisian code-switching, deduplicating, reserving a token-proportional holdout, and packing to the token budget.

---

## 1. Farah21/reddit-tunisia-qa (Primary seed)

### Dataset Overview
* **Source:** `Farah21/reddit-tunisia-qa`
* **Revision:** `b18b9ba1ef98fdfe1f1d1868fff9f62700c0821f`
* **Local file:** `data/raw/Farah21__reddit-tunisia-qa/train-00000-of-00001.parquet`
* **Total Rows:** ~4,541–5,500 QA pairs reported (count after normalization)
* **Format:** Alpaca (`instruction`/`input`/`output`), Reddit-derived instruction-tuning QA pairs.
* **License:** TBD — verify SPDX id before use (TunisIA-Co-Lab open initiative — confirm exact terms).
* **Permitted use:** TBD — confirm redistribution/training terms before use.

### Structural Analysis
* An explicit `dialect` label column is present, which makes this the strongest franco_tunisian fit: rows are pre-labeled `mixed` (~5,104), `french` (~117), `tunisian_arabizi` (~165), `arabic` (~110), `tunisian_arabic` (~4).
* Small volume — treat as the primary seed rather than the sole source.
* Already instruction-formatted, so conversion to the canonical schema (`messages`, `category`, `source`, `language`) is straightforward.

### Coverage Analysis
The dialect labels allow direct filtering for code-switching rows (`mixed`, `french`, `tunisian_arabizi`). The `arabic` (~110) and `tunisian_arabic` (~4) rows are not code-switching and should be excluded from the franco_tunisian slice.

### Proportion Contributing to the franco_tunisian Budget
Too small to carry the slice alone (~5K rows max). It is a high-quality, already-labeled seed that anchors the `mixed`/`french` code-switching signal; volume must come from the larger pools below.

### Quality Note
License and permitted use are TBD and must be resolved before acquisition is considered final. Small `french` (~117) and `tunisian_arabizi` (~165) sub-counts mean this pool contributes diversity more than volume.

---

## 2. Syrinesmati/tunisian-question-response-dataset (Volume supplement)

### Dataset Overview
* **Source:** `Syrinesmati/tunisian-question-response-dataset`
* **Revision:** `1ae283f9d0d3271727bdd45c145d94e5eb80c1b8`
* **Local file:** `data/raw/Syrinesmati__tunisian-question-response-dataset/train-00000-of-00001.parquet`
* **Total Rows:** ~31,000 QA pairs reported (count after normalization)
* **Format:** `instruction`, `response`, `category` — Arabic script with natural French code-switching mixed in where appropriate to the topic.
* **License:** CC-BY-4.0 (permits use in model training with attribution; safe for Apache-2.0 weight release).
* **Attribution:** Tunisian Question-Response Dataset — Syrine Smati.

### Structural Analysis
* AI-generated, then reviewed and validated by native Tunisian speakers.
* Fields: `instruction`, `response`, `category` — requires mapping to canonical `messages`/`category`/`source`/`language`.
* Code-switching scope is present but not exhaustive across all domains — filter for French-CS rows rather than using the whole pool.

### Coverage Analysis
~31,000 Tunisian Darija QA pairs in Arabic script. A meaningful fraction contains natural French code-switching, making this the largest clearly-licensed (CC-BY-4.0) volume supplement for the slice.

### Proportion Contributing to the franco_tunisian Budget
A strong CC-BY-4.0 volume candidate. The filter step (keep French-code-switched rows) will reduce the usable count; the post-filter total is TBD until normalization and language-ID filtering are run.

### Quality Note
Reviewed by native speakers, so quality is higher than raw scraped corpora, but code-switching is incidental rather than systematic — expect a mix of pure-Darija and code-switched rows that must be separated at selection time.

---

## 3. linagora/Tunisian_Derja_Dataset (TunSwitchCodeSwitching config)

### Dataset Overview
* **Source:** `linagora/Tunisian_Derja_Dataset`
* **Revision:** `a6defa513c5a33d68be82a7baf6da4edd43866a5`
* **Local file:** `data/raw/linagora__Tunisian_Derja_Dataset/train-00001-of-00002.parquet`
* **Config:** `TunSwitchCodeSwitching` — the dedicated code-switching subset (~394,160 lines reported).
* **Total Rows:** ~394,160 lines (count after normalization)
* **Format:** Raw text (single `text` field), **not** instruction-formatted — requires conversion to canonical schema.
* **License:** CC-BY-SA-4.0 — share-alike; **LEGAL REVIEW REQUIRED** before use.
* **Attribution:** Tunisian Derja Dataset — Wajdi Ghezaiel and Jean-Pierre Lorré (LINAGORA).

### Structural Analysis
* Aggregated Tunisian Derja LM corpus with 14 sub-corpora; the TunSwitchCodeSwitching config is the dedicated code-switching subset.
* Single `text` field — no task labels, no chat turns; conversion to `messages` (user→assistant) is required before it can serve as SFT data.

### Coverage Analysis
~394K lines is the largest single code-switching-specific pool by raw count. It directly targets the franco_tunisian purpose (Tunisian↔French code-switching) but carries the CC-BY-SA-4.0 share-alike license, which may conflict with Apache-2.0 weight release.

### Proportion Contributing to the franco_tunisian Budget
High volume (~394K lines) but license-blocked pending legal review. If cleared, it would be the dominant volume source; if not, the slice must be funded entirely by CC-BY-4.0/TBD pools.

### Quality Note
Raw LM corpus text — noisy, no task structure, requires heavy cleaning and instruction formatting before selection. Share-alike terms are the primary risk, not data quality.

---

## 4. hamzabouajila/tunisian-derja-unified-raw-corpus

### Dataset Overview
* **Source:** `hamzabouajila/tunisian-derja-unified-raw-corpus`
* **Revision:** `e542d360022e90caee6946392f674ed013c42201`
* **Local file:** `data/raw/hamzabouajila__tunisian-derja-unified-raw-corpus/train-00000-of-00002.parquet`
* **Total Rows:** ~802,659 examples reported (count after normalization)
* **Format:** Raw text — raw and noisy (URLs, emojis, short texts <10 chars, no task labels).
* **License:** CC-BY-SA-4.0 — share-alike; **LEGAL REVIEW REQUIRED** before use.
* **Attribution:** Tunisian Derja Unified Raw Corpus — Hamza Bouajila.

### Structural Analysis
* Aggregated Derja corpus from multiple HF sources (linagora/Tunisian_Derja_Dataset, AzizBelaweid/Tunisian_Language_Dataset, arbml/Tunisian_Dialect_Corpus, abdouuu/tunisian_chatbot_data, khaled123/tuniset, …).
* Preserves French/English code-switching but is raw and noisy; no task labels, no chat structure.

### Coverage Analysis
~802K examples is the largest raw pool by count. It aggregates several other Tunisian Derja sources, so there is likely **significant overlap** with linagora/Tunisian_Derja_Dataset — cross-source deduplication will be essential.

### Proportion Contributing to the franco_tunisian Budget
Largest raw volume but CC-BY-SA-4.0 (license-blocked pending review) and very noisy. Even after aggressive cleaning, overlap with the linagora pool means its *incremental* unique contribution is uncertain until dedup is run.

### Quality Note
Heavy cleaning required (drop URLs, emojis, sub-10-char texts, quality heuristics). Not usable as SFT data without normalization to the canonical schema and instruction formatting.

---

## 5. linagora/linto-dataset-text-ar-tn (Code-switching category)

### Dataset Overview
* **Source:** `linagora/linto-dataset-text-ar-tn`
* **Revision:** `25dbb0aff3e5d4db1db5a69f7a4077280b442f4b`
* **Local file:** `data/raw/linagora__linto-dataset-text-ar-tn/train-00000-of-00001.parquet`
* **Total Rows:** ~4.5M lines reported (~65K code-switching); count after normalization
* **Format:** Raw text (ASR-oriented) — noisy and not instruction-formatted.
* **License:** CC-BY-4.0 (safe for Apache-2.0 weight release).
* **Attribution:** LinTO Textual Dataset in Arabic Tunisian — Hedi Naouara, Jean-Pierre Lorré, Jérôme Louradour (LINAGORA).

### Structural Analysis
* Tunisian dialect textual corpus compiled from films, TV series, rap lyrics, documentaries, stories, blogs and websites.
* Explicit code-switching category (~65K lines, Tunisian↔French/English).
* ASR-oriented raw text — requires conversion to canonical schema and quality filtering before use as SFT data.

### Coverage Analysis
~4.5M total lines but only ~65K in the explicit code-switching category — that category is the franco_tunisian-relevant subset. CC-BY-4.0 makes it license-safe.

### Proportion Contributing to the franco_tunisian Budget
~65K code-switching lines is a solid, license-safe supplement, but raw/ASR text means a large fraction will be dropped by cleaning (short texts, non-instructional content, noise). The post-cleaning usable count is TBD.

### Quality Note
Noisy ASR text; code-switching category is the target subset. Requires conversion to `messages` and quality filtering. License-safe (CC-BY-4.0), unlike the two linagora/hamzabouajila CC-BY-SA-4.0 pools.

---

## 6. Selection & Token-Budget Strategy (16k Context Window)

**Reference strategy:** `docs/data/retention_mohamed.md`, Section 3 — the master reference. Basic vocabulary (token, context window, padding, packing, block-diagonal mask) is defined there (Section 3.1). This section applies the same rules to `franco_tunisian` and adds only what is specific to it; the packing and padding rules are identical and are summarized here.

**Decision context:** The model trains with a **16,384-token context window** (~15,500 usable per example after chat-template overhead). None of the five pools' formatted examples are expected to approach this limit (raw text, short QA pairs), so — as with the retention slices — **length is not the problem.** Supply is adequate (~860K+ raw rows); the remaining work is normalization, code-switching filtering, dedup, holdout reservation, and token-budget selection.

### 6.0 Vocabulary (read this first)

| Term | Plain-language meaning |
|---|---|
| **Token** | A small chunk of text (~3–4 characters). Models read tokens, not letters or words. |
| **Context window** | The maximum number of tokens the model can process at once — its "desk size". Ours is 16,384 tokens. |
| **Example** | One training record (one French/Tunisian code-switching conversation). |
| **Padding** | Filling unused space in a sequence with meaningless filler tokens so all sequences in a batch have the same length (required by the GPU, but wasted compute). |
| **Packing** | Placing several short examples into one sequence so the window is filled with real data instead of filler. |
| **Block-diagonal mask** | The safety rule that prevents packed neighbors from "seeing" each other; each example keeps its own attention scope. |

### 6.0b The two problems this strategy solves

**Problem 1 — Equal example counts do NOT mean equal learning.**
The five pools have very different row counts (~5K to ~4.5M) and very different example lengths (short Reddit QA vs. long ASR lines). Selecting equal example counts from all pools would skew the token mass toward the long/raw pools and under-represent the short, high-quality instruction pairs. **Tokens are the currency; examples are not.**

**Problem 2 — Short examples waste the context window.**
The GPU needs every sequence in a batch to have the same shape. A short code-switching QA pair stretched into a 16k slot would be ~99% padding — pure waste. Packing removes almost all of this waste. This slice is packing-efficient: most examples are short, so many fit per 16k sequence at >99% real data.

### 6.1 The anchoring rule applied to `franco_tunisian`

Using the reference rule, with coding as the measuring stick:

> **T_category = (category's share ÷ coding's share) × T_code**

From `configs/data/mixture.yaml`, `franco_tunisian = 0.12`; coding is 25% of the 0.20 retention slice, i.e. **0.05** of the whole mixture. Therefore:

> **T_franco = (0.12 ÷ 0.05) × T_code = 2.4 × T_code ≈ 18.5M tokens** (T_code ≈ 7.7M)

The franco_tunisian token budget is ~18.5M tokens, to be filled from the five acquired pools whole-example by whole-example.

> **Caveat:** Real token totals must be measured with Gemma's own tokenizer on the **formatted** example (`soup data doctor --model …`) before any budget decision. Approximate counts are insufficient.

### 6.2 Supply assessment

| Pool | Approx rows | Notes |
|---|---|---|
| reddit_tunisia_qa | ~4,541–5,500 | Dialect-labeled; seed, not sole source |
| tunisian_question_response | ~31,000 | CC-BY-4.0; filter for French-CS rows |
| linagora_tunisian_derja_dataset (TunSwitchCodeSwitching) | ~394,160 | CC-BY-SA-4.0 — legal review |
| tunisian_derja_unified_raw_corpus | ~802,659 | CC-BY-SA-4.0 — legal review; overlaps linagora pool |
| linto_text_ar_tn | ~4.5M total (~65K code-switching) | CC-BY-4.0; noisy ASR text |

Combined raw volume is ample (~860K+ relevant rows after counting), but two pools are license-blocked and all five need normalization + code-switching filtering + dedup. The slice is **selection-constrained**: choose which cleared pools to use, in what token proportion, within per-source caps.

### 6.3 Phase 0 — Normalize into the canonical schema first

Before any selection, every pool must be normalized (Task 4B adapters) into the canonical schema (`docs/data/schema.yaml`): fields `id`, `messages`, `category`, `source`, `language` (+ optional `script`, `code_switching`, `variation_group`).

1. **Convert to `messages`:** All pools except reddit_tunisia_qa are raw text or `instruction`/`response` pairs — convert to `system`/`user`/`assistant` turns ending on `assistant`, per the schema's `messages` ordering rules.
2. **Tag `code_switching: true`** for rows that genuinely mix French and Tunisian/Arabic; `language` = `fr`, `ar`, `arabizi`, or `mixed` per the schema enum.
3. **Tag `category`:** use the schema's category enum (mathematics, reasoning, coding, general_instruction, knowledge_qa) where identifiable; franco_tunisian examples are predominantly `general_instruction`/`knowledge_qa`.
4. **Record `source`** per manifest source id for provenance and per-source caps.
5. **Cross-source dedup:** the hamzabouajila pool aggregates the linagora pool and others — run exact + near-duplicate (Soup MinHash 0.85) dedup within and across sources, recording the field set used.
6. **Record-level disjointness from eval:** verify no overlap with `data/manifests/eval.yaml` before any training run.

### 6.4 Phase 1 — Fill the budget

**Step 1 — Wait for the anchor.** T_code is measured, never estimated — no franco_tunisian selection starts before the coding slice has been formatted and tokenized with Gemma's tokenizer.

**Step 2 — Count the reference tokens, the right way.** Use the same token metric as every other slice (assistant tokens preferred; total acceptable if recorded). Re-measure with Gemma's own tokenizer on the **formatted** example before selection.

**Step 3 — Fill the franco_tunisian budget to T_franco = 2.4 × T_code (whole examples only).**
* **Never cut an example.** Whole examples only; any formatted example over ~15,500 Gemma tokens is dropped whole, never truncated.
* **Prefer matching assistant tokens** (the true learning signal). Record both total and assistant tokens in the manifest.
* **Stop inside a tolerance band of ±2%.** If the next example would overshoot, stop or swap in a smaller one.
* **Stay deterministic:** fixed order, stable sort by id, seed 42 — reruns must be byte-identical.
* **Filter for genuine code-switching:** do not fill the budget with pure-Darija or pure-French rows; the `code_switching` tag and dialect labels are the selection axis.

**Step 4 — Reserve the holdout BEFORE anything else.** Carve out the holdout first, token-proportional across sources, never trained on, never packed with training data (`data/processed/franco_tunisian/holdout.jsonl`).

**Step 5 — Stratify by source inside the budget.** Per-source mixture weights are deliberately undecided (`data/manifests/franco_tunisian.yaml` `mixture.weights: {}`). When decided, weights are **token shares of the franco_tunisian budget — not example shares.** Set `max_share_per_source` deliberately so no single source dominates (e.g., cap the two CC-BY-SA pools or exclude them pending legal review).

**Step 6 — Add cleaning filters:** drop URLs, emojis, short texts (<10 chars), and reject formatted examples over ~15,500 Gemma tokens whole.

### 6.5 Phase 2 — Pack and pad

There are no special cases; a single rule covers everything:

> **Keep filling each 16k sequence with whole examples for as long as one fits. When nothing fits anymore, pad the small remaining gap.**

* **Step 1 — Sort biggest-first.** Longest examples claim their sequences first; small ones plug the gaps ("first-fit decreasing" packing).
* **Step 2 — Fill the sequences.** Place each example into the first sequence where it fits. Short code-switching QA pairs pack densely — sequences end up >99% full of real data.
* **Step 3 — Pad only the leftovers.** The final partial sequence of the slice gets padding; this is the only place padding appears.
* **Step 4 — Isolate shelf-mates (critical).** Packed examples must never attend to each other: block-diagonal attention mask, position ids restarting per example. **Verify the training framework supports document-masked packing for Gemma 3 specifically** — Gemma 3 uses hybrid attention (sliding-window layers + periodic global layers), and the mask must be respected in both. TRL and Axolotl handle this; a naive custom loop does not.

### 6.6 Rules that apply everywhere

1. **Tokens are the currency.** Every budget, share, and holdout split is measured in tokens.
2. **Never cut an example.** It fits whole, or it is dropped whole (>~15,500 formatted Gemma tokens).
3. **Tokenize with Gemma's tokenizer, on the formatted example.** Generic counters misestimate mixed French/Tunisian text.
4. **Stratify inside every budget** (source / dialect / length band) before counting tokens.
5. **Deterministic always:** same inputs + seed 42 = byte-identical outputs.
6. **Document everything in the manifest:** per-source token totals, assistant-token totals, example counts, dedup field set and threshold, code-switching filter counts, packing efficiency, padding waste.

### 6.7 Holdout

Reserved **first**, before any selection, token-proportional across sources, never trained on, never packed with training data (`data/processed/franco_tunisian/holdout.jsonl`).

### 6.8 Hygiene checks

1. **Contamination vs. eval sets — required.** `data/manifests/franco_tunisian.yaml` sets `contamination_check_required: true`; verify record-level disjointness from `data/manifests/eval.yaml` before training.
2. **Contamination vs. public benchmarks — not meaningful as shipped.** `soup data decontaminate --benchmarks` accepts benchmark names only; real n-gram checking requires `--benchmark-file`.
3. **`soup data doctor` — deferred.** Requires `--model` (a tokenizer id); Gemma 3 is gated. Run as soon as the tokenizer is available for the Gemma-token recount.
4. **Language/script filtering:** `soup data langdetect` is Latin-script-centric and returns `"unknown"` for Arabic/Arabizi. Add a local Arabic-script / Arabizi check rather than delegating to Soup.
5. **License gate:** CC-BY-SA-4.0 pools (linagora_tunisian_derja_dataset, tunisian_derja_unified_raw_corpus) require legal review before production use; CC-BY-4.0/TBD pools are the safe path.

### 6.9 Implementation notes

* **`data/manifests/franco_tunisian.yaml`:** decide `mixture.weights` (token shares), set `seed: 42`, pin normalization adapter versions, record `output.path: data/processed/franco_tunisian/train.jsonl`.
* **`configs/data/mixture.yaml`:** `franco_tunisian: 0.12` is already set; token-aware blending/validation must be implemented before production (the count-based validator is retained for synthetic smoke tests).
* **`src/data/schema.py` / `src/data/validate.py`:** normalization adapters must emit the canonical schema with `code_switching`, `script`, `variation_group` where applicable.
* **`src/data/dedupe.py`:** record the field set alongside every near-duplicate count; prefer explicit `--field` over Soup's "all text fields" default.
* **`data/manifests/eval.yaml`:** must declare real sources before the contamination check can run; it is currently blocked, not passing.
* **Manifest:** record T_code, the token metric, per-source token totals and example counts, the dedup field set/threshold, code-switching filter counts, minimum-length drop count, packing efficiency, padding waste, and the contamination/license checks still outstanding.

---

## Conclusion & Next Steps

All five French/Tunisian code-switching candidate pools are acquired and pinned in `data/manifests/franco_tunisian.yaml`. The slice has ample raw volume (~860K+ rows) but is selection-constrained: normalization to the canonical schema, genuine code-switching filtering, cross-source dedup, holdout reservation, and token-budget selection remain.

Next steps:
1. **Write normalization adapters (Task 4B)** for each of the five pools into the canonical schema (`id`, `messages`, `category`, `source`, `language`, `code_switching`, `script`).
2. **Resolve licenses:** clear the CC-BY-SA-4.0 pools (linagora_tunisian_derja_dataset, tunisian_derja_unified_raw_corpus) or exclude them pending legal review.
3. **Run the full Soup preflight** (validate/stats/dedup/langdetect) on the normalized pools and record per-source token totals with Gemma's tokenizer.
4. **Decide `mixture.weights`** in `data/manifests/franco_tunisian.yaml` (token shares of the ~18.5M franco_tunisian budget) and set per-source caps.
5. **Run the eval-contamination check** as soon as `eval.yaml` declares real sources.
6. **Select and pack** to T_franco = 2.4 × T_code, reserving the holdout first, then document decisions in `docs/DATA.md`.
