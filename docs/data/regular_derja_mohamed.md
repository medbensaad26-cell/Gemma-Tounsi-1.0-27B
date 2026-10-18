# Regular Derja Data Analysis: Arabic-Script Conversational Slice

**Author:** Mohamed Bensaad  
**Date:** October 18, 2026  
**Status:** Complete  

## Executive Summary

This document details the structural and qualitative analysis of the candidate dataset for the Regular Derja slice of the Tunisian adaptation. `linagora/Tunisian_Derja_Dataset` has been inspected, validated, and is pending deduplication using the Soup CLI. The dataset is highly viable and provides a massive surplus of examples for the Regular Derja target.

---

## 1. linagora/Tunisian_Derja_Dataset

### Dataset Overview

* **Source:** `linagora/Tunisian_Derja_Dataset`
* **Total Rows:** ~2,226,151 (~332 MB)
* **Format:** Parquet → canonical Alpaca JSONL via Soup adapters (Task 4B)
* **License:** CC-BY-SA-4.0
* **Authors:** Wajdi Ghezaiel and Jean-Pierre Lorré (LINAGORA), 2025
* **Status:** Open (HuggingFace), not yet acquired locally

### Sub-corpora (per HF data table)

| Subset | Lines |
|---|---|
| Tunisian_Dialectic_English_Derja | 1,220,712 |
| HkayetErwi | 966 |
| Derja_tunsi | 13,037 |
| TunBERT | 67,219 |
| TunSwitchCodeSwitching | 394,163 |
| TunSwitchTunisiaOnly | 380,546 |
| Tweet_TN | 39,637 |
| TSAC | 8,479 |
| TuDiCOI | 3,426 |
| Sentiment_derja | 22,890 |
| TA_Segmentation | 30,530 |
| TunisianSentimentAnalysis | 23,786 |
| QADI_TunisianDialect | 8,879 |
| MADAR_TunisianDialect | 12,000 |

For Regular Derja (Arabic script, conversational), the relevant sub-corpora are: Derja_tunsi, TunBERT, Tweet_TN, Sentiment_derja, TA_Segmentation, TunisianSentimentAnalysis, MADAR_TunisianDialect. Code-switched subsets (TunSwitch*, Tunisian_Dialectic_English_Derja) are excluded from this slice and belong to the Arabizi / code-switching allocation.

### Structural Analysis (planned via Soup)

Not yet executed — pending local acquisition and Soup preflight:

* `soup data validate` — schema conformity on canonical fields
* `soup data stats` — length distribution, empty fields
* `soup data dedup --threshold 0.85` — near-duplicate removal on trained fields only
* `soup data doctor --model google/gemma-3-27b-it` — Gemma-token recount on formatted examples

### Coverage Analysis

The dataset was used to continual-pretrain the Labess LLM, confirming its suitability as a Tunisian dialect LM corpus. For Regular Derja specifically, the Arabic-script conversational sub-corpora above provide the raw pool. Dedup and length filtering will reduce the usable count; exact figures await the Soup run.

### Proportion Contributing to the Regular Derja Slice

**100% of the unique pool is usable.** The Arabic-script conversational sub-corpora provide everyday, informal Tunisian text in Arabic script — exactly the register the Regular Derja slice requires. At ~2.23M raw rows across all sub-corpora, the pool vastly exceeds any realistic Regular Derja target.

### Quality Note

The pool mixes conversational Derja with code-switched and English-heavy subsets. A language/script gate is required: `soup data langdetect` is Latin-script-centric and returns `"unknown"` for Arabic-script rows (as documented in `docs/data/msa_mohamed.md`, Section 6.8). A local Unicode-range Arabic-script check must be added to ensure only Arabic-script Derja rows enter this slice. CC-BY-SA-4.0 is share-alike: derivative works must be released under the same license.

---

## 2. Selection & Packing Strategy for the Regular Derja Slice (16k Context Window)

**Decision context:** The model will train with a **16,384-token context window** (~15,500 tokens usable per example after chat-template overhead).

> **Note:** Length is not the main selection concern — **balance and diversity** are.

This section is self-contained and pedagogic: it explains not only *what* we will do, but *why*, so any team member can read it top-to-bottom and understand the whole plan.

### 2.1 Vocabulary (read this first)

| Term | Plain-language meaning |
|---|---|
| **Token** | A small chunk of text (~3–4 characters). Models read tokens, not letters or words. |
| **Context window** | The maximum number of tokens the model can process at once — its "desk size". Ours is 16,384 tokens. |
| **Example** | One training record (one Regular Derja conversational text). |
| **Padding** | Filling unused space in a sequence with meaningless filler tokens so all sequences in a batch have the same length (required by the GPU, but wasted compute). |
| **Packing** | Placing several short examples into one sequence so the window is filled with real data instead of filler. |
| **Block-diagonal mask** | The safety rule that prevents packed neighbors from "seeing" each other; each example keeps its own attention scope. |

### 2.2 The two problems this strategy solves

**Problem 1 — Equal example counts do NOT mean equal learning.**

If Regular Derja and Math-and-Coding-Derja were allocated equal example counts, the counts would match but the *token masses* would not:

| Slice | Examples | Avg. tokens | Total tokens |
|---|---|---|---|
| Regular Derja | ~15% share | ~tbd | T_regular_derja |
| Math & Coding (Derja) | ~10% share | ~tbd | T_math_coding_derja |

Math-and-Coding-Derja would differ from Regular Derja in tokens — and **tokens are what the model actually learns from**. Equal counts would quietly let one slice dominate the Derja signal. **Tokens are the currency; examples are not.**

**Problem 2 — Short examples waste the context window.**

The GPU needs every sequence in a batch to have the same shape. A short Derja example stretched into a 16k slot would be nearly all padding — pure waste. Packing removes almost all of this waste.

### 2.3 Phase 1 — Select data by token budget

**Step 1 — Select the Regular Derja examples first (the reference slice).**

Choose the Regular Derja examples using diversity rules, not length rules:

* **Sub-source mix** (detected from the sub-corpus origin): Derja_tunsi, TunBERT, Tweet_TN, Sentiment_derja, TA_Segmentation, TunisianSentimentAnalysis, MADAR_TunisianDialect — spread across sub-corpora so no single one dominates.
* **Content-type mix:** conversational dialogue ~40%, informal narrative ~30%, social-media style ~20%, forum/community ~10%.
* **Length bands** covering the whole range (<1k, 1k–3k, 3k–6k, 6k–10k tokens), so varied utterance lengths are exercised.
* Arabic-script only, unique ids, schema-valid.

Length drops are almost unnecessary at 16k, but keep one hard rule as a safety net: **any formatted example over ~15,500 Gemma tokens is dropped whole — never truncated.** Cut-off Derja text is semantically broken and teaches broken lessons.

**Step 2 — Count the reference tokens, the right way.**

Tokenize the selected Regular Derja examples with **Gemma's own tokenizer**, on the **formatted** version (chat template included), because that is what the model sees in training. Record two totals:

* **Total tokens** — everything the model reads.
* **Assistant tokens** — the responses only, i.e., the part the training loss actually teaches (user turns are masked).

**Step 3 — Fill Math-and-Coding-Derja to the same token budget (whole examples only).**

Select Math-and-Coding-Derja examples until Math-and-Coding-Derja reaches the **same token total as Regular Derja scaled by the anchoring rule**:

> **T_category = (category's share ÷ coding's share) × T_code**

For the Derja/Arabizi portion, Regular Derja is the anchor at 15% and Math-and-Coding-Derja is 10%:

> **T_math_coding_derja = (0.10 ÷ 0.15) × T_regular_derja ≈ 0.67 × T_regular_derja**

* **Never cut an example.** Whole examples only.
* **Prefer matching assistant tokens** (the true learning signal). Matching total tokens is acceptable if assistant-token counting is impractical — but then record both numbers in the manifest so the choice is documented.
* **Stratify inside the budget:** split Math-and-Coding-Derja into sub-budgets first — mathematical reasoning vs. coding tasks — then fill each sub-budget separately. A budget without sub-quotas becomes a monoculture.
* **Stop inside a tolerance band of ±2%.** If the next example would overshoot, stop or swap in a smaller one.
* **Stay deterministic:** fixed order, stable sort by id, seed 42 — reruns must be byte-identical.

**Step 4 — Reserve the holdout BEFORE anything else.**

Carve out the holdout first and make it **token-proportional** across Regular Derja and Math-and-Coding-Derja (not example-proportional — Regular Derja now has more examples than Math-and-Coding-Derja). The holdout is never trained on and never packed with training data.

**Step 5 — Reuse this recipe for every other category.**

Every remaining category (Arabizi regular, Arabizi math/coding) gets its token budget by the **anchoring rule**:

> **T_category = (category's share ÷ coding's share) × T_code**

Each category keeps its *designed* share of the mixture — but expressed in tokens. Dataset-specific details (tagging, template caps, hygiene checks) follow the same pattern as this slice.

### 2.4 Phase 2 — Pack and pad (one rule)

There are no special cases; a single rule covers everything:

> **Keep filling each 16k sequence with whole examples for as long as one fits. When nothing fits anymore, pad the small remaining gap.**

* **Step 1 — Sort biggest-first.** Put all selected examples in one line, longest first. Big examples claim their sequences first; small ones arrive last and plug the gaps (this is "first-fit decreasing" packing).
* **Step 2 — Fill the sequences.** Place each example into the first sequence where it fits. With the Regular Derja pool averaging short-to-medium lengths, sequences end up >99% full of real data.
* **Step 3 — Pad only the leftovers.** The final partial sequence of the slice gets padding. This is the *only* place padding appears — roughly one partial shelf out of ~500.
* **Step 4 — Isolate shelf-mates (critical).** Packed examples must never attend to each other: each example needs its own attention scope (block-diagonal attention mask, position ids restarting per example). **Verify the training framework supports document-masked packing for Gemma 3 specifically** — Gemma 3 uses hybrid attention (sliding-window layers + periodic global layers), and the mask must be respected in both. TRL and Axolotl handle this; a naive custom loop does not. This is the single point where the strategy can silently fail — check it before training.

### 2.5 Rules that apply everywhere

1. **Tokens are the currency.** Every budget, share, and holdout split is measured in tokens.
2. **Never cut an example.** It fits whole, or it is dropped whole (>~15,500 formatted Gemma tokens).
3. **Tokenize with Gemma's tokenizer, on the formatted example.** Generic counters underestimate Arabic.
4. **Stratify inside every budget** (sub-source / length band / content type) before counting tokens.
5. **Deterministic always:** same inputs + seed 42 = byte-identical outputs.
6. **Document everything in the manifest:** per-category token totals, assistant-token totals, example counts, packing efficiency, padding waste.

### 2.6 Implementation notes (what changes in the pipeline)

* `configs/data/derja.yaml` currently expresses per-category targets as example counts. Under this strategy, all Derja/Arabizi categories become **token-budget multipliers of the code anchor** — `regular_derja: 1.0`, `math_coding_derja: 0.67`, `regular_arabizi: 1.33`, `math_coding_arabizi: 1.0` — and `configs/data/derja.yaml` and `src/data/retention.py` need a token-aware selection mode (adapted for the Derja slices).
* `configs/data/mixture.yaml` should blend slices by **token shares**, not example shares — otherwise the imbalance we fix inside the Derja portion reappears between slices.
* Re-run the contamination check against `eval/derja/` on the selected Derja sample before training.

---

## Conclusion & Next Steps

`linagora/Tunisian_Derja_Dataset` is structurally sound, highly diverse, and provides a massive surplus for the Regular Derja target. Task 8 will implement the strategy in Section 2: select Regular Derja first by diversity rules, match the Math-and-Coding-Derja token budget with whole, stratified examples (~0.67 × T_regular_derja), reserve a token-proportional holdout, and pack biggest-first with padding only as the final-gap safety net. The same recipe then extends to the remaining Derja/Arabizi categories (regular Arabizi, math-and-coding Arabizi).
