# Regular Arabizi Data Analysis: Latin-Script Conversational Slice

**Author:** Mohamed  
**Date:** October 18, 2026  
**Status:** Complete  

## Executive Summary

This document details the structural and qualitative analysis of the candidate datasets for the Regular Arabizi slice of the Tunisian adaptation. Both TUNIZI v2 and TArC have been inspected, validated, and are pending deduplication using the Soup CLI. Both datasets are highly viable and provide a surplus of examples for the Regular Arabizi target.

---

## 1. TUNIZI v2 (Regular Arabizi)

### Dataset Overview

* **Source:** `iCompass-ai/TUNIZI` (Zenodo record 4275240)
* **Total Rows:** ~100,000 comments
* **Format:** Plain text with sentiment labels (positive / negative / neutral)
* **License:** Open (iCompass, WANLP 2021)
* **Status:** Open, not yet acquired locally

### Structural Analysis (planned via Soup)

Not yet executed — pending local acquisition and Soup preflight:

* `soup data validate` — schema conformity on canonical fields
* `soup data stats` — length distribution, empty fields
* `soup data dedup --threshold 0.85` — near-duplicate removal on trained fields only
* `soup data doctor --model google/gemma-3-27b-it` — Gemma-token recount on formatted examples

### Category & Difficulty Distribution

The dataset contains manually annotated sentiment polarity:

* **Positive:** ~52% (~52,000 comments)
* **Negative:** ~43% (~43,000 comments)
* **Neutral:** ~5% (~5,000 comments)

Topics span movies, politics, sport, comedy, TV shows, TV series, arts, and Tunisian music videos.

### Proportion Contributing to Regular Arabizi Slice

**100% of the unique pool is usable.** At ~100k comments, TUNIZI v2 vastly exceeds any realistic Regular Arabizi target. It is the primary reference slice for Regular Arabizi — the larger, volume anchor against which Math-and-Coding-Arabizi will be matched.

### Quality Note

Comments were collected from public social media via Common Crawl-based scraping. Links, emoji symbols, and punctuation were removed during preprocessing. Annotation was performed by ten Tunisian native speakers (two male PhD holders, eight others). Negative comments may include offensive or insulting content; a toxicity filter should be applied before selection.

---

## 2. TArC — Tunisian Arabish Corpus (Regular Arabizi)

### Dataset Overview

* **Source:** `eligugliotta/tarc`
* **Total Rows:** 4,797 sentences / 43,327 tokens
* **Format:** Tab-separated files by genre (forum.tab, social.tab, blog.tab, rap.tab) with columns: data, arabish, class, coda, token, pos, governorate, age, gender
* **License:** CC BY-NC-SA 4.0 (**non-commercial**)
* **Status:** Open (GitHub), not yet acquired locally

### Structural Analysis (planned via Soup)

Not yet executed — pending local acquisition and Soup preflight:

* `soup data validate` — schema conformity on canonical fields
* `soup data stats` — length distribution, empty fields
* `soup data dedup --threshold 0.85` — near-duplicate removal on trained fields only
* `soup data doctor --model google/gemma-3-27b-it` — Gemma-token recount on formatted examples

### Category & Difficulty Distribution

TArC provides multi-layer linguistic annotation:

* **Token classification:** arabizi / foreign / emotag
* **CODA transliteration:** Arabizi → Arabic script (CODA convention)
* **Tokenization:** CODAfied text tokenization
* **POS tagging:** Penn Arabic Treebank style for arabizi tokens
* **Genres:** Forum (755 sentences), Social (3,162 sentences), Blog (366 sentences), Rap (514 sentences)
* **Metadata:** Governorate of provenance, age range ([-25], [25-35], [35-50], [50+]), gender (M/F)

### Proportion Contributing to Regular Arabizi Slice

**100% of the unique pool is usable.** TArC's 4,797 sentences provide high-quality, linguistically annotated Arabizi — ideal for validation, held-out evaluation, and supplementing TUNIZI v2 with annotated genre diversity.

### Quality Note

TArC is CC BY-NC-SA 4.0 — share-alike with a non-commercial clause. Derivative works must be released under the same license, and commercial model release is restricted. If commercial release is a requirement, TArC should be excluded or used only for non-commercial evaluation. Metadata coverage is incomplete (some rows have `/` for governorate/age/gender).

---

## 3. Selection & Packing Strategy for Regular Arabizi (16k Context Window)

**Added:** October 18, 2026  
**Decision context:** The model will train with a **16,384-token context window** (~15,500 tokens usable per example after chat-template overhead).

> **Note:** Length is not the main selection concern — **balance and diversity** are.

This section is self-contained and pedagogic: it explains not only *what* we will do, but *why*, so any team member can read it top-to-bottom and understand the whole plan.

### 3.1 Vocabulary (read this first)

| Term | Plain-language meaning |
|---|---|
| **Token** | A small chunk of text (~3–4 characters). Models read tokens, not letters or words. |
| **Context window** | The maximum number of tokens the model can process at once — its "desk size". Ours is 16,384 tokens. |
| **Example** | One training record (one Arabizi conversational text, one sentiment comment). |
| **Padding** | Filling unused space in a sequence with meaningless filler tokens so all sequences in a batch have the same length (required by the GPU, but wasted compute). |
| **Packing** | Placing several short examples into one sequence so the window is filled with real data instead of filler. |
| **Block-diagonal mask** | The safety rule that prevents packed neighbors from "seeing" each other; each example keeps its own attention scope. |

### 3.2 The two problems this strategy solves

**Problem 1 — Equal example counts do NOT mean equal learning.**

If Regular Arabizi and Math-and-Coding-Arabizi were allocated equal example counts, the counts would match but the *token masses* would not:

| Slice | Examples | Avg. tokens | Total tokens |
|---|---|---|---|
| Regular Arabizi (TUNIZI v2 + TArC) | ~20% share | ~tbd | T_regular_arabizi |
| Math & Coding (Arabizi) | ~15% share | ~tbd | T_math_coding_arabizi |

Math-and-Coding-Arabizi would differ from Regular Arabizi in tokens — and **tokens are what the model actually learns from**. Equal counts would quietly let one slice dominate the Arabizi signal. **Tokens are the currency; examples are not.**

**Problem 2 — Short examples waste the context window.**

The GPU needs every sequence in a batch to have the same shape. A short Arabizi example stretched into a 16k slot would be nearly all padding — pure waste. Packing removes almost all of this waste.

### 3.3 Phase 1 — Select data by token budget

**Step 1 — Select the Regular Arabizi examples first (the reference slice).**

Choose the Regular Arabizi examples using diversity rules, not length rules:

* **Sub-source mix** (detected from dataset origin): TUNIZI v2 (~100k sentiment comments) and TArC (4,797 annotated sentences across forum/social/blog/rap) — spread across sources so no single one dominates.
* **Content-type mix:** sentiment-annotated social comments ~70%, annotated linguistic corpus ~30%.
* **Length bands** covering the whole range (<1k, 1k–3k, 3k–6k, 6k–10k tokens), so varied utterance lengths are exercised.
* Latin-script Arabizi only, unique ids, schema-valid.

Length drops are almost unnecessary at 16k, but keep one hard rule as a safety net: **any formatted example over ~15,500 Gemma tokens is dropped whole — never truncated.** Cut-off Arabizi text is semantically broken and teaches broken lessons.

**Step 2 — Count the reference tokens, the right way.**

Tokenize the selected Regular Arabizi examples with **Gemma's own tokenizer**, on the **formatted** version (chat template included), because that is what the model sees in training. Record two totals:

* **Total tokens** — everything the model reads.
* **Assistant tokens** — the responses only, i.e., the part the training loss actually teaches (user turns are masked).

**Step 3 — Fill Math-and-Coding-Arabizi to the same token budget (whole examples only).**

Select Math-and-Coding-Arabizi examples until Math-and-Coding-Arabizi reaches the **same token total as Regular Arabizi scaled by the anchoring rule**:

> **T_category = (category's original share ÷ coding's share) × T_code**

For the Arabizi portion, Regular Arabizi is the anchor at 20% and Math-and-Coding-Arabizi is 15%:

> **T_math_coding_arabizi = (0.15 ÷ 0.20) × T_regular_arabizi = 0.75 × T_regular_arabizi**

* **Never cut an example.** Whole examples only.
* **Prefer matching assistant tokens** (the true learning signal). Matching total tokens is acceptable if assistant-token counting is impractical — but then record both numbers in the manifest so the choice is documented.
* **Stratify inside the budget:** split Math-and-Coding-Arabizi into sub-budgets first — mathematical reasoning vs. coding tasks — then fill each sub-budget separately. A budget without sub-quotas becomes a monoculture.
* **Stop inside a tolerance band of ±2%.** If the next example would overshoot, stop or swap in a smaller one.
* **Stay deterministic:** fixed order, stable sort by id, seed 42 — reruns must be byte-identical.

**Step 4 — Reserve the holdout BEFORE anything else.**

Carve out the holdout first and make it **token-proportional** across Regular Arabizi and Math-and-Coding-Arabizi (not example-proportional — Regular Arabizi now has more examples than Math-and-Coding-Arabizi). The holdout is never trained on and never packed with training data.

**Step 5 — Reuse this recipe for every other category.**

Every remaining category (regular Derja, math-and-coding Derja) gets its token budget by the **anchoring rule**:

> **T_category = (category's share ÷ coding's share) × T_code**

Each category keeps its *designed* share of the mixture — but expressed in tokens. Dataset-specific details (tagging, template caps, hygiene checks) follow the same pattern as this slice.

### 3.4 Phase 2 — Pack and pad (one rule)

There are no special cases; a single rule covers everything:

> **Keep filling each 16k sequence with whole examples for as long as one fits. When nothing fits anymore, pad the small remaining gap.**

* **Step 1 — Sort biggest-first.** Put all selected examples in one line, longest first. Big examples claim their sequences first; small ones arrive last and plug the gaps (this is "first-fit decreasing" packing).
* **Step 2 — Fill the sequences.** Place each example into the first sequence where it fits. With the Regular Arabizi pool averaging short-to-medium lengths, sequences end up >99% full of real data.
* **Step 3 — Pad only the leftovers.** The final partial sequence of the slice gets padding. This is the *only* place padding appears — roughly one partial shelf out of ~500.
* **Step 4 — Isolate shelf-mates (critical).** Packed examples must never attend to each other: each example needs its own attention scope (block-diagonal attention mask, position ids restarting per example). **Verify the training framework supports document-masked packing for Gemma 3 specifically** — Gemma 3 uses hybrid attention (sliding-window layers + periodic global layers), and the mask must be respected in both. TRL and Axolotl handle this; a naive custom loop does not. This is the single point where the strategy can silently fail — check it before training.

### 3.5 Rules that apply everywhere

1. **Tokens are the currency.** Every budget, share, and holdout split is measured in tokens.
2. **Never cut an example.** It fits whole, or it is dropped whole (>~15,500 formatted Gemma tokens).
3. **Tokenize with Gemma's tokenizer, on the formatted example.** Generic counters underestimate Arabizi.
4. **Stratify inside every budget** (sub-source / length band / content type) before counting tokens.
5. **Deterministic always:** same inputs + seed 42 = byte-identical outputs.
6. **Document everything in the manifest:** per-category token totals, assistant-token totals, example counts, packing efficiency, padding waste.

### 3.6 Implementation notes (what changes in the pipeline)

* `configs/data/derja.yaml` currently expresses per-category targets as example counts. Under this strategy, all Derja/Arabizi categories become **token-budget multipliers of the code anchor** — `regular_derja: 1.0`, `math_coding_derja: 0.67`, `regular_arabizi: 1.0`, `math_coding_arabizi: 0.75` — and `configs/data/derja.yaml` and `src/data/retention.py` need a token-aware selection mode (adapted for the Derja/Arabizi slices).
* `configs/data/mixture.yaml` should blend slices by **token shares**, not example shares — otherwise the imbalance we fix inside the Arabizi portion reappears between slices.
* Re-run the contamination check against `eval/arabizi/` on the selected Arabizi sample before training.

---

## Conclusion & Next Steps

TUNIZI v2 and TArC are structurally sound, highly diverse, and provide a surplus for the Regular Arabizi target. Task 8 will implement the strategy in Section 3: select Regular Arabizi first (TUNIZI v2 by diversity rules, supplemented by TArC annotated sentences), match the Math-and-Coding-Arabizi token budget with whole, stratified examples (~0.75 × T_regular_arabizi), reserve a token-proportional holdout, and pack biggest-first with padding only as the final-gap safety net. The same recipe then extends to the remaining Derja/Arabizi categories (regular Derja, math-and-coding Derja).
