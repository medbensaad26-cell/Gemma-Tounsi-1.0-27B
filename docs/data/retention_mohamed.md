# Retention Data Analysis: Mathematics & Coding (Task 4)

**Author:** Mohamed
**Date:** August 29, 2026  
**Updated:** September 4, 2026 — added the Task 8 selection & packing strategy (Section 3)  
**Status:** Complete  

## Executive Summary
This document details the structural and qualitative analysis of the candidate datasets for the Mathematics and Coding slices of the Capability Retention dataset. Both MetaMathQA and Code-Feedback have been fully inspected, validated, and deduplicated using the Soup CLI. Both datasets are highly viable and provide a massive surplus of examples for the 5,000 math and 5,000 coding targets.

---

## 1. MetaMathQA (Mathematics)

### Dataset Overview
* **Source:** `meta-math/MetaMathQA`
* **Total Rows:** 395,000
* **Format:** Alpaca (Converted from original JSON to `instruction`/`output` JSONL for Soup compatibility).

### Structural Analysis
* **Valid Rows:** 395,000 / 395,000 (100% valid).
* **Empty Fields:** 0.
* **Length Distribution:** 
  * Average: ~230 tokens
  * Median: 848 characters
  * Max: 2,464 tokens (Fits comfortably within standard 4k context windows).
* **Duplicates:**
  * Exact duplicates: 8,954
  * Near-duplicates (at 0.85 threshold): 45,901
  * **Unique Rows Remaining:** 349,099

### Category & Difficulty Distribution
The dataset includes a `type` column indicating the augmentation method and source difficulty:
* **GSM8K-derived (Grade-school word problems):** ~240,000 rows (60.8%)
  * Types: `GSM_Rephrased`, `GSM_AnsAug`, `GSM_SV`, `GSM_FOBAR`
* **MATH-derived (Competition-level algebra/geometry):** ~155,000 rows (39.2%)
  * Types: `MATH_AnsAug`, `MATH_Rephrased`, `MATH_FOBAR`, `MATH_SV`

### Proportion Contributing to 5k Math Target
**100% of the unique pool is usable.** The 60/40 split between elementary (GSM8K) and advanced (MATH) problems provides a perfectly balanced foundation. We can easily select a proportional or difficulty-weighted 5,000 examples from the 349k unique rows.

---

## 2. Code-Feedback (Coding)

### Dataset Overview
* **Source:** `m-a-p/Code-Feedback`
* **Total Rows:** 66,383
* **Format:** ChatML (Native format, no conversion required).

### Structural Analysis
* **Valid Rows:** 66,383 / 66,383 (100% valid).
* **Empty Fields:** 0.
* **Length Distribution:**
  * Average: ~1,549 tokens
  * Median: 5,356 characters
  * Max: 11,304 tokens
* **Duplicates:**
  * Exact duplicates: 0
  * Near-duplicates (at 0.85 threshold): 0

### Category & Difficulty Distribution
The dataset does not contain explicit category tags. However, inspection of the `messages` column confirms it contains multi-turn coding interactions across various languages (Python, Ruby, etc.), covering code generation, debugging, and implementation tasks.

### Proportion Contributing to 5k Coding Target
**100% of the dataset is usable.** With 66,383 completely unique, high-quality coding examples, this dataset vastly exceeds the 5,000 coding target. 

### Quality Note for Task 8 (Cleaning)
While the average length (1,549 tokens) is healthy, the maximum length (11,304 tokens) exceeds standard context windows. During Task 8, we must apply a length filter (e.g., capping at 4,096 or 8,192 tokens) to prevent OOM errors during training.

---

## 3. Selection & Packing Strategy for Task 8 (16k Context Window)

**Added:** September 4, 2026
**Decision context:** The model will train with a **16,384-token context window** (~15,500 tokens usable per example after chat-template overhead).

> **Note:** This section supersedes the 4,096/8,192-token cap suggested in Section 2's quality note. With a 16k window, essentially all of Code-Feedback fits; length is no longer the main selection concern — **balance and diversity** are.

This section is self-contained and pedagogic: it explains not only *what* we will do, but *why*, so any team member can read it top-to-bottom and understand the whole plan.

### 3.1 Vocabulary (read this first)

| Term | Plain-language meaning |
|---|---|
| **Token** | A small chunk of text (~3–4 characters). Models read tokens, not letters or words. |
| **Context window** | The maximum number of tokens the model can process at once — its "desk size". Ours is 16,384 tokens. |
| **Example** | One training record (one math problem, or one full coding conversation). |
| **Padding** | Filling unused space in a sequence with meaningless filler tokens so all sequences in a batch have the same length (required by the GPU, but wasted compute). |
| **Packing** | Placing several short examples into one sequence so the window is filled with real data instead of filler. |
| **Block-diagonal mask** | The safety rule that prevents packed neighbors from "seeing" each other; each example keeps its own attention scope. |

### 3.2 The two problems this strategy solves

**Problem 1 — Equal example counts do NOT mean equal learning.**
If we select 5,000 math + 5,000 coding examples, the counts match but the *token masses* do not:

| Slice | Examples | Avg. tokens | Total tokens |
|---|---|---|---|
| Math (MetaMathQA) | 5,000 | ~230 | ~1.15M |
| Coding (Code-Feedback) | 5,000 | ~1,549 | ~7.7M |

Coding would outweigh math by roughly **6–7×** in tokens — and tokens are what the model actually learns from. Equal counts would quietly let coding dominate the retention signal and let math ability fade. **Tokens are the currency; examples are not.**

**Problem 2 — Short examples waste the context window.**
The GPU needs every sequence in a batch to have the same shape. A 230-token math example stretched into a 16k slot would be ~98.5% padding — pure waste. Packing removes almost all of this waste.

### 3.3 Phase 1 — Select data by token budget

**Step 1 — Select the coding examples first (the reference slice).**
Choose the 5,000 coding examples using diversity rules, not length rules:

* **Language mix** (detected from code fences): Python ~50%, JS/TS ~15%, Java/C/C++ ~15%, Go/Rust ~5%, other (Ruby, PHP, Shell, SQL, …) ~15%.
* **Task-type mix:** generation ~50%, debugging/fixing ~30%, refactor/optimize/explain ~20%.
* **Length bands** covering the whole range (<1k, 1k–3k, 3k–6k, 6k–10k, 10k+ tokens), so long-context skill is exercised.
* English-only, unique ids, schema-valid, multi-turn structure preserved.

Length drops are almost unnecessary at 16k (the dataset max is ~11,304 generic tokens), but keep one hard rule as a safety net: **any formatted example over ~15,500 Gemma tokens is dropped whole — never truncated.** Cut-off code is semantically broken and teaches broken lessons.

**Step 2 — Count the reference tokens, the right way.**
Tokenize the 5,000 coding examples with **Gemma's own tokenizer**, on the **formatted** version (chat template included), because that is what the model sees in training. Record two totals:

* **Total tokens** — everything the model reads.
* **Assistant tokens** — the responses only, i.e., the part the training loss actually teaches (user turns are masked).

**Step 3 — Fill math to the same token budget (whole examples only).**
Select math examples until math reaches the **same token total as coding**:

* **Never cut an example.** Whole examples only.
* **Prefer matching assistant tokens** (the true learning signal). Matching total tokens is acceptable if assistant-token counting is impractical — but then record both numbers in the manifest so the choice is documented.
* **Expect ~30,000–32,000 math examples** (~7.7M tokens ÷ ~250 tokens/example). The 349k unique-row pool makes this easy.
* **Stratify inside the budget:** split it into sub-budgets first — ~60% GSM8K-derived / ~40% MATH-derived, with per-type quotas (keep SV + FOBAR ≤ ~30% combined; they are templated and repetitive in bulk) — then fill each sub-budget separately. A budget without sub-quotas becomes a monoculture.
* **Stop inside a tolerance band of ±2%.** If the next example would overshoot, stop or swap in a smaller one.
* **Stay deterministic:** fixed order, stable sort by id, seed 42 — reruns must be byte-identical.

**Step 4 — Reserve the holdout BEFORE anything else.**
Carve out the holdout first and make it **token-proportional** across categories (not example-proportional — math now has ~6× more examples than coding). The holdout is never trained on and never packed with training data.

**Step 5 — Reuse this recipe for every other category.**
Every remaining category gets its token budget by the **anchoring rule**:

> **T_category = (category's original share ÷ coding's share) × T_code**

Each category keeps its *designed* share of the retention slice — but expressed in tokens. For the SlimOrca categories (reasoning, general instruction, knowledge QA) this yields **0.8 / 0.6 / 0.6 × T_code**, preserving the designed 4:3:3 ratio exactly. Dataset-specific details (tagging, template caps, hygiene checks) are in `docs/data/retention_haithem.md`, Section 2.

### 3.4 Phase 2 — Pack and pad (one rule)

There are no special cases; a single rule covers everything:

> **Keep filling each 16k sequence with whole examples for as long as one fits. When nothing fits anymore, pad the small remaining gap.**

* **Step 1 — Sort biggest-first.** Put all selected examples in one line, longest first. Big examples claim their sequences first; small ones arrive last and plug the gaps (this is "first-fit decreasing" packing).
* **Step 2 — Fill the sequences.** Place each example into the first sequence where it fits. With ~30k+ math examples averaging ~250 tokens, sequences end up >99% full of real data.
* **Step 3 — Pad only the leftovers.** The final partial sequence of each slice gets padding. This is the *only* place padding appears — roughly one partial shelf out of ~500.
* **Step 4 — Isolate shelf-mates (critical).** Packed examples must never attend to each other: each example needs its own attention scope (block-diagonal attention mask, position ids restarting per example). **Verify the training framework supports document-masked packing for Gemma 3 specifically** — Gemma 3 uses hybrid attention (sliding-window layers + periodic global layers), and the mask must be respected in both. TRL and Axolotl handle this; a naive custom loop does not. This is the single point where the strategy can silently fail — check it before training.

### 3.5 Rules that apply everywhere

1. **Tokens are the currency.** Every budget, share, and holdout split is measured in tokens.
2. **Never cut an example.** It fits whole, or it is dropped whole (>~15,500 formatted Gemma tokens).
3. **Tokenize with Gemma's tokenizer, on the formatted example.** Generic counters underestimate code.
4. **Stratify inside every budget** (type / language / task / length band) before counting tokens.
5. **Deterministic always:** same inputs + seed 42 = byte-identical outputs.
6. **Document everything in the manifest:** per-category token totals, assistant-token totals, example counts, packing efficiency, padding waste.

### 3.6 Implementation notes (what changes in the pipeline)

* `configs/data/retention.yaml` currently expresses per-category targets as example counts (math 5,000 + coding 5,000 within the 20,000 total). Under this strategy, all five categories become **token-budget multipliers of the code anchor** — `coding: 1.0`, `mathematics: 1.0`, `reasoning: 0.8`, `general_instruction: 0.6`, `knowledge_qa: 0.6` — and the config and `src/data/retention.py` need a token-aware selection mode.
* `configs/data/mixture.yaml` should blend slices by **token shares**, not example shares — otherwise the imbalance we fix inside retention reappears between slices.
* Re-run the contamination check against `eval/retention/` on the larger math sample before training.

---

## Conclusion & Next Steps

Both datasets are structurally sound, highly diverse, and fully validated. Task 8 will implement the strategy in Section 3: select coding first (5,000 examples by diversity rules), match the math token budget with whole, stratified examples (~31k), reserve a token-proportional holdout, and pack biggest-first with padding only as the final-gap safety net. The same recipe then extends to the remaining retention categories (reasoning, instruction-following, knowledge QA).
