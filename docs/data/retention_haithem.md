# Retention Data Analysis: Reasoning + General Instruction (Task 5)

**Author:** Haithem  
**Date:** August 29, 2026  
**Updated:** September 4, 2026 — added the Task 9 selection & token-budget strategy (Section 2)  
**Status:** Complete (SlimOrca analyzed)  

## Executive Summary
This document details the structural and qualitative analysis of the candidate dataset for the Reasoning, General Instruction, and Knowledge/QA slices of the Capability Retention dataset. SlimOrca has been fully inspected, validated, and deduplicated using the Soup CLI. It provides a massive, highly diverse pool of examples that easily covers the 4,000 reasoning, 3,000 general instruction, and 3,000 knowledge/QA targets.

---

## 1. SlimOrca (Reasoning, Instruction, Knowledge)

### Dataset Overview
* **Source:** `Open-Orca/SlimOrca`
* **File:** `oo-labeled_correct.gpt4.sharegpt.jsonl`
* **Total Rows:** 517,982
* **Format:** ShareGPT (Native `conversations` format with `from`/`value` keys. No conversion required).

### Structural Analysis
* **Valid Rows:** 517,982 / 517,982 (100% valid).
* **Empty Fields:** 0.
* **Length Distribution:** 
  * Average: ~461 tokens
  * Median: 1,499 characters
  * Max: 10,006 tokens
* **Duplicates:**
  * Exact duplicates: 0
  * Near-duplicates (at 0.85 threshold): 13,978
  * **Unique Rows Remaining:** 504,004

### Coverage Analysis (Reasoning, Instruction, Knowledge)
SlimOrca is a distilled subset of the OpenOrca dataset (based on FLAN). While it does not contain explicit category tags like MetaMathQA, its provenance guarantees broad coverage:
* **Reasoning:** Contains Chain-of-Thought (CoT) and complex logic prompts derived from FLAN.
* **General Instruction:** Covers a wide variety of standard NLP tasks (summarization, translation, rewriting).
* **Knowledge/QA:** Includes open-domain question answering and factual recall prompts.
* *Note: Exact category counts will be determined during Task 9 (Cleaning) when we parse the `system` and `user` prompts to tag and balance the final 10,000 examples.*

### Proportion Contributing to Targets
**100% of the unique pool is usable.** With over 500,000 unique, high-quality examples, SlimOrca vastly exceeds the combined 10,000 target for reasoning, general instruction, and knowledge.

---

## 2. Selection & Token-Budget Strategy for Task 9 (16k Context Window)

**Added:** September 4, 2026  
**Reference strategy:** `docs/data/retention_mohamed.md`, Section 3 — the master reference. Basic vocabulary (token, context window, padding, packing, block-diagonal mask) is defined there (Section 3.1). This section applies the same strategy to SlimOrca and adds only what is specific to this dataset; the packing and padding rules are identical and are only summarized here.

**Decision context:** The model trains with a **16,384-token context window** (~15,500 usable per example after chat-template overhead). SlimOrca's longest example (10,006 tokens) fits comfortably, so **length is not the problem** — the real work is **tagging, token balance, and diversity**.

> **Note:** This section supersedes the "10,000 examples" framing in Section 1. The original 4,000 / 3,000 / 3,000 *example* targets become 4:3:3 **token budgets**: the designed ratio is preserved exactly, but it is now measured in the currency the model actually learns with (≈ 33,400 examples total).

### 2.1 The anchoring rule (the one formula to remember)

The reference strategy fixed **T_code** = the measured token total of the 5,000 selected coding examples (~7.7M tokens) as the **anchor**, and filled math to 1.0 × T_code. SlimOrca's categories follow the **same rule**: each budget is the category's original designed share of the retention slice, expressed relative to coding's share.

> **T_category = (category's original share ÷ coding's share) × T_code**

| Category | Original share | Budget | ≈ Tokens | ≈ Examples |
|---|---|---|---|---|
| Coding — anchor (Task 8) | 25% | 1.0 × T_code | ~7.7M | 5,000 |
| Mathematics (Task 8) | 25% | 1.0 × T_code | ~7.7M | ~31,000 |
| **Reasoning** | 20% | **0.8 × T_code** | ~6.2M | **~13,400** |
| **General instruction** | 15% | **0.6 × T_code** | ~4.6M | **~10,000** |
| **Knowledge/QA** | 15% | **0.6 × T_code** | ~4.6M | **~10,000** |

**Sanity checks:**
* 0.8 : 0.6 : 0.6 = **4 : 3 : 3** — the designed SlimOrca ratio survives exactly.
* ≈ 33,400 examples is only ~6.6% of the 504,004 unique rows — no supply problem.
* The full retention slice becomes exactly 4.0 × T_code (~30.8M tokens), so the original whole-mixture percentages survive honestly — coding 5%, math 5%, reasoning 4%, instruction 3%, knowledge 3% — now in **tokens**.

In one sentence: *the example-count recipe is translated into token language, using code as the measuring stick.*

### 2.2 Phase 0 — Tag every example first (SlimOrca's extra prerequisite)

MetaMathQA arrived with a `type` column; code has detectable languages. SlimOrca has **neither** — and a per-category token budget cannot be filled until every example carries exactly one label. This tagging happens **before** any budgeting:

1. **Reasoning** — the prompt or response shows chain-of-thought structure (e.g., "Let's think step by step", "Step 1:", multi-step logical deduction) or the task is logical/mathematical deduction.
2. **Knowledge/QA** — otherwise, if the task is factual question answering or recall.
3. **General instruction** — everything else (summarization, rewriting, translation, open-ended tasks).

Rules of the tagging phase:
* **Deterministic** — same input, same tag, always: fixed rules plus the priority order above; no human judgment at scale.
* **Audited** — manually inspect ~200 random examples per tagged category, estimate accuracy, record the result in the manifest.
* **Borderline cases** (e.g., a CoT-heavy factual answer) are settled by the priority order — reasoning first — and that is fine.

### 2.3 Phase 1 — Fill the budgets (same recipe as math)

1. **Wait for the anchor.** T_code is *measured*, never estimated: Task 8 selects the 5,000 code examples, formats them (chat template), and tokenizes them with Gemma's tokenizer — only then can SlimOrca budgeting start. No category is selected before the anchor exists.
2. **Use the same token metric as math.** Whatever metric math matched on (assistant tokens preferred; total tokens acceptable), SlimOrca must use the same one — one metric, one anchor, five budgets. Standardize/strip system prompts **before** tokenizing; they are masked in training, which is one more reason assistant-token matching is the better metric.
3. **Whole examples only, ±2% tolerance.** Fill each budget with whole examples; stop before overshooting, or swap in a smaller example to land inside the band.
4. **Stratify inside each budget.** SlimOrca's system prompts are FLAN task templates — hash the system prompt and cap examples per template (e.g., ≤ 200 per unique template). This is SlimOrca's equivalent of MetaMathQA's "≤ 2 variants per seed problem" rule; it prevents task-template monoculture.
5. **Stay deterministic** — fixed order, stable sort by id, seed 42; reruns must be byte-identical.

### 2.4 Phase 2 — Pack and pad (unchanged; summarized)

Packing and padding follow the shared rules in the reference strategy: sort **biggest-first**, keep filling each 16k sequence with whole examples while one fits, isolate shelf-mates with **block-diagonal attention masks** (verified for Gemma 3's hybrid attention), and **pad only the final partial sequence** of each slice. SlimOrca's ~461-token average packs at >99% efficiency — dozens of examples per 16k shelf.

### 2.5 SlimOrca-specific hygiene checks

1. **Cross-dataset contamination with the math slice.** SlimOrca is FLAN-derived, and FLAN's CoT submixes include GSM8K and MATH — the same sources MetaMathQA was built from. Run a near-duplicate check between the SlimOrca *reasoning* picks and the MetaMathQA picks (not just within SlimOrca), plus the usual check against `eval/retention/`.
2. **Translation tasks vs. the English-only gate.** FLAN includes translation exercises whose *content* holds other languages even though the *instruction* is English. Recommendation: **keep them** (the instruction is English, the script is Latin, and translating is an English-capability exercise) — but record the decision explicitly in the manifest.
3. **Near-duplicates.** Remove the 13,978 near-duplicates (0.85 threshold) reported in Section 1 *before* tagging and budgeting.

### 2.6 Holdout

Reserved **first**, before any selection, and **token-proportional** across all five retention categories — which, thanks to the anchoring rule, means the same 25 / 25 / 20 / 15 / 15 token split as the training budgets. The holdout is never trained on and never packed with training data.

### 2.7 Implementation notes

* `configs/data/retention.yaml`: per-category targets become **token-budget multipliers of the code anchor** — `reasoning: 0.8`, `general_instruction: 0.6`, `knowledge_qa: 0.6` — consistent with the reference strategy's implementation note.
* Tagging lives in the adapter stage (upstream of selection); `src/data/retention.py` needs the token-aware selection mode described in the reference strategy.
* **Manifest:** record T_code, the token metric used (assistant vs. total), per-category token totals and example counts, tag-rule version + audit results, template-cap counts, and the cross-contamination check outcome.

---

## Conclusion & Next Steps

SlimOrca is structurally sound, highly diverse, and fully validated. Task 9 will implement Section 2: tag every example deterministically (reasoning → knowledge → instruction, with a ~200-example audit per category), fill the 0.8 / 0.6 / 0.6 × T_code budgets with whole, stratified examples (≈ 13,400 / 10,000 / 10,000), run the SlimOrca-specific hygiene checks (cross-contamination vs. the math picks, the translation-task decision, template caps), and reserve a token-proportional holdout — with packing and padding following the shared rules in `docs/data/retention_mohamed.md`, Section 3.
