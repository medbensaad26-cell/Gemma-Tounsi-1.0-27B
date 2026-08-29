# Retention Data Analysis: Reasoning + General Instruction (Task 5)

**Author:** Haithem  
**Date:** August 30, 2026  
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

## Conclusion & Next Steps
SlimOrca is structurally sound, highly diverse, and fully validated. It is ready to move to **Task 9 (Clean Reasoning + Instruction + Knowledge)**, where we will apply length filtering, standardize system prompts, and finalize the selection logic for the 10,000 total reasoning/instruction/knowledge examples.