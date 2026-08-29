# Retention Data Analysis: Mathematics & Coding (Task 4)

**Author:** Haithem  
**Date:** August 29, 2026  
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

## Conclusion & Next Steps
Both datasets are structurally sound, highly diverse, and fully validated. They are ready to move to **Task 8 (Clean Mathematics + Coding)**, where we will apply length filtering and finalize the selection logic for the 10,000 total retention examples.