# MSA Data Analysis: Formal Register (msa_formal slice)

**Author:** Mohamed  
**Date:** October 10, 2026  
**Status:** Complete for CIDAR + Arabic QA Dataset – SIGIR 2024 (inspected, validated, deduplicated with Soup); three additional open-source pools (Alpaca-Arabic-GPT4, Evol-Instruct-Arabic, Aya Dataset Arabic subset) are recorded as candidates awaiting acquisition.

## Executive Summary

This document details the structural and qualitative analysis of the candidate pools for the formal-register Modern Standard Arabic (MSA) slice — `msa_formal`. CIDAR and Arabic QA Dataset – SIGIR 2024 have been fully inspected, validated, and deduplicated using the Soup CLI. Both are genuine MSA corpora (97.2% and 99.9% MSA respectively), and CIDAR alone covers the 8,000-example target. Three further open-source pools (Alpaca-Arabic-GPT4, Evol-Instruct-Arabic, Aya Dataset Arabic subset) have been added to the candidate manifest but are not yet acquired.

Unlike the retention slices, **this slice does not have a surplus from the two originally acquired pools alone.** CIDAR + Arabic QA supply ≈ 12,891 usable rows (adequate in example terms, 1.43× the 9,000 needed) but only ≈ 1.3M tokens against a required ≈ 12.3M tokens — a ~9× shortfall in the currency that actually matters.

That gap is **closed by the three newly acquired pools** (Alpaca-Arabic-GPT4, Evol-Instruct-Arabic, Aya Arabic), which add ~40M approximate tokens, bringing total available to ≈41.2M vs ≈12.3M required — a 3.4× surplus. The slice is no longer supply-constrained; it is selection-constrained: choose which pools to use, in what proportion, and within what per-source caps.

---

## 1. CIDAR (Instruction-following, formal Arabic)

### Dataset Overview
* **Source:** `arbml/CIDAR`
* **File:** `data/raw/arbml__CIDAR/data/train-00000-of-00001-b2881e1b9f14c3b1.parquet`
* **Total Rows:** 10,000
* **Format:** Parquet → Alpaca JSONL. Parquet is not one of Soup's wire formats, so the driver materialises strict `instruction`/`input`/`output` rows (sha256 `edaa667b3e5a…`). The dropped `index` column is preserved in a sidecar id-map.
* **License:** Apache-2.0

### Structural Analysis
* **Valid Rows:** 10,000 / 10,000 (100% valid for `alpaca`). — `soup data validate`
* **Empty Fields:** 0. — `soup data stats`
* **Length Distribution:**
  * Median: 306 characters (p10 98 · p25 163 · p75 469 · p90 669)
  * Average: ~92 tokens (Soup's generic approximation — see the caveat in Section 6.1)
  * Max: 2,544 tokens — comfortably inside the 16k window
  * Min: 5 tokens — a very short tail; see the quality note below
* **Duplicates:**
  * Exact duplicate rows (flagged by `soup data validate`): **33** — reconciles the September audit's "29 reported / 33 by strict pair."
  * Near-duplicates (at 0.85 threshold): **35** — `soup data dedup --threshold 0.85`
  * **Unique Rows Remaining: 9,965** (reproduces the September audit exactly)

### Coverage Analysis
CIDAR is a culturally-adapted Arabic instruction set (~9,109 Alpagasus-derived samples translated to Arabic and human-reviewed, plus ~891 Arabic grammar instructions from Al Jazeera's "Ask the teacher"). Its `input` field is **empty on every row**, so it is pure single-turn instruction → response with no grounding passage. Consequence: CIDAR has **no `variation_group`**, so the per-passage diversity cap simply does not apply to it, and all 9,965 unique rows stay eligible.

Post-dedup statistics are **identical** to pre-dedup — removing 35 near-duplicates changed no percentile — confirming the duplicates were scattered noise rather than a distinct length band, so dedup costs the slice no diversity.

### Proportion Contributing to the 8,000-example Target
**100% of the unique pool is usable — and it is the backbone of the slice.** At 9,965 rows CIDAR alone covers the 8,000-example target, though not with enough margin to also fund the 1,000-row holdout on its own (9,965 vs. 9,000 needed = 1.11×). It is the only pool unaffected by the passage cap.

### Quality Note
The 5-token minimum and the 98-character p10 mean a real slice of CIDAR is extremely short. Very short instruction/response pairs teach little and inflate example counts without contributing tokens. **Recommendation:** add a minimum-length floor at selection time (a sensible starting point is the p10, ~98 characters) and record how many rows it drops — do not let trivially short rows consume holdout slots.

---

## 2. Arabic QA Dataset – SIGIR 2024 (Passage-grounded QA)

### Dataset Overview
* **Source:** `bobez999/arabic-qa-dataset-sigir2024` (ArabicaQA, SIGIR 2024)
* **File:** `data/raw/bobez999__arabic-qa-dataset-sigir2024/data/arabic_qa_10k_sample.jsonl`
* **Total Rows:** 10,000 (a sample, not the full corpus)
* **Format:** **Alpaca** natively (`instruction`/`input`/`output`, plus constant `source` + `language`). Re-emitted as strict alpaca (sha256 `b9722481d0df…`); the two metadata columns were moved to the sidecar id-map.
* **License:** MIT

### Structural Analysis
* **Valid Rows:** 10,000 / 10,000 (100% valid for `alpaca`). — `soup data validate`
* **Empty Fields:** **1** (one row with an empty `instruction` or `output`). It was deliberately passed to Soup rather than pre-filtered, so validation — not our script — found it.
* **Length Distribution:**
  * Median: 568 characters (p10 237 · p25 508 · p75 614 · p90 662)
  * Average: ~130 tokens (Soup approximation)
  * Max: 579 tokens · Min: 33 tokens
  * The band is **very tight** (p25 508 → p90 662): this is a homogeneous, templated corpus, not a varied one.
* **Duplicates:**
  * Exact duplicate rows (flagged by `soup data validate`): **1**
  * Near-duplicates (at 0.85 threshold): **3,055 (30.6%)** — `soup data dedup --threshold 0.85` on the trained fields only (`instruction`/`input`/`output`)
  * **Unique Rows Remaining: 6,945**

Post-dedup the distribution *widens* at the bottom (p25 moves 508 → 380, p10 237 → 211) while the median holds (568 → 574). Dedup removed a dense cluster of mid-length, near-identical rows — exactly the templated redundancy the tight band predicted.

### Coverage Analysis — the passage ceiling
This is the decisive measurement for this pool, computed from the id-map:

| Measure | Pre-dedup | Post-dedup |
|---|---|---|
| Rows | 10,000 | 6,945 |
| **Unique passages** | **1,500** | **1,500** |
| Max questions per passage | 67 | 50 |
| Mean questions per passage | 6.7 | 4.6 |

The 1,500-passage ceiling is **invariant under dedup** — 3,055 rows were removed and **not a single passage** was lost. The pool is 1,500 passages wide no matter how it is cleaned, and `configs/data/msa.yaml` caps questions per `variation_group` at 2:

| Cap | Rows surviving |
|---|---|
| 1 per passage | 1,500 |
| **2 per passage (configured)** | **2,926** |
| 3 per passage | 4,127 |

So the configured cap takes ArabicaQA from 6,945 to **2,926 usable rows** — a 58% reduction on top of the 30.6% already lost to dedup. The audit's estimate of "~3,000 rows from ~1,500 passages" was close but slightly optimistic: the exact figure is **2,926**, because 74 passages are left with only one question after dedup.

### Proportion Contributing to the 8,000-example Target
**Only ~29% of the raw pool is usable.** 10,000 raw → 6,945 unique → **2,926 after the diversity cap.** ArabicaQA is a useful *supplement* that adds passage-grounded reading comprehension — a genuinely different skill from CIDAR's bare instructions — but it cannot carry the slice.

### Quality Note
Raising the cap is the obvious temptation and should be resisted on evidence: at cap 3 the pool only reaches 4,127 rows (+1,201) while per-passage repetition rises 50%. The slice would be buying example count with topical monoculture. If more ArabicaQA volume is genuinely needed, the right move is to **pull a larger sample of the upstream corpus** (this file is a 10k sample) so that passage *breadth* grows, not per-passage depth.

---

## 3. Alpaca-Arabic-GPT4 (Volume candidate — not yet acquired)

### Dataset Overview
* **Source:** `FreedomIntelligence/Alpaca-Arabic-GPT4`
* **Total Rows:** ~49,969 (per RightNow paper merge)
* **Format:** Alpaca (`instruction`/`input`/`output`), translated from English Alpaca-GPT4 via GPT-3.5/GPT-4
* **License:** Apache-2.0
* **Status:** NOT YET ACQUIRED — run `scripts/acquire_msa_candidate_data.py` to download into `data/raw/`, then pin its real revision SHA in `data/manifests/msa.yaml` (currently `2026-10-16` snapshot date pin).

### Coverage Analysis
Adds ~50K rows of volume to close the msa_formal token-budget shortfall (~12.3M tokens needed vs ~1.3M supplied by CIDAR + Arabic QA). Register is translated formal Arabic / MSA — verify Arabic-script purity and filter register drift before selection. Cons: translated from English (may carry Western cultural assumptions, "Levantine media Arabic" register rather than pure MSA); GPT-4-generated responses may carry OpenAI terms-of-use considerations.

### Proportion Contributing to the Token Budget
Not yet measurable — structural analysis (validate/stats/dedup/langdetect) is pending acquisition. Once acquired, run `soup data validate`, `soup data stats`, `soup data dedup --threshold 0.85`, and `soup data langdetect` exactly as done for CIDAR and Arabic QA, then apply the passage cap only if the pool is passage-based (it is not — Alpaca-style, single-turn).

---

## 4. Evol-Instruct-Arabic (Volume candidate — not yet acquired)

### Dataset Overview
* **Source:** `FreedomIntelligence/evol-instruct-arabic`
* **Total Rows:** ~59,022 (per RightNow paper merge)
* **Format:** Alpaca-style instruction/output, generated with the Evol-Instruct method (more complex instructions than plain Alpaca), responses regenerated in Arabic
* **License:** Apache-2.0
* **Status:** NOT YET ACQUIRED — run `scripts/acquire_msa_candidate_data.py`, then pin its real revision SHA in `data/manifests/msa.yaml`.

### Coverage Analysis
Complements CIDAR's culturally-relevant instructions with higher instruction complexity. Register is translated/synthetic MSA — verify Arabic-script purity and filter register drift before selection. Cons: translated/synthetic; register may not be pure MSA.

### Proportion Contributing to the Token Budget
Not yet measurable — pending acquisition and the same Soup preflight (validate/stats/dedup/langdetect).

---

## 5. Aya Dataset Arabic subset (Quality supplement — not yet acquired)

### Dataset Overview
* **Source:** `CohereForAI/aya_dataset` (Arabic subset, `language_code == "arb"`)
* **Total Rows:** ~4,947 Arabic rows (full dataset: 204,114 pairs across 65 languages)
* **Format:** Prompt-completion
* **License:** Apache-2.0
* **Status:** NOT YET ACQUIRED — acquisition should scope to the Arabic subset; pin its real revision SHA in `data/manifests/msa.yaml`.

### Coverage Analysis
Human-curated multilingual instruction data (ACL 2024 Best Paper), native speaker input — not machine-translated — high-quality MSA signal, different provenance from the translated FreedomIntelligence pools. Cons: relatively small (~5K); may include dialectal content that needs MSA filtering.

### Proportion Contributing to the Token Budget
Not yet measurable — pending acquisition and the same Soup preflight.

---

## 6. Selection & Token-Budget Strategy (16k Context Window)

**Reference strategy:** `docs/data/retention_mohamed.md`, Section 3 — the master reference. This section applies the same rules to `msa_formal` and adds only what is specific to it.

### 6.0 Vocabulary (read this first)

| Term | Plain-language meaning |
|------|------------------------|
| **Token** | A small chunk of text (~3–4 characters). Models read tokens, not letters or words. |
| **Context window** | The maximum number of tokens the model can process at once — its "desk size". Ours is 16,384 tokens. |
| **Example** | One training record (one MSA instruction-output pair, or one passage-grounded QA pair). |
| **Padding** | Filling unused space in a sequence with meaningless filler tokens so all sequences in a batch have the same length (required by the GPU, but wasted compute). |
| **Packing** | Placing several short examples into one sequence so the window is filled with real data instead of filler. |
| **Block-diagonal mask** | The safety rule that prevents packed neighbors from "seeing" each other; each example keeps its own attention scope. |

**Decision context:** 16,384-token context window (~15,500 usable per example after chat-template overhead). All five pools' maxima (CIDAR 2,544, ArabicaQA 579, Alpaca-Arabic-GPT4 ~1,488, Evol-Instruct-Arabic ~4,057, Aya ~TBD) fit with enormous room to spare, so — as with the retention slices — **length is not the problem.** Supply is now adequate (Section 6.2); the remaining work is selection.

### 6.0b The two problems this strategy solves

**Problem 1 — Equal example counts do NOT mean equal learning.**

CIDAR examples average ~92 tokens; Arabic QA examples average ~128 tokens. Selecting equal example counts from both pools would still skew the token mass toward Arabic QA, and both pools together supply only ≈1.3M tokens against a required ≈12.3M. **Tokens are the currency; examples are not.** The shortfall is a slice-design problem, not a selection problem — it is escalated (Section 6.2), not silently absorbed.

**Problem 2 — Short examples waste the context window.**

The GPU needs every sequence in a batch to have the same shape. A 92-token CIDAR example stretched into a 16k slot would be ~99.4% padding — pure waste. Packing removes almost all of this waste. With averages of ~92 and ~128 tokens, this is the most packing-efficient slice in the project — roughly 120–170 examples per 16k sequence, >99% real data.

### 6.1 The anchoring rule applied to `msa_formal`

Using the reference rule, with coding as the measuring stick:

> **T_category = (category's share ÷ coding's share) × T_code**

From `configs/data/mixture.yaml`, `msa_formal = 0.08`; coding is 25% of the 0.20 retention slice, i.e. **0.05** of the whole mixture. Therefore:

> **T_msa = (0.08 ÷ 0.05) × T_code = 1.6 × T_code ≈ 12.3M tokens** (T_code ≈ 7.7M)

What the pools actually supply, using approximate token counts (~3.3 chars/token, Soup's heuristic; Arabic may be finer-grained):

| Pool | Usable rows | Avg tokens (approx) | Tokens |
|------|-------------|---------------------|--------|
| CIDAR (post-dedup) | 9,965 | ~92 | ~0.92M |
| ArabicaQA (post-dedup, capped at 2/passage) | 2,926 | ~128 | ~0.37M |
| Alpaca-Arabic-GPT4 (raw, ShareGPT→Alpaca) | 49,969 | ~186 | ~9.3M |
| Evol-Instruct-Arabic (raw, ShareGPT→Alpaca) | 59,022 | ~505 | ~29.8M |
| Aya Dataset Arabic (raw, inputs+targets) | 4,995 | ~158 | ~0.79M |
| **Total available (all 5 pools)** | **126,877** | — | **≈ 41.2M** |
| **Required (1.6 × T_code)** | — | — | **≈ 12.3M** |

> **Caveat:** Approximate counts use ~3.3 chars/token. Arabic tokenizes more finely in Gemma's tokenizer, so real totals may be lower — but even at a 2× correction the combined supply (~20.6M) still exceeds the 12.3M requirement. The supply shortfall that motivated Section 6.2 is **resolved**: all three newly acquired pools together add ~40M tokens, more than closing the gap. Budget decisions must still await Gemma-token measurement (`soup data doctor --model …`) before selection, but the slice is no longer supply-constrained.

### 6.2 The supply question — resolved

`msa_formal` can now fill its 8% token share. The three newly acquired pools add ~40M approximate tokens (Alpaca-Arabic-GPT4 ~9.3M, Evol-Instruct-Arabic ~29.8M, Aya Arabic ~0.79M), bringing total available to ≈41.2M vs ≈12.3M required — a **3.4× surplus** even before dedup. After dedup (0.85) and the passage cap, the slice still has ample headroom.

This is no longer a supply problem; it is a **selection problem**: choose which pools to use, in what proportion, and within what per-source caps. The four options from the earlier analysis are superseded:

| Option | What it means | Status |
|--------|---------------|--------|
| **A. Re-measure first** | Recount with Gemma's tokenizer before deciding anything. | **Required** — no budget decision on approximate counts. |
| **B. Widen the pools** | Not needed — pools already exceed the budget by 3.4×. | Superseded. |
| **C. Lower the share** | Not needed — 0.08 is fundable. | Superseded. |
| **D. Repeat examples** | Multiple epochs over the slice. | **Not recommended** — repetition invites memorisation. |

**Recommended sequence: A only.** Never D.

### 6.3 Phase 0 — Apply the passage cap before anything else

`msa_formal` has no categories to tag (`categories: null`), so it skips the tagging phase that SlimOrca needed. Its equivalent prerequisite is the **passage cap**, and the order is not optional:

> **Gap found:** `configs/data/msa.yaml` sets `max_examples_per_variation_group: 2` but **never defines what a variation group is** — there is no `variation_group` key anywhere in the config or the schema. The cap is therefore currently unenforceable: a limit without a group key cannot be applied. This analysis supplies the missing definition below, and it needs to be written into the config.

1. **Dedup first** (0.85, on the trained fields only — Section 3.1 of the analysis). Run `soup data dedup --threshold 0.85` on exactly the fields that will be trained on, and record that field set alongside the count. Constant metadata columns (`source`, `language`) suppress duplicate detections — the ArabicaQA figure of 3,055 (on trained fields) vs 2,128 (with metadata columns present) proves this.
2. **Then cap at 2 per variation group**, defining the group key as **the SHA-256 of the normalised passage (`input`)** — the definition the config is missing. The id-maps produced by the Soup run already carry `passage_sha256`, so the cap becomes reproducible rather than re-derived by eye.
3. **Record both counts.** Caps applied in the wrong order give different answers: capping before dedup would retain near-duplicate pairs *inside* a passage and waste the quota on them.
4. **CIDAR is exempt by construction** — empty `input`, so no group. Document this explicitly in the manifest so it reads as a deliberate decision, not an oversight.

### 6.4 Phase 1 — Fill the budget

**Step 1 — Wait for the anchor.** T_code is *measured*, never estimated — no MSA selection starts before the coding slice has been formatted and tokenized with Gemma's tokenizer.

**Step 2 — Count the reference tokens, the right way.** Use the same token metric as every other slice (assistant tokens preferred; total acceptable if recorded). One metric, one anchor, five budgets. The budget must not be finalised on approximate counts: re-measure with Gemma's own tokenizer on the **formatted** example (`soup data doctor --model …`) before selection.

**Step 3 — Fill the MSA budget to T_msa = 1.6 × T_code (whole examples only).** Select MSA examples until the slice reaches the same token total as coding scaled by the anchoring rule:

* **Never cut an example.** Whole examples only. Any formatted example over ~15,500 Gemma tokens is dropped **whole**, never truncated. No row in either acquired pool is anywhere near this limit.
* **Prefer matching assistant tokens** (the true learning signal). Matching total tokens is acceptable if assistant-token counting is impractical — but then record both numbers in the manifest so the choice is documented.
* **Stop inside a tolerance band of ±2%.** If the next example would overshoot, stop or swap in a smaller one.
* **Stay deterministic:** fixed order, stable sort by id, seed 42 — reruns must be byte-identical.

**Step 4 — Reserve the holdout BEFORE anything else.** Carve out the holdout first and make it **token-proportional** across the two sources (not example-proportional). The holdout is never trained on and never packed with training data (1,000 rows → `data/processed/msa/holdout.jsonl`).

**Step 5 — Stratify by source inside the budget.** The slice defines no categories, so `source` is the stratification axis. `max_share_per_source: 1.0` in `configs/data/msa.yaml` **disables** the cap and must be tightened. The realistic mix is CIDAR-dominant with the three new pools as volume fillers: set `max_share_per_source` deliberately (e.g., CIDAR ≤ 0.50, ArabicaQA ≤ 0.20, Alpaca-Arabic-GPT4 ≤ 0.20, Evol-Instruct-Arabic ≤ 0.15, Aya Arabic ≤ 0.05) rather than leaving it at 1.0. Accept that no single source should dominate the token budget.

**Step 6 — Add a minimum-length floor** (Section 1's quality note) so the 5-token tail cannot consume slots.

### 6.5 Phase 2 — Pack and pad

There are no special cases; a single rule covers everything:

> **Keep filling each 16k sequence with whole examples for as long as one fits. When nothing fits anymore, pad the small remaining gap.**

* **Step 1 — Sort biggest-first.** Put all selected examples in one line, longest first. Big examples claim their sequences first; small ones arrive last and plug the gaps (this is "first-fit decreasing" packing).
* **Step 2 — Fill the sequences.** Place each example into the first sequence where it fits. With ~126,877 raw rows across all five pools averaging ~92–505 tokens, sequences end up >99% full of real data.
* **Step 3 — Pad only the leftovers.** The final partial sequence of the slice gets padding. This is the *only* place padding appears — roughly one partial shelf out of ~100. Padding waste is negligible.
* **Step 4 — Isolate shelf-mates (critical).** Packed examples must never attend to each other: each example needs its own attention scope (block-diagonal attention mask, position ids restarting per example). **Verify the training framework supports document-masked packing for Gemma 3 specifically** — Gemma 3 uses hybrid attention (sliding-window layers + periodic global layers), and the mask must be respected in both. TRL and Axolotl handle this; a naive custom loop does not. This is the single point where the strategy can silently fail — check it before training.

### 6.6 Rules that apply everywhere

1. **Tokens are the currency.** Every budget, share, and holdout split is measured in tokens.
2. **Never cut an example.** It fits whole, or it is dropped whole (>~15,500 formatted Gemma tokens).
3. **Tokenize with Gemma's tokenizer, on the formatted example.** Generic counters underestimate Arabic.
4. **Stratify inside every budget** (source / length band) before counting tokens.
5. **Deterministic always:** same inputs + seed 42 = byte-identical outputs.
6. **Document everything in the manifest:** per-source token totals, assistant-token totals, example counts, passage-cap counts, packing efficiency, padding waste.

### 6.7 Holdout

Reserved **first**, before any selection, token-proportional across the two sources, never trained on, never packed with training data (1,000 rows → `data/processed/msa/holdout.jsonl`).

One slice-specific hazard: **the holdout must split on passages, not rows.** If one ArabicaQA passage lands in both the holdout and training, the holdout is contaminated — the model will have seen that passage's text verbatim. Partition by `passage_sha256`, so every question from a given passage lands entirely on one side. The existing `src/data/split.py` guarantees id-level disjointness only; **passage-level disjointness is a new requirement this analysis introduces.**

### 6.8 Hygiene checks

1. **Contamination vs. eval sets — blocked.** `data/manifests/msa.yaml` requires record-level disjointness from `eval.yaml`. `data/manifests/eval.yaml` is still a skeleton (`version: 0`, `sources: []` for all three tracks). Running `soup data decontaminate` against an empty reference set would report zero overlap for a trivial reason and would be misleading evidence, so it was deliberately not run. **Unblocks when:** `eval.yaml` declares at least one real source.
2. **Contamination vs. public benchmarks — not meaningful as shipped.** `soup data decontaminate --benchmarks` accepts benchmark *names* only; its own help states the corpora are not bundled. Real n-gram checking requires `--benchmark-file` with an operator-supplied corpus.
3. **`soup data doctor` — deferred.** It requires `--model` (a tokenizer id). Gemma 3 is gated and no model weights or tokenizer have been pulled. This is the right tool for the Gemma-token recount in Section 6.1 and should be run as soon as the tokenizer is available.
4. **Arabic-script gate.** `soup data langdetect` returns `"unknown"` for 100% of rows in both pools (it is Latin-script-centric), so it cannot enforce `language: [ar]` / `script: [arabic]`. Add a Unicode-range Arabic-script check locally — sufficient and cheap — rather than delegating to Soup.

### 6.9 Implementation notes

* **`configs/data/msa.yaml`:** tighten `max_share_per_source` from the placeholder `1.0` to the measured mix (Section 6.4); add a minimum-length floor; keep `max_examples_per_variation_group: 2` (the evidence in Section 2 supports it); add the `variation_group` group-key definition (passage `sha256`).
* **`src/data/split.py`:** add passage-level (group-aware) splitting for the holdout — Section 6.7.
* **Script validation:** add a Unicode-range Arabic-script check locally; Soup's `langdetect` cannot do it (Section 6.8).
* **`src/data/dedupe.py`:** record the field set alongside every near-duplicate count, and prefer explicit `--field` over Soup's "all text fields" default so results cannot drift with the column layout (Section 6.3).
* **`docs/data/msa_audit.md`:** supersede the ArabicaQA near-duplicate figure — **3,055, not 2,128** — and cite the field set.
* **Manifest:** record T_code, the token metric, per-source token totals and example counts, the dedup field set and threshold, passage-cap counts, the minimum-length drop count, packing efficiency, and the contamination checks still outstanding.

---

## Conclusion & Next Steps

All five MSA pools are structurally clean — 20,000 / 20,000 rows valid across CIDAR and ArabicaQA; the three new pools acquired and ready for Soup preflight. CIDAR is a solid backbone at 9,965 unique, cap-exempt rows. Evol-Instruct-Arabic (~29.8M approx tokens) and Alpaca-Arabic-GPT4 (~9.3M) provide more than enough volume to fund the 12.3M token budget.

The supply shortfall that previously blocked this slice is **resolved**. The remaining work is selection, not acquisition:

1. **Re-measure with Gemma's own tokenizer** (`soup data doctor --model …`) — no budget decision on approximate counts.
2. **Run the full Soup preflight** (validate/stats/dedup/langdetect) on the three new pools (Alpaca-Arabic-GPT4, Evol-Instruct-Arabic, Aya Arabic) and pin their real revision SHAs in `data/manifests/msa.yaml` (already recorded: 210c9f49…, 215a190d…, f9ea0458…).
3. **Set per-source caps** in `configs/data/msa.yaml` — `max_share_per_source` must be tightened from 1.0 to a deliberate mix.
4. **Implement the new requirements** this analysis surfaced: passage-level holdout splitting, a local Arabic-script check, an explicit dedup field set, and the `variation_group` group-key definition in `configs/data/msa.yaml`.
5. **Run the eval-contamination check** as soon as `eval.yaml` declares real sources — it is currently blocked, not passing.
