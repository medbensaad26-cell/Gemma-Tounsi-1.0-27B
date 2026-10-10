# MSA Data Analysis: Formal Register (`msa_formal` slice)

**Analysis run:** `scripts/soup_analyze_msa.py` (14 Soup invocations, 0 failures)
**Tooling:** Soup **v0.73.3** inside the pinned container `gemma-tounsi-soup:0.73.3`
**Date:** October 10, 2026
**Status:** Complete for the two available pools — CIDAR and ArabicaQA. PALM and ArSyra remain **blocked** (gated / licence; see `data/manifests/msa.yaml` → `blocked_candidates`).
**Evidence:** `data/processed/msa/soup/soup_report.json`, raw logs in `data/processed/msa/soup/logs/`, captured `--help` text in `data/processed/msa/soup/help/`.

## Executive Summary

Both available MSA candidate pools have been inspected, validated, and deduplicated with the Soup CLI. Both are **structurally sound**: 20,000 / 20,000 rows are valid `alpaca`, and only one row in the whole corpus has an empty field.

Unlike the retention slices, however, **this slice does not have a surplus.** Three findings drive every recommendation below:

1. **ArabicaQA is far more redundant than previously recorded.** At the 0.85 threshold it loses **3,055 rows (30.6%)**, not the 2,128 recorded in `docs/data/msa_audit.md`. The old number was measured on a file that carried two constant metadata columns; this is reproduced and explained in Section 3.1.
2. **The passage ceiling is now exact, and it survives dedup.** The September audit already identified that ArabicaQA's 10,000 questions rest on only 1,500 unique passages and recommended a ≤ 2-per-passage cap yielding "~3,000 rows"; this run confirms that ceiling and sharpens it. Critically, dedup removes 3,055 rows but **not a single passage**, and after dedup the cap leaves exactly **2,926** rows — not ~3,000 — because 74 passages have only one question left. The cap must be applied *after* dedup, an ordering no document had specified.
3. **The slice cannot fill its 8% token share.** Against the anchoring rule used by the retention strategy, `msa_formal` needs ≈ **12.3M tokens** but the pools supply ≈ **1.3M** — a shortfall of roughly **9×**. In *example* terms the slice is fine (12,891 available vs. 9,000 needed); in the currency that actually matters it is not. Section 4.2 sets out the options.

---

## 1. CIDAR (Instruction-following, formal Arabic)

### Dataset Overview
* **Source:** `arbml/CIDAR`
* **File:** `data/raw/arbml__CIDAR/data/train-00000-of-00001-b2881e1b9f14c3b1.parquet`
* **Total Rows:** 10,000
* **Format:** Parquet → **Alpaca** JSONL. Parquet is not one of Soup's wire formats, so the driver materialises strict `instruction`/`input`/`output` rows (sha256 `edaa667b3e5a…`). The dropped `index` column is preserved in a sidecar id-map, not discarded.

### Structural Analysis
* **Valid Rows:** 10,000 / 10,000 (100% valid for `alpaca`).
* **Empty Fields:** 0.
* **Length Distribution:**
  * Median: 306 characters (p10 98 · p25 163 · p75 469 · p90 669)
  * Average: ~92 tokens *(Soup's generic approximation — see the caveat in Section 4.1)*
  * Max: 2,544 tokens — comfortably inside the 16k window
  * Min: 5 tokens — **there is a very short tail; see the quality note below**
* **Duplicates:**
  * Exact duplicate rows (flagged by `soup data validate`): **33** — this matches the September audit's strict `(instruction, output)` pair count exactly, reconciling the "29 reported / 33 by strict pair" note recorded there.
  * Near-duplicates (at 0.85 threshold): **35**
  * **Unique Rows Remaining: 9,965** (reproduces September's figure exactly)

### Coverage Analysis
CIDAR is a culturally-adapted Arabic instruction set. Its `input` field is **empty on every row**, so it is pure single-turn instruction → response with no grounding passage. Consequence: CIDAR has **no `variation_group`**, so the per-passage diversity cap simply does not apply to it, and all 9,965 unique rows stay eligible.

Post-dedup statistics are **identical** to pre-dedup (median 306 chars, avg ~92 tokens, max 2,544). Removing 35 near-duplicates changed no percentile — confirming the duplicates were scattered noise rather than a distinct length band, so dedup costs the slice no diversity.

### Proportion Contributing to the 8,000-example Target
**100% of the unique pool is usable — and it is the backbone of the slice.** At 9,965 rows CIDAR alone covers the 8,000-example target, though not with enough margin to also fund the 1,000-row holdout on its own (9,965 vs. 9,000 needed = 1.11×). It is the only pool unaffected by the passage cap.

### Quality Note
The 5-token minimum and the 98-character p10 mean a real slice of CIDAR is extremely short. Very short instruction/response pairs teach little and inflate example counts without contributing tokens. **Recommendation:** add a minimum-length floor at selection time (a sensible starting point is the p10, ~98 characters) and record how many rows it drops — do not let trivially short rows consume holdout slots.

---

## 2. ArabicaQA (Passage-grounded QA)

### Dataset Overview
* **Source:** `bobez999/arabic-qa-dataset-sigir2024` (ArabicaQA, SIGIR 2024)
* **File:** `data/raw/bobez999__arabic-qa-dataset-sigir2024/data/arabic_qa_10k_sample.jsonl`
* **Total Rows:** 10,000 (a sample, not the full corpus)
* **Format:** **Alpaca** natively (`instruction`/`input`/`output`, plus constant `source` + `language`). Re-emitted as strict alpaca (sha256 `b9722481d0df…`); the two metadata columns were moved to the sidecar id-map — a change that turns out to matter a great deal (Section 3.1).

### Structural Analysis
* **Valid Rows:** 10,000 / 10,000 (100% valid for `alpaca`).
* **Empty Fields:** **1** (one row with an empty `instruction` or `output`). It was deliberately passed to Soup rather than pre-filtered, so validation — not our script — would be the thing that finds it.
* **Length Distribution:**
  * Median: 568 characters (p10 237 · p25 508 · p75 614 · p90 662)
  * Average: ~130 tokens *(Soup approximation)*
  * Max: 579 tokens · Min: 33 tokens
  * The band is **very tight** (p25 508 → p90 662): this is a homogeneous, templated corpus, not a varied one.
* **Duplicates:**
  * Exact duplicate rows (flagged by `soup data validate`): **1**
  * Near-duplicates (at 0.85 threshold): **3,055 (30.6%)**
  * **Unique Rows Remaining: 6,945**

Post-dedup the distribution *widens* at the bottom (p25 moves 508 → 380, p10 237 → 211) while the median holds (568 → 574). Dedup removed a dense cluster of mid-length, near-identical rows — exactly the templated redundancy the tight band predicted.

### Coverage Analysis — the passage ceiling
This is the decisive measurement for this pool, and it was computed from the id-map rather than assumed:

| Measure | Pre-dedup | Post-dedup |
|---|---|---|
| Rows | 10,000 | 6,945 |
| **Unique passages** | **1,500** | **1,500** |
| Max questions per passage | 67 | 50 |
| Mean questions per passage | 6.7 | 4.6 |

The 1,500-passage ceiling was first reported in the September audit (§4); what is new here is that it is **invariant under dedup** — 3,055 rows were removed and **not a single passage** was lost. The pool is 1,500 passages wide no matter how it is cleaned, and `configs/data/msa.yaml` caps questions per `variation_group` at 2:

| Cap | Rows surviving |
|---|---|
| 1 per passage | 1,500 |
| **2 per passage (configured)** | **2,926** |
| 3 per passage | 4,127 |

So the configured cap takes ArabicaQA from 6,945 to **2,926 usable rows** — a 58% reduction on top of the 30.6% already lost to dedup. The audit's estimate of "~3,000 rows from ~1,500 passages" was close but slightly optimistic: the exact figure is **2,926**, because 74 passages are left with only one question after dedup. Small as that gap is, it is the difference between an estimate and a count, and the selection code should use the count.

### Proportion Contributing to the 8,000-example Target
**Only ~29% of the raw pool is usable, and it cannot be the primary source.** 10,000 raw → 6,945 unique → **2,926 after the diversity cap**. ArabicaQA is a useful *supplement* that adds passage-grounded, long-form reading comprehension — a genuinely different skill from CIDAR's bare instructions — but it cannot carry the slice.

### Quality Note
Raising the cap is the obvious temptation and should be resisted on evidence: at cap 3 the pool only reaches 4,127 rows (+1,201) while per-passage repetition rises 50%. The slice would be buying example count with topical monoculture. If more ArabicaQA volume is genuinely needed, the right move is to **pull a larger sample of the upstream corpus** (this file is a 10k sample) so that passage *breadth* grows, not per-passage depth.

---

## 3. Cross-Pool Findings

### 3.1 Near-duplicate counts depend on the field set — reproduced and corrected

`docs/data/msa_audit.md` records 2,128 near-duplicates removed from ArabicaQA. This run measured **3,055**. The discrepancy was not left as a mystery; it was reproduced:

| Soup input | Columns fed to `dedup` | Result at 0.85 |
|---|---|---|
| `arabic_qa_with_meta.jsonl` (September's shape) | instruction, input, output, **source, language** | 10,000 → 7,872 (**−2,128**) |
| `arabic_qa_alpaca.jsonl` (this run) | instruction, input, output | 10,000 → 6,945 (**−3,055**) |

The September figure reproduces **exactly** when the two extra columns are present. CIDAR, whose only extra column was the integer `index`, reproduced identically in both runs (−35 = −35), which is the control that confirms the field set is the variable.

`soup data dedup` hashes *all text fields* by default, and `source` / `language` are **constant** across all 10,000 rows (`ArabicaQA-SIGIR2024`, `ar`). Their presence suppresses **927 duplicate detections (30% of the true count)**. The precise mechanism inside Soup's MinHash implementation is **not established here** and this document does not guess at it; what is established, reproducibly, is the effect and its size.

> **Rule going forward:** deduplicate on **exactly the fields that will be trained on**, and record that field set alongside the number. A near-duplicate count without its field set is not a reproducible measurement. **3,055 is the number this slice should use**, because constant metadata columns are never trained on.

### 3.2 `soup data langdetect` is unusable for Arabic script — confirmed, not assumed

The audit suspected this; it is now verified against 0.73.3. Soup writes a `_language` field, and across **all 20,000 rows in both pools the value is `"unknown"`** (100%). The heuristic is Latin-script-centric and contributes nothing here.

**Consequence:** the `language: [ar]` and `script: [arabic]` gates in `configs/data/msa.yaml` **cannot** be enforced with Soup. They need an Arabic-aware check — a Unicode-range script test is sufficient and cheap, and should be added to the local validator rather than delegated.

### 3.3 Checks that could not be run — and why

Honesty about gaps matters more than a longer list of green checks:

* **Contamination vs. our eval sets — blocked.** `data/manifests/msa.yaml` requires record-level disjointness from `eval.yaml`. `data/manifests/eval.yaml` is still a skeleton (`version: 0`, `sources: []` for all three tracks). Running `soup data decontaminate` against an empty reference set would report zero overlap *for a trivial reason* and would be misleading evidence, so it was deliberately not run. **Unblocks when:** `eval.yaml` declares at least one real source.
* **Contamination vs. public benchmarks — not meaningful as shipped.** `soup data decontaminate --benchmarks` accepts benchmark *names* only; its own help states the **corpora are not bundled**. Real n-gram checking requires `--benchmark-file` with an operator-supplied corpus, which this project does not yet have.
* **`soup data doctor` — deferred.** It requires `--model` (a tokenizer id). Gemma 3 is gated and, per `docs/ENVIRONMENT.md`, no model weights or tokenizer have been pulled. This is the right tool for the Gemma-token recount in Section 4.1 and should be run as soon as the tokenizer is available.
* **`soup data topics` — deferred.** It needs an embedding model download; the container runs offline by design. Worth revisiting, as it would quantify CIDAR's topical breadth directly.

---

## 4. Selection & Token-Budget Strategy (16k Context Window)

**Reference strategy:** `docs/data/retention_mohamed.md`, Section 3 — the master reference. Vocabulary (token, context window, padding, packing, block-diagonal mask) is defined there in Section 3.1 and is not repeated. `docs/data/retention_haithem.md`, Section 2 applies the same rules to a second slice. This section applies them to `msa_formal` and adds only what is specific to it.

**Decision context:** 16,384-token context window (~15,500 usable per example). Both pools' maxima (2,544 and 579 approximate tokens) fit with enormous room to spare, so — as with the retention slices — **length is not the problem.** For `msa_formal` the problem is **supply**.

### 4.1 The anchoring rule applied to `msa_formal`

Using the reference rule, with coding as the measuring stick:

> **T_category = (category's share ÷ coding's share) × T_code**

From `configs/data/mixture.yaml`, `msa_formal = 0.08`; coding is 25% of the 0.20 retention slice, i.e. **0.05** of the whole mixture. Therefore:

> **T_msa = (0.08 ÷ 0.05) × T_code = 1.6 × T_code ≈ 12.3M tokens** (T_code ≈ 7.7M)

What the pools actually supply, using Soup's approximate token averages:

| Pool | Usable rows | Avg tokens (approx) | Tokens |
|---|---|---|---|
| CIDAR (post-dedup) | 9,965 | ~92 | ~0.92M |
| ArabicaQA (post-dedup, capped at 2/passage) | 2,926 | ~128 | ~0.37M |
| **Total available** | **12,891** | — | **≈ 1.29M** |
| **Required (1.6 × T_code)** | — | — | **≈ 12.3M** |

> **Caveat, stated plainly:** Soup's "Tokens (approx)" is a generic heuristic (~3.3 chars/token here) and **underestimates Arabic**, which Gemma's tokenizer splits more finely. A realistic correction factor of ×1.5–2 raises the supply to ≈ 1.9–2.6M tokens. **This does not change the conclusion** — the shortfall is ~5× even on the most generous assumption. The budget must not be finalised on approximate counts: re-measure with Gemma's own tokenizer on the **formatted** example (`soup data doctor --model …`, per Section 3.3) before selection.

### 4.2 The supply problem — and the four honest options

**`msa_formal` cannot fill an 8% token share from the data now in hand.** It is short by roughly **9×** on measured approximate tokens, or ~5× after the most generous Arabic tokenizer correction. This is a slice-design decision, not something selection can fix, so it is escalated rather than silently absorbed:

| Option | What it means | Cost |
|---|---|---|
| **A. Re-measure first** | Recount with Gemma's tokenizer before deciding anything. | None — **do this first, always.** |
| **B. Widen the pools** | Pull a larger ArabicaQA sample (this is a 10k sample of a bigger corpus) to grow passage *breadth*; unblock PALM via its licence/gating. | Acquisition work; PALM's CC-BY-NC-ND licence may be a hard stop. |
| **C. Lower the share** | Reduce `msa_formal` below 0.08 to what the data can honestly fund, and redistribute to slices with surplus. | Changes the designed mixture; needs sign-off. |
| **D. Repeat examples** | Multiple epochs over the slice to reach the token budget. | **Not recommended** — repetition on a 1,500-passage pool invites memorisation; if used at all, cap at ~2 epochs and record it. |

**Recommended sequence: A → B → C.** Never D alone.

Note the contrast that makes this slice unusual: in **example** terms there is adequate headroom (12,891 available vs. 9,000 needed = **1.43×**), which is exactly why the shortfall has gone unnoticed — `target_examples: 8000` looks comfortably satisfiable. The gap only appears once the share is read in tokens, as `configs/data/msa.yaml` itself specifies (`token_budget.unit: tokens`).

### 4.3 Phase 0 — Apply the passage cap before anything else

`msa_formal` has no categories to tag (`categories: null`), so it skips the tagging phase that SlimOrca needed. Its equivalent prerequisite is the **passage cap**, and the order is not optional:

> **Gap found:** `configs/data/msa.yaml` sets `max_examples_per_variation_group: 2` but **never defines what a variation group is** — there is no `variation_group` key anywhere in the config or the schema. The cap is therefore currently unenforceable: a limit without a group key cannot be applied. This analysis supplies the missing definition below, and it needs to be written into the config.

1. **Dedup first** (0.85, on the trained fields only — Section 3.1).
2. **Then cap at 2 per variation group**, defining the group key as **the SHA-256 of the normalised passage (`input`)** — the definition the config is missing. The id-maps this run produced already carry `passage_sha256`, so the cap becomes reproducible rather than re-derived by eye.
3. **Record both counts.** Caps applied in the wrong order give different answers: capping before dedup would retain near-duplicate pairs *inside* a passage and waste the quota on them.
4. **CIDAR is exempt by construction** — empty `input`, so no group. Document this explicitly in the manifest so it reads as a deliberate decision, not an oversight.

### 4.4 Phase 1 — Fill the budget

1. **Wait for the anchor.** T_code is *measured*, never estimated — no MSA selection starts before the coding slice has been formatted and tokenized with Gemma's tokenizer.
2. **Use the same token metric as every other slice** (assistant tokens preferred; total acceptable if recorded). One metric, one anchor.
3. **Whole examples only, ±2% tolerance.** Any formatted example over ~15,500 Gemma tokens is dropped **whole**, never truncated. No row in either pool is anywhere near this limit.
4. **Stratify by source inside the budget.** The slice defines no categories, so `source` is the stratification axis. Note that `max_share_per_source: 1.0` in `configs/data/msa.yaml` currently **disables** the cap — it was left loose pending exactly this analysis. The data now supports tightening it: the realistic mix is about **77% CIDAR / 23% ArabicaQA** (9,965 : 2,926). Set the cap deliberately rather than leaving it at 1.0, and accept that CIDAR must dominate.
5. **Add a minimum-length floor** (Section 1's quality note) so the 5-token tail cannot consume slots.
6. **Stay deterministic** — fixed order, stable sort by id, seed 42; reruns byte-identical.

### 4.5 Phase 2 — Pack and pad (unchanged; summarized)

Identical to the shared rules in the reference strategy: sort **biggest-first**, keep filling each 16k sequence while a whole example fits, isolate shelf-mates with **block-diagonal attention masks** (verified for Gemma 3's hybrid sliding-window + global attention), and **pad only the final partial sequence** of the slice.

With averages of ~92 and ~128 tokens, this is the most packing-efficient slice in the project — roughly **120–170 examples per 16k sequence**, >99% real data. The entire slice is on the order of ~100 packed sequences, so padding waste is negligible.

### 4.6 Holdout

Reserved **first**, before any selection, token-proportional across the two sources, never trained on, never packed with training data (1,000 rows → `data/processed/msa/holdout.jsonl`).

One slice-specific hazard: **the holdout must split on passages, not rows.** If one ArabicaQA passage lands in both the holdout and training, the holdout is contaminated — the model will have seen that passage's text verbatim. Partition by `passage_sha256`, so every question from a given passage lands entirely on one side. The existing `src/data/split.py` guarantees id-level disjointness only; **passage-level disjointness is a new requirement this analysis introduces.**

### 4.7 Implementation notes

* **`configs/data/msa.yaml`:** tighten `max_share_per_source` from the placeholder `1.0` to the measured mix (Section 4.4); add a minimum-length floor; keep `max_examples_per_variation_group: 2` (the evidence in Section 2 supports it).
* **`src/data/split.py`:** add passage-level (group-aware) splitting for the holdout — Section 4.6.
* **Script validation:** add a Unicode-range Arabic-script check locally; Soup's `langdetect` cannot do it (Section 3.2).
* **`src/data/dedupe.py`:** record the field set alongside every near-duplicate count, and prefer explicit `--field` over Soup's "all text fields" default so results cannot drift with the column layout (Section 3.1).
* **`docs/data/msa_audit.md`:** supersede the ArabicaQA near-duplicate figure — **3,055, not 2,128** — and cite the field set.
* **Manifest:** record T_code, the token metric, per-source token totals and example counts, the dedup field set and threshold, passage-cap counts, the minimum-length drop count, packing efficiency, and the contamination checks still outstanding.

---

## 5. Reproducing This Analysis

Every number above is re-derivable. This is the main thing this document adds over the September audit, whose Soup figures existed only as prose.

```bash
# The full run (14 Soup invocations) inside the pinned container
docker compose run --rm soup-cpu python scripts/soup_analyze_msa.py

# Materialise Soup inputs and print the plan without running Soup
python scripts/soup_analyze_msa.py --dry-run

# Re-capture the real `soup data` flag surface as evidence
docker compose run --rm soup-cpu python scripts/soup_analyze_msa.py --probe-help
```

| Artifact | Path | What it is |
|---|---|---|
| Machine-readable report | `data/processed/msa/soup/soup_report.json` | every command, exit code, duration, parsed numbers, and caveats |
| Run plan | `data/processed/msa/soup/soup_run_plan.json` | the 14 commands with a written justification for each |
| Raw logs | `data/processed/msa/soup/logs/*.log` (14 files) | verbatim stdout/stderr — **the authoritative record** |
| Captured help text | `data/processed/msa/soup/help/*.txt` (18 files) | the real 0.73.3 flag surface, so no flag in the plan is guessed |
| Soup inputs | `data/processed/msa/soup/*_alpaca.jsonl` | strict-alpaca files Soup consumed, each with a recorded SHA-256 |
| Provenance id-maps | `data/processed/msa/soup/*_idmap.jsonl` | dropped columns + `passage_sha256` per row |
| Field-set experiment | `data/processed/msa/soup/arabic_qa_with_meta*.jsonl` | the §3.1 control that reproduced September's −2,128 |

Three design rules make the run trustworthy rather than merely green:

1. **No invented flags.** Only flags read from the captured `--help` output are used. Commands whose surface was not verified are not run.
2. **Raw output is the source of truth.** Parsed numbers in the report each carry a `"parsed": true|false` flag; an unparsed value is reported as `null`, never as a plausible guess. Row counts of files Soup wrote are counted from the files themselves.
3. **Non-destructive.** The September artifacts under `data/processed/msa/` were left untouched; this run writes to the `soup/` subdirectory, which is what made the §3.1 side-by-side comparison possible.

> Note: `data/processed/` is gitignored, so these artifacts are local. Re-run the command above to regenerate them; the inputs are the immutable files in `data/raw/` and the run is deterministic.

---

## Conclusion & Next Steps

Both MSA pools are structurally clean — 20,000 / 20,000 rows valid, one empty field, no encoding problems — and CIDAR is a solid backbone at 9,965 unique, cap-exempt rows. But this slice is **not** the comfortable surplus the retention slices enjoy: ArabicaQA collapses from 10,000 to **2,926** usable rows once genuine redundancy (−3,055) and its 1,500-passage ceiling (−4,019) are both accounted for.

The blocking issue is **supply measured in tokens**: `msa_formal` needs ≈12.3M and has ≈1.3M. Immediate next steps, in order:

1. **Re-measure with Gemma's own tokenizer** (`soup data doctor --model …`) once the tokenizer is available — no budget decision on approximate counts.
2. **Decide the share question** (Section 4.2: widen the pools, or lower the 0.08 share). This needs a decision, not a workaround.
3. **Correct the audit record** — ArabicaQA's near-duplicate count is 3,055 on the trained fields.
4. **Implement the three new requirements** this analysis surfaced: passage-level holdout splitting, a local Arabic-script check, and an explicit dedup field set.
5. **Run the eval-contamination check** as soon as `eval.yaml` declares real sources — it is currently blocked, not passing.
