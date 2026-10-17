# Data — sources, provenance & preparation

> **STATUS:** Retention candidate sources have been acquired and audited. Final production
> selection is pending; the pipeline is tested on synthetic fixtures. Other inventory sections
> remain placeholders.

This is the authoritative, human-readable record of every dataset in the project. The machine-
readable counterpart lives in `data/manifests/`; the two must always agree.

## 1. Principles

1. **No undeclared data.** A corpus that is not documented here and declared in a manifest must
   never reach a training file.
2. **Licensing is checked before use**, not after. If permitted use is unclear, the source is not
   used.
3. **Raw data is immutable.** `data/raw/` holds sources exactly as obtained; all cleaning happens
   downstream so decisions can be revised without re-downloading.
4. **Evaluation data is never training data.** See §5.
5. **Reproducibility over convenience.** Pinned revisions, fixed seeds, recorded checksums.

## 2. Data inventory

*TODO: one subsection per source, using this template.*

```
### <source id>
- Type:            hf_dataset | url | local | scraped | synthetic | human-authored
- Location:        <dataset id / URL / path>
- Revision:        <commit SHA / snapshot date>       # must be pinned
- Size:            <records / tokens>
- Script:          derja_arabic | arabizi | mixed
- Domain:          <conversational, news, social, instructions, ...>
- License:         <SPDX id or explicit terms>
- Permitted use:   <why training use is allowed>
- Attribution:     <required credit, if any>
- Collection:      <how it was gathered; consent considerations if applicable>
- Known issues:    <quality, dialect skew, noise, PII risk>
```

### 2.1 Tunisian adaptation data (`data/manifests/train.yaml`)

Candidates for the `arabizi` and `arabic_derja` slices. Revisions are pinned
(commit SHAs verified on 2026-10-17). Franco-Tunisian pools are declared in
§2.4; where a source is used by both slices (Farah21, Linto) each slice takes a
disjoint subset and rows are never counted twice. Licenses are as declared by
the dataset cards; sources marked provisional need a license trace before any
production run. Attribution is required for all CC-BY sources and is recorded in
the model card.

| ID | Location @ revision | Type / size | Script | License | Status |
|----|--------------------|-------------|--------|---------|--------|
| farah21-tunisia-tech-arabizi | `Farah21/Tunisia_Tech` @ `be5ca6c5…8be` | hf_dataset, 5,500 Q&A rows (4,950 train / 550 test), Alpaca-style, Reddit r/Tunisia | mostly English; 165 rows labelled `tunisian_arabizi`, 4 `tunisian_arabic` | CC-BY-4.0 | **Arabizi-labelled train rows only.** `Farah21/reddit-tunisia-qa` is a strict subset (all 4,541 ids) and is the franco source in §2.4; shared rows are used once. Test split excluded from training. Replies are other users' comments, not assistant answers. |
| linto-text-ar-tn-derja | `linagora/linto-dataset-text-ar-tn` @ `25dbb0af…f4b` | hf_dataset, 991,371 raw rows in 20 subsets | Arabic script (77–100% per subset) | Apache-2.0 on card; **provisional** (same subsets appear in a CC-BY-SA repo) | **Per-subset review.** Candidates: Derja_tunsi, HkayetTounsia, KisatiAna, TunBERT, TuDiCOI, HkayetErwi, TunHistoires, Chakhabitt, TRCD, TSAC (toxicity filter), Tweet_TN (PII filter). Excluded: T-HSAB (hate speech), TN_RAP_lyrics (copyright), BRAD (not Tunisian, to verify), CombinedNames (content unclear), Tunisien_Dialect_Summary-llama2 (model-generated). TunSwitch ×2 (775K rows) belongs to §2.4; origin check pending. |
| ctab-arabizi | CTAB, Zenodo record 4781769 (v4, 2021-05-11) | url, 13 files (~306 kB text), one raw Arabizi Facebook message per line | Arabizi | CC-BY-4.0 | **Accepted, very small** (est. ~100K tokens). File names suggest a sample; check tunisiya.org for a larger release. PII filtering required. Message count and md5 check pending acquisition. |

Quarantined / excluded (not declared in `train.yaml`):

| Source | Reason |
|--------|--------|
| `linagora/Tunisian_Derja_Dataset` (CC-BY-SA-4.0) | Same subsets as Linto (identical row counts on 8). Share-alike may conflict with an Apache-2.0 release; needs legal review. |
| `hamzabouajila/tunisian-derja-unified-raw-corpus` (CC-BY-SA-4.0) | Re-upload of Linagora and other datasets; derivative. |
| `tunis-ai/MADAR-TUN` (CC-BY-NC-3.0 on card; paper says CC-BY-SA-4.0) | Non-commercial; conflict unresolved. |
| `chaymafourati/tunizi` (license unknown) | Repo holds a loader script only; Kaggle copy says GPL-2. |
| `khaled123/Tunisian_Dialectic_English_Derja` (RAIL-M) | Use restrictions, mixed provenance. |

Known gaps (drive the rest of Phase 1):
1. **Arabizi is scarce.** Verified real Arabizi is ~165 labelled Farah21 rows
   plus CTAB (~100K tokens estimated), against a slice that needs tens of
   millions of tokens at 35% of the mixture. The slice needs synthetic
   Derja→Arabizi transliteration and authored data.
2. **No assistant-style data.** Candidates are raw comments, tweets and
   stories; schema adapters must build instruction/chat tasks around them.
3. **Safety/privacy.** Reddit and social-media text needs PII and toxicity
   filtering before any training use.

### 2.2 Replay / rehearsal data (`data/manifests/retention.yaml`)

This is **training** data, disjoint from retention evaluation. Pinned revisions, licenses and
local paths are recorded in `data/manifests/retention.yaml`:
- **MetaMathQA:** mathematics, 395,000 candidate rows, MIT.
- **Code-Feedback:** coding, 66,383 candidate rows, Apache-2.0.
- **SlimOrca:** reasoning, general instruction and knowledge QA, 517,982 candidate rows, MIT.

Production follows `docs/data/retention_mohamed.md` §3 and `docs/data/retention_haithem.md` §2,
configured in `configs/data/retention.yaml`:
1. Deduplicate and tag candidates, then reserve a token-proportional holdout first
   (25/25/20/15/15 across coding/math/reasoning/instruction/knowledge; absolute size still to set).
2. Select 5,000 diverse coding conversations and measure `T_code` using Gemma's tokenizer
   with the chat template. Prefer assistant tokens; total tokens are an acceptable fallback
   used consistently across all categories. Record both.
3. Fill coding/math/reasoning/instruction/knowledge budgets at **1.0/1.0/0.8/0.6/0.6 × T_code**
   within ±2%, with whole examples and source-specific diversity quotas. There is no fixed
   20k-example production target.
4. Apply math type/seed caps and SlimOrca template caps/tagging audits. Keep Latin-script
   translation exercises with English instructions; document the decision. Near-deduplicate
   at 0.85 and cross-check SlimOrca reasoning against math and all selections against evaluation.
5. Drop examples above 15,500 formatted Gemma tokens; pack biggest-first into 16,384-token
   sequences with isolated attention and reset position IDs, verified for Gemma 3 hybrid
   attention. Never pack holdout with training. Record packing efficiency and padding waste.

Retention is **20% of training tokens** (`configs/data/mixture.yaml`). Token-aware selection,
blending and packing are not yet implemented; legacy count fields support synthetic tests only.

### 2.3 Evaluation data (`data/manifests/eval.yaml`)

*TODO — TounsiBench, Arabizi benchmark, retention benchmark. **Evaluation only.***

### 2.4 French/Tunisian code-switching data (`data/manifests/franco_tunisian.yaml`)

This is **training** data for the `franco_tunisian` slice (12% of the training
cocktail per `configs/data/mixture.yaml`). Five candidate pools are declared
in `data/manifests/franco_tunisian.yaml`; none have been acquired or selected
yet. See that manifest for per-source details (license, row count, notes).

| # | Source | Size (reported) | Code-switching | License | Status |
|---|--------|-----------------|----------------|---------|--------|
| 1 | Farah21/reddit-tunisia-qa | ~4,541–5,500 QA pairs | `mixed` / `french` dialect labels | TBD — verify | Identified |
| 2 | Syrinesmati/tunisian-question-response-dataset | ~31,000 QA pairs | French CS present, not exhaustive | CC-BY-4.0 | Identified |
| 3 | linagora/Tunisian_Derja_Dataset (TunSwitchCodeSwitching) | ~394,160 lines | Dedicated CS config | CC-BY-SA-4.0 | Identified |
| 4 | hamzabouajila/tunisian-derja-unified-raw-corpus | ~802,659 examples | French/English CS present | CC-BY-SA-4.0 | Identified |
| 5 | linagora/linto-dataset-text-ar-tn | ~4.5M lines (65K CS) | Explicit CS category | CC-BY-4.0 | Identified |

License note: sources #3 and #4 are CC-BY-SA-4.0 (share-alike) and require
legal review before use in a model released under Apache-2.0. Sources #2 and #5
are CC-BY-4.0 (attribution only; safe for Apache-2.0 weight release). Source #1
license must be verified before acquisition.

Run `scripts/acquire_franco_tunisian_candidate_data.py` to download the pools
into `data/raw/`, then pin their revisions in `data/manifests/franco_tunisian.yaml`.


## 3. Preparation pipeline

```
manifests/*.yaml  ──►  data/raw/  ──►  data/processed/  ──►  data/splits/  ──►  *.jsonl
```

Executed by `scripts/prepare_data.sh` for synthetic fixtures; production corpus adapters and
token-aware selection/packing remain pending.

*TODO: document each stage as it is implemented.*

- [ ] **Fetch** — idempotent downloads into `data/raw/`, checksum-verified.
- [ ] **Normalize** — Unicode form for Arabic script; tatweel/diacritics/punctuation handling;
      Arabizi casing and digit-substitution normalization.
- [ ] **Filter** — length bounds, language identification, quality heuristics, PII handling.
- [ ] **Deduplicate** — exact and near-duplicate, within and across sources.
- [ ] **Split** — deterministic, seeded.
- [ ] **Emit** — `data/train.jsonl`, `data/retention.jsonl`.

## 4. Record schema

**Not yet fixed** — it must match what Soup expects, which will be confirmed once the Soup
environment is validated.

*TODO: document field names, the chat/turn structure, system-prompt handling, and how
Arabic-script vs. Arabizi examples are tagged.*

## 5. Contamination policy

- Every training mixture is cross-checked against **every** evaluation set.
- Any overlap **aborts** data preparation. Overlap is treated as a defect in the mixture
  definition, not something to silently filter away.
- The retention replay data and the retention evaluation set are disjoint at the source level,
  and this is verified again at the record level.

*TODO: document the matching method used (exact hash, normalized hash, n-gram overlap threshold)
and the exact commands to reproduce the check.*

## 6. Storage & versioning

- Nothing under `data/` is committed except `data/manifests/*.yaml`.
- Large files live outside Git; see `.gitignore`.
- Reproducibility comes from *manifests + pinned revisions + preparation script*, not from
  committed data.

*TODO: decide where prepared datasets are archived for release reproducibility (private bucket,
Hub dataset repo, etc.) and record checksums here.*
