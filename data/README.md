# `data/` — datasets, manifests & provenance

This directory owns **how data enters the project**: where every corpus comes from, how it is
licensed, how it is normalized, and which mixture is fed to training. It does not own the training
loop — that belongs to Soup.

> **Status:** the data-engineering pipeline is implemented and proven end to end against
> **synthetic fixtures only**. No real corpus has been downloaded yet. Dataset-specific
> adapters arrive in Task 4B.

## Layout

```
data/
├── manifests/          # TRACKED — declarative source & mixture definitions
│   ├── train.yaml      #   Tunisian adaptation training mixture
│   ├── retention.yaml  #   replay/rehearsal mixture (general capabilities)
│   ├── msa.yaml        #   MSA formal-register candidate sources (msa_formal slice)
│   ├── eval.yaml       #   evaluation sets — EVALUATION ONLY, never trained on
│   └── retention/      #   GENERATED — selection manifests (provenance receipts)
├── raw/                # IGNORED — untouched downloads, exactly as obtained
├── authored/           # IGNORED content — human-written Tunisian data
│   ├── arabizi/        #   Latin-script Derja
│   ├── derja/          #   Arabic-script Derja
│   ├── franco_tunisian/#   French/Tunisian code-switching
│   ├── msa/            #   Modern Standard Arabic (formal register)
│   └── retention/      #   English capability-preservation material
├── processed/          # IGNORED — normalized/filtered/selected intermediates
├── splits/             # IGNORED — deterministic train/val splits
├── synthetic/          # TRACKED — tiny synthetic fixtures for testing the pipeline
│   ├── generate.py     #   deterministic generator (regenerates byte-identically)
│   ├── raw/            #   synthetic canonical JSONL inputs
│   └── expected/       #   expected counts, asserted by the test suite
├── train.jsonl         # IGNORED — final training file consumed by Soup
└── retention.jsonl     # IGNORED — final replay file consumed by Soup
```

**Tracked:** `manifests/*.yaml`, `synthetic/**`, and this README. Everything else is large,
regenerable, or license-restricted, and lives outside version control. Reproducibility comes
from the manifests plus `scripts/prepare_data.sh`, not from committed data.

## The four kinds of data here

### 1. Raw — external, never committed
Downloads exactly as obtained, treated as **immutable** once fetched. Never edited in place, so
any preprocessing decision can be revisited without re-downloading. Nothing here is committed:
licenses often forbid redistribution and the files are large.

### 2. Authored — written by native Tunisian contributors
The one thing that cannot be downloaded. Real Derja, Arabizi and Franco-Tunisian data written by
people who actually speak it, and the primary defense against fluent-sounding but unnatural output.
Content stays gitignored until reviewed and explicitly added (`git add -f`), so contributor work is
never committed by accident.

### 3. Processed — generated, reproducible
Output of `scripts/prepare_data.sh`: normalized, validated, deduplicated, categorized, selected and
split. **Never edit by hand.** If something here is wrong, fix the pipeline and regenerate — a
hand-patched file is unreproducible and silently invalidates every downstream claim.

### 4. Synthetic — fixtures, never training data
Small, deterministic, machine-generated records used *only* to test the pipeline. Every record is
tagged `"quality": {"synthetic": true}` and sourced as `synthetic_*`, so it is greppable and cannot
be mistaken for real data. It contains deliberate defects — duplicates, malformed rows, invalid
categories, a below-quota slice — because a pipeline that has never rejected anything has not been
tested.

## Manifests record provenance

`manifests/*.yaml` are the **tracked declaration** of what may enter the project: source, revision,
license, permitted use. No undeclared corpus enters a mixture.

`manifests/retention/` holds **generated** selection manifests — receipts that record, for a given
run, the seed, the config used, per-category counts, source diversity and the exact ids selected.
Together they answer "where did this training file come from?" without needing the data itself.

## Rules

1. **Never commit real data.** If a file is bigger than a manifest, it does not belong in Git. The
   only exception is `data/synthetic/`, which is tiny, generated, and required by the tests.
2. **Provenance is mandatory.** Every corpus must be declared in a manifest with its source,
   license, and permitted use *before* it is used.
3. **Evaluation data is never training data.** `manifests/eval.yaml` is consumed only by the
   evaluation tracks in `eval/`. TounsiBench and the retention *evaluation* set must never appear
   in `train.jsonl` or `retention.jsonl`. Evaluation data stays isolated — it is not inspected
   during development iteration.
4. **Retention training ≠ retention evaluation.** `retention.jsonl` is replay data used *during*
   training. The retention *benchmark* lives in `eval/retention/` and is drawn from disjoint
   sources. See rule 3.
5. **Holdout is never trained on.** The holdout is reserved *before* training selection, and the
   split logic raises if any id appears on both sides.
6. **Leakage must be checked, not assumed.** Deduplication and contamination checks are part of
   data preparation, not an afterthought. On overlap the pipeline **aborts** — it never silently
   drops the offending rows.
7. **Deterministic output.** Same manifests + same pinned sources ⇒ byte-identical outputs (fixed
   seeds, sorted inputs, pinned revisions).

## What retention and MSA actually are

> **`retention/` = English capability preservation.**
> It exists so Tunisian adaptation does not degrade Gemma's original general abilities
> (mathematics, coding, reasoning, instruction following, knowledge QA). It is **primarily
> English** and does **not** require Tunisian output. It is *not* a Tunisian-language dataset.

> **`msa/` = formal/register coverage.**
> Modern Standard Arabic teaches register-shifting into formal Arabic. This is a **separate
> purpose** from retention, and MSA is **never counted as retention**.

## Data flow

```
manifests/*.yaml ─┐
authored/         ├─► raw/ ─► processed/ ─► splits/ ─► train.jsonl
external sources ─┘                                    retention.jsonl
```

Implemented as: normalize → validate → deduplicate → categorize → quality filter →
balanced selection → train/holdout split → mixture validation → Soup-compatible JSONL.
Generic operations (dedup, filtering, sampling, splitting, format validation) are delegated to
Soup; this repository implements only the Gemma-Tounsi-specific logic (canonical schema, slice
definitions, technical quotas, retention selection policy).

## Format

Internally, every example is a canonical JSONL record — see **[`docs/DATA_SCHEMA.md`](../docs/DATA_SCHEMA.md)**
for the full specification and **[`configs/data/schema.yaml`](../configs/data/schema.yaml)** for the
machine-readable source of truth. The final export converts these to the **ShareGPT** format that
Soup 0.73.3 consumes (`{"conversations": [{"from": "human", "value": "..."}, ...]}`).

## Usage

```bash
# run the whole pipeline over synthetic fixtures (no downloads, no training)
./scripts/prepare_data.sh

# same, inside the pinned Soup container
./scripts/prepare_data.sh --in-docker

# individual stages
python -m src.data.validate data/synthetic/raw/arabizi.jsonl
python -m src.data.stats    data/synthetic/raw/*.jsonl
```

## Next steps

- [ ] Identify candidate Derja / Arabizi corpora and verify their licenses.
- [ ] Fill in `manifests/train.yaml`, `manifests/retention.yaml`, `manifests/eval.yaml`.
- [ ] Write per-dataset adapters that normalize external schemas into the canonical schema.
- [ ] Begin collecting authored Tunisian data from native contributors.
- [ ] Record full provenance and preprocessing decisions in `docs/DATA.md`.
