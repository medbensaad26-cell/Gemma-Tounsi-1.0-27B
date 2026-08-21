# `data/` — datasets, manifests & provenance

This directory owns **how data enters the project**: where every corpus comes from, how it is
licensed, how it is normalized, and which mixture is fed to training. It does not own the training
loop — that belongs to Soup.

> **Status:** structure only. No datasets have been sourced, downloaded, or generated yet.

## Layout

```
data/
├── manifests/          # TRACKED — declarative source & mixture definitions
│   ├── train.yaml      #   Tunisian adaptation training mixture
│   ├── retention.yaml  #   replay/rehearsal mixture (general capabilities)
│   └── eval.yaml       #   evaluation sets — EVALUATION ONLY, never trained on
├── raw/                # IGNORED — untouched downloads, exactly as obtained
├── processed/          # IGNORED — normalized/filtered intermediates
├── splits/             # IGNORED — deterministic train/val splits
├── train.jsonl         # IGNORED — final training file consumed by Soup
└── retention.jsonl     # IGNORED — final replay file consumed by Soup
```

**Only `manifests/*.yaml` is tracked in Git.** Everything else here is large, regenerable, or
license-restricted, and is stored outside version control. Reproducibility comes from the
manifests plus `scripts/prepare_data.sh`, not from committed data.

## Data flow

```
manifests/*.yaml  ──►  raw/  ──►  processed/  ──►  splits/  ──►  train.jsonl
   (tracked)          (as-is)     (normalized)    (deterministic)  retention.jsonl
```

Each stage is append-only with respect to the previous one: `raw/` is never edited in place, so
any preprocessing decision can be revisited without re-downloading.

## Rules

1. **Never commit data.** If a file is bigger than a manifest, it does not belong in Git.
2. **Provenance is mandatory.** Every corpus must be declared in a manifest with its source,
   license, and permitted use before it is used. No undeclared data enters a mixture.
3. **Evaluation data is never training data.** `manifests/eval.yaml` is consumed only by the
   evaluation tracks in `eval/`. TounsiBench and the retention *evaluation* set must never appear
   in `train.jsonl` or `retention.jsonl`.
4. **Retention training ≠ retention evaluation.** `retention.jsonl` is replay/rehearsal data used
   *during* training to preserve general capabilities. The retention *benchmark* lives in
   `eval/retention/` and is drawn from disjoint sources. See rule 3.
5. **Leakage must be checked, not assumed.** Deduplication and contamination checks between
   training mixtures and every evaluation set are part of data preparation, not an afterthought.
6. **Deterministic output.** Given the same manifests and the same sources, preparation must
   produce byte-identical outputs (fixed seeds, sorted inputs, pinned revisions).

## Format

The record schema for `train.jsonl` / `retention.jsonl` is **not yet fixed**: it must match what
Soup expects, which will be confirmed when the Soup environment is validated.

<!-- TODO: after validating Soup, document the exact JSONL record schema here
     (field names, chat/turn structure, handling of system prompts, and how
     Arabic-script vs. Arabizi examples are tagged). -->

## Next steps

- [ ] Identify candidate Derja / Arabizi corpora and verify their licenses.
- [ ] Fill in `manifests/train.yaml`, `manifests/retention.yaml`, `manifests/eval.yaml`.
- [ ] Confirm Soup's expected input format and document the schema above.
- [ ] Implement `scripts/prepare_data.sh` (download → normalize → split → emit JSONL).
- [ ] Implement dedup / contamination checks between training and evaluation data.
- [ ] Record full provenance and preprocessing decisions in `docs/DATA.md`.
