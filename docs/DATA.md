# Data — sources, provenance & preparation

> **STATUS: PLACEHOLDER.** No datasets have been selected, licensed, downloaded, or processed.
> This document defines *what must be recorded* before any data is used.

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

*TODO — Derja (Arabic script) and Arabizi sources.*

### 2.2 Replay / rehearsal data (`data/manifests/retention.yaml`)

*TODO — general-capability data replayed during training. This is TRAINING data; it must be
disjoint from the retention evaluation set.*

### 2.3 Evaluation data (`data/manifests/eval.yaml`)

*TODO — TounsiBench, Arabizi benchmark, retention benchmark. **Evaluation only.***

## 3. Preparation pipeline

```
manifests/*.yaml  ──►  data/raw/  ──►  data/processed/  ──►  data/splits/  ──►  *.jsonl
```

Executed by `scripts/prepare_data.sh` (not yet implemented).

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
