# Arabizi — Latin-script Tunisian benchmark

Dedicated track for Tunisian written in Latin characters with digit substitutions (`3` for ع,
`7` for ح, `9` for ق, …). This is how a large share of everyday Tunisian text is actually written,
and it behaves differently enough from Arabic-script Derja to deserve its own benchmark rather
than a subset of TounsiBench.

> **Status:** placeholder. No tasks, data, or scores defined yet.

## ⚠️ Evaluation only

This benchmark is **evaluation-only**. It must never enter a training mixture or be referenced by
any Soup training configuration.

## Why it is separate

Arabizi has no standardized orthography: the same word can be spelled many ways by different
writers, and even inconsistently by the same writer. Measuring it separately keeps two distinct
failure modes visible:

- the model understands Tunisian but only in Arabic script
- the model handles Arabizi surface forms but loses meaning

## Scope (to be defined)

- comprehension of Arabizi input
- generation of natural Arabizi output
- robustness to orthographic variation (spelling variants of the same word)
- digit-letter substitution handling
- code-switching (Arabizi ↔ French ↔ English ↔ MSA)
- optional: transliteration consistency between Arabizi and Arabic script

<!-- TODO: turn the list above into concrete tasks with sizes and metrics.
     Decide explicitly whether transliteration is scored as its own task and,
     if so, how a "correct" transliteration is defined given the absence of a
     standard orthography. -->

## Planned contents of this directory

| Path | Purpose | Git |
| --- | --- | --- |
| `tasks/` | Task definitions & prompt templates | tracked |
| `scoring/` | Metric implementations | tracked |
| `data/` | Benchmark items | ignored |
| `predictions/` | Raw model outputs | ignored |
| `results/` | Scores & reports | ignored |

## Next steps

- [ ] Decide task inventory and answer formats.
- [ ] Collect orthographic-variation cases; record provenance in `data/manifests/eval.yaml`.
- [ ] Define metrics tolerant of legitimate spelling variation.
- [ ] Define the base Gemma 3 27B baseline run for comparison.
- [ ] Add a contamination check against all training mixtures.
