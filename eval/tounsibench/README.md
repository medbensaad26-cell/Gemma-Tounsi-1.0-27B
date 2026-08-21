# TounsiBench — Tunisian Arabic / Derja (Arabic script)

Primary quality benchmark for Gemma Tounsi 1.0: does the model actually understand and produce
natural Tunisian Derja written in Arabic script?

> **Status:** placeholder. No tasks, data, or scores defined yet.

## ⚠️ Evaluation only

TounsiBench is **evaluation-only**. It must never be used as training data, must never be
referenced by `soup.yaml`, `data/manifests/train.yaml`, or `data/manifests/retention.yaml`, and
must never be reachable by any Soup run. Any overlap between TounsiBench and a training mixture is
a bug that must fail the data pipeline.

## Scope (to be defined)

Intended coverage, to be finalized as a task inventory in `data/manifests/eval.yaml`:

- comprehension of Derja input
- generation of natural, non-MSA-drifting Derja output
- instruction following in Derja
- dialect authenticity (Tunisian-specific lexicon, not generic Maghrebi or MSA)
- code-switching with French/MSA where natural
- register and politeness handling
- cultural/local knowledge grounded in Tunisia

<!-- TODO: turn the list above into concrete tasks with sizes, input/output
     formats, and per-task metrics. Distinguish automatically scorable tasks
     from ones needing human or model-as-judge scoring, and state the
     inter-annotator agreement protocol for human-scored tasks. -->

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
- [ ] Source or author benchmark items; record provenance in `data/manifests/eval.yaml`.
- [ ] Define metrics and the judging protocol (automatic vs. human vs. LLM-judge).
- [ ] Define the base Gemma 3 27B baseline run for comparison.
- [ ] Add a contamination check against all training mixtures.
