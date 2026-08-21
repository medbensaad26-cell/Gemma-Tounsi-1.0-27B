# `eval/` — evaluation tracks

This directory owns **how the model is measured**. It holds benchmark *definitions*, task
specifications, and scoring logic. It does not hold bulk evaluation data or results — those are
generated artifacts and are gitignored.

> **Status:** structure only. No benchmarks defined, no runs performed, no results exist.

## The three tracks

| Track | Directory | Measures |
| --- | --- | --- |
| **TounsiBench** | `tounsibench/` | Tunisian Arabic / Derja in Arabic script |
| **Arabizi** | `arabizi/` | Latin-script Tunisian, incl. code-switching |
| **Retention** | `retention/` | Preservation of Gemma's general capabilities |

The tracks are deliberately independent: each has its own data, tasks, and metrics, and each can be
run on its own. A Tunisian gain that costs general capability must be visible, so TounsiBench and
Arabizi scores are never aggregated with retention scores into a single number.

## Non-negotiable rules

1. **Evaluation-only.** Everything referenced from `eval/` — including TounsiBench — is
   evaluation-only. It must never be reachable from any training configuration, mixture, or
   manifest used by Soup. See `data/manifests/eval.yaml`.
2. **Retention eval ≠ retention training.** Replay/rehearsal data used *during* training is
   declared in `data/manifests/retention.yaml` and lives in `data/retention.jsonl`. The retention
   *benchmark* lives here and must be drawn from disjoint sources. Measuring the model on its own
   replay data would make the retention number meaningless.
3. **Contamination is a failure, not a filter.** If an evaluation item is found in a training
   mixture, preparation must fail loudly so the mixture can be fixed.
4. **Baselines are mandatory.** Every reported score is paired with the same measurement on the
   unmodified base Gemma 3 27B, at pinned revisions and identical decoding settings. A score
   without its baseline is not a result.
5. **No fabricated results.** Numbers appear in this repository only after a real, reproducible
   run.

## What is tracked vs. ignored

| Tracked | Ignored |
| --- | --- |
| Task/benchmark definitions | Bulk evaluation data (`eval/**/data/`) |
| Prompt templates & scoring rules | Model predictions (`eval/**/predictions/`) |
| Per-track README documentation | Scores & reports (`eval/**/results/`) |

## Next steps

- [ ] Define the TounsiBench task inventory (categories, sizes, answer formats).
- [ ] Define the Arabizi benchmark (orthographic variation, code-switching).
- [ ] Select retention suites and verify disjointness from replay data.
- [ ] Specify metrics, decoding settings, and the judging protocol per track.
- [ ] Implement `scripts/evaluate.sh` as a thin runner over these tracks.
- [ ] Write up the full protocol in `docs/EVALUATION.md`.
