# Evaluation — protocol & reporting

> **STATUS: PLACEHOLDER.** No benchmarks have been defined and no evaluations have been run.
> **This document contains no results and must not be filled with estimated or expected numbers.**

## 1. Why three tracks

Adapting a 27B model to a low-resource dialect creates a trade-off: dialect gains can come at the
cost of general capability. A single aggregate score hides that. So the model is measured on three
independent tracks, reported separately:

| Track | Directory | Question it answers |
| --- | --- | --- |
| **TounsiBench** | `eval/tounsibench/` | Does it handle real Derja in Arabic script? |
| **Arabizi** | `eval/arabizi/` | Does it handle Latin-script Tunisian? |
| **Retention** | `eval/retention/` | What did adaptation break? |

Scores from these tracks are never combined into one headline number.

## 2. Hard rules

1. **Evaluation-only data.** Every set used here is declared in `data/manifests/eval.yaml` and must
   never appear in `data/train.jsonl` or `data/retention.jsonl`. TounsiBench in particular must
   never be trainable.
2. **Retention eval ≠ retention replay.** Replay data used during training
   (`data/manifests/retention.yaml`) is disjoint from the retention benchmark. Otherwise the
   retention number measures memorization, not preservation.
3. **Contamination aborts.** Detected overlap is a pipeline failure, not something to filter out.
4. **Baseline required.** Every score is reported next to the same measurement on unmodified base
   Gemma 3 27B, using identical prompts, decoding settings, seeds, and harness version.
5. **No fabricated numbers.** A result exists only if a reproducible run produced it.

## 3. Protocol

*TODO — define before the first evaluation run so results are comparable across runs.*

- [ ] Prompt templates and chat formatting per track
- [ ] Decoding settings (temperature, top-p, max tokens) — fixed and recorded
- [ ] Seeds and number of repetitions
- [ ] Scoring method per task: automatic metric / human rating / model-as-judge
- [ ] If model-as-judge: judge model, revision, rubric, and its known biases
- [ ] If human rating: annotator guidelines, number of annotators, agreement measure
- [ ] Handling of refusals, empty outputs, and format violations

## 4. Metrics

*TODO per track. Metrics must be chosen before results are seen, to prevent post-hoc selection of
whichever metric looks best.*

| Track | Metric(s) | Notes |
| --- | --- | --- |
| TounsiBench | *TODO* | must penalize MSA drift, not just token overlap |
| Arabizi | *TODO* | must tolerate legitimate orthographic variation |
| Retention | *TODO* | reported as delta vs. base model, per capability |

## 5. Retention gate

Retention is a **release gate**, not just a reported figure: a regression beyond the agreed
threshold blocks the release (enforced in `scripts/release.sh`).

*TODO: set per-capability maximum acceptable regression. Thresholds must be decided after real
baseline numbers exist — not invented now.*

## 6. Reporting format

Every report must include:

- [ ] model under test (adapter/merged path + revision)
- [ ] base model identifier + revision
- [ ] this repository's git commit and the harness version
- [ ] decoding settings and seed
- [ ] per-task scores, not only aggregates
- [ ] deltas vs. the base model
- [ ] number of items evaluated per task
- [ ] known caveats

## 7. Reproducing an evaluation

```bash
# Not yet implemented
scripts/evaluate.sh --track all --model <path>
```

Predictions and results are generated artifacts: they are gitignored and stored outside Git. Only
task definitions and scoring logic are tracked.
