# Retention — general-capability regression guard

Measures what adaptation **costs**. Fine-tuning a 27B model on a low-resource dialect can quietly
damage its general instruction-following, reasoning, multilingual, and safety behaviour. This track
exists to make that damage measurable and to block a release that trades too much away.

> **Status:** placeholder. No suites selected, no thresholds set, no results exist.

## ⚠️ Two different things called "retention"

| | Retention **training** | Retention **evaluation** |
| --- | --- | --- |
| Purpose | replay/rehearsal *during* training | regression measurement *after* training |
| Declared in | `data/manifests/retention.yaml` | `data/manifests/eval.yaml` |
| Artifact | `data/retention.jsonl` | this directory |
| May overlap? | **No — must be disjoint** | **No — must be disjoint** |

If replay data leaked into this benchmark, the model would be graded on examples it was trained on
and retention would look perfect regardless of actual forgetting. Disjointness is enforced at the
source level first, then verified with a record-level contamination check.

## Scope (to be defined)

Capabilities to guard, kept aligned 1:1 with `capabilities_to_preserve` in
`data/manifests/retention.yaml`:

- instruction following and output-format compliance
- general reasoning
- multilingual ability (notably English, French, MSA)
- factual knowledge
- safety behaviour and refusal quality
- optional: code, long-context handling

<!-- TODO: select concrete public suites per capability, pin their revisions,
     and record why each was chosen. Prefer suites with a documented protocol
     over ad-hoc prompt sets. -->

## Reporting rule

Retention is always reported as a **delta against the unmodified base Gemma 3 27B**, measured with
identical prompts, decoding settings, and harness version. An absolute score alone says nothing
about forgetting.

<!-- TODO: define the maximum acceptable regression per capability and make it a
     release gate in scripts/release.sh. Do not invent thresholds before real
     baseline numbers exist. -->

## Planned contents of this directory

| Path | Purpose | Git |
| --- | --- | --- |
| `tasks/` | Suite definitions & prompt templates | tracked |
| `scoring/` | Metric implementations | tracked |
| `data/` | Benchmark items | ignored |
| `predictions/` | Raw model outputs | ignored |
| `results/` | Scores, deltas & reports | ignored |

## Next steps

- [ ] Select suites per guarded capability and pin revisions.
- [ ] Verify source-level disjointness from retention replay data.
- [ ] Implement the record-level contamination check.
- [ ] Produce the base-model baseline before any adapted-model run.
- [ ] Set regression thresholds and wire them into the release gate.
