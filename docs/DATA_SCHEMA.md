# Canonical Data Schema

The record format every internal training example in **Gemma Tounsi 1.0 27B** must
conform to, from raw ingestion through to the final Soup-compatible export.

The machine-readable source of truth is [`configs/data/schema.yaml`](../configs/data/schema.yaml).
The validator reads that file directly, so the accepted values documented here and
the enforcement in code cannot drift apart.

This schema is **dataset-independent**. Adapters for external corpora are added
later and must normalize *into* this schema — never the other way around.

---

## 1. Record structure

One example = one JSON object on one line of a UTF-8 JSONL file.

```json
{
  "id": "authored-arabizi-000123",
  "messages": [
    { "role": "user", "content": "3andi 12 dinar w zedt 8, 9adech el majmou3?" },
    { "role": "assistant", "content": "El majmou3 howa 20 dinar." }
  ],
  "category": "mathematics",
  "source": "authored_arabizi_batch1",
  "language": "arabizi",
  "subcategory": "arithmetic_word_problem",
  "script": "latin",
  "code_switching": false,
  "difficulty": "easy",
  "quality": { "reviewed": true, "native_authored": true },
  "variation_group": "arith-020"
}
```

### Required fields

| Field | Type | Rules |
|---|---|---|
| `id` | string | Non-empty, **globally unique** within a file. Matches `^[A-Za-z0-9._:-]+$` (advisory). Duplicates are an **error**, not a warning. |
| `messages` | array | ≥ 2 turns, valid roles, correct ordering, non-empty content. See §2. |
| `category` | string | One of the five values in §3. |
| `source` | string | Non-empty provenance tag. Free-form, but must identify where the example came from. |
| `language` | string | One of the five values in §3. |

### Optional fields

| Field | Type | Purpose |
|---|---|---|
| `subcategory` | string | Finer label beneath `category`. Free-form, but be consistent. |
| `script` | string | Writing system: `latin`, `arabic`, `mixed`. Strongly recommended for Tunisian slices. |
| `code_switching` | boolean | Does the example deliberately mix languages/scripts? |
| `difficulty` | string | `easy`, `medium`, `hard`. |
| `quality` | object | Boolean flags: `reviewed`, `native_authored`, `synthetic`, `flagged`. |
| `variation_group` | string | Links orthographic/paraphrase variants of the same underlying item, so variants can be kept together across a split. |

Any field **not** listed above is rejected (`unknown_field`). This is deliberate:
silent typos like `catagory` would otherwise pass unnoticed. Extending the schema
means editing `configs/data/schema.yaml`, which is a reviewable change.

---

## 2. `messages` rules

Turns model a chat conversation:

```json
{ "role": "user", "content": "..." }
```

| Rule | Detail |
|---|---|
| Minimum turns | 2 — at least one `user` and one `assistant` |
| Allowed roles | `system`, `user`, `assistant` |
| System turn | Optional; may appear **only** as the first turn, at most once |
| First non-system turn | Must be `user` |
| Alternation | `user` and `assistant` must strictly alternate |
| Final turn | Must be `assistant` (it is the training target) |
| Content | Every turn's `content` must be a non-empty, non-whitespace string |

A conversation that ends on `user` has no target to learn from, and one that starts
with `assistant` teaches the model to answer before being asked. Both are errors.

---

## 3. Accepted values

### `category` — technical/functional tagging

| Value | Meaning |
|---|---|
| `mathematics` | Arithmetic, algebra, word problems, quantitative reasoning |
| `reasoning` | Logic, inference, multi-step deduction, planning |
| `coding` | Code generation, explanation, debugging |
| `general_instruction` | Open-ended instruction following, summarizing, rewriting, formatting |
| `knowledge_qa` | Factual question answering, world/cultural knowledge |

**Technical categories** — `mathematics`, `reasoning`, `coding` — are the three that
count toward the cross-cutting quota in §5.

> The retention configuration uses the label `instruction_following`; it maps to the
> canonical `general_instruction` via `category_aliases` in
> [`configs/data/retention.yaml`](../configs/data/retention.yaml). The canonical name
> is the one that appears in records.

### `language`

| Value | Meaning |
|---|---|
| `arabizi` | Latin-script Tunisian Derja (`3`, `7`, `9`, …) |
| `ar` | Arabic script — Derja *or* MSA, disambiguated by slice/`script` |
| `fr` | French, including Franco-Tunisian code-switching |
| `en` | English — primarily the retention slice |
| `mixed` | Genuinely mixed with no dominant language |

### `script`

`latin` · `arabic` · `mixed`

### `difficulty`

`easy` · `medium` · `hard`

### `quality` flags

| Flag | Meaning |
|---|---|
| `reviewed` | A human reviewed this example |
| `native_authored` | Written by a native Tunisian contributor |
| `synthetic` | Machine-generated (e.g. the test fixtures in `data/synthetic/`) |
| `flagged` | Suspect; excluded from final mixtures |

Absence of a flag means *unknown*, not *false*.

---

## 4. Project slices

Every processed example belongs to exactly one slice for mixture accounting.
Shares are defined in [`configs/data/mixture.yaml`](../configs/data/mixture.yaml).

| Slice | Share | Purpose |
|---|---|---|
| `arabizi` | 0.35 | Latin-script Tunisian Derja |
| `arabic_derja` | 0.25 | Arabic-script Tunisian Derja |
| `franco_tunisian` | 0.12 | French/Tunisian code-switching |
| `msa_formal` | 0.08 | **Formal register coverage** |
| `retention` | 0.20 | **English capability preservation** |

### Two distinctions that matter

> **`retention` = English capability preservation.**
> It exists so that Tunisian adaptation does not degrade the base model's original
> general abilities. It is **primarily English**, it does **not** require Tunisian
> output, and it is **not** a Tunisian-language dataset. Never describe retention as
> Tunisian adaptation.

> **`msa_formal` = formal/register coverage.**
> Modern Standard Arabic teaches the model to shift register. This is a **separate
> purpose** from retention and is **never counted as retention**, even though both
> are non-Derja.

The mixture validator enforces both statements: retention must be declared
`english_capability_preservation`, `msa_formal` must be
`formal_register_coverage`, and non-English records in the retention slice are a
hard failure.

---

## 5. Cross-cutting technical quota

At least **20%** of examples in **both** the `arabizi` and `arabic_derja` slices must
be in a technical category (`mathematics`, `reasoning`, `coding`).

The objective is *"make Gemma's existing capabilities work naturally in Tunisian."*
A model that chats fluently in Derja but cannot do arithmetic in it has not met that
objective. The quota is checked as `>=` (exactly 20.0% passes) and applies to the
Tunisian slices only — `franco_tunisian`, `msa_formal` and `retention` have no
technical minimum.

---

## 6. Data lifecycle: training vs. development vs. evaluation

Three kinds of data, kept strictly apart.

| Kind | Location | Used for | Rule |
|---|---|---|---|
| **Training** | `data/processed/**/train.jsonl` → final export | Gradient updates | The only data the model learns from |
| **Development (holdout)** | `data/processed/**/holdout.jsonl` | Tuning, monitoring, regression checks during development | **Never trained on.** Reserved *before* training selection |
| **Final evaluation** | `eval/**` (per `data/manifests/eval.yaml`) | Reported benchmark results | Isolated from both of the above. Never inspected during iteration |

Guarantees enforced in code:

- The retention holdout is reserved **before** training selection, so a record can
  never reach both partitions.
- `SplitResult` raises on any id present in both sides.
- The mixture validator fails when a holdout/eval id appears in a training slice.
- Splits are deterministic (fixed seed + stable hashing), so the partition is
  reproducible and auditable rather than re-rolled per run.

Retention *training* data and the retention *evaluation* benchmark are different
things drawn from disjoint sources. Confusing them invalidates the central claim
of the project — that capabilities were preserved.

---

## 7. Validation rules

`python -m src.data.validate FILE...`

| Code | Meaning |
|---|---|
| `invalid_json` | Line is not parsable JSON |
| `invalid_type` | Line parses but is not a JSON object |
| `missing_field` | A required field is absent |
| `empty_id` | `id` is empty or not a string |
| `duplicate_id` | `id` already seen in this file |
| `empty_messages` | `messages` is absent, not a list, or empty |
| `too_few_turns` | Fewer than the minimum number of turns |
| `invalid_role` | Role outside `system`/`user`/`assistant` |
| `invalid_ordering` | System not first, wrong first/last role, or broken alternation |
| `empty_content` | A turn's content is empty or whitespace |
| `invalid_category` | `category` outside the accepted list |
| `invalid_language` | `language` outside the accepted list |
| `invalid_script` | `script` outside the accepted list |
| `invalid_difficulty` | `difficulty` outside the accepted list |
| `invalid_quality` | `quality` is not an object, or has unknown/non-boolean flags |
| `invalid_type_field` | A field has the wrong JSON type |
| `unknown_field` | Field not part of the canonical schema |

Behaviour:

- Every error carries the **line number**, the offending **field**, and the record
  **id** where known.
- A record's defects are **all** reported, not just the first.
- Bad records are **never silently dropped**. The validator reports them and exits
  non-zero; the caller decides what to do.

---

## 8. Final export format

Training-time consumption is Soup's job, so the canonical record is converted to
the **ShareGPT** conversation format that Soup 0.73.3 accepts:

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

| Canonical role | ShareGPT `from` |
|---|---|
| `system` | `system` |
| `user` | `human` |
| `assistant` | `gpt` |

Project metadata (`category`, `language`, `quality`, …) is dropped by default since
Soup does not consume it; pass `--keep-metadata` to retain `id`/`category`/`source`/
`language` for debugging. ShareGPT is chosen over `alpaca` because it represents
multi-turn conversations without loss.

---

## 9. Related files

| Path | Role |
|---|---|
| [`configs/data/schema.yaml`](../configs/data/schema.yaml) | Machine-readable schema (source of truth) |
| [`configs/data/retention.yaml`](../configs/data/retention.yaml) | Retention targets & selection policy |
| [`configs/data/mixture.yaml`](../configs/data/mixture.yaml) | Slice shares & cross-cutting quotas |
| `src/data/schema.py` | Schema loading + per-record validation |
| `src/data/validate.py` | File/dataset-level validation CLI |
| `src/data/stats.py` | Dataset statistics |
| `src/data/dedupe.py` | Deduplication (delegates near-dups to Soup) |
| `src/data/split.py` | Deterministic train/holdout splitting |
| `src/data/retention.py` | Retention selection |
| `src/data/mixture.py` | Mixture validation |
| `src/data/export.py` | Canonical → Soup ShareGPT export |
| [`data/README.md`](../data/README.md) | Directory layout & provenance rules |
