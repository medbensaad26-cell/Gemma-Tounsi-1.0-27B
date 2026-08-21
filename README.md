# Gemma Tounsi 1.0 — 27B

Tunisian-adapted **Gemma 3 27B**, trained with **QLoRA**, with dedicated evaluation for
Tunisian Arabic (Derja), Arabizi, and preservation of the base model's general capabilities.

## Objective

Gemma Tounsi 1.0 27B adapts Gemma 3 27B to Tunisian Arabic — including Derja written in Arabic
script and Arabizi (Latin-script Tunisian) — without degrading the general instruction-following
and reasoning capabilities of the base model. This repository owns the *data*, *evaluation*, and
*release* side of the project: dataset preparation and provenance, mixture definitions,
experiment configuration, benchmark definitions, and release orchestration. Model training
itself is delegated to the open-source **Soup** training engine, which is consumed as an
external dependency.

## Project status

**Stage 0 — repository architecture.** Scaffolding only.

| Area | Status |
| --- | --- |
| Repository architecture | ✅ in place |
| Soup environment validation | ⬜ not started |
| Dataset sourcing & provenance | ⬜ not started |
| Data preparation pipeline | ⬜ not started |
| Training configuration (`soup.yaml`) | ⬜ placeholder only |
| Evaluation harnesses | ⬜ not started |
| Released model | ⬜ not released |

Nothing in this repository trains a model or downloads a dataset yet. Training hyperparameters
(LoRA rank, learning rate, batch size, target modules, mixture ratios, GPU settings) are
deliberately **undecided** and will be fixed only after the Soup environment is validated.

## Architecture overview

```
gemma-tounsi-1.0/
├── soup.yaml            # training config consumed by Soup (placeholder)
├── Dockerfile           # runtime image (placeholder)
├── docker-compose.yml   # local/GPU orchestration (placeholder)
├── data/                # dataset preparation, manifests & provenance
├── eval/                # evaluation tracks (definitions, not results)
├── configs/releases/    # frozen configs for released models
├── scripts/             # thin orchestration wrappers
├── docs/                # model card, data, training & evaluation docs
└── tests/               # tests for data/eval tooling
```

### Separation of concerns

| This repository owns | Soup owns |
| --- | --- |
| Dataset preparation & provenance | Training loop |
| Dataset mixture definitions | QLoRA implementation |
| Experiment configuration | Model loading & quantization |
| Evaluation | Checkpointing |
| Release orchestration | GPU / training infrastructure |
| Documentation | |

Soup is **never** vendored, cloned, or reimplemented here. Shell scripts in `scripts/` are thin
wrappers that will eventually invoke Soup; they must not duplicate its functionality.

<!-- TODO: pin the upstream Soup repository URL and the exact commit SHA used for v1.0. -->

## Planned evaluation tracks

Evaluation is split into three independent tracks, each with its own directory under `eval/`:

1. **TounsiBench** (`eval/tounsibench/`) — Tunisian Arabic / Derja in Arabic script.
2. **Arabizi** (`eval/arabizi/`) — Latin-script Tunisian, including code-switching and
   orthographic variation.
3. **Retention** (`eval/retention/`) — measures preservation of Gemma's general capabilities
   after adaptation (regression guard).

**Evaluation data is evaluation-only.** TounsiBench and the retention *evaluation* set must never
enter any training mixture. The retention *training* (replay/rehearsal) data lives under `data/`
and is kept strictly separate from the retention evaluation set. See `eval/README.md`.

## Training

Training is performed by [Soup](#architecture-overview), an external open-source training engine.
This repository provides the configuration (`soup.yaml`), the data manifests, and the wrapper in
`scripts/train.sh`. See `docs/TRAINING.md`.

## Large files & secrets

Datasets, model checkpoints, Hugging Face caches, experiment outputs, and generated artifacts are
**stored outside Git** and are gitignored. Only small metadata stays tracked: manifests
(`data/manifests/*.yaml`), release configs (`configs/releases/*`), benchmark definitions, scripts,
and documentation.

Secrets are never committed. Copy `.env.example` to `.env` and fill it locally:

```bash
cp .env.example .env
```

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model card for the released model |
| [`docs/DATA.md`](docs/DATA.md) | Data sources, provenance, licensing, preprocessing |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Soup setup, training procedure, reproduction steps |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Evaluation protocol, metrics, reporting rules |

## License

Source code in this repository is licensed under the Apache License 2.0 — see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Model weights, datasets, and the base Gemma 3 model are governed by their
own separate licenses and terms of use.

## Citation

See [CITATION.cff](CITATION.cff).
