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
| Docker/Soup training environment | ✅ implemented & validated (except GPU) |
| Soup environment validation | 🟨 validated on CPU; **GPU still unverified** |
| Dataset sourcing & provenance | ⬜ not started |
| Data preparation pipeline | ⬜ not started |
| Training configuration (`soup.yaml`) | 🟨 smoke-test placeholder only |
| Evaluation harnesses | ⬜ not started |
| Released model | ⬜ not released |

Nothing in this repository trains a model or downloads a dataset yet. Training hyperparameters
(LoRA rank, learning rate, batch size, target modules, mixture ratios, GPU settings) are
deliberately **undecided** and will be fixed only after the Soup environment is validated.

## Architecture overview

```
gemma-tounsi-1.0/
├── soup.yaml            # training config consumed by Soup (smoke-test placeholder)
├── Dockerfile           # runtime image (pinned official Soup image)
├── docker-compose.yml   # local/GPU orchestration
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

## Training environment

Soup runs **entirely inside Docker**, with this repository mounted at `/workspace`. The host needs
only Docker — no local PyTorch, CUDA, or `soup-cli` install.

| | |
| --- | --- |
| Base image | `ghcr.io/makazhanalpamys/soup:0.73.3` (official, **pinned** — never `latest`) |
| Base digest | `sha256:4536816b4975a4b3abdcc3bd0761e94cb0c5f82cdee675f10f750fa19c4843d1` |
| Soup version | `soup v0.73.3` |
| Python | `3.10.12` (Soup supports 3.10–3.12; 3.13 is **not** supported) |
| Fine-tuning | QLoRA / SFT — `training.quantization: 4bit` + `training.lora` |
| Layer streaming | **off** (BETA; evaluated only after the baseline path works) |

`0.73.3` was chosen because it is the highest tag published to
`ghcr.io/makazhanalpamys/soup` and matches the latest `soup-cli` release on PyPI.

### Quick start

```bash
cp .env.example .env      # then paste your HF_TOKEN (gated Gemma 3 access)

docker compose build                                            # build
docker compose run --rm soup soup version                       # check Soup
docker compose run --rm soup soup doctor                        # health check
./scripts/doctor.sh                                             # all env checks
./scripts/train.sh                                              # smoke-test run
```

Two compose services share one definition: **`soup`** reserves all NVIDIA GPUs (the training
service, requires the NVIDIA Container Toolkit), and **`soup-cpu`** is identical without the GPU
reservation, for environment validation on a machine that has no NVIDIA adapter — not for training.

Outputs written under `/workspace` appear directly on the host (e.g. `output/smoke-test/`) and are
gitignored. Secrets come from `.env` at run time and are never baked into the image.

> **Current limitation:** the validation host has no NVIDIA GPU, so `torch.cuda.is_available()`
> is still `False` and no training has been run. Build, Soup, Python, mounts, secret passthrough
> and config validation all pass. See [`docs/TRAINING.md`](docs/TRAINING.md) §9.

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
