<p align="center">
  <img src="assets/Gemma Tounsi Logo.png" alt="Gemma Tounsi 1.0" width="300">
</p>

<h1 align="center">Gemma Tounsi 1.0 — 27B</h1>

<p align="center">
  <strong>A Tunisian-Arabic adaptation of Gemma 3 27B, built with QLoRA and engineered for measurable capability retention.</strong>
</p>

<p align="center">
  Tunisian Derja · Arabizi · Franco-Tunisian · MSA · Capability Retention
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Gemma%203%2027B-blue" alt="Gemma 3 27B">
  <img src="https://img.shields.io/badge/Method-QLoRA-blue" alt="QLoRA">
  <img src="https://img.shields.io/badge/Status-Research%20Build-orange" alt="Research Build">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="Apache 2.0">
</p>

Gemma Tounsi 1.0 adapts Gemma 3 27B to Tunisian Arabic — ****Derja in Arabic script, **Arabizi** (Latin-script Tunisian), and Franco-Tunisian code-switching — *without degrading the base model's general instruction-following and reasoning capabilities*. This repository owns the ****data****, ****evaluation****, and ****release**** side of the project: dataset preparation and provenance, mixture definitions, experiment configuration, benchmark definitions, and release orchestration. Training itself is delegated to the open-source ****[Soup](https://github.com/MakazhanAlpamys/Soup)**** engine, consumed as a pinned external dependency — never vendored or reimplemented here.

> ****Project status: research build in progress.**** No model has been trained or released yet. Every claim in this repository is backed by an implemented, tested artifact — and nothing is claimed before it has been measured. See [Project status](#project-status).

---

**

**## Table of contents**

1\. [Objective](#objective)

2\. [Project status](#project-status)

3\. [Design principles](#design-principles)

4\. [Architecture overview](#architecture-overview)

5\. [The training mixture](#the-training-mixture)

6\. [Data pipeline](#data-pipeline)

7\. [Training environment](#training-environment)

8\. [Evaluation tracks](#evaluation-tracks)

9\. [Quick start](#quick-start)

10\. [Testing](#testing)

11\. [Documentation](#documentation)

12\. [Repository layout](#repository-layout)

13\. [Large files & secrets](#large-files--secrets)

14\. [License](#license)

15\. [Citation](#citation)

**---**

**## Objective**

Adapt ****Gemma 3 27B**** (\`google/gemma-3-27b-it\`) to Tunisian Arabic with ****QLoRA**** (4-bit quantization + LoRA

adapters), so that the resulting model can:

\- understand and generate ****Derja**** written in Arabic script,

\- understand and generate ****Arabizi**** — Latin-script Tunisian with its \`3\`/\`7\`/\`9\` orthography and heavy

  code-switching,

\- shift register into formal Arabic (MSA) and handle French/Tunisian mixing,

\- ****keep**** the base model's general capabilities — mathematics, coding, reasoning, instruction following,

  knowledge QA — which is why an English **retention/replay** slice is a first-class part of the mixture and

  retention is a hard ****release gate****, not a footnote.

The success criterion is **"make Gemma's existing capabilities work naturally in Tunisian"** — a model that

chats fluently in Derja but cannot do arithmetic in it has not met the objective. That is enforced as a

cross-cutting technical quota inside the Tunisian slices (see [The training mixture](#the-training-mixture)).

**### Separation of concerns**

\| This repository owns | Soup owns |

\| --- | --- |

\| Dataset preparation, provenance & manifests | Training loop |

\| Mixture definitions & validation | QLoRA implementation |

\| Experiment configuration (\`soup.yaml\`) | Model loading & quantization |

\| Evaluation harnesses & release gating | Checkpointing |

\| Documentation | GPU / training infrastructure |

**## Project status**

\| Area | Status |

\| --- | --- |

\| Repository architecture | ✅ complete |

\| Docker + Soup training environment | ✅ implemented & validated on CPU ([docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)) |

\| GPU validation & training | ⬜ pending — no NVIDIA adapter on the validation host so far (needs ≥ 48 GB VRAM for 27B QLoRA) |

\| Data-engineering pipeline (\`src/data/\`) | ✅ implemented & tested end to end on synthetic fixtures (\`tests/test_data_pipeline.py\`) |

\| Retention candidate corpora | ✅ acquired, license-checked, pinned, validated & deduplicated — MetaMathQA, Code-Feedback, SlimOrca |

\| Tunisian adaptation data (Derja / Arabizi / Franco / MSA) | 🟨 specs frozen; sourcing, adapters & authored-data collection in progress |

\| Training configuration (\`soup.yaml\`) | 🟨 clearly-labelled ****smoke-test placeholder**** — real hyperparameters deliberately undecided until GPU validation |

\| Evaluation harnesses | ⬜ track structure only — no benchmarks defined, no runs performed |

\| Model training & release | ⬜ not started — no model exists yet |

****No fabricated results.**** Scores, claims, and example outputs appear in this repository only after a real,

reproducible run produced them. \`docs/MODEL_CARD.md\` is an explicit placeholder until release.

**## Design principles**

These rules are enforced in code and tests, not just documented:

1\. ****No undeclared data.**** A corpus that is not declared in a manifest (\`data/manifests/*.yaml\`) never reaches

   a training file. Licensing is checked **before** use.

2\. ****Raw data is immutable.**** \`data/raw/\` holds sources exactly as obtained (marked read-only on disk); all

   cleaning happens downstream so decisions can be revised without re-downloading.

3\. ****Evaluation data is never training data.**** TounsiBench and the retention **benchmark** are evaluation-only

   and disjoint from training inputs at the source level. Detected contamination ****aborts**** the pipeline —

   it is never silently filtered away.

4\. ****Holdout is reserved before selection**** and can never appear in a training file (\`SplitResult\` raises on

   any shared id).

5\. ****Fail loudly, never auto-rebalance.**** The mixture validator refuses to proceed on any constraint

   violation instead of silently adjusting shares.

6\. ****Deterministic outputs.**** Same manifests + pinned sources + fixed seeds ⇒ byte-identical outputs.

7\. ****Generic operations are delegated to Soup**** (deduplication, dataset validation); this repository

   implements only the Gemma-Tounsi-specific logic.

**## Architecture overview**

\`\`\`text

                    ┌─────────────────────────────────────────────────────┐

                    │                THIS REPOSITORY                     │

                    │                                                    │

  manifests ──►  raw ──►  processed ──►  splits ──►  train.jsonl          │

  (provenance)   (immutable)  (src/data/)          retention.jsonl       │

                    │                                    │              │

                    │  eval/  (3 tracks, eval-only)      ▼              │

                    │  configs/  (specs, releases)   soup.yaml           │

                    └──────────────────────────┬─────────────────────────┘

                                               │  pinned image, /workspace mount

                                               ▼

                                    ┌────────────────────┐

                                    │  Soup 0.73.3       │

                                    │  QLoRA / SFT       │

                                    │  (Docker, GPU)     │

                                    └────────────────────┘

\`\`\`

\`\`\`text

gemma-tounsi-1.0/

├── soup.yaml              # training config consumed by Soup (smoke-test placeholder)

├── Dockerfile             # thin layer over the pinned official Soup image

├── docker-compose.yml     # soup (GPU) + soup-cpu (validation) services

├── configs/

│   ├── data/              # machine-readable specs: schema, mixture, retention, msa

│   └── releases/          # frozen configs for released models

├── data/                  # datasets, manifests & provenance  (see data/README.md)

├── src/data/              # data-engineering pipeline (schema → export)

├── eval/                  # three evaluation tracks (definitions, not results)

├── scripts/               # thin orchestration wrappers (no training logic)

├── docs/                  # model card, data, schema, training, evaluation, environment

└── tests/                 # pipeline tests, all runnable offline

\`\`\`

**## The training mixture**

The authoritative, dataset-independent definition lives in [\`configs/data/mixture.yaml\`](configs/data/mixture.yaml)

and is enforced by \`src/data/mixture.py\`:

\| Slice | Share | Purpose |

\| --- | --- | --- |

\| \`arabizi\` | 0.35 | Latin-script Tunisian Derja |

\| \`arabic_derja\` | 0.25 | Arabic-script Tunisian Derja |

\| \`franco_tunisian\` | 0.12 | French / Tunisian code-switching |

\| \`msa_formal\` | 0.08 | ****Formal register coverage**** — **never counted as retention** |

\| \`retention\` | 0.20 | ****English capability preservation**** — **not a Tunisian dataset** |

Two distinctions the validator enforces in code:

\- ****\`retention\` = English capability preservation.**** Replay/rehearsal data protecting the base model's

  general abilities. It is primarily English and requires no Tunisian output.

\- ****\`msa_formal\` = formal/register coverage.**** A separate purpose; non-English records in the retention

  slice are a hard failure.

****Cross-cutting technical quota:**** at least ****20 %**** of both the \`arabizi\` and \`arabic_derja\` slices must be

technical (\`mathematics\` / \`reasoning\` / \`coding\`), checked as \`>=\` against the actual processed contents.

The retention slice itself is specified in [\`configs/data/retention.yaml\`](configs/data/retention.yaml):

20,000 training examples with hard per-category targets (5,000 math, 5,000 coding, 4,000 reasoning,

3,000 instruction following, 3,000 knowledge QA) plus a stratified 2,500-example holdout reserved before

selection. Candidate pools already acquired and validated cover these targets with a large surplus:

\| Candidate pool | Capability | Rows | License |

\| --- | --- | --- | --- |

\| [MetaMathQA](https\://huggingface.co/datasets/meta-math/MetaMathQA) | mathematics | 395,000 | MIT |

\| [Code-Feedback](https\://huggingface.co/datasets/m-a-p/Code-Feedback) | coding | 66,383 | Apache-2.0 |

\| [SlimOrca](https\://huggingface.co/datasets/Open-Orca/SlimOrca) | reasoning / instruction / knowledge | 517,982 | MIT |

All three are pinned to exact revisions in [\`data/manifests/retention.yaml\`](data/manifests/retention.yaml),

stored immutable under \`data/raw/\`, and were inspected and deduplicated with the Soup CLI (analysis in

\`docs/data/\`).

**## Data pipeline**

One entry point, [\`scripts/prepare_data.sh\`](scripts/prepare_data.sh), currently proven end to end against

synthetic fixtures (deliberately containing duplicates, malformed rows and a below-quota slice — a pipeline

that has never rejected anything has not been tested):

\`\`\`text

canonicalization → validation → statistics → deduplication (exact local, near-dup via Soup)

  → retention selection + holdout reservation → mixture validation → Soup-compatible JSONL

\`\`\`

Every internal example is a ****canonical JSONL record**** (\`id\`, \`messages\`, \`category\`, \`source\`, \`language\`,

plus optional \`script\`, \`code_switching\`, \`difficulty\`, \`quality\`, \`variation_group\`). The full specification

is [\`docs/DATA_SCHEMA.md\`](docs/DATA_SCHEMA.md), with the machine-readable source of truth in

[\`configs/data/schema.yaml\`](configs/data/schema.yaml) — the validator reads that file directly, so docs and

enforcement cannot drift apart. The final export converts canonical records to the ****ShareGPT**** format

Soup 0.73.3 consumes.

\`\`\`bash

\# whole pipeline over synthetic fixtures (no downloads, no GPU, no training)

./scripts/prepare_data.sh

./scripts/prepare_data.sh --in-docker     # same, inside the pinned Soup container

\# individual stages

python -m src.data.validate data/synthetic/raw/arabizi.jsonl

python -m src.data.stats    data/synthetic/raw/*.jsonl

\`\`\`

Real corpus adapters and Tunisian authored-data ingestion are wired in next; the stage order will not change.

**## Training environment**

Soup runs ****entirely inside Docker****, with this repository mounted at \`/workspace\`. The host needs only

Docker — no local PyTorch, CUDA, or \`soup-cli\` install.

\| | |

\| --- | --- |

\| Base model | \`google/gemma-3-27b-it\` (gated — requires \`HF_TOKEN\`) |

\| Base image | \`ghcr.io/makazhanalpamys/soup:0.73.3\` (official, ****pinned by digest**** — never \`latest\`) |

\| Soup version | \`soup v0.73.3\` · schema source of truth: \`src/soup_cli/config/schema.py\` @ \`v0.73.3\` |

\| Python | \`3.10.12\` (Soup supports 3.10–3.12; 3.13 is ****not**** supported) |

\| Stack | torch \`2.13.0+cu130\` · transformers \`4.57.6\` · peft \`0.20.0\` · trl \`0.28.0\` · bitsandbytes \`0.50.1\` |

\| Fine-tuning | QLoRA / SFT — \`training.quantization: 4bit\` + \`training.lora\` |

\| Layer streaming | ****off**** (BETA; evaluated only after the baseline path works) |

Two compose services share one definition: ****\`soup\`**** reserves all NVIDIA GPUs (the training service,

requires the NVIDIA Container Toolkit), and ****\`soup-cpu\`**** is identical without the GPU reservation for

environment validation on GPU-less hosts — not for training.

\`docker compose up\` is intentionally inert (default command is \`soup --help\`): no run ever starts implicitly.

\> ****Current limitation:**** the validation host has no NVIDIA GPU, so \`torch.cuda.is_available()\` is still

\> \`False\` and no training has been run. Build, Soup, Python, mounts, secret passthrough and QLoRA config

\> validation all pass. Training requires a CUDA host with the NVIDIA Container Toolkit and roughly

\> ****≥ 48 GB VRAM****. Full report: [\`docs/ENVIRONMENT.md\`](docs/ENVIRONMENT.md) and

\> [\`docs/TRAINING.md\`](docs/TRAINING.md) §9.

\`soup.yaml\` is a clearly-labelled ****smoke-test placeholder**** validated field-by-field against Soup's real

schema. Every hyperparameter (LoRA rank/alpha, learning rate, batch size, sequence length, epochs, mixture

ratios) is deliberately ****undecided**** and will be fixed only after the GPU environment is validated.

**## Evaluation tracks**

Evaluation is split into three independent tracks under \`eval/\`, each with its own data, tasks and metrics:

\| Track | Directory | Question it answers |

\| --- | --- | --- |

\| ****TounsiBench**** | \`eval/tounsibench/\` | Does it handle real Derja in Arabic script? |

\| ****Arabizi**** | \`eval/arabizi/\` | Does it handle Latin-script Tunisian, incl. code-switching? |

\| ****Retention**** | \`eval/retention/\` | What did adaptation break? |

Rules that make the numbers meaningful:

\- ****Baselines are mandatory.**** Every score is paired with the same measurement on unmodified base

  Gemma 3 27B, at pinned revisions and identical decoding settings.

\- Track scores are ****never aggregated**** into a single headline number — a Tunisian gain that costs general

  capability must stay visible.

\- Retention is a ****release gate****: a regression beyond the agreed threshold blocks the release

  (\`scripts/release.sh\`).

Track structure is in place; benchmark definitions are the next major work item. See

[\`eval/README.md\`](eval/README.md) and [\`docs/EVALUATION.md\`](docs/EVALUATION.md).

**## Quick start**

\`\`\`bash

git clone https\://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B.git

cd Gemma-Tounsi-1.0-27B

\# 1. secrets (gated Gemma 3 access) — .env is gitignored, never committed

cp .env.example .env          # then paste your HF_TOKEN

\# 2. environment

docker compose build          # builds gemma-tounsi-soup:0.73.3

docker compose run --rm soup soup version      # -> soup v0.73.3

docker compose run --rm soup soup doctor       # -> All checks passed

./scripts/doctor.sh           # full validation (SERVICE=soup-cpu on GPU-less hosts)

\# 3. data pipeline (offline, deterministic, synthetic fixtures)

python -m pytest tests/ -v    # schema, quotas, contamination, export, reproducibility

./scripts/prepare_data.sh     # end-to-end synthetic run

\# 4. smoke-test training run (needs a GPU host + prepared data/train.jsonl)

./scripts/train.sh

\`\`\`

Local development of the data pipeline needs only Python 3.10+ with \`pytest\` and \`pyyaml\`; the training

stack itself never touches the host.

Outputs written under \`/workspace\` appear directly on the host (e.g. \`output/smoke-test/\`) and are

gitignored.

**## Testing**

\`\`\`bash

python -m pytest tests/ -v

\`\`\`

\`tests/test_data_pipeline.py\` runs ****entirely offline**** — no external dataset, no network, no GPU, no

training — and asserts real behavioural guarantees: required fields and message ordering, duplicate-id

handling, exact retention category targets, determinism, holdout/train disjointness, all eight mixture

constraints (including the 20 % technical quota at the exact boundary), ShareGPT export correctness, and

end-to-end pipeline reproducibility. Negative cases are tested too: the malformed fixture **must** be rejected

and the below-quota mixture **must** fail.

**## Documentation**

\| Document | Contents |

\| --- | --- |

\| [\`docs/MODEL_CARD.md\`](docs/MODEL_CARD.md) | Model card for the released model (placeholder until release) |

\| [\`docs/DATA.md\`](docs/DATA.md) | Data sources, provenance, licensing, preprocessing record |

\| [\`docs/DATA_SCHEMA.md\`](docs/DATA_SCHEMA.md) | The canonical record schema — full specification |

\| [\`docs/TRAINING.md\`](docs/TRAINING.md) | Soup setup, configuration, reproduction & validation record |

\| [\`docs/EVALUATION.md\`](docs/EVALUATION.md) | Evaluation protocol, metrics, reporting rules |

\| [\`docs/ENVIRONMENT.md\`](docs/ENVIRONMENT.md) | Full environment validation report (CPU host) |

\| [\`data/README.md\`](data/README.md) | Data directory layout, provenance & contamination rules |

\| [\`eval/README.md\`](eval/README.md) | Evaluation track structure and non-negotiable rules |

**## Repository layout**

\| Path | Role |

\| --- | --- |

\| \`src/data/\` | \`schema\` · \`validate\` · \`stats\` · \`dedupe\` (Soup delegation) · \`split\` · \`retention\` · \`mixture\` · \`export\` |

\| \`configs/data/\` | Machine-readable specs: \`schema.yaml\`, \`mixture.yaml\`, \`retention.yaml\`, \`msa.yaml\` |

\| \`data/manifests/\` | Tracked declarations of every permitted corpus (source, revision, license) |

\| \`data/synthetic/\` | Deterministic fixtures for testing — never training data |

\| \`scripts/\` | Thin wrappers: \`doctor.sh\`, \`prepare_data.sh\`, \`train.sh\`, \`evaluate.sh\`, \`release.sh\` |

\| \`configs/releases/\` | Frozen configurations for released models |

**## Large files & secrets**

Datasets, model checkpoints, Hugging Face caches, experiment outputs and generated artifacts are **stored

outside Git** and gitignored. Only small metadata stays tracked: manifests, configs, benchmark definitions,

scripts, documentation, and the synthetic fixtures required by the tests.

Secrets are never committed and never baked into an image. Copy \`.env.example\` to \`.env\` (gitignored and

docker-ignored) and fill it locally; compose injects it at **run** time via \`env_file\`.

**## License**

Source code in this repository is licensed under the ****Apache License 2.0**** — see [LICENSE](LICENSE) and

[NOTICE](NOTICE). Model weights, datasets, and the base Gemma 3 model are governed by their own separate

licenses and terms of use (Gemma 3 is gated; accept its license at

\<https\://huggingface.co/google/gemma-3-27b-it>).

**## Citation**

If you use Gemma Tounsi 1.0 or this repository, please cite it — see [CITATION.cff](CITATION.cff):

\`\`\`bibtex

@software{gemma_tounsi_1_0_27b,

  title  = {Gemma Tounsi 1.0 (27B): a Tunisian-adapted Gemma 3 27B model},

  author = {Mohamed BENSAAD and Haithem NASR},

  url    = {https\://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B},

  note   = {Apache-2.0; training performed with the external Soup engine}
