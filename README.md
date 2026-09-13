<p align="center">
  <img src="assets/Gemma Tounsi Logo.png" alt="Gemma Tounsi 1.0" width="300">
</p>

<h1 align="center">Gemma Tounsi 1.0 — 27B</h1>

<p align="center">
  <strong>A Tunisian-Arabic adaptation of Gemma 3 27B, built with QLoRA and engineered for measurable capability retention.</strong>
</p>

<p align="center">
  <a href="https://github.com/google/gemma_pytorch">
    <img src="https://img.shields.io/badge/Base%20Model-Gemma%203%2027B-1f6feb?style=flat-square" alt="Base Model">
  </a>
  <a href="https://github.com/MakazhanAlpamys/Soup">
    <img src="https://img.shields.io/badge/Training-QLoRA%20%2B%20SFT-1f6feb?style=flat-square" alt="Training">
  </a>
  <img src="https://img.shields.io/badge/Soup-0.73.3-1f6feb?style=flat-square" alt="Soup 0.73.3">
  <img src="https://img.shields.io/badge/Status-Research%20Build-f59e0b?style=flat-square" alt="Research Build">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0">
</p>

<p align="center">
  Tunisian Derja · Arabizi · Franco-Tunisian · MSA · Capability Retention
</p>

---

## Overview

**Gemma Tounsi 1.0** adapts `google/gemma-3-27b-it` to Tunisian Arabic across:

- **Derja in Arabic script**
- **Arabizi** — Latin-script Tunisian using the characteristic `3` / `7` / `9` orthography
- **Franco-Tunisian code-switching**
- **MSA / formal Arabic register**

The project is designed around a central objective:

> **Make Gemma's existing capabilities work naturally in Tunisian.**

A model that speaks fluent Tunisian but loses the ability to perform mathematics, coding, reasoning, instruction following, or knowledge QA has **not** met the project's objective.

This repository owns the **data, evaluation, and release** side of the project: dataset preparation and provenance, mixture definitions, experiment configuration, benchmark definitions, and release orchestration. Training is delegated to the open-source **[Soup](https://github.com/MakazhanAlpamys/Soup)** engine as a pinned external dependency; Soup is not vendored or reimplemented here.

> **Research status:** No model has been trained or released yet. No performance claims are made before reproducible measurement. The repository distinguishes implemented and tested infrastructure from work that is still pending.

---

## Table of Contents

- [Objective](#objective)
- [Project Status](#project-status)
- [Design Principles](#design-principles)
- [Architecture](#architecture)
- [Training Mixture](#training-mixture)
- [Data Pipeline](#data-pipeline)
- [Training Environment](#training-environment)
- [Evaluation](#evaluation)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Documentation](#documentation)
- [Repository Layout](#repository-layout)
- [Large Files and Secrets](#large-files-and-secrets)
- [License](#license)
- [Citation](#citation)

---

## Objective

Gemma Tounsi 1.0 uses **QLoRA** — 4-bit quantization with LoRA adapters — to adapt Gemma 3 27B so that it can:

1. Understand and generate **Tunisian Derja in Arabic script**.
2. Understand and generate **Tunisian Arabizi**, including `3` / `7` / `9` orthography and heavy code-switching.
3. Handle **French/Tunisian code-switching** naturally.
4. Shift appropriately into **MSA / formal Arabic**.
5. Preserve the base model's general capabilities:
   - mathematics
   - coding
   - reasoning
   - instruction following
   - knowledge QA

Capability preservation is treated as a **release requirement**, not an afterthought. An English retention/replay slice is therefore a first-class component of the training mixture, while technical examples are deliberately distributed throughout the Tunisian adaptation slices.

---

## Project Status

| Area | Status |
|---|:---:|
| Repository architecture | ✅ Complete |
| Docker + Soup training environment | ✅ Implemented and validated on CPU |
| GPU validation and training | ⬜ Pending |
| Data-engineering pipeline | ✅ Implemented and tested end to end on synthetic fixtures |
| Retention candidate corpora | ✅ Acquired, license-checked, pinned, validated and deduplicated |
| Tunisian adaptation data | 🟨 Specifications frozen; sourcing, adapters and authored-data collection in progress |
| Training configuration | 🟨 Smoke-test placeholder; final hyperparameters deliberately undecided |
| Evaluation harnesses | ⬜ Track structure exists; benchmark definitions pending |
| Model training and release | ⬜ Not started |

### Current limitation

The validation host currently has no NVIDIA GPU. Consequently:

- `torch.cuda.is_available()` remains `False`
- no 27B training run has been performed
- GPU validation is still pending

The Docker environment, Soup installation, Python environment, mounts, secret passthrough and QLoRA configuration validation have passed on the available CPU host.

> **VRAM planning target:** approximately **≥48 GB VRAM** for the baseline 27B QLoRA configuration. Final requirements will be established during GPU validation rather than treated as a universal hardware requirement.

**No fabricated results.** Scores, benchmark claims and example outputs will only be added after a real, reproducible run. `docs/MODEL_CARD.md` remains a placeholder until release.

---

## Design Principles

These principles are enforced through implementation and tests wherever applicable.

### 1. No undeclared data

A corpus that is not declared in `data/manifests/*.yaml` cannot enter a training file. Licensing is checked before use.

### 2. Raw data is immutable

`data/raw/` contains sources exactly as obtained. Cleaning and transformation happen downstream so preprocessing decisions can be revised without re-downloading source data.

### 3. Evaluation data is never training data

TounsiBench and the retention benchmark are evaluation-only and are kept disjoint from training inputs at the source level. Detected contamination aborts the pipeline rather than being silently filtered.

### 4. Holdout is reserved before selection

The retention holdout is reserved before training-set selection and cannot appear in training data. `SplitResult` raises on shared IDs.

### 5. Fail loudly

The mixture validator rejects constraint violations instead of silently rebalancing the dataset.

### 6. Deterministic outputs

The same manifests, pinned sources and fixed seeds are intended to produce byte-identical outputs.

### 7. Delegate generic operations

Generic dataset operations such as deduplication and dataset validation are delegated to Soup where appropriate. This repository implements Gemma-Tounsi-specific logic.

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                         GEMMA TOUNSI REPOSITORY                       │
│                                                                       │
│  manifests → raw → processed → splits → train.jsonl                 │
│                                        └→ retention.jsonl             │
│                                                                       │
│  eval/  ────────────────────────────────────────────────┐             │
│  configs/ ──────────────────────────────────────────────┤             │
│  soup.yaml ─────────────────────────────────────────────┘             │
└───────────────────────────────────────┬───────────────────────────────┘
                                        │
                              pinned Docker image
                                        │
                                        ▼
                         ┌─────────────────────────┐
                         │       Soup 0.73.3       │
                         │       QLoRA / SFT        │
                         │       Docker + GPU       │
                         └─────────────────────────┘
```

### Separation of concerns

| Gemma Tounsi repository | Soup |
|---|---|
| Dataset preparation | Training loop |
| Provenance and manifests | QLoRA implementation |
| Mixture definitions | Model loading and quantization |
| Experiment configuration | Checkpointing |
| Evaluation harnesses | GPU / training infrastructure |
| Release gating | — |
| Documentation | — |

---

## Training Mixture

The authoritative definition lives in [`configs/data/mixture.yaml`](configs/data/mixture.yaml). Production shares are measured in **tokens**; `src/data/mixture.py` currently validates example counts for synthetic smoke tests, so token-aware blending remains pending.

| Slice | Token share | Purpose |
|---|---:|---|
| `arabizi` | **35%** | Latin-script Tunisian Derja |
| `arabic_derja` | **25%** | Arabic-script Tunisian Derja |
| `franco_tunisian` | **12%** | French / Tunisian code-switching |
| `msa_formal` | **8%** | Formal-register coverage; never counted as retention |
| `retention` | **20%** | English capability preservation |

### Retention vs. MSA

These slices have deliberately different purposes:

- **`retention`** = English capability preservation through replay/rehearsal data. It primarily protects the base model's general abilities and does not require Tunisian output.
- **`msa_formal`** = formal/register coverage, separate from retention. Retention records use English instructions and Latin script; Latin-script translation exercises are kept, with the decision recorded in the selection manifest.

> **Shared selection strategy (token budgets).** Both slices are selected by the same recipe
> (`docs/data/retention_mohamed.md` §3, `docs/data/retention_haithem.md` §2, applied to MSA in
> `docs/data/msa_audit.md`): budgets and holdouts are measured in **tokens** (not example counts),
> whole examples only (anything over ~15,500 formatted Gemma tokens is dropped, never truncated),
> diversity caps inside each budget (template caps for SlimOrca, ≤ 2 questions per passage for MSA
> QA pools), token-proportional holdouts reserved first, deterministic selection (seed 42), and
> biggest-first packing into 16k sequences with block-diagonal attention masks.

### Technical quota

At least **20% of both the `arabizi` and `arabic_derja` slices** must be technical examples covering:

- mathematics
- reasoning
- coding

The quota is checked against the actual processed contents.

### Retention dataset

Production retention uses **token budgets**, following [Mohamed §3](docs/data/retention_mohamed.md#3-selection--packing-strategy-for-task-8-16k-context-window) and [Haithem §2](docs/data/retention_haithem.md#2-selection--token-budget-strategy-for-task-9-16k-context-window). Reserve the holdout first, then select **5,000 diverse coding conversations** and measure `T_code` with Gemma's tokenizer and chat template. Prefer assistant tokens; total tokens are an acceptable fallback, using one metric consistently and recording both totals.

| Category | Token budget | Share within retention |
|---|---:|---:|
| Coding (anchor) | 1.0 × T_code | 25% |
| Mathematics | 1.0 × T_code | 25% |
| Reasoning | 0.8 × T_code | 20% |
| General instruction | 0.6 × T_code | 15% |
| Knowledge QA | 0.6 × T_code | 15% |
| **Total** | **4.0 × T_code** | **100%** |

Fill budgets with whole examples within **±2%**; example counts emerge from measured tokens, not fixed quotas. The holdout follows the same **25/25/20/15/15 token proportions**, is reserved before training selection, and is never packed with training. Its absolute token budget must be set before production.

- **Coding:** diversify language, task type and length bands; preserve multi-turn conversations.
- **Math:** roughly 60% GSM8K / 40% MATH token sub-budgets, per-type quotas, SV + FOBAR ≤30% combined, and ≤2 variants per seed problem.
- **SlimOrca:** deterministic tagging (reasoning → knowledge QA → general instruction), audit ~200 examples per category, cap original system-prompt templates at 200 examples, and standardize system prompts before tokenizing.
- **Hygiene:** near-deduplicate at 0.85 before tagging/budgeting; cross-check SlimOrca reasoning against MetaMathQA and check all selections against retention evaluation data.
- **Packing:** drop whole examples over 15,500 formatted Gemma tokens; first-fit decreasing into 16,384-token sequences, isolated attention and reset position IDs. Verify masking in both Gemma 3 sliding-window and global layers; pad only final partial sequences as specified in the reports.
- **Manifest:** record both token totals, counts, anchor/metric, tagging audits, template caps, translation decision, contamination checks, packing efficiency and padding waste.

**Implementation status:** these are production specifications, not a completed selection run. Token-aware selection/packing remains pending; the old 20,000/2,500 count fields are retained only for synthetic pipeline compatibility.

Current candidate pools:

| Dataset | Capability | Rows | License |
|---|---|---:|---|
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | Mathematics | 395,000 | MIT |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | Coding | 66,383 | Apache-2.0 |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | Reasoning / instruction / knowledge | 517,982 | MIT |

All three are pinned to exact revisions in [`data/manifests/retention.yaml`](data/manifests/retention.yaml), stored immutably under `data/raw/`, and validated and deduplicated as documented under `docs/data/`.

---

## Data Pipeline

The data pipeline has one main entry point:

```text
canonicalization
      ↓
validation
      ↓
statistics
      ↓
deduplication
  ├─ exact local
  └─ near-duplicate via Soup
      ↓
holdout reservation
      ↓
retention selection
      ↓
mixture validation
      ↓
Soup-compatible JSONL export
```

The pipeline is currently proven end to end against synthetic fixtures deliberately containing:

- duplicate records
- malformed rows
- a below-quota slice

A pipeline that has never rejected invalid input has not been meaningfully tested.

### Canonical record

Internal examples use a canonical JSONL schema containing:

- `id`
- `messages`
- `category`
- `source`
- `language`

Optional fields include:

- `script`
- `code_switching`
- `difficulty`
- `quality`
- `variation_group`

The specification is documented in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md), with the machine-readable source of truth in [`configs/data/schema.yaml`](configs/data/schema.yaml).

The final export converts canonical records into the ShareGPT-compatible format consumed by Soup 0.73.3.

### Run the pipeline

```bash
# End-to-end synthetic pipeline
./scripts/prepare_data.sh

# Same pipeline inside the pinned Soup container
./scripts/prepare_data.sh --in-docker

# Individual stages
python -m src.data.validate data/synthetic/raw/arabizi.jsonl
python -m src.data.stats data/synthetic/raw/*.jsonl
```

Real corpus adapters and Tunisian authored-data ingestion are the next implementation stage; the pipeline stage order is designed to remain unchanged.

---

## Training Environment

Soup runs entirely inside Docker with this repository mounted at `/workspace`.

The host therefore does not need a local PyTorch, CUDA or `soup-cli` installation for training.

| Component | Configuration |
|---|---|
| Base model | `google/gemma-3-27b-it` |
| Model access | Gated; requires `HF_TOKEN` |
| Soup image | `ghcr.io/makazhanalpamys/soup:0.73.3` |
| Soup version | `0.73.3` |
| Python | `3.10.12` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `4.57.6` |
| PEFT | `0.20.0` |
| TRL | `0.28.0` |
| bitsandbytes | `0.50.1` |
| Fine-tuning | QLoRA / SFT |
| Quantization | 4-bit |
| Layer streaming | **Off**; BETA path evaluated only after baseline |

The Docker Compose setup provides two services:

- **`soup`** — GPU training service; requires the NVIDIA Container Toolkit.
- **`soup-cpu`** — GPU-less validation service; not intended for training.

`docker compose up` is intentionally inert. Its default command is `soup --help`, so starting the Compose project does not implicitly launch training.

### Training configuration

`soup.yaml` is currently a **smoke-test placeholder** validated field-by-field against Soup's schema.

Final values for:

- LoRA rank / alpha
- learning rate
- batch size
- sequence length
- epochs
- mixture ratios

are deliberately undecided until GPU validation.

---

## Evaluation

Evaluation is organized into three independent tracks:

| Track | Directory | Question |
|---|---|---|
| **TounsiBench** | `eval/tounsibench/` | Does the model handle real Derja in Arabic script? |
| **Arabizi** | `eval/arabizi/` | Does it handle Latin-script Tunisian and code-switching? |
| **Retention** | `eval/retention/` | What did adaptation break? |

### Evaluation rules

#### Baselines are mandatory

Every fine-tuned score is compared against the same measurement on the unmodified Gemma 3 27B base model using pinned revisions and identical decoding settings.

#### Scores are not collapsed into one headline number

Tunisian-language gains and general-capability regressions must remain independently visible.

#### Retention is a release gate

A regression beyond the agreed threshold blocks release through `scripts/release.sh`.

> Benchmark definitions are currently the next major evaluation milestone. The repository contains the track structure and reporting rules, but no benchmark results exist yet.

See [`eval/README.md`](eval/README.md) and [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B.git
cd Gemma-Tounsi-1.0-27B
```

### 2. Configure gated model access

```bash
cp .env.example .env
```

Add your Hugging Face token to `.env`.

> `.env` is gitignored and must never be committed.

### 3. Validate the environment

```bash
docker compose build

docker compose run --rm soup soup version
docker compose run --rm soup soup doctor

./scripts/doctor.sh
```

On a GPU-less host, use the CPU validation service as documented in `docs/ENVIRONMENT.md`.

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Run the synthetic data pipeline

```bash
./scripts/prepare_data.sh
```

### 6. Train

Training requires a validated CUDA host and prepared training data:

```bash
./scripts/train.sh
```

Training outputs under `/workspace` appear directly on the host and are gitignored.

---

## Testing

```bash
python -m pytest tests/ -v
```

The data-pipeline test suite runs entirely offline:

- no external dataset downloads
- no network
- no GPU
- no model training

The tests cover:

- required fields and message ordering
- duplicate-ID handling
- exact retention category targets
- deterministic processing
- train/holdout disjointness
- all eight mixture constraints
- the 20% technical quota at the exact boundary
- ShareGPT export correctness
- end-to-end reproducibility

Negative cases are tested as well:

- malformed fixtures **must** be rejected
- below-quota mixtures **must** fail

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model card; placeholder until release |
| [`docs/DATA.md`](docs/DATA.md) | Data sources, provenance, licensing and preprocessing |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | Canonical record schema |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Soup setup, configuration, reproduction and validation |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Evaluation protocol, metrics and reporting |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Full environment validation report |
| [`data/README.md`](data/README.md) | Data layout, provenance and contamination rules |
| [`eval/README.md`](eval/README.md) | Evaluation tracks and non-negotiable rules |

---

## Repository Layout

```text
Gemma-Tounsi-1.0-27B/
│
├── assets/
│   └── gemma-tounsi-logo.png
│
├── configs/
│   ├── data/
│   │   ├── schema.yaml
│   │   ├── mixture.yaml
│   │   ├── retention.yaml
│   │   └── msa.yaml
│   └── releases/
│       └── ...
│
├── data/
│   ├── manifests/
│   ├── raw/
│   ├── synthetic/
│   └── ...
│
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATA.md
│   ├── DATA_SCHEMA.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── ENVIRONMENT.md
│
├── eval/
│   ├── tounsibench/
│   ├── arabizi/
│   └── retention/
│
├── scripts/
│   ├── doctor.sh
│   ├── prepare_data.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── release.sh
│
├── src/
│   └── data/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── soup.yaml
├── LICENSE
├── NOTICE
├── CITATION.cff
└── README.md
```

---

## Large Files and Secrets

Datasets, model checkpoints, Hugging Face caches, experiment outputs and generated artifacts are stored outside Git and gitignored.

Tracked repository content is limited to lightweight, reproducible project assets such as:

- manifests
- configuration
- benchmark definitions
- scripts
- documentation
- synthetic fixtures required by tests

Secrets are never committed or baked into the Docker image.

Use:

```bash
cp .env.example .env
```

and provide the token locally. Docker Compose injects it at run time through `env_file`.

---

## License

The source code in this repository is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Model weights, datasets and the base Gemma 3 model are governed by their own licenses and terms of use. Gemma 3 is gated and requires acceptance of its applicable license terms.

---

## Citation

If you use Gemma Tounsi 1.0 or this repository, please cite the project using [`CITATION.cff`](CITATION.cff):

```bibtex
@software{gemma_tounsi_1_0_27b,
  title  = {Gemma Tounsi 1.0 (27B): a Tunisian-adapted Gemma 3 27B model},
  author = {Mohamed BENSAAD and Haithem NASR},
  url    = {https://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B},
  note   = {Apache-2.0; training performed with the external Soup engine}
}
```

---

<p align="center">
  <strong>Gemma Tounsi 1.0</strong><br>
  <sub>Making Gemma's existing capabilities work naturally in Tunisian.</sub>
</p>
