<p align="center">
  <img src="assets/Gemma Tounsi Logo.png" alt="Gemma Tounsi 1.0" width="300">
</p>

<h1 align="center">Gemma Tounsi 1.0 — 27B</h1>

<p align="center">
  <strong>Making Gemma's existing capabilities work naturally in Tunisian.</strong>
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

**Gemma Tounsi 1.0** adapts `google/gemma-3-27b-it` to Tunisian Arabic across Derja (Arabic script), Arabizi (Latin script with `3`/`7`/`9` orthography), Franco-Tunisian code-switching, and MSA — without sacrificing the base model's general capabilities.

A model that speaks fluent Tunisian but loses mathematics, coding, reasoning, instruction following, or knowledge QA has not met the objective. Capability preservation is a release requirement, not an afterthought.

This repository owns the **data, evaluation, and release** side of the project: dataset preparation and provenance, mixture definitions, experiment configuration, benchmark definitions, and release orchestration. Training is delegated to the open-source **[Soup](https://github.com/MakazhanAlpamys/Soup)** engine as a pinned external dependency; Soup is not vendored or reimplemented here.

> **Research status.** No model has been trained or released yet. No performance claims are made before reproducible measurement. The repository distinguishes implemented and tested infrastructure from work still pending.

---

## Project Status

| Area | Status |
|---|---|
| Repository architecture | ✅ Complete |
| Docker + Soup training environment | ✅ Validated on CPU |
| GPU validation and training | ⬜ Pending |
| Data-engineering pipeline | ✅ End-to-end on synthetic fixtures |
| Retention candidate corpora | ✅ Acquired, license-checked, pinned, validated, deduplicated |
| Tunisian adaptation data | 🟨 Specs frozen; sourcing, adapters, authored data in progress |
| Training configuration | 🟨 Smoke-test placeholder; hyperparameters undecided |
| Evaluation harnesses | ⬜ Track structure exists; benchmarks pending |
| Model training and release | ⬜ Not started |

---

## The Problem

Fine-tuning a 27B model on a low-resource dialect risks degrading the very capabilities that make it useful. Dialect gains can come at the cost of general reasoning, coding, and knowledge — and a single aggregate score hides that trade-off.

## The Solution

Gemma Tounsi 1.0 treats capability preservation as a first-class design constraint:

- **QLoRA** (4-bit quantization + LoRA adapters) for parameter-efficient adaptation.
- An **English retention slice** (20% of training tokens) replayed during training to protect general abilities.
- A **20% technical quota** inside both Tunisian slices (math, reasoning, coding) so capabilities work *in* Tunisian, not just alongside it.
- Three independent evaluation tracks — TounsiBench, Arabizi, Retention — reported separately, never collapsed into one headline number.
- A **retention release gate**: any regression beyond the agreed threshold blocks publication.

---

## Training Mixture

The authoritative definition lives in `configs/data/mixture.yaml`. Shares are fractions of final training tokens and sum to 1.0.

| Slice | Share | Purpose |
|---|---:|---|
| `arabizi` | 35% | Latin-script Tunisian Derja |
| `arabic_derja` | 25% | Arabic-script Tunisian Derja |
| `franco_tunisian` | 12% | French/Tunisian code-switching |
| `msa_formal` | 8% | Formal MSA register (never counted as retention) |
| `retention` | 20% | English capability preservation |

### Cross-cutting technical quota

At least **20%** of both the `arabizi` and `arabic_derja` slices must be technical examples covering mathematics, reasoning, and coding. The quota is checked against actual processed contents — `>=` boundary, exactly 20.0% passes.

### Retention budget (token-anchored)

| Category | Token budget | Share within retention |
|---|---:|---:|
| Coding (anchor) | 1.0 × T_code | 25% |
| Mathematics | 1.0 × T_code | 25% |
| Reasoning | 0.8 × T_code | 20% |
| General instruction | 0.6 × T_code | 15% |
| Knowledge QA | 0.6 × T_code | 15% |
| **Total** | **4.0 × T_code** | **100%** |

Retention candidate pools are acquired and pinned: MetaMathQA (395K, MIT), Code-Feedback (66K, Apache-2.0), SlimOrca (518K, MIT). Final token-aware selection and packing are specified but not yet implemented at production scale.

---

## Data Pipeline

One entry point, seven stages:

```text
canonicalization → validation → statistics → deduplication
                                      ├─ exact (local)
                                      └─ near-dup (via Soup)
       → holdout reservation → retention selection
       → mixture validation → Soup-compatible JSONL export
```

Run end-to-end on synthetic fixtures:

```bash
./scripts/prepare_data.sh
```

Individual stages:

```bash
python -m src.data.validate data/synthetic/raw/arabizi.jsonl
python -m src.data.stats data/synthetic/raw/*.jsonl
```

Real corpus adapters and Tunisian authored-data ingestion are the next implementation stage; the pipeline stage order is frozen.

---

## Architecture

```text
manifests → raw → processed → splits → train.jsonl
                                  └→ retention.jsonl

eval/ ─────────────────────────────────────┐
configs/ ──────────────────────────────────┤
soup.yaml ─────────────────────────────────┘
                                     │
                               pinned Docker image
                                     │
                                     ▼
                    ┌─────────────────────────┐
                    │     Soup 0.73.3         │
                    │     QLoRA / SFT         │
                    │     Docker + GPU        │
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

---

## Training Environment

Soup runs entirely inside Docker with this repository mounted at `/workspace`. The host needs only Docker — no local PyTorch, CUDA, or `soup-cli`.

| Component | Configuration |
|---|---|
| Base model | `google/gemma-3-27b-it` (gated; requires `HF_TOKEN`) |
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
| Layer streaming | Off (BETA; evaluated only after baseline) |

Two Compose services:

- **`soup`** — GPU training service; requires the NVIDIA Container Toolkit.
- **`soup-cpu`** — GPU-less validation service; not for training.

`docker compose up` is intentionally inert (`soup --help` default); no run starts implicitly.

`soup.yaml` is currently a smoke-test placeholder, validated field-by-field against Soup's schema. Final hyperparameters (LoRA rank/alpha, LR, batch size, sequence length, epochs, mixture ratios) are deliberately undecided until GPU validation.

---

## Evaluation

Three independent tracks, reported separately — never aggregated into one headline number.

| Track | Directory | Question |
|---|---|---|
| **TounsiBench** | `eval/tounsibench/` | Does the model handle real Derja in Arabic script? |
| **Arabizi** | `eval/arabizi/` | Does it handle Latin-script Tunisian and code-switching? |
| **Retention** | `eval/retention/` | What did adaptation break? |

### Non-negotiable rules

- **Baselines are mandatory.** Every fine-tuned score is paired with the same measurement on unmodified Gemma 3 27B, at pinned revisions and identical decoding settings.
- **Evaluation data is never training data.** Contamination aborts the pipeline; it is not silently filtered.
- **Retention is a release gate.** A regression beyond the agreed threshold blocks release through `scripts/release.sh`.

Benchmark definitions are the next major evaluation milestone. Track structure and reporting rules exist; no results exist yet.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B.git
cd Gemma-Tounsi-1.0-27B

# 2. Configure gated model access
cp .env.example .env
# Add your Hugging Face token to .env (gitignored, never committed)

# 3. Validate the environment
docker compose build
docker compose run --rm soup soup version
docker compose run --rm soup soup doctor
./scripts/doctor.sh

# 4. Run tests (offline — no network, no GPU, no training)
python -m pytest tests/ -v

# 5. Run the synthetic data pipeline
./scripts/prepare_data.sh

# 6. Train (requires a validated CUDA host and prepared data/train.jsonl)
./scripts/train.sh
```

Training outputs appear under `/workspace` (bind-mounted to the host) and are gitignored.

---

## Design Principles

1. **No undeclared data.** A corpus not declared in `data/manifests/*.yaml` cannot enter a training file. Licensing is checked before use.
2. **Raw data is immutable.** `data/raw/` holds sources exactly as obtained; cleaning happens downstream.
3. **Evaluation data is never training data.** Detected contamination aborts the pipeline.
4. **Holdout is reserved before selection.** Cannot appear in training data; `SplitResult` raises on shared IDs.
5. **Fail loudly.** The mixture validator rejects constraint violations instead of silently rebalancing.
6. **Deterministic outputs.** Same manifests, pinned sources, and fixed seeds produce byte-identical outputs.
7. **Delegate generic operations.** Dedup and validation are delegated to Soup; this repository implements Gemma-Tounsi-specific logic.

---

## Testing

```bash
python -m pytest tests/ -v
```

The suite runs entirely offline and covers schema validation, malformed/duplicate handling, statistics, retention selection, MSA specification, deterministic splitting, all eight mixture constraints, Soup interface compatibility, ShareGPT export, end-to-end pipeline reproducibility, and negative cases (malformed fixtures must be rejected; below-quota mixtures must fail).

---

## Repository Layout

```
Gemma-Tounsi-1.0-27B/
├── assets/
│   └── Gemma Tounsi Logo.png
├── configs/
│   ├── data/
│   │   ├── schema.yaml
│   │   ├── mixture.yaml
│   │   ├── retention.yaml
│   │   ├── msa.yaml
│   │   └── franco_tunisian.yaml
│   └── releases/
├── data/
│   ├── manifests/
│   ├── raw/
│   ├── authored/
│   ├── processed/
│   ├── splits/
│   ├── synthetic/
│   ├── train.jsonl
│   └── retention.jsonl
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATA.md
│   ├── DATA_SCHEMA.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── ENVIRONMENT.md
├── eval/
│   ├── tounsibench/
│   ├── arabizi/
│   └── retention/
├── scripts/
│   ├── doctor.sh
│   ├── prepare_data.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── release.sh
├── src/
│   └── data/
│       ├── schema.py
│       ├── validate.py
│       ├── stats.py
│       ├── dedupe.py
│       ├── split.py
│       ├── retention.py
│       ├── mixture.py
│       └── export.py
├── tests/
│   └── test_data_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── soup.yaml
├── soup-env.lock
├── LICENSE
├── NOTICE
├── CITATION.cff
└── README.md
```

---

## Large Files and Secrets

Datasets, model checkpoints, Hugging Face caches, experiment outputs, and generated artifacts are stored outside Git and gitignored. Tracked content is limited to lightweight, reproducible project assets: manifests, configuration, benchmark definitions, scripts, documentation, and synthetic test fixtures.

Secrets are never committed or baked into the Docker image. `HF_TOKEN` and `WANDB_API_KEY` are injected at run time from `.env` (gitignored) via `env_file`.

---

## License

The source code in this repository is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Model weights, datasets, and the base Gemma 3 model are governed by their own licenses and terms of use. Gemma 3 is gated and requires acceptance of its applicable license terms.

---

## Citation

If you use Gemma Tounsi 1.0 or this repository, please cite it using [`CITATION.cff`](CITATION.cff):

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