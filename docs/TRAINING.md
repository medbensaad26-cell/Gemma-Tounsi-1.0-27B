# Training — Soup setup & procedure

> **STATUS: PLACEHOLDER.** No training has been run. The Soup environment has not yet been
> installed or validated, and **no hyperparameters have been chosen**.

## 1. Division of responsibility

Training is delegated entirely to **Soup**, an external open-source training engine.

| This repository | Soup |
| --- | --- |
| `soup.yaml` (configuration) | training loop |
| prepared datasets (`data/*.jsonl`) | QLoRA implementation |
| `scripts/train.sh` (thin wrapper) | model loading & quantization |
| provenance & reproducibility records | checkpointing |
| evaluation & release gating | GPU / distributed infrastructure |

**Soup is never cloned into, vendored in, or reimplemented inside this repository.** It is
installed as a dependency (see the Dockerfile). If something in the training loop needs to change,
that change belongs upstream in Soup — not in a local fork inside this repo.

<!-- TODO: record the upstream Soup repository URL, the pinned commit SHA, and a link to its
     configuration reference. -->

## 2. Environment

*TODO after the Soup environment is validated.*

- [ ] Hardware requirements for QLoRA on a 27B model (GPU memory, count, interconnect)
- [ ] CUDA / driver versions
- [ ] Python version and pinned dependencies
- [ ] Soup installation procedure and pinned commit
- [ ] Docker image digest (`Dockerfile`, `docker-compose.yml`)
- [ ] Access to the gated Gemma 3 base model (`HF_TOKEN` via `.env`)

## 3. Configuration

The training configuration lives in [`../soup.yaml`](../soup.yaml), currently an inert placeholder.

**Deliberately undecided at this stage:** LoRA rank/alpha/dropout, target modules, learning rate
and schedule, batch size and gradient accumulation, sequence length, epochs, quantization details,
and mixture proportions. These will be set only after the Soup environment is validated, using
Soup's real schema — never guessed.

## 4. Procedure

*TODO once implemented. Intended flow:*

```bash
# 1. Prepare data (not yet implemented)
scripts/prepare_data.sh

# 2. Train via Soup (not yet implemented)
scripts/train.sh

# 3. Evaluate all three tracks (not yet implemented)
scripts/evaluate.sh --track all --model <adapter path>
```

## 5. Reproducibility record

Every run must capture enough context for an independent party to reproduce it:

- [ ] this repository's git commit
- [ ] Soup commit SHA
- [ ] base model identifier + revision
- [ ] resolved training configuration
- [ ] dataset checksums and record counts
- [ ] random seeds
- [ ] GPU/driver/container versions
- [ ] full training logs

For a released model, the frozen configuration is committed to `configs/releases/<version>.yaml`.

## 6. Safety gates

Before a run starts, `scripts/train.sh` must verify:

- [ ] `soup.yaml` is a real configuration, not the placeholder
- [ ] `data/train.jsonl` and `data/retention.jsonl` exist and are non-empty
- [ ] **no evaluation data is reachable from the training inputs** — TounsiBench, the Arabizi
      benchmark, and the retention evaluation set are evaluation-only

## 7. Outputs

Checkpoints, adapters, and logs are written outside Git (see `.gitignore`) to the location
configured via `.env`.

<!-- TODO: document the checkpoint layout Soup produces and how adapters are selected for
     evaluation and release. -->
