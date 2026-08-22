# Training — Soup setup & procedure

> **STATUS:** the Docker/Soup environment is implemented and validated. **No training has been
> run**, and **no final hyperparameters have been chosen**. `soup.yaml` is currently a clearly
> labelled *smoke-test* configuration, not the Gemma Tounsi 1.0 recipe.

## 1. Division of responsibility

Training is delegated entirely to **Soup**, an external open-source training engine.

| This repository | Soup |
| --- | --- |
| `soup.yaml` (configuration) | training loop |
| prepared datasets (`data/*.jsonl`) | QLoRA implementation |
| `scripts/train.sh` (thin wrapper) | model loading & quantization |
| provenance & reproducibility records | checkpointing |
| evaluation & release gating | GPU / distributed infrastructure |

**Soup is never cloned into, vendored in, or reimplemented inside this repository.** It is consumed
as a pinned upstream container image. If something in the training loop needs to change, that
change belongs upstream in Soup — not in a local fork here.

- Upstream: <https://github.com/MakazhanAlpamys/Soup> · docs <https://trysoup.dev/docs>
- Config schema (single source of truth): `src/soup_cli/config/schema.py` at tag `v0.73.3`

## 2. Why Docker?

The host needs **only Docker** — no local PyTorch, CUDA, or `soup-cli` install.

1. **Reproducibility.** The environment is one pinned image digest, not a machine-specific pile of
   `pip install` results. Anyone rebuilding gets the same torch/transformers/peft/trl/bitsandbytes.
2. **No host pollution.** A 27B QLoRA stack pulls in a large, version-sensitive dependency tree.
   Nothing of it lands on the host.
3. **Secrets stay out of the image.** `HF_TOKEN` / `WANDB_API_KEY` are injected at *run* time from
   `.env` (gitignored). Nothing is baked into a layer.
4. **The training engine stays external.** Pinning an upstream image is what keeps Soup a
   dependency rather than a fork.

### Image

`Dockerfile` is a deliberately thin layer over the **official** Soup image:

| | |
| --- | --- |
| Base image | `ghcr.io/makazhanalpamys/soup:0.73.3` |
| Base digest | `sha256:4536816b4975a4b3abdcc3bd0761e94cb0c5f82cdee675f10f750fa19c4843d1` |
| Local tag | `gemma-tounsi-soup:0.73.3` |
| Soup version | `soup v0.73.3` |
| Python | `3.10.12` (Soup supports 3.10–3.12; **3.13 is not supported**) |
| CUDA | 12.1 runtime from the upstream base; `torch 2.13.0+cu130` |
| `WORKDIR` | `/workspace` |

**Why the tag `0.73.3`:** it is the highest version published to
`ghcr.io/makazhanalpamys/soup` and it matches the latest `soup-cli` release on PyPI. `latest` is
deliberately **not** used — a moving tag would break reproducibility. The project-specific layer
only sets `WORKDIR`, resets the upstream `ENTRYPOINT ["soup"]` (so that
`docker compose run --rm soup soup doctor` is not mangled into `soup soup doctor`), and sets an
inert default command. It copies **no** datasets, weights, or secrets.

## 3. How Soup is accessed inside the container

`docker-compose.yml` defines two services from one shared definition:

| Service | GPU | Purpose |
| --- | --- | --- |
| `soup` | reserves all NVIDIA GPUs | **the training service** |
| `soup-cpu` | none | environment validation on a host with no NVIDIA adapter. **Not for training.** |

Every invocation is a one-shot `docker compose run --rm <service> <command>`:

```bash
docker compose run --rm soup soup doctor
docker compose run --rm soup soup train --config /workspace/soup.yaml
docker compose run --rm soup bash          # interactive shell (tty enabled)
```

Mounts:

| Host | Container | Notes |
| --- | --- | --- |
| `./` (repository root) | `/workspace` | read-write; **outputs appear directly on the host** |
| `${HF_CACHE:-./.cache/huggingface}` | `/root/.cache/huggingface` | persistent HF cache |

`docker compose up` is intentionally inert (default command is `soup --help`): no run ever starts
implicitly.

## 4. Hugging Face authentication

Gemma 3 is a **gated** model, so a token with access is required.

```bash
cp .env.example .env      # then edit .env and paste your token
```

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

- `.env` is gitignored (and `.dockerignore`d) and **must never be committed**. This repository
  ships only `.env.example`, with empty placeholders.
- Compose injects it via `env_file` at run time, so `HF_TOKEN` exists as an environment variable
  inside the container and in **no image layer**.
- `env_file` is marked `required: false`, so the stack still builds and validates before `.env`
  exists.
- Verify presence **without revealing the value**:

  ```bash
  docker compose run --rm soup sh -lc 'test -n "$HF_TOKEN" && echo "HF_TOKEN is set" || echo "HF_TOKEN is not set"'
  ```

## 5. Running the health check

```bash
docker compose build                        # once
docker compose run --rm soup soup doctor    # Soup's own environment report
```

Or run every check at once (container start, Soup version, Python version, `soup doctor`, CUDA,
mount, token presence):

```bash
./scripts/doctor.sh
SERVICE=soup-cpu ./scripts/doctor.sh        # host without an NVIDIA adapter
```

`scripts/doctor.sh` **only validates**; it never starts training.

### Inspecting the Soup version

```bash
docker compose run --rm soup soup version           # -> soup v0.73.3
docker compose run --rm soup soup version --full    # + system info
docker compose run --rm soup soup version --json    # machine-readable
```

> It is `soup version`, **not** `soup --version` — that option does not exist and exits non-zero.

Python version:

```bash
docker compose run --rm soup python -c "import sys; print(sys.version)"
```

## 6. Running a smoke test

The point of the smoke test is to prove the *path*, not to produce a model: Docker works, Soup
works, CUDA works, Gemma 3 27B loads, and the QLoRA configuration is accepted.

```bash
# 1. Environment
./scripts/doctor.sh

# 2. Is the config accepted by Soup's real schema? (no training, no download)
docker compose run --rm soup python -c \
  "from soup_cli.config.schema import SoupConfig; import yaml; \
   SoupConfig(**yaml.safe_load(open('/workspace/soup.yaml'))); print('CONFIG ACCEPTED')"

# 3. Smoke-test run (needs a GPU and a prepared data/train.jsonl)
./scripts/train.sh
```

`scripts/train.sh` is a thin wrapper: it runs `soup doctor`, then
`soup train --config /workspace/soup.yaml`. It contains no training logic and installs nothing on
the host. Extra arguments are forwarded to `soup train`; `CONFIG=<path>` selects another config.

**Prerequisite:** `data/train.jsonl` does not exist yet (it is produced by
`scripts/prepare_data.sh`, which is still scaffolding), so step 3 cannot complete yet. No dataset
is downloaded by this repository.

## 7. Where outputs appear on the host

The repository root is bind-mounted at `/workspace`, so anything Soup writes under `/workspace`
is written straight into this directory:

| Config value | Container path | Host path |
| --- | --- | --- |
| `output: ./output/smoke-test` | `/workspace/output/smoke-test` | `./output/smoke-test/` |

Checkpoints, adapters, logs and the HF cache are all gitignored. Keep `output` under `/workspace`
— a path outside it would live only inside the ephemeral container and vanish with `--rm`.

## 8. Configuration

[`../soup.yaml`](../soup.yaml) is a **smoke-test placeholder**, validated field-by-field against
`SoupConfig` at `v0.73.3`. No field is invented.

Active keys: `base`, `task`, `backend`, `data.{train,format,val_split,max_length}`,
`training.{epochs,lr,batch_size,gradient_accumulation_steps,quantization,lora.*,stream_layers,seed,logging_steps}`,
`output`. QLoRA is expressed as `training.quantization: 4bit` plus `training.lora`;
`lora.target_modules: auto` lets peft resolve Gemma 3's module names, so none is guessed.

**Still deliberately undecided:** LoRA rank/alpha/dropout, explicit target modules, learning rate
and schedule, batch size, gradient accumulation, sequence length, epochs, quantization details,
and mixture proportions. The values in `soup.yaml` are placeholders, **not** decisions.

### Layer streaming is deliberately OFF

`training.stream_layers: false` (also the schema default). Soup's layer streaming is **BETA**, and
it requires `quantization: none` — which would disable the very QLoRA path we need to prove first.
It will be evaluated separately, only after the baseline path works end to end.

## 9. Validation results

Performed on a Windows 10 laptop with Docker Desktop (Docker 29.5.2, Compose v5.1.3):

| Check | Result |
| --- | --- |
| `docker compose build` | ✅ built `gemma-tounsi-soup:0.73.3` |
| `soup version` | ✅ `soup v0.73.3` |
| `soup doctor` | ✅ *All checks passed* — torch 2.13.0+cu130, transformers 4.57.6, peft 0.20.0, trl 0.28.0, bitsandbytes 0.50.1, accelerate 1.14.0 |
| Python version | ✅ `3.10.12` (supported) |
| CUDA / GPU | ❌ **not available on this host** — see below |
| Repository mounted at `/workspace` | ✅ full tree visible |
| `HF_TOKEN` passthrough | ✅ correctly reports *not set* / *is set*, value never printed |
| `soup.yaml` accepted by `SoupConfig` | ✅ QLoRA (`4bit` + LoRA r=16/α=32) accepted, `stream_layers=False` |

### Unresolved: no GPU on the validation host

The validation machine has an **Intel UHD Graphics 620** and no NVIDIA adapter, so
`docker compose run --rm soup ...` fails at container init:

```
nvidia-container-cli: initialization error: WSL environment detected but no adapters were found
```

This is the GPU reservation behaving **correctly** — the hardware prerequisite is simply absent.
It is a host limitation, not a defect in this configuration, and it is **not** worked around for
training: `scripts/train.sh` deliberately uses the GPU service and will fail loudly on such a host.
The `soup-cpu` service exists only so the non-GPU checks above can still be validated.

**Still to be confirmed on real NVIDIA hardware:**

- [ ] `torch.cuda.is_available()` is `True` inside the container
- [ ] `soup doctor` reports the GPU backend instead of *CPU only*
- [ ] Gemma 3 27B actually loads with 4-bit quantization
- [ ] the smoke-test run reaches at least one training step

Requirements for that machine: NVIDIA driver + **NVIDIA Container Toolkit**, and enough VRAM for a
27B QLoRA run. Note the validation host reported only **4 GB RAM** to the container, which is far
below what 27B needs.

## 10. Reproducibility record

Every run must capture enough context for an independent party to reproduce it:

- [x] pinned Soup image tag **and** digest (§2)
- [x] Soup version (`soup version`) and Python version
- [ ] this repository's git commit
- [ ] base model identifier + revision
- [ ] resolved training configuration
- [ ] dataset checksums and record counts
- [ ] random seeds (`training.seed`)
- [ ] GPU/driver/container versions
- [ ] full training logs

Soup can emit some of this itself (e.g. `soup train --repro-receipt <out.json>`). For a released
model, the frozen configuration is committed to `configs/releases/<version>.yaml`.

## 11. Safety gates

Before a real run starts, the following must hold — these are **not yet implemented** in
`scripts/train.sh`, which is currently the minimal doctor→train wrapper:

- [ ] `soup.yaml` is a real configuration, not the smoke-test placeholder
- [ ] `data/train.jsonl` and `data/retention.jsonl` exist and are non-empty
- [ ] **no evaluation data is reachable from the training inputs** — TounsiBench, the Arabizi
      benchmark, and the retention evaluation set are evaluation-only
