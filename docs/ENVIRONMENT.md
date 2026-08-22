# Environment Validation — Gemma Tounsi 1.0 (27B)

Validation of the training stack **before** any dataset download or training run.

    Docker → Docker Compose → NVIDIA GPU → CUDA/PyTorch → Soup
           → Hugging Face auth → Gemma 3 27B access → QLoRA readiness

- **Date:** 2026-08-22
- **Validated Soup image:** `ghcr.io/makazhanalpamys/soup:0.73.3`
- **Local image:** `gemma-tounsi-soup:0.73.3`
- **Scope:** environment only. No dataset was downloaded, no training was started.

> Secrets are never recorded here. Only the *presence* of `HF_TOKEN` is reported,
> never its value.

---

## Host environment

| Item | Value |
| --- | --- |
| OS | Windows 10 (Docker Desktop, WSL2 backend) |
| Docker | `29.5.2`, build `79eb04c` |
| Docker Compose | `v5.1.3` |
| Docker server | `29.5.2` — OS/Arch `linux/x86_64` |
| Default runtime | `runc` |
| Registered runtimes | `runc`, `io.containerd.runc.v2`, `nvidia` |
| WSL2 kernel (as seen by Soup) | `6.18.33.2-microsoft-standard-WSL2` |
| GPU model (from Docker) | **not detectable — no NVIDIA device is exposed** |
| GPU model (from host WMI) | `Intel(R) UHD Graphics 620` (integrated, non-CUDA) |
| RAM visible to container | 4 GB |
| Disk free in container | 360 GB |

`docker compose config` resolves without errors and both services (`soup`,
`soup-cpu`) materialise correctly from the `x-soup-common` anchor.

### Host notes

- The Docker daemon was **not running** at the start of validation
  (`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`).
  Docker Desktop was started, after which the daemon responded normally.
  Note that `docker compose config` succeeds even with the daemon down, because
  it only parses YAML locally — it is not proof that Docker works.
- The `nvidia` container runtime **is** registered, so the NVIDIA Container
  Toolkit is installed. The blocker is the absence of NVIDIA *hardware*.

---

## Container environment

| Component | Version |
| --- | --- |
| Soup | **`0.73.3`** (`soup version` → `soup v0.73.3`) |
| Python | **`3.10.12`** (GCC 11.4.0) |
| PyTorch | `2.13.0+cu130` |
| CUDA (PyTorch build) | `13.0` |
| CUDA (reported by `soup env lock`) | `12.1.0` |
| Transformers | `4.57.6` |
| PEFT | `0.20.0` |
| TRL | `0.28.0` |
| Datasets | `5.0.1` |
| bitsandbytes | `0.50.1` (installed) |
| Accelerate | `1.14.0` |
| huggingface-hub | `0.36.2` |
| tokenizers | `0.22.2` |

Python `3.10.12` is inside Soup's supported **3.10–3.12** range (Soup requires
`<3.13`), so the pinned image satisfies the interpreter constraint.

`soup doctor` reports **"All checks passed! Your environment is ready."** — every
`Required: yes` dependency is `OK`. Nothing had to be installed: the Soup image
ships the full training stack, so no package was added to the image.

### Warnings — not hidden

1. **GPU backend is `CPU only`.** `soup doctor` prints
   `Warning: Training will be slow without GPU.` This is the blocker below.
2. **RAM is 4 GB.** Far below what a 27B QLoRA run needs; a symptom of the same
   underpowered validation host.
3. **CUDA version mismatch in reporting.** PyTorch is built against CUDA `13.0`,
   while `soup env lock` records `12.1.0` (inherited from the upstream base image
   `nvidia/cuda:12.1.0-runtime-ubuntu22.04`). Cosmetic while no GPU is present,
   but it must be re-checked on real GPU hardware.
4. **Optional packages absent:** `wandb`, `deepspeed`, `unsloth`, `Pillow`,
   `torchao`, `sglang`, `librosa`. All are marked `optional` by `soup doctor` and
   none is required for the baseline `transformers` QLoRA/SFT path.

---

## GPU validation

| Check | Result |
| --- | --- |
| GPU visible in container | **NO** |
| GPU count | `0` |
| GPU model | none |
| Total VRAM | none |
| `torch.cuda.is_available()` | **`False`** |
| `nvidia-smi` in container | not reachable (container cannot start with GPU reservation) |

The acceptance criterion `torch.cuda.is_available() == True` is **NOT met**.

### Diagnosis

Running the GPU service fails at container creation:

```
docker compose run --rm soup soup version
=> Error response from daemon: failed to create task for container:
   ... error running prestart hook #0: exit status 1,
   stderr: Auto-detected mode as 'legacy'
   nvidia-container-cli: initialization error:
   WSL environment detected but no adapters were found
```

Root cause: **this host has no NVIDIA GPU.** The only display adapter is an
`Intel(R) UHD Graphics 620`, which cannot run CUDA. The NVIDIA container runtime
is installed and correctly registered; it simply finds no adapter to pass through.

This is **not a configuration defect**:

- `docker-compose.yml` is correct — the `soup` service reserves
  `driver: nvidia, count: all, capabilities: [gpu]`, which is the documented way
  to request GPUs, and it is precisely this reservation that fails.
- The failure is the documented, intended behaviour recorded in the compose file
  header, which predicts this exact `no adapters were found` error.
- Nothing in the project was changed to mask it, and no host modification outside
  the project was attempted.

All remaining checks were therefore run through the **`soup-cpu`** service, the
validation-only service the architecture already provides for exactly this case.

### What must be fixed before training

1. Run on a host with a real NVIDIA GPU. A 27B QLoRA/SFT run needs roughly
   **≥ 48 GB VRAM** (e.g. A100 80GB / H100); ~24 GB cards are not sufficient at
   this scale without further compromises.
2. Install the NVIDIA driver plus the NVIDIA Container Toolkit on that host
   (on WSL2, a driver with WSL CUDA support).
3. Confirm with `docker compose run --rm soup nvidia-smi`, then re-run
   `./scripts/doctor.sh` and re-generate `soup-env.lock` on that machine.

---

## Hugging Face

| Check | Result |
| --- | --- |
| `HF_TOKEN` configured | **NO** |
| Authenticated (`whoami`) | **NO** — no token in the environment |
| `.env` present | no (only `.env.example` exists) |

`docker-compose.yml` declares `env_file: [{path: .env, required: false}]`, so the
stack starts without `.env` and `HF_TOKEN` is simply absent inside the container.
That is why authentication is unconfigured — not a code defect.

No credentials were invented and no token value was printed at any point.

### To configure (operator action, outside this task)

1. `cp .env.example .env`
2. Put a Hugging Face access token in `HF_TOKEN=` inside `.env`
   (`.env` is gitignored and must never be committed).
3. Accept the Gemma license at <https://huggingface.co/google/gemma-3-27b-it>
   with the same account.

---

## Gemma

| Check | Result |
| --- | --- |
| Model ID | `google/gemma-3-27b-it` (from `soup.yaml` → `base`) |
| Model identifier valid | **YES** — `model_info` resolved the repo |
| Repository state | `gated: manual`, `private: False` |
| Metadata/config accessible | **NO** — blocked by authentication |

The repository exists and the identifier in `soup.yaml` is correct: an
unauthenticated `HfApi().model_info()` call resolved it. Fetching the actual
config metadata fails:

```
AutoConfig.from_pretrained("google/gemma-3-27b-it")
=> OSError: You are trying to access a gated repo.
   401 Client Error.
   Cannot access gated repo for url
   https://huggingface.co/google/gemma-3-27b-it/resolve/main/config.json
   Access to model google/gemma-3-27b-it is restricted.
   You must have access to it and be authenticated to access it.
```

This is the expected licensing/authentication gate for Gemma, and it was **not**
bypassed. It resolves once a token from a license-accepted account is present.
Only metadata was requested — **no model weights were downloaded**.

---

## QLoRA readiness

| Requirement | Status |
| --- | --- |
| `torch` | OK (`2.13.0+cu130`) |
| `transformers` | OK (`4.57.6`) |
| `peft` | OK (`0.20.0`) |
| `bitsandbytes` | OK (`0.50.1`) — required by `quantization: 4bit` |
| `trl` | OK (`0.28.0`) |
| `datasets` | OK (`5.0.1`) |
| `accelerate` | OK (`1.14.0`) |

**Required packages available: YES.**

Beyond imports, the QLoRA configuration objects were constructed successfully
(config objects only — no model was loaded, no LoRA attached to the real 27B):

- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", …)` → OK
- `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, CAUSAL_LM)` → OK
- `trl.SFTConfig` import → OK

This proves the 4-bit + LoRA code path in `soup.yaml` is constructible in this
runtime. It does **not** prove it runs — that needs the GPU.

---

## Soup preflight commands available

Confirmed against the **installed** `soup 0.73.3` via `--help` (no command from
another version is assumed):

| Command | Purpose |
| --- | --- |
| `soup version` | Show Soup CLI version. **`soup --version` does not exist** (`No such option '--version'`); use the subcommand. |
| `soup doctor` | Check system dependencies, GPU and compatibility. |
| `soup env lock` | Snapshot the current env into a lock file (`-o`, default `./soup-env.lock`). |
| `soup env status` | Print the currently-locked env summary. |
| `soup env check` | Compare current env against the lock and report ABI drift. |
| `soup env fix` | Render a reproducible install plan from `soup-env.lock`. |
| `soup profile` | Estimate memory, speed and GPU requirements BEFORE training (`-c`, `-g`, `--json`). |
| `soup plan` | Render a pre-flight plan (cost / ETA / SHA / VRAM), writes `soup.tfstate`. |
| `soup apply` | Execute the planned run, refusing on drift vs `soup.tfstate`. |
| `soup advise run\|explain\|compare` | Pre-flight decision: PROMPT_ENG / RAG / SFT / DPO / GRPO. |
| `soup lock write\|show\|check` | Shared run lockfile (`soup.lock`); `check` exits 3 on drift. |
| `soup data validate\|inspect\|stats\|doctor\|split\|dedup\|decontaminate\|convert\|merge` | Dataset preflight tooling (a subset; `soup data --help` lists ~45 subcommands). |
| `soup train` | Start training from a `soup.yaml` config. **Not run in this task.** |

Useful for later tasks: `soup profile` and `soup plan` estimate VRAM *before*
committing to a run, and `soup data validate` / `soup data doctor` check the
dataset and chat template without training.

---

## Environment lock

- **Path:** `soup-env.lock` (repository root)
- **Generated with:** `docker compose run --rm soup-cpu soup env lock`
- **Mechanism:** official `soup env lock` (default output `./soup-env.lock`)
- **Result:** `Locked 9 packages to soup-env.lock` — `Python: 3.10.12 | Platform: linux-x86_64 | CUDA: 12.1.0`
- **Validated with:** `soup env check` → **`ABI-clean. No drift detected.`**

No pre-existing lock file was present, so nothing was overwritten. The file
contains only package names/versions and platform metadata — no secrets. It is
tracked by Git (not matched by any `.gitignore` rule), which is intended for a
reproducibility artifact.

**This lock was captured on CPU-only hardware.** It must be regenerated on the
real GPU host, where the CUDA-linked wheels will differ.

---

## Final status

**BLOCKED**

Everything that can be validated without NVIDIA hardware passes. The single
blocker is physical: no NVIDIA GPU on this host.

| # | Requirement | Status |
| --- | --- | --- |
| 1 | Docker works | PASS (`29.5.2`; daemon had to be started) |
| 2 | Docker Compose works | PASS (`v5.1.3`, config resolves) |
| 3 | Soup starts | PASS |
| 4 | Soup version recorded | PASS (`0.73.3`) |
| 5 | Python version recorded | PASS (`3.10.12`) |
| 6 | GPU visible in container | **FAIL — no NVIDIA adapter on host** |
| 7 | CUDA/PyTorch works | PARTIAL — PyTorch `2.13.0+cu130` imports; `torch.cuda.is_available()` is `False` |
| 8 | Repository mounted correctly | PASS (`/workspace`) |
| 9 | HF authentication status known | PASS (known: not configured) |
| 10 | Gemma 3 27B metadata accessible | **FAIL — gated, 401 without a token** |
| 11 | QLoRA dependencies available | PASS |
| 12 | Soup preflight commands identified | PASS |
| 13 | Environment lock generated | PASS (`soup-env.lock`, ABI-clean) |
| 14 | No datasets downloaded | PASS (HF cache is 0 bytes) |
| 15 | No training started | PASS |
| 16 | No secrets exposed | PASS |

### Remaining blockers

1. **No NVIDIA GPU (hard blocker).** Training cannot run here. Needs a CUDA host
   with the NVIDIA Container Toolkit and enough VRAM for 27B QLoRA (≥ 48 GB).
2. **Hugging Face token not configured.** Create `.env` from `.env.example` and
   set `HF_TOKEN`.
3. **Gemma 3 27B is gated.** Accept the license at
   <https://huggingface.co/google/gemma-3-27b-it> with the token's account, then
   re-run the Gemma metadata check.

Blockers 2 and 3 are operator/credential actions. Blocker 1 requires different
hardware. None is a defect in this repository: `docker-compose.yml`, `Dockerfile`
and `soup.yaml` were all confirmed correct during validation and were left
unchanged.

### Reproducing this validation

```bash
# On a host WITHOUT an NVIDIA GPU (validation only):
SERVICE=soup-cpu ./scripts/doctor.sh

# On the real GPU host (also verifies GPU passthrough):
./scripts/doctor.sh
```
