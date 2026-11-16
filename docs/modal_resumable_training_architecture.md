# Gemma Tounsi 1.0 (27B) — Modal Resumable Training Architecture

**Project:** Gemma-Tounsi-1.0-27B
**Soup version:** 0.73.3 (unchanged — no upgrade)
**Modal SDK:** 1.6.4
**Modal profile:** medbensaad26-cell
**Selected GPU:** NVIDIA L40S 48GB
**Base model:** google/gemma-3-27b-it (gated)

---

## 1. Final Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL WINDOWS HOST                                    │
│                                                                               │
│  • Python 3.14.3 (venv: %USERPROFILE%\.venvs\gemma-modal)                   │
  • Modal SDK 1.6.4 authenticated (profile: medbensaad26-cell)                  │
  • soup_modal_app.py (generated, not yet deployed)                              │
  • scripts/upload_modal_data.ps1 (dataset upload, not yet run)                  │
  • scripts/export_checkpoint_bundle.py (checkpoint portability)                 │
  • scripts/verify_checkpoint_bundle.py (checkpoint integrity)                   │
  • tests/test_modal_architecture.py (37 regression tests, all pass)             │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                              │ modal run --detach
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODAL CLOUD (L40S 48GB)                             │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Modal App: gemma-tounsi-train                                      │   │
│  │                                                                       │   │
│  │  Image: debian_slim + soup-cli[train]==0.73.3                       │   │
│  │         + model-size patch applied at build                         │   │
│  │                                                                       │   │
│  │  Volumes (mounted, persistent across runs):                         │   │
│  │    • gemma-tounsi-data    → /mnt/data/                              │   │
│  │    • gemma-tounsi-checkpoints → /checkpoints/                       │   │
│  │    • gemma-tounsi-hf-cache  → /hf-cache/                            │   │
│  │                                                                       │   │
│  │  Secret: gemma-hf (provides HF_TOKEN)                               │   │
│  │                                                                       │   │
│  │  Function: train()                                                   │   │
│  │    • Validates dataset + SHA256                                      │   │
│  │    • Validates HF_TOKEN                                              │   │
│  │    • Background commit thread (30s interval)                        │   │
│  │    • soup train --config /root/soup.modal.yaml --yes --resume auto   │   │
│  │    • Soft-stop on MAX_RUNTIME_SECONDS or STOP_AFTER_STEPS             │   │
│  │    • Final Volume commit + checkpoint manifest                       │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Eight Blockers and Resolution

| Blocker | Issue | Resolution |
|---------|-------|------------|
| **1. Model-size patch** | `model_size_from_name("google/gemma-3-27b-it")` returns 7B instead of 27B | Patch files (`patches/soup-0.73.3-model-size.patch` + `scripts/apply_soup_compat_patch.py`) copied into Modal image via `add_local_file`, applied after `pip install`, smoke-tested in image build |
| **2. HF authentication** | Gemma 3 27B is gated; needs HF_TOKEN | `modal.Secret.from_name("gemma-hf")` attached to `@app.function(secrets=[hf_secret])`. Validated in `validate_hf_token()` — fails immediately if missing. Secret must be created manually: `modal secret create gemma-hf HF_TOKEN=...` |
| **3. Dataset mounting** | No dataset in Modal container; relative paths break | Modal Volume `gemma-tounsi-data` mounted at `/mnt/data`. Dataset at `/mnt/data/soup_trainval.jsonl`. SHA256 validation before training. Test split excluded. Upload via `scripts/upload_modal_data.ps1` |
| **4. Checkpoint storage** | Ephemeral container loses checkpoints | Modal Volume `gemma-tounsi-checkpoints` mounted at `/checkpoints`. Output dir: `/checkpoints/gemma-tounsi-1.0-27b`. Background commit thread commits Volumes every 30s. Final commit after training. |
| **5. HF cache** | Model re-downloaded every run | Modal Volume `gemma-tounsi-hf-cache` mounted at `/hf-cache`. `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` env vars set. Cache committed after model downloads. |
| **6. Automatic resume** | Training restarts from scratch after interruption | `--resume auto` passed to `soup train`. Soup 0.73.3 finds latest `checkpoint-*` with `trainer_state.json`. Checkpoint manifest with SHA256 checksums generated after each session. |
| **7. Runtime/budget** | 24h Modal timeout; no budget visibility | `MAX_RUNTIME_SECONDS=86400` (Modal max). `STOP_AFTER_STEPS` optional step-based stop. SIGTERM sent for graceful stop at checkpoint boundary. No Modal billing API — monitor via Modal web dashboard. |
| **8. Output paths** | Relative/ephemeral paths lose data | All paths absolute: `/mnt/data/...`, `/checkpoints/...`, `/hf-cache/...`. No relative paths. Volumes never deleted by cleanup. |

---

## 3. Volume Names and Mount Paths

| Volume Name | Mount Path | Purpose |
|-------------|------------|---------|
| `gemma-tounsi-data` | `/mnt/data` | Training dataset (`soup_trainval.jsonl`) + SHA256 fingerprint |
| `gemma-tounsi-checkpoints` | `/checkpoints` | LoRA checkpoints, final adapter, training output |
| `gemma-tounsi-hf-cache` | `/hf-cache` | Hugging Face model/tokenizer cache |

**Volumes must be created manually before first run:**
```powershell
modal volume create gemma-tounsi-data
modal volume create gemma-tounsi-checkpoints
modal volume create gemma-tounsi-hf-cache
```

**Dataset must be uploaded manually before first run:**
```powershell
.\scripts\upload_modal_data.ps1
```

**Secret must be created manually before first run:**
```powershell
modal secret create gemma-hf HF_TOKEN=<your_token>
```

---

## 4. Secret Requirements

| Secret Name | Key | Purpose | Created By |
|-------------|-----|---------|------------|
| `gemma-hf` | `HF_TOKEN` | Hugging Face User Access Token for gated Gemma 3 27B model | User (manual) |

**Never:**
- Hardcode the token in `soup_modal_app.py` or `soup.modal.yaml`
- Print the token in logs
- Commit the token to Git
- Store the token in the Docker image

---

## 5. Remote Image Contents

The Modal image is built from `modal.Image.debian_slim()`:

1. `pip_install("soup-cli[train]==0.73.3")` — Soup 0.73.3 with training extras (torch, transformers, peft, trl, datasets, bitsandbytes, accelerate)
2. `add_local_file("patches/soup-0.73.3-model-size.patch", remote_path="/tmp/...")` — Patch file
3. `add_local_file("scripts/apply_soup_compat_patch.py", remote_path="/tmp/...")` — Apply script
4. `run_commands("python3 /tmp/apply_soup_compat_patch.py")` — Apply patch
5. `run_commands('python3 -c "from soup_cli.utils.gpu import model_size_from_name as m; assert m("google/gemma-3-27b-it") == 27..."')` — Smoke test

**Critical:** The patch is applied in the remote image, not inherited from the local Docker build. This is why `add_local_file` + `run_commands` are required.

---

## 6. Checkpoint Lifecycle

### Checkpoint Save (Soup 0.73.3 HF Trainer)

Soup 0.73.3 saves checkpoints every `save_steps=100` via HF Trainer. Each `checkpoint-*` directory contains:

| File | Purpose |
|------|---------|
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | LoRA configuration |
| `optimizer.pt` | AdamW optimizer state |
| `scheduler.pt` | Learning-rate scheduler state |
| `trainer_state.json` | Trainer state (global_step, epoch, etc.) |
| `rng_state.pth` | RNG state |
| `training_args.bin` | Training arguments |
| `tokenizer.json` | Tokenizer (final save) |

### Checkpoint Commit (Modal Volume)

Modal Volumes buffer writes locally and require explicit `.commit()` for durability. The architecture implements:

1. **Background commit thread:** Commits all 3 Volumes every 30 seconds during training
2. **Final commit:** After training completes (or is interrupted via SIGTERM)
3. **Manifest generation:** `checkpoint_manifest.json` with metadata and SHA256 checksums

### Checkpoint Manifest

Written to `/checkpoints/gemma-tounsi-1.0-27b/checkpoint_manifest.json`:

```json
{
  "project": "gemma-tounsi-1.0-27b",
  "base_model": "google/gemma-3-27b-it",
  "soup_version": "0.73.3",
  "python_version": "3.10.12",
  "dependencies": {"torch": "2.x", "transformers": "4.x", "peft": "x.x", ...},
  "training_config_fingerprint": "sha256_of_soup.modal.yaml",
  "dataset_fingerprint": "sha256_of_soup_trainval.jsonl",
  "checkpoints": [
    {
      "step": 100,
      "path": "checkpoint-100",
      "files": [
        {"name": "adapter_model.safetensors", "size": 1234, "sha256": "..."}
      ]
    }
  ]
}
```

---

## 7. Resume Procedure

### Resume on Modal (same cloud)

```powershell
modal run --detach soup_modal_app.py
```

Soup 0.73.3's `--resume auto` finds the latest `checkpoint-*` directory containing `trainer_state.json` in `/checkpoints/gemma-tounsi-1.0-27b/` and resumes from that global step.

### Resume on Alibaba Cloud (portable)

1. Download checkpoint from Modal Volume:
   ```powershell
   modal volume get gemma-tounsi-checkpoints ./local-checkpoints/
   ```

2. Verify integrity:
   ```powershell
   python scripts/verify_checkpoint_bundle.py --bundle-dir ./local-checkpoints/checkpoint-XXX_bundle/
   ```

3. Export bundle:
   ```powershell
   python scripts/export_checkpoint_bundle.py \
       --checkpoint-dir ./local-checkpoints/checkpoint-XXX \
       --output-dir ./checkpoint_bundles/
   ```

4. Transfer bundle to Alibaba Cloud (SCP, OSS, etc.)

5. Set up Soup 0.73.3 environment on Alibaba (Python 3.10, soup-cli 0.73.3, model-size patch)

6. Resume:
   ```powershell
   soup train --config soup.modal.yaml --yes --resume auto
   ```

**GPU provider compatibility:** Checkpoint files (`.safetensors`, `.pt`, `.json`) are standard HF/PEFT/PyTorch formats and are portable across GPU providers. The main compatibility concern is software version alignment (same transformers, peft, bitsandbytes, soup-cli versions).

---

## 8. Detached Execution

### Launch

```powershell
modal run --detach soup_modal_app.py
```

This starts training in the cloud and returns immediately. Your computer can be shut down.

### Monitor

| Task | Command |
|------|---------|
| Get app info | `modal app list` |
| View logs | `modal app logs gemma-tounsi-train` |
| Check status | `modal app list` (look for running/stopped status) |
| List volumes | `modal volume list` |
| Inspect checkpoint | Download via `modal volume get` |

### Reconnect another day

```powershell
modal run --detach soup_modal_app.py
```

Soup `--resume auto` finds the latest checkpoint and continues from there.

### Runtime limit handling

- **24-hour Modal timeout:** Enforced by `timeout=86400` on `@app.function`
- **Soft-stop:** SIGTERM sent when `MAX_RUNTIME_SECONDS` or `STOP_AFTER_STEPS` is reached
- **Graceful stop:** Training stops at the next checkpoint boundary (save_steps=100)
- **Previous checkpoint preserved:** The last committed checkpoint remains available for resume

### Cost control

- No automatic budget limit — monitor via Modal web dashboard
- Estimated cost: ~$3–$5 for 2 epochs on L40S (rough estimate)
- Storage: ~$0.15/GB-month for checkpoint volumes
- Stop training early with `STOP_AFTER_STEPS` if budget is a concern

---

## 9. Alibaba Cloud Portability

### Portable artifacts

| Artifact | Format | Portable |
|----------|--------|----------|
| LoRA adapter weights | `.safetensors` | ✓ |
| Optimizer state | `optimizer.pt` | ✓ |
| Scheduler state | `scheduler.pt` | ✓ |
| Trainer state | `trainer_state.json` | ✓ |
| RNG state | `rng_state.pth` | ✓ |
| Training config | YAML | ✓ |
| Tokenizer | `.json` + vocab | ✓ |

### Non-portable (Modal-specific)

| Component | Alternative on Alibaba |
|-----------|----------------------|
| Modal Volume | Alibaba Cloud NAS / OSS |
| Modal Secret | Alibaba Cloud KMS / Secret Manager |
| `modal run` CLI | Standard `soup train` command |
| Modal App object | Direct container execution |

### Migration steps

1. Export checkpoint bundle from Modal
2. Transfer to Alibaba Cloud Object Storage (OSS)
3. Provision GPU instance (L40S equivalent: Alibaba Cloud gn7i or gn7)
4. Install Python 3.10 + soup-cli 0.73.3 + model-size patch
5. Set HF_TOKEN as environment variable
6. Mount dataset and checkpoint storage
7. Run `soup train --config soup.modal.yaml --yes --resume auto`

---

## 10. Remaining Limitations

1. **No Modal billing API:** Cannot automatically detect remaining promotional credit. Monitor via Modal web dashboard.
2. **VRAM estimate accuracy:** The ~17.9 GB estimate is computed by Soup's formula using the corrected 27B size. Not independently validated against measured memory usage.
3. **Modal volume creation:** Volumes and secrets must be created manually before first run. Not automated in the application code.
4. **Dataset upload:** Must be uploaded manually via `scripts/upload_modal_data.ps1` before training.
5. **Python 3.10 EOL:** Modal will drop Python 3.10 support in v1.7.0. Plan to upgrade to 3.11+ before then.
6. **No automatic checkpoint download:** Checkpoints must be manually downloaded from Modal Volumes for Alibaba Cloud migration.

---

## 11. Files Created or Modified

| File | Action | Purpose |
|------|--------|---------|
| `soup_modal_app.py` | Created | Modal application with all 8 blockers fixed |
| `soup.modal.yaml` | Created | Modal-specific training config (absolute paths) |
| `scripts/upload_modal_data.ps1` | Created | PowerShell script to upload dataset to Modal Volume |
| `scripts/export_checkpoint_bundle.py` | Created | Export checkpoint bundle for Alibaba Cloud portability |
| `scripts/verify_checkpoint_bundle.py` | Created | Verify checkpoint bundle integrity |
| `tests/test_modal_architecture.py` | Created | 37 regression tests (all pass) |
| `docs/modal_resumable_training_architecture.md` | Created | This document |
| `Dockerfile` | Not modified | Patch already integrated in previous step |
| `soup.yaml` | Not modified | Original training config preserved |
| `data/splits/*` | Not modified | Original datasets preserved |

---

## 12. Next Steps (User Action Required)

1. Create Modal Volumes: `modal volume create gemma-tounsi-data gemma-tounsi-checkpoints gemma-tounsi-hf-cache`
2. Create Modal Secret: `modal secret create gemma-hf HF_TOKEN=<your_token>`
3. Upload dataset: `.\scripts\upload_modal_data.ps1`
4. Deploy: `modal run --detach soup_modal_app.py`
5. Monitor: `modal app logs gemma-tounsi-train`
6. Download checkpoints when done: `modal volume get gemma-tounsi-checkpoints ./`

**Do not push to GitHub until ready.** The `.modal.toml` authentication file is outside the repository and contains credentials.
