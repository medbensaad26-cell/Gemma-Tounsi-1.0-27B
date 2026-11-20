# Gemma Tounsi 1.0 (27B) — Pre-Training Configuration Audit

**Project:** Gemma-Tounsi-1.0-27B  
**Task:** Pre-training preparation ONLY — no training launched, no GPU allocated, no credits consumed.  
**Base model:** `google/gemma-3-27b-it` (gated)  
**Method:** QLoRA (4-bit quantized LoRA fine-tuning) via Soup 0.73.3  
**Cloud target:** Modal — exactly one NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)  
**Audit status:** ✅ Configuration complete; Blackwell compatibility VERIFIED via smoke test. Cloud-side prerequisites (Modal Volumes, Secret, dataset upload, credit balance) remain to be confirmed before launch.

---

## 1. Training architecture overview

Training is delegated to **Soup 0.73.3**, an external pinned training engine (not vendored or forked in this repo). The repository provides:

| Component | Role |
|---|---|
| `soup.custom_75_25_english.yaml` | Experimental Soup training config (this experiment; production `soup.yaml` untouched) |
| `modal_train_custom_75_25_english.py` | Modal entrypoint: builds image, mounts persistent Volumes, runs `soup train`, commits checkpoints, writes manifest |
| `data/experiments/downsampled/custom_75_25_english/train.jsonl` | Training input (custom 75/25 English variant) |
| `data/splits/validation.jsonl` | Original 5% validation split (untouched) |
| `data/splits/test.jsonl` | Test set (excluded from training) |
| Modal Volumes | `gemma-tounsi-data`, `gemma-tounsi-checkpoints`, `gemma-tounsi-hf-cache` |
| Modal Secret | `gemma-hf` (provides `HF_TOKEN` for the gated base model) |

Execution path (remote, Modal):
1. `modal run --detach modal_train_custom_75_25_english.py`
2. Modal builds the training image (`debian_slim` Python 3.12 + `soup-cli[train]==0.73.3` + model-size patch).
3. The `train()` function runs on **one** `RTX-PRO-6000` GPU with 3 persistent Volumes mounted and the `gemma-hf` Secret attached.
4. It writes the embedded Modal config, validates the dataset (SHA256), validates `HF_TOKEN`, then shells out to `soup train --config ... --yes [--resume auto]`.
5. A polling loop commits each completed checkpoint (has `trainer_state.json`) to the checkpoint Volume, enforces a 23 h soft-stop with a 30 min finalization reserve before Modal's 24 h hard timeout, and on exit commits the HF cache Volume and writes `checkpoint_manifest.json`.
6. Resume on a later run: `soup train --resume auto` picks up the latest complete checkpoint.

**Your personal computer can be shut down** — the Modal Function runs detached in the cloud. Reconnect with `modal run --detach ...` again; `--resume auto` continues from the latest committed checkpoint.

---

## 2. Exact training dataset path

`data/experiments/downsampled/custom_75_25_english/train.jsonl` (repository-relative).  
On the Modal container: `/mnt/data/custom_75_25_english/train.jsonl` (Volume `gemma-tounsi-data`).

---

## 3. Actual dataset statistics (verified)

Source: `data/experiments/downsampled/custom_75_25_english/manifest.json` (produced by `scripts/create_custom_75_25_english.py`, integrity-verified).

| Metric | Value |
|---|---|
| Total examples | 384,572 |
| Total training tokens | 45,926,282 |
| Disk size | 293,746,534 bytes (~280 MB) |
| Avg tokens / example | ~119.4 |
| Arabizi | 183,423 examples (8,940,111 tokens) |
| Arabic (`ar`) | 161,837 examples (25,475,341 tokens) |
| Mixed | 11,905 examples (541,632 tokens) |
| English (`en`) | 27,407 examples (10,969,198 tokens) |

All expected values match exactly. Token counts computed with the local Gemma 3 27B tokenizer via `apply_chat_template(chatml/gemma3, add_special_tokens=False)` (total formatted sequence, user + assistant).

**Integrity (all PASS):**
- every output example exists in source
- no duplicates in output
- preserved categories complete (arabizi / ar / mixed at 100%)
- English ~25% (27,407 / 109,626)
- **no validation/test leakage**
- all source datasets unchanged (checksums verified)

The file contains **no** `data/splits/validation.jsonl` or `data/splits/test.jsonl` examples. It was derived strictly from the 75% downsampled variant; original splits were neither modified nor re-split.

---

## 4. Validation and test paths

| Split | Path | Examples | Used in training? |
|---|---|---|---|
| Training | `data/experiments/downsampled/custom_75_25_english/train.jsonl` | 384,572 | YES |
| Validation | `data/splits/validation.jsonl` | 37,036 | NO (post-training eval only) |
| Test | `data/splits/test.jsonl` | 37,038 | NO (excluded) |

**Soup 0.73.3 limitation (documented):** `DataConfig` does NOT support a separate local validation file. `val_split` is applied internally to the train file only. Because the custom train file is pure training data (no validation examples mixed in), this experiment sets `val_split: 0.0` so 100% of the custom data is used for training and **no examples are withheld for an internal validation split**. `data/splits/validation.jsonl` is preserved untouched and reserved for post-training evaluation; it is not referenced by the training config. This guarantees zero validation/test leakage into training by construction. The test set is never included in training, checkpoint selection, or hyperparameter tuning.

`soup_trainval.jsonl` (train+val combined) is NOT used as training data in this experiment.

---

## 5. Base model

`google/gemma-3-27b-it` (gated; requires HF_TOKEN with accepted Gemma license).  
Identifier verified in `soup.yaml`, `soup.modal.yaml`, and the experimental config. Model weights are **not** downloaded locally; they are fetched into the Modal HF cache Volume at runtime.

---

## 6. QLoRA configuration

Mirroring production `soup.yaml` (no arbitrary hyperparameter changes):

| Parameter | Value |
|---|---|
| Quantization | 4-bit (`bitsandbytes` `BitsAndBytesConfig`, `load_in_4bit=True`) |
| Compute dtype | bfloat16 (Soup default for 4-bit QLoRA) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `auto` (peft resolves Gemma 3 module names) |
| Gradient checkpointing | `true` |
| Micro-batch size | 1 |
| Gradient accumulation steps | 8 (effective batch 8) |
| Learning rate | 1.0e-5 |
| Scheduler | Soup default (linear warmup decay) |
| Warmup | Soup default |
| Weight decay | Soup default |
| Max sequence length | 8192 |
| Packing | not configured (Soup default) |
| Optimizer | Soup default (AdamW) |
| Training precision | 4-bit base + bf16 LoRA |
| Epochs | 2 |
| Seed | 42 |
| Save steps | 100 |
| Logging steps | 10 |
| Train on responses only | `true` |
| Train on EOT | `true` |
| Layer streaming | `false` (beta; disabled) |

**Appropriateness for Gemma 3 27B on one 96 GB RTX PRO 6000:** 4-bit QLoRA with r=16, batch 1, accum 8, gradient checkpointing, and 8192 seq length is a conservative, memory-safe configuration. 96 GB VRAM is ample for a 27B 4-bit base (~13–14 GB weights) plus LoRA adapters and activations. Memory is **not** the bottleneck; dependency/Blackwell compatibility (section 8) is the main technical risk.

**Note / discrepancy found:** production `soup.yaml` uses `lora.target_modules: auto`, while the base64-embedded `soup.modal.yaml` (existing Modal config) decodes to explicit `q_proj, k_proj, v_proj, o_proj`. The experimental config follows `soup.yaml` (`auto`). This pre-existing inconsistency between `soup.yaml` and `soup.modal.yaml` is documented, not introduced here.

---

## 7. RTX PRO 6000 configuration

- GPU identifier: `gpu="RTX-PRO-6000"` (exactly one GPU, **no fallback list**, no silent substitution).
- Modal docs and pricing page confirm `RTX-PRO-6000` is a valid Modal GPU identifier (96 GB, Blackwell), available on Modal since April 2026.
- Modal SDK in use: **1.6.4** (string GPU shorthand supported; legacy `modal.gpu.*` objects deprecated).
- Price: **$3.03 / GPU-hour** on-demand (verified against Modal pricing page, Nov 18 2026: `$0.000842/sec`).
- Architecture: Blackwell, compute capability 12.0 (sm_120), 96 GB GDDR7 with ECC.
- Modal host driver: CUDA Driver API 13.3.

**Blackwell dependency compatibility (risk — see section 8):**
- PyTorch: Blackwell (sm_120) requires a torch build with CUDA >= 12.8 (torch 2.7+ / cu128+). The Soup 0.73.3 Docker image pins `torch 2.13.0+cu130`; the Modal image resolves `soup-cli[train]==0.73.3` dependencies **unpinned**, so the torch/CUDA build in the Modal container is not locked and must be verified.
- bitsandbytes 0.50.1: **no confirmed Blackwell 4-bit (nf4) support** — this is the principal compatibility risk for QLoRA on Blackwell. May fail or fall back to slow CPU offloading. Do **not** blindly upgrade; verify before training and upgrade only if tested.
- transformers 4.57.6, accelerate 1.14.0, peft 0.20.0: expected compatible with Blackwell given a suitable CUDA toolchain.
- Python 3.12 in the Modal image (within Soup's supported 3.10–3.12 range).

---

## 8. CUDA and dependency compatibility

| Component | Validated version | Notes |
|---|---|---|
| Soup | 0.73.3 | Pinned; no upgrade |
| Python (Modal image) | 3.12 | Soup supports >=3.10,<3.13 |
| GPU | RTX-PRO-6000 (Blackwell, CC 12.0 / sm_120) | 95.01 GB VRAM verified |
| CUDA driver (Modal host) | 13.3 | Driver API; containers inherit |
| Container CUDA runtime | 13.2 | Verified via smoke test (`torch.version.cuda`) |
| torch | 2.15.0+cu132 | **VERIFIED** — sm_120 support confirmed |
| transformers | 5.23.0 | **VERIFIED** |
| peft | 0.21.3 | **VERIFIED** |
| bitsandbytes | 0.50.2 | **VERIFIED** — Blackwell 4-bit NF4 confirmed working |
| accelerate | 1.16.0 | **VERIFIED** |
| huggingface-hub | — | Required for gated base model access |

**Compatibility status: ALL CLEARED.** The smoke test confirmed on real RTX PRO 6000 hardware:
- `torch.cuda.is_available() == True`, compute capability 12.0 (sm_120)
- `bnb.nn.Linear4bit(quant_type="nf4")` forward + backward PASS
- PEFT LoRA forward + backward + gradients PASS
- All 7 mandatory smoke-test checks PASS

The local validation host (Windows laptop, Intel UHD 620, no NVIDIA GPU) cannot run CUDA; this compatibility was confirmed on the Modal RTX PRO 6000 via the smoke test.

---

## 9. Persistent storage design

Three Modal Volumes, all persistent across runs and container lifecycles:

| Volume | Mount path | Purpose |
|---|---|---|
| `gemma-tounsi-data` | `/mnt/data` | Training dataset (`custom_75_25_english/train.jsonl` + SHA256 fingerprint) |
| `gemma-tounsi-checkpoints` | `/checkpoints` | LoRA checkpoints, final adapter, training output, manifest |
| `gemma-tounsi-hf-cache` | `/hf-cache` | Hugging Face model/tokenizer cache (base model downloaded once, reused across runs) |

Environment variables set in the container: `HF_HOME=/hf-cache`, `HF_HUB_CACHE=/hf-cache/hub`, `TRANSFORMERS_CACHE=/hf-cache/transformers`.

**Authentication:** `HF_TOKEN` supplied via Modal Secret `gemma-hf` (attached to the Function). Never hardcoded, never printed, never committed. `.env.modal` (local, gitignored) holds a token for local use only; `.env` is gitignored and must never be committed.

Checkpoints and final adapter are written to `/checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/` on the persistent checkpoint Volume — **not** `/tmp` or ephemeral container storage. Shutting down your personal computer does **not** affect the remote job.

Volumes and the Secret must be created manually before the first run (documented in section 10 / `checkpoint_and_recovery.md`). Nothing is auto-created that would spend credits unexpectedly.

---

## 10. Checkpoint format and save intervals

Checkpoints are written by Soup/HF Trainer every `save_steps=100`. Each `checkpoint-*` directory contains:

| File | Purpose |
|---|---|
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | LoRA configuration |
| `optimizer.pt` | AdamW optimizer state |
| `scheduler.pt` | Learning-rate scheduler state |
| `trainer_state.json` | Trainer state (`global_step`, `epoch`, etc.) — written LAST |
| `rng_state.pth` | RNG state |
| `training_args.bin` | Training arguments |
| `tokenizer.json` (+ config) | Tokenizer (final save) |

A checkpoint is considered **complete and resumable only when `trainer_state.json` exists** (HF Trainer writes it last, so its presence guarantees all other checkpoint files were written and closed).

**Save/commit policy:**
- Periodic checkpoint saving every 100 steps (Soup `save_steps`).
- Modal Volume committed **only after** a checkpoint is complete (`trainer_state.json` present), never periodically in the background (avoids committing partial checkpoints).
- A threading lock serializes commits; already-committed steps are tracked to avoid redundant commits.
- Failed commits preserve the previous valid `latest-valid-checkpoint` pointer.
- Final adapter saved at end of training; manifest generated after all commits.

**Retention:** Not configured aggressively — completed checkpoints are retained; only incomplete directories are ignored by resume logic. The only resumable checkpoint is not auto-deleted.

**Checkpoint manifest:** `checkpoint_manifest.json` in the output directory records project, base model, base model revision, Soup/Python/dependency versions, config fingerprint, dataset fingerprint, and per-checkpoint file list with SHA256 + sizes. This is the portability envelope for migration to another provider.

---

## 11. Resume procedure

**Resume on Modal (same cloud):**
```powershell
modal run --detach modal_train_custom_75_25_english.py
```
`soup train --resume auto` finds the latest `checkpoint-*` directory containing `trainer_state.json` in `/checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/` and resumes from that global step, restoring LoRA weights, optimizer state, scheduler state, trainer state, RNG state, and training arguments.

**Shutdown safety:** Your local PC can be shut down at any time; the detached Modal Function keeps running. Reconnect later and run the same command to resume.

**Migrate to another provider (portable):**
1. `modal volume get gemma-tounsi-checkpoints ./local-checkpoints/`
2. `python scripts/verify_checkpoint_bundle.py --bundle-dir ...` (integrity)
3. `python scripts/export_checkpoint_bundle.py --checkpoint-dir .../checkpoint-NNN --output-dir ./checkpoint_bundles/`
4. Transfer the bundle (standard `.safetensors`/`.pt`/`.json` HF/PEFT/PyTorch formats) to the other provider.
5. Recreate a Soup 0.73.3 environment (Python 3.10–3.12, soup-cli 0.73.3, model-size patch) and run `soup train --config <config> --yes --resume auto`.

**Limitation:** Automatic recovery is **not** promised if the architecture did not implement it — here, resume is implemented via `--resume auto` + committed checkpoints, but there is no external watchdog restarting a dead Modal Function. Modal Function timeouts and crashes are handled by the soft-stop/commit logic; a hard container failure loses only the in-flight (uncommitted) step.

---

## 12. Data integrity checks

- Dataset SHA256 computed and stored at `/mnt/data/custom_75_25_english/train.jsonl.sha256`; validated before training starts; mismatch aborts.
- Checkpoint files checksummed in `checkpoint_manifest.json` (per-file SHA256 + size).
- `scripts/verify_checkpoint_bundle.py` verifies all bundle SHA256 checksums, file sizes, and critical-file presence.
- Original `data/splits/validation.jsonl` and `data/splits/test.jsonl` checksums recorded and verified unchanged by `scripts/create_custom_75_25_english.py`.
- No validation/test examples in the training file (integrity check `no_validation_test_leakage: true`).

---

## 13. Cost estimates and limitations

**Verified pricing (Modal on-demand, Nov 18 2026):**
- RTX PRO 6000: **$3.03 / GPU-hour** (`$0.000842 / sec`).
- Exactly one GPU requested; no multi-GPU or fallback.

**Additional cost categories (not billed by GPU):**
- CPU / RAM for the Modal container (included in Function pricing; Modal charges per-second for GPU + allocated CPU/memory).
- Modal Volume storage (~$0.15 / GB-month per existing project docs; checkpoints + 96 GB model cache are the main consumers).
- No data transfer charges documented for Volume upload/download at this level.

**Credit balance:** User-confirmed **$29.10 USD** available (treated as the
absolute maximum project budget — not permission to spend it). See
`docs/training/modal_budget_plan.md` for the full allocation:
$1.00 smoke test / $24.00 training / $2.00 final artifact export & inference
verification / $2.10 emergency reserve (untouched).
Full detail, throughput analysis, and cost scenarios are in the budget plan.

**Estimated runtime / total cost:** Cannot be estimated reliably without
measured tokens/sec on the RTX PRO 6000 (no GPU available locally, no
training run performed). Cost = runtime(hours) × $3.03 + CPU/RAM/storage/
keep-alive/image-build. GPU-only: $24.00 ≈ 7.92 GPU-hours (28,500 s); one
epoch (45.93M tokens) requires ≥1,611 tokens/sec GPU-only — and realistic
4-bit QLoRA 27B throughput on a single card is uncertain and likely below
that. **One full epoch is not financially realistic at the all-in $24.00
training budget; the configured 2-epoch run (91.85M tokens) is impossible
within $29.10.** Use `STOP_AFTER_STEPS` or `MAX_RUNTIME_SECONDS` (23 h) to
bound spend; monitor via the Modal web dashboard (no billing API
integration — documented limitation).

**Limitations:**
- No GPU on the local validation host; CUDA/Blackwell compatibility (especially bitsandbytes 4-bit) is unverified locally.
- Soup 0.73.3 has no external validation file; `validation.jsonl` is post-training only (section 4).
- `soup-cli[train]` dependencies are unpinned in the Modal image (version drift vs. the pinned Docker stack possible).
- Volumes/Secret must be created manually before first run; dataset must be uploaded manually.
- No automatic budget monitoring or external watchdog restart.

---

## 14. Remaining blockers

| # | Blocker | Severity | Status | Required action before launch |
|---|---|---|---|---|
| B1 | `bitsandbytes` Blackwell 4-bit support on RTX PRO 6000 | HIGH | ✅ CLEARED | Smoke test confirmed NF4 forward/backward on sm_120. No action needed. |
| B2 | Modal torch/CUDA build sm_120 support | MEDIUM | ✅ CLEARED | Smoke test confirmed torch 2.15.0+cu132 reports compute capability 12.0. No action needed. |
| B3 | Workspace credit balance verification | HIGH | ⏳ PENDING | User will provide latest Modal dashboard balance. Do not assume $29.10 remains. Confirm before launch. |
| B4 | Modal resources not yet created | MEDIUM | ⏳ PENDING | Create Volumes (`gemma-tounsi-data`, `gemma-tounsi-checkpoints`, `gemma-tounsi-hf-cache`) and Secret (`gemma-hf`) manually; upload the dataset via `scripts/upload_modal_data.ps1` (adapted to `custom_75_25_english/train.jsonl` path). |
| B5 | `validation.jsonl` not consumable by Soup 0.73.3 during training | LOW (documented) | ✅ ACCEPTED | Accept post-training-only validation; `val_split: 0.0` in experimental config. No action needed. |

---

## 15. Exact training launch command

Do **not** run until blockers B1–B4 are cleared and you explicitly instruct launch.

```powershell
# Prerequisites (run once, manually):
#   modal volume create gemma-tounsi-data
#   modal volume create gemma-tounsi-checkpoints
#   modal volume create gemma-tounsi-hf-cache
#   modal secret create gemma-hf HF_TOKEN=<your_token>
#   .\scripts\upload_modal_data.ps1   (upload custom_75_25_english/train.jsonl to /mnt/data/custom_75_25_english/train.jsonl)

# Launch (detached — safe to shut down your PC afterwards):
modal run --detach modal_train_custom_75_25_english.py

# Monitor:
modal app logs gemma-tounsi-custom-train
modal volume list

# Resume later (same command; --resume auto continues from latest committed checkpoint):
modal run --detach modal_train_custom_75_25_english.py
```

`modal run --detach` returns immediately; no GPU is allocated by this document and no credits are consumed by reading it.

---

## Readiness verdict

**CONDITIONALLY READY — TWO CLOUD-SIDE PREREQUISITES REMAIN**

The configuration files are complete and locally validated. Blackwell compatibility is verified on real hardware. Training must not launch until the following are resolved:

1. **Confirm credit balance via Modal dashboard (B3)** — The user will provide the latest balance. Do not assume $29.10 remains. Confirm before any billable operation.
2. **Create Modal Volumes + `gemma-hf` Secret and upload the dataset (B4)** — Create `gemma-tounsi-data`, `gemma-tounsi-checkpoints`, `gemma-tounsi-hf-cache` Volumes and `gemma-hf` Secret manually. Upload `data/experiments/downsampled/custom_75_25_english/train.jsonl` to `/mnt/data/custom_75_25_english/train.jsonl` on the data Volume (adapt `scripts/upload_modal_data.ps1` to the custom dataset path).

Once B3 and B4 are cleared, the experiment is fully configured and the launch command in section 15 is ready to execute. Do not launch training or allocate the RTX PRO 6000 until then.
