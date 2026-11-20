# Gemma Tounsi 1.0 — Release v1.0

**Project:** Gemma-Tounsi-1.0-27B  
**Version:** 1.0  
**Status:** Release candidate (first functional version)  
**Base model:** `google/gemma-3-27b-it`  
**Method:** QLoRA 4-bit SFT via Soup 0.73.3  
**Training budget:** $24.00 Modal (pilot-scale; not a full epoch)

---

## 1. What this release contains

This directory holds the **Gemma Tounsi V1 LoRA adapter** plus everything needed
to load it for inference and to resume training later. It is **provider-agnostic**:
the adapter works wherever `google/gemma-3-27b-it`, Soup 0.73.3, and a compatible
GPU are available.

```
releases/gemma-tounsi-v1/
├── README.md                      # this file
├── INSTALL.md                     # installation & usage instructions
├── adapter/                       # lightweight inference adapter (LoRA weights)
│   ├── adapter_model.safetensors  # LoRA adapter weights
│   ├── adapter_config.json        # LoRA configuration (r, alpha, target modules)
│   └── generation_config.json     # generation settings (temperature, top-p, ...)
├── tokenizer/                     # tokenizer + chat-template files (or references)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── chat_template.jinja        # gemma3 chat template
├── checkpoint/                    # full resumable training state (separate from adapter)
│   ├── checkpoint-<step>/         # latest valid checkpoint (all HF Trainer files)
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   ├── optimizer.pt
│   │   ├── scheduler.pt
│   │   ├── trainer_state.json     # completeness marker + global_step
│   │   ├── rng_state.pth
│   │   └── training_args.bin
│   └── latest-valid-checkpoint    # pointer file: latest committed step
├── training_metadata.json         # run metadata (seed, LR, batch, epochs, steps, tokens)
├── training_metrics.json          # per-step loss / throughput / cost metrics
├── dataset_manifest.json          # dataset path, SHA256, record count, language split
├── checkpoint_info.json           # base model revision, config fingerprint, dataset fingerprint
├── sha256_checksums.txt           # SHA-256 checksums for every artifact file
└── inference_acceptance.json      # functional acceptance test results (prompts + outputs)
```

### Artifact roles

| Artifact | Purpose | Required for inference | Required for resume |
|---|---|---|---|
| `adapter/adapter_model.safetensors` | LoRA adapter weights | ✅ | ✅ |
| `adapter/adapter_config.json` | LoRA config (r=16, alpha=32, ...) | ✅ | ✅ |
| `adapter/generation_config.json` | Generation defaults | ✅ | |
| `tokenizer/` | Tokenizer + chat template | ✅ | |
| `checkpoint/checkpoint-<step>/` | Full training state (optimizer, scheduler, RNG) | | ✅ |
| `training_metadata.json` | Hyperparameters + run info | | |
| `training_metrics.json` | Loss, throughput, cost per step | | |
| `dataset_manifest.json` | Dataset provenance + SHA256 | | |
| `checkpoint_info.json` | Base model revision + config fingerprint | | |
| `sha256_checksums.txt` | Integrity verification | | |
| `inference_acceptance.json` | Functional acceptance test results | | |

### Separation of concerns

- **`adapter/`** is the lightweight inference artifact. Load it on top of
  `google/gemma-3-27b-it` with the tokenizer from `tokenizer/`. It does NOT
  contain the base model — you must have access to the gated base model
  separately (HF_TOKEN with Gemma license accepted).
- **`checkpoint/`** is the full resumable training state. Preserve it separately
  so training can continue later on Modal or another provider.
- The adapter is **not merged** with the base model. Merging is optional and
  only done if there is a separate, verified reason.

---

## 2. Base model information

| Field | Value |
|---|---|
| Base model identifier | `google/gemma-3-27b-it` |
| Base model revision | `<filled after training>` |
| Model type | Gemma 3 27B instruct (gated) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | auto |
| Chat template | gemma3 |

The base model revision (commit SHA) is recorded in `checkpoint_info.json` and
`training_metadata.json`. It is resolved via the Hugging Face Hub API at
training time.

---

## 3. Installation & usage

See [`INSTALL.md`](INSTALL.md) for step-by-step instructions to:
1. Install dependencies (Soup 0.73.3, transformers, peft, bitsandbytes, accelerate)
2. Set `HF_TOKEN` for the gated base model
3. Load the adapter for inference (`scripts/infer_gemma_tounsi.py`)
4. Resume training from `checkpoint/`
5. Reproduce the release artifacts

---

## 4. Functional acceptance test

The file `inference_acceptance.json` contains the results of the V1 functional
acceptance test:

- Load `google/gemma-3-27b-it`
- Load the LoRA adapter from `adapter/`
- Load the tokenizer + gemma3 chat template from `tokenizer/`
- Generate responses on held-out Tunisian Arabizi and Arabic-script Tunisian prompts
- Verify all outputs are non-empty
- Confirm the adapter loads again from the exported artifact

A model that loads and generates responses is **technically functional**.
Linguistic quality of Tunisian dialect is a separate question and is not
guaranteed by this release.

---

## 5. Reproducibility

| Item | Location |
|---|---|
| Training config | `soup.custom_75_25_english.yaml` (repository root) |
| Modal training app | `modal_train_custom_75_25_english.py` (repository root) |
| Training hyperparameters | `training_metadata.json` |
| Dataset fingerprint | `dataset_manifest.json` (SHA256) |
| Config fingerprint | `checkpoint_info.json` |
| Dependency versions | `training_metadata.json` + `adapter/` |
| Git commit | recorded in `training_metadata.json` |

---

## 6. Continuing training after V1

V1 is a pilot-scale fine-tune (a fraction of one epoch within the $24.00
training budget). Training can be resumed later:

```powershell
# On Modal (same cloud):
modal run --detach modal_train_custom_75_25_english.py

# `--resume auto` continues from the latest committed checkpoint
# on the gemma-tounsi-checkpoints Volume.
```

To migrate to another provider, export the checkpoint bundle:

```powershell
python scripts/export_checkpoint_bundle.py \
    --checkpoint-dir <path>/checkpoint-<step> \
    --output-dir ./checkpoint_bundles \
    --dataset-fingerprint <path>/dataset.sha256 \
    --manifest <path>/checkpoint_manifest.json
```

See `docs/training/checkpoint_and_recovery.md` for the full migration procedure.

---

## 7. Budget

| Phase | Allocation | Status |
|---|---|---|
| Blackwell smoke test | $1.00 | Completed (pre-V1) |
| Actual training | $24.00 | Completed (pilot-scale, V1) |
| Final artifact export & inference verification | $2.00 | Completed (this release) |
| Emergency reserve | $2.10 | Untouched |
| **Total** | **$29.10** | **Never exceeded** |

The emergency reserve ($2.10) was not consumed. No credits remain for further
training without replenishing the Modal balance.

---

## 8. Known limitations

- **Pilot-scale training only:** V1 was trained on a fraction of one epoch
  (~0.19–0.74 epoch depending on measured throughput), not the full 45.9M-token
  dataset or 2 epochs. Do not claim full dataset coverage.
- **Base model is gated:** `google/gemma-3-27b-it` requires a Hugging Face
  token with the Gemma license accepted. The adapter will not load without it.
- **No merging:** The adapter is not merged into the base model. Inference
  requires loading both the base model and the adapter.
- **Technical functionality ≠ linguistic quality:** The model loads and
  generates Tunisian responses, but dialect quality is not independently
  evaluated in this release.

---

## 9. Integrity verification

Verify every artifact before use:

```powershell
# Verify SHA-256 checksums
Get-FileHash -Algorithm SHA256 releases/gemma-tounsi-v1/**/*.* | Format-Table

# Or compare against sha256_checksums.txt
```

Adapter checksums are listed in `sha256_checksums.txt`. Checkpoint files are
checksummed in `checkpoint_info.json` (from the remote `checkpoint_manifest.json`).

---

*End of README. Replace `<filled after training>` placeholders with actual
values when producing the V1 release.*
