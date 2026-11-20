# Gemma Tounsi 1.0 (27B) — Checkpoint and Recovery Guide

**Experiment:** `custom_75_25_english`  
**GPU:** one NVIDIA RTX PRO 6000 Blackwell (96 GB) on Modal  
**Framework:** Soup 0.73.3 QLoRA (HF Trainer)  
**Persistent storage:** Modal Volumes (`gemma-tounsi-checkpoints` at `/checkpoints`)

---

## 1. What a resumable checkpoint contains

Each completed `checkpoint-NNN` directory (saved every `save_steps=100`) includes all state required to resume training faithfully:

| File | Required for resume | Purpose |
|---|---|---|
| `adapter_model.safetensors` | Yes | LoRA adapter weights |
| `adapter_config.json` | Yes | LoRA configuration (rank, alpha, target modules) |
| `optimizer.pt` | Yes | AdamW optimizer state |
| `scheduler.pt` | Yes | Learning-rate scheduler state |
| `trainer_state.json` | **Yes (completeness marker)** | Trainer state: `global_step`, `epoch`, etc. Written LAST by HF Trainer |
| `rng_state.pth` | Yes | RNG state (exact batch / dropout reproducibility) |
| `training_args.bin` | Yes | Training arguments |
| `tokenizer.json` (+ config) | Recommended | Tokenizer files |

**Completeness rule:** A checkpoint is resumable **only if `trainer_state.json` exists**. HF Trainer writes `trainer_state.json` last, after all other checkpoint files, so its presence guarantees the checkpoint directory was fully written and closed. Incomplete directories (missing `trainer_state.json`) are invisible to resume logic and are never committed to the Volume.

---

## 2. Where checkpoints live (persistent, not ephemeral)

Remote path inside the Modal container:

```
/checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/
└── checkpoint-100/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   ├── rng_state.pth
│   ├── training_args.bin
│   └── tokenizer.json
├── checkpoint-200/
├── latest-valid-checkpoint          (pointer file: latest committed step)
├── checkpoint_manifest.json         (metadata + per-file SHA256)
└── ...
```

This directory is on the **`gemma-tounsi-checkpoints` Modal Volume**, mounted at `/checkpoints`. Volume writes are durable only after an explicit `.commit()`. Checkpoints are **not** stored in `/tmp`, ephemeral container storage, or non-persistent Docker layers.

Your personal computer can be shut down at any time; the remote Modal Function and its Volumes keep running. Checkpoints survive container termination.

---

## 3. Save and commit intervals

| Event | Action |
|---|---|
| Every 100 training steps | HF Trainer writes `checkpoint-NNN` (all files incl. `trainer_state.json`) |
| After each completed checkpoint | Modal polling loop commits the checkpoint Volume for newly completed checkpoints only |
| At soft-stop (23 h) / `STOP_AFTER_STEPS` / hard timeout reserve | SIGTERM to Soup after the newest committed checkpoint; final Volume commit + manifest |
| At training end | Final adapter saved; HF cache Volume committed; manifest written |

**Commit safety guarantees:**
- No periodic background commits (never commits a partially written checkpoint).
- Commits happen only after a checkpoint is complete (`trainer_state.json` present).
- A threading lock serializes Volume commits.
- Already-committed steps are tracked to avoid redundant commits.
- A failed commit preserves the previous `latest-valid-checkpoint` pointer.

---

## 4. Checkpoint manifest (portability envelope)

`checkpoint_manifest.json` is written to the output directory after every session. It records:

- `project`, `base_model` (`google/gemma-3-27b-it`), `base_model_revision`
- `soup_version` (`0.73.3`), `python_version`
- `dependencies` (torch, transformers, peft, bitsandbytes, soup-cli versions)
- `training_config_fingerprint` (SHA256 of the Modal config YAML)
- `dataset_fingerprint` (SHA256 of the training dataset)
- `checkpoints[]`: each entry with `step`, `path`, and per-file `name`/`size`/`sha256`

This manifest plus the checkpoint files is the portable envelope for migrating to another cloud provider. All checkpoint formats are standard HF/PEFT/PyTorch (`.safetensors`, `.pt`, `.json`) and provider-agnostic.

---

## 5. Resume procedure

### 5.1 Resume on Modal (same cloud) — recommended

```powershell
modal run --detach modal_train_custom_75_25_english.py
```

- `soup train --resume auto` finds the latest `checkpoint-*` with `trainer_state.json` in `/checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/` and resumes from that `global_step`.
- LoRA weights, optimizer state, scheduler state, trainer state, and RNG state are restored.
- You can shut down your PC between runs; reconnect and re-run the same command.

### 5.2 Resume after an interruption / timeout

If the Modal Function stopped (soft-stop, `STOP_AFTER_STEPS`, or hard-timeout reserve):

```powershell
modal run --detach modal_train_custom_75_25_english.py
```

Resume continues from the **latest committed checkpoint**, never from step zero. In-flight (uncommitted) work at the moment of interruption is lost — this is by design (only committed checkpoints are recovery points).

### 5.3 Migrate checkpoints to another provider

1. Download from the Modal checkpoint Volume:
   ```powershell
   modal volume get gemma-tounsi-checkpoints ./local-checkpoints/
   ```
2. Verify integrity:
   ```powershell
   python scripts/verify_checkpoint_bundle.py --bundle-dir ./local-checkpoints/checkpoint-NNN_bundle/
   ```
   (Or verify `checkpoint_manifest.json` SHA256 entries directly against the downloaded files.)
3. Export a portable bundle:
   ```powershell
   python scripts/export_checkpoint_bundle.py \
       --checkpoint-dir ./local-checkpoints/checkpoint-NNN \
       --output-dir ./checkpoint_bundles/ \
       --dataset-fingerprint ./local-checkpoints/custom_75_25_english/train.jsonl.sha256 \
       --manifest ./local-checkpoints/checkpoint_manifest.json
   ```
4. Transfer the bundle to the other provider (SCP, object storage, etc.).
5. On the target: install Soup 0.73.3 (Python 3.10–3.12), apply the model-size patch, set `HF_TOKEN`, place the dataset and checkpoint, then:
   ```powershell
   soup train --config <your-config>.yaml --yes --resume auto
   ```

**Migration requirements:** same Soup version (0.73.3), same model-size patch, compatible dependency versions (especially torch/CUDA for Blackwell if migrating to another Blackwell GPU), and the `HF_TOKEN` for the gated base model.

---

## 6. Checkpoint integrity verification

- `scripts/verify_checkpoint_bundle.py` checks: bundle manifest is valid JSON, all listed files exist, SHA256 and size match for every file, critical files present (`adapter_model.safetensors`, `trainer_state.json`, `bundle_manifest.json`, `dependency_versions.txt`).
- `scripts/export_checkpoint_bundle.py` computes per-file SHA256 and records dependency versions for portability.
- The remote `checkpoint_manifest.json` provides SHA256 + size for every file in every committed checkpoint; validate after download before resuming.

**Do not resume from a checkpoint that fails checksum verification** — fall back to the previous committed checkpoint (tracked via `latest-valid-checkpoint`).

---

## 7. Files required for resumption (minimum set)

To resume training you need, at minimum, the latest **complete** checkpoint directory plus the training config and dataset:

```
checkpoint-NNN/
    adapter_model.safetensors   (required)
    adapter_config.json         (required)
    optimizer.pt                (required)
    scheduler.pt                (required)
    trainer_state.json          (required — completeness marker)
    rng_state.pth               (required)
    training_args.bin           (required)
soup.custom_75_25_english.yaml  (training config — same hyperparameters)
data/.../train.jsonl            (training dataset — same fingerprint)
```

Tokenizer and model config are re-fetched from the HF cache / Hub at runtime (cached on the `gemma-tounsi-hf-cache` Volume).

---

## 8. Retention policy (do NOT delete the only resumable checkpoint)

- Completed checkpoints are retained indefinitely on the Volume; nothing auto-deletes the latest resumable checkpoint.
- Incomplete directories (no `trainer_state.json`) are ignored by resume and never committed — they do not create a false recovery point.
- Do not configure aggressive checkpoint pruning that would remove the only valid recovery point. Keep at least the latest 2–3 completed checkpoints until training is verified complete.

---

## 9. Limitations

- **No external watchdog:** If the Modal Function crashes hard (not via the soft-stop path), no process automatically restarts it. Re-run `modal run --detach ...` to resume from the latest committed checkpoint.
- **In-flight work loss:** The step executing at interruption (uncommitted) is lost; recovery resumes at the last committed checkpoint.
- **Resume requires the same config fingerprint:** Changing hyperparameters between runs invalidates the config fingerprint; resume with the same `soup.custom_75_25_english.yaml`.
- **Volume/Secret must exist:** `gemma-tounsi-checkpoints`, `gemma-tounsi-data`, `gemma-tounsi-hf-cache`, and `gemma-hf` must be created before the first run (manual step).
- **Blackwell compatibility:** `bitsandbytes 0.50.1` 4-bit support on RTX PRO 6000 is unverified (see `pre_training_audit.md` blocker B1) — confirm before relying on long runs.
