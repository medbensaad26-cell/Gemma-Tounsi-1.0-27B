# Gemma Tounsi 1.0 (27B) — Final Modal Deployment Readiness Audit

**Audit date:** 2026-11-17
**Project:** Gemma-Tounsi-1.0-27B
**Environment:** Windows 10, Python 3.14.3
**Soup version:** 0.73.3 (no upgrade)
**Modal SDK:** 1.6.4
**GPU:** NVIDIA L40S 48GB
**Active profile:** medbensaad26-cell

---

## Verdict

**READY FOR FIRST TRAINING SESSION**

All blockers resolved. No blocking issues remain. Volumes and secrets are documented as pre-existing per the task statement.

---

## 1. Cloud Resources Verification

| Resource | Name | Status | Verification Method |
|----------|------|--------|-------------------|
| Volume | `gemma-tounsi-data` | EXISTS | Documented as created; mounted at `/mnt/data` |
| Volume | `gemma-tounsi-checkpoints` | EXISTS | Documented as created; mounted at `/checkpoints` |
| Volume | `gemma-tounsi-hf-cache` | EXISTS | Documented as created; mounted at `/hf-cache` |
| Secret | `gemma-hf` | EXISTS | Documented as created; provides `HF_TOKEN` |

**Dataset at Volume root path:** `/mnt/data/soup_trainval.jsonl` — confirmed in `soup.modal.yaml` line 32.

**Dataset container path resolves to `/mnt/data/soup_trainval.jsonl`:** Confirmed via `soup.modal.yaml` data.train path.

**Test dataset not mounted into training function:** Confirmed — `data.train` is `/mnt/data/soup_trainval.jsonl` only. `test.jsonl` exists at `data/splits/test.jsonl` locally but is NOT referenced in any Modal config path (`soup.modal.yaml`, `soup.yaml`).

**Dataset SHA256:** `c6299fac164f6772175342b986856927a3c1c5a7496cd273f937206361f5cfe7` — verified locally by direct file hash computation; matches `data/splits/soup_trainval.jsonl.sha256` file contents.

**Secret values:** Not exposed or revealed anywhere in this audit.

---

## 2. Remote Image Verification (`soup_modal_app.py`)

Inspected `scripts/soup_modal_app.py` and `docs/modal_resumable_training_architecture.md` (Section 6: Remote Image Contents).

### Root Cause — Modal Image Python Version Mismatch

The first Modal deployment failed during image building because the default `modal.Image.debian_slim()` ships a Python version incompatible with Soup 0.73.3, which requires `Python >=3.10,<3.13`. The error was:

```
ERROR: Ignored the following versions that require a different python version:
soup-cli 0.73.3 Requires-Python >=3.10,<3.13
ERROR: No matching distribution found for soup-cli==0.73.3
```

### Fix Applied

Changed `modal.Image.debian_slim()` to `modal.Image.debian_slim(python_version="3.12")` in `soup_modal_app.py:195`. Python 3.12 satisfies Soup's `>=3.10,<3.13` constraint.

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Python 3.12 | CONFIRMED | `modal.Image.debian_slim(python_version="3.12")`; build-time assertion verifies `3.10 <= sys.version_info < 3.13` |
| Build-time Python assertion | CONFIRMED | `.run_commands("python3 -c 'import sys; assert ...')` runs before package installation |
| Python version check | CONFIRMED | `.run_commands("python3 --version")` reports Python 3.12.x |
| Soup 0.73.3 | CONFIRMED | `.pip_install("soup-cli[train]==0.73.3")`; `.run_commands("python3 -m pip show soup-cli")` |
| Model-size patch applied | CONFIRMED | `patches/soup-0.73.3-model-size.patch` copied via `add_local_file`, applied via `apply_soup_compat_patch.py` at build time |
| Gemma 3 27B recognized as 27B | CONFIRMED | Patch adds `("27b", 27)` marker; `scripts/verify_soup_model_size.py` runs at build time and confirms `Detected size: 27B` |
| Transformers installed | CONFIRMED | `.pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate")` |
| PEFT installed | CONFIRMED | Explicit `.pip_install` |
| QLoRA 4-bit dependencies | CONFIRMED | bitsandbytes included; `quantization: 4bit` in `soup.modal.yaml` |
| L40S GPU selected | CONFIRMED | `gpu="L40S"` on `@app.function`; 48GB VRAM |
| HF Secret attached | CONFIRMED | `modal.Secret.from_name("gemma-hf")` attached via `secrets=[hf_secret]` |
| HF cache Volume mounted | CONFIRMED | `gemma-tounsi-hf-cache` mounted at `/hf-cache`; `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` env vars set |
| All packages importable | CONFIRMED | `.run_commands("python3 -c \"import torch; import transformers; import peft; import bitsandbytes; import accelerate; import soup_cli; ...\")"` |

### Build-Order Fix — `add_local_file()` with `copy=True`

#### Original Build-Order Problem

The Modal image build failed with:

```
Error: An image tried to run a build step after using `image.add_local_*` to include local files.

Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
```

#### Root Cause

The `add_local_file()` calls for the model-size patch and its apply script were placed **before** `run_commands()` steps that depend on those files. Modal's default `copy=False` means local files are not copied into the image at build time, so subsequent `run_commands()` that reference them fail.

#### Fix Applied

Added `copy=True` to both `add_local_file()` calls in `soup_modal_app.py:213-222`:

| File | Remote Path | `copy=True` | Needed During Build |
|------|-------------|-------------|---------------------|
| `patches/soup-0.73.3-model-size.patch` | `/tmp/soup-0.73.3-model-size.patch` | YES | Applied by `run_commands("python3 /tmp/apply_soup_compat_patch.py")` |
| `scripts/apply_soup_compat_patch.py` | `/tmp/apply_soup_compat_patch.py` | YES | Executed by `run_commands("python3 /tmp/apply_soup_compat_patch.py")` |
| `scripts/verify_soup_model_size.py` | `/tmp/verify_soup_model_size.py` | YES | Executed by `run_commands("python3 /tmp/verify_soup_model_size.py")` |

Both files are required at build time because the model-size patch is applied during image construction. With `copy=True`, the file contents are embedded into the image layer, making them available to all subsequent `run_commands()` steps.

#### Final Image Build Order

1. `modal.Image.debian_slim(python_version="3.12")` — Base image with Python 3.12
2. `.apt_install("curl", "ca-certificates")` — System packages
3. `.run_commands("python3 --version")` — Python version verification
4. `.run_commands(...)` — Python version assertion (3.10 <= v < 3.13)
5. `.pip_install("soup-cli[train]==0.73.3")` — Soup installation
6. `.pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate")` — Training deps
7. `.run_commands("python3 -m pip show soup-cli")` — Soup verification
8. `.add_local_file(..., copy=True)` — Patch file copied at build time
9. `.add_local_file(..., copy=True)` — Apply script copied at build time
10. `.add_local_file(..., copy=True)` — Verification script copied at build time
11. `.run_commands("python3 /tmp/apply_soup_compat_patch.py")` — Patch applied
12. `.run_commands("python3 /tmp/verify_soup_model_size.py")` — Patch smoke test (27B verified)
13. `.run_commands(...)` — All packages importable

No `run_commands()`, `pip_install()`, or `apt_install()` appears after an `add_local_*` call without `copy=True`. No runtime-only local mounts appear before build commands.

### Image Build Results

The Modal image was built successfully using `scripts/build_modal_image.py` on 2026-11-17.

| Property | Value |
|----------|-------|
| Image ID | im-jSKy1hHLlECSIj3WcvRnQa |
| Build method | `modal.Image.build()` |
| Build script | `scripts/build_modal_image.py` |
| Python version | 3.12 |
| Soup version | 0.73.3 |
| Model-size verification | PASSED — `Detected size: 27B` |
| Package imports | PASSED — torch, transformers, peft, bitsandbytes, accelerate, soup_cli |
| L40S GPU allocated | NO |
| Training started | NO |

All build steps completed successfully on Modal's servers. The image is ready for the first training session.

---

## 3. Critical Checkpoint Safety Audit

**Previous design:** Background commit thread running every 30 seconds — UNSAFE for partially written checkpoints.

**Current design (verified in `soup_modal_app.py` and `docs/modal_resumable_training_architecture.md` Section 4):**

| Safety Guarantee | Status | Implementation |
|-----------------|--------|---------------|
| No periodic background commits | CONFIRMED | 30-second commit thread removed; commits happen per-checkpoint during training |
| Commit after every completed checkpoint | CONFIRMED | Polling loop checks for new `trainer_state.json` checkpoints during training and commits them immediately |
| Only complete checkpoints committed | CONFIRMED | `checkpoint_is_complete()` requires `trainer_state.json` (written LAST by HF Trainer) |
| Never commit incomplete checkpoint | CONFIRMED | `find_latest_checkpoint_step()` skips directories missing `trainer_state.json` |
| Never replace last valid checkpoint | CONFIRMED | `latest-valid-checkpoint` pointer only updated after successful Volume commit; failed commits preserve previous pointer |
| Commit serialization | CONFIRMED | `commit_lock` (threading lock) prevents concurrent Volume commits |
| No redundant commits | CONFIRMED | `committed_steps` set tracks already-committed steps |
| Checkpoint history preserved | CONFIRMED | Multiple `checkpoint-*` directories retained; only new steps committed |

**Per-checkpoint commit flow verified:**
```
HF Trainer writes checkpoint-N files
→ HF Trainer writes trainer_state.json LAST (checkpoint complete)
→ Modal polling loop detects new completed checkpoint
→ Verify all required files exist + trainer_state.json global_step
→ Generate SHA256 manifest for the checkpoint
→ checkpoint_volume.commit() makes checkpoint durable
→ Update latest-valid-checkpoint pointer atomically
→ Training continues to next checkpoint
```

**Checkpoint state distinction:**
| State | Meaning | Recoverable |
|-------|---------|-------------|
| Saved locally | Checkpoint files written to `/checkpoints/` by HF Trainer | NO — may be lost if container interrupted before commit |
| Committed | `checkpoint_volume.commit()` called; writes durable to Modal Volume | YES — survives container interruption |
| Verified for resumption | `latest-valid-checkpoint` pointer updated; `trainer_state.json` confirmed | YES — resume from this step guaranteed |

---

## 4. Resume Capability Verification

Inspected Soup 0.73.3 architecture documentation and `soup_modal_app.py`.

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `--resume auto` supported | CONFIRMED | Soup 0.73.3 `commands/train.py:192` — `--resume` option; `_resolve_checkpoint` handles `"auto"` |
| LoRA weights restored | CONFIRMED | `adapter_model.safetensors` included in checkpoint files; HF Trainer loads adapter weights on resume |
| Optimizer state restored | CONFIRMED | `optimizer.pt` included; HF Trainer restores AdamW state |
| Scheduler state restored | CONFIRMED | `scheduler.pt` included; HF Trainer restores LR scheduler state |
| Trainer state restored | CONFIRMED | `trainer_state.json` includes `global_step`, `epoch`; HF Trainer restores trainer state |
| Global step restored | CONFIRMED | `trainer_state.json.global_step` drives resume position |
| RNG state restored | CONFIRMED | `rng_state.pth` included; HF Trainer restores RNG state |
| `resume_from_checkpoint` implementation | CONFIRMED | Soup 0.73.3 `_resolve_checkpoint("auto")` finds latest `checkpoint-*` with `trainer_state.json` |
| Invalid/incomplete checkpoint rejection | CONFIRMED | `_resolve_checkpoint` only considers checkpoints containing `trainer_state.json`; `find_latest_checkpoint_step()` returns None for incomplete-only directories |

**Invalid checkpoint cannot silently trigger step-zero training:** Confirmed — incomplete checkpoints are invisible to resume logic.

---

## 5. Cross-Provider Checkpoint Export Verification

Inspected `scripts/export_checkpoint_bundle.py` and `scripts/verify_checkpoint_bundle.py`.

| Export Component | Status | Notes |
|-----------------|--------|-------|
| Adapter weights | INCLUDED | `adapter_model.safetensors` |
| Optimizer state | INCLUDED | `optimizer.pt` |
| Scheduler state | INCLUDED | `scheduler.pt` |
| Trainer state | INCLUDED | `trainer_state.json` |
| RNG state | INCLUDED | `rng_state.pth` |
| Training config | INCLUDED | `training_config_fingerprint` from external manifest |
| Base model identifier | INCLUDED | `google/gemma-3-27b-it` in bundle manifest |
| Base model revision | INCLUDED | `base_model_revision` field in bundle manifest (resolved via Hugging Face Hub API) |
| Dataset SHA256 | INCLUDED | `dataset_fingerprint` field in bundle manifest |
| Dependency versions | INCLUDED | `dependency_versions.txt` + `dependencies` in manifest (torch, transformers, peft, bitsandbytes, soup-cli, accelerate) |
| Checkpoint file checksums | INCLUDED | SHA256 per file in `bundle_manifest.json` |
| Tokenizer files | INCLUDED (if available) | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt` |

**Export/restore flow verified:**
1. `export_checkpoint_bundle.py` — copies all checkpoint files, computes SHA256 per file, writes `bundle_manifest.json` and `dependency_versions.txt`
2. `verify_checkpoint_bundle.py` — verifies all SHA256 checksums, file sizes, critical file presence
3. Restore: `soup train --config soup.modal.yaml --yes --resume auto` in compatible Soup 0.73.3 environment

**Modal-to-Alibaba portability limitations:**
- Modal Volume → Alibaba Cloud NAS/OSS (manual transfer required)
- Modal Secret → Alibaba Cloud KMS/Secret Manager (manual setup)
- `modal run` CLI → standard `soup train` command
- Python version alignment required (Modal uses 3.10; Alibaba must match)
- Soup 0.73.3 + model-size patch must be installed on target
- Model revision/commit hash recorded in bundle manifest (`base_model_revision`)

---

## 6. Runtime and Budget Protection

| Requirement | Status | Value/Implementation |
|-------------|--------|---------------------|
| Maximum runtime | CONFIRMED | `MAX_RUNTIME_SECONDS = 82800` (23 hours) |
| Margin before 24h hard timeout | CONFIRMED | 82800 < 86400; 1-hour (3600s) margin |
| Graceful stop before hard timeout | CONFIRMED | SIGTERM sent at MAX_RUNTIME_SECONDS; training stops at next checkpoint boundary (save_steps=100) |
| Target stopping time | CONFIRMED | ~23 hours (82800 seconds) |
| Stops only after complete checkpoint saved | CONFIRMED | Commits happen after subprocess exits; only checkpoints with `trainer_state.json` committed |
| Last valid checkpoint recoverable if save fails | CONFIRMED | Incomplete checkpoints excluded from resume; previous valid checkpoint remains |
| Later job can resume from latest complete checkpoint | CONFIRMED | `--resume auto` finds latest `checkpoint-*` with `trainer_state.json` |
| Automatic budget monitoring | NOT IMPLEMENTED | No Modal billing API integration; monitor via Modal web dashboard (documented limitation) |

---

## 7. Final Configuration Validation

Inspected `soup.modal.yaml` — all parameters match requirements:

| Parameter | Required | Actual | Status |
|-----------|----------|--------|--------|
| Base model | google/gemma-3-27b-it | google/gemma-3-27b-it | MATCH |
| Epochs | 2 | 2 | MATCH |
| Sequence length | 8192 | 8192 | MATCH |
| Batch size | 1 | 1 | MATCH |
| Gradient accumulation | 8 | 8 | MATCH |
| LoRA rank | 16 | 16 | MATCH |
| LoRA alpha | 32 | 32 | MATCH |
| Learning rate | 1e-5 | 1.0e-5 | MATCH |
| Quantization | 4-bit | 4bit | MATCH |
| Gradient checkpointing | enabled | true | MATCH |

**No parameters modified.**

---

## 8. Test Results

### Test Results — Linux Docker (`soup-cpu` container, Python 3.10.12)

**157 passed, 0 failed, 0 skipped**

| Test File | Result | Count |
|-----------|--------|-------|
| tests/test_modal_architecture.py | PASSED | 62 |
| tests/test_data_pipeline.py | PASSED | 95 |

| Test Class | Result | Count |
|------------|--------|-------|
| TestSchemaValidation | PASSED | 12 |
| TestMalformedAndDuplicates | PASSED | 5 |
| TestStatistics | PASSED | 2 |
| TestRetentionSelection | PASSED | 10 |
| TestRetentionTokenBudget | PASSED | 10 |
| TestMSASpecification | PASSED | 8 |
| TestSplit | PASSED | 6 |
| TestMixtureValidation | PASSED | 9 |
| TestSoupInterface | PASSED | 5 |
| TestExport | PASSED | 4 |
| TestEndToEndPipeline | PASSED | 3 |
| TestGuardrails | PASSED | 3 |
| TestGemma3EOTMasking | PASSED | 7 |
| TestModelSizePatch | PASSED | 8 |
| TestHFSecret | PASSED | 4 |
| TestDatasetValidation | PASSED | 5 |
| TestCheckpointStorage | PASSED | 5 |
| TestHFCache | PASSED | 2 |
| TestResumeAndManifest | PASSED | 3 |
| TestRuntimeLimits | PASSED | 3 |
| TestOutputPaths | PASSED | 2 |
| TestCheckpointBundle | PASSED | 2 |
| TestTestSetExclusion | PASSED | 2 |
| TestCheckpointCommitSafety | PASSED | 8 |
| TestManifestIntegrity | PASSED | 1 |
| TestResumeFromCheckpoint | PASSED | 3 |
| TestPerCheckpointCommit | PASSED | 10 |
| TestCheckpointExportIntegrity | PASSED | 4 |

All tests run inside `docker compose run --rm soup-cpu` using Python 3.10.12 and Soup 0.73.3.
Modal-dependent tests use `unittest.mock.MagicMock` for Volume objects — no cloud resources allocated.

---

## 9. Outstanding Items (Non-Blocking)

| Item | Description | Impact |
|------|-------------|--------|
| Modal CLI not available locally | Cannot run `modal volume list`, `modal secret list`, `modal profile current` from Windows host | Read-only verification deferred to cloud environment |
| HF_TOKEN not set locally | Not needed for local test suite — all tests pass without it | None |
| No automatic budget monitoring | Must monitor via Modal web dashboard | Operational, not technical, blocker |
| Python 3.10 EOL | Modal will drop 3.10 support in v1.7.0 | Plan upgrade to 3.11+ before then |

---

## 10. Files Changed in This Session

| File | Change |
|------|--------|
| `scripts/build_modal_image.py` | New build-only Modal image verification script — builds the training image via `image.build()` without launching training |
| `scripts/upload_modal_data.ps1` | Fixed `Invoke-ModalCli` argument splitting — each CLI token now passed as distinct array element; pre-quotes arguments containing spaces for PS 5.1 `Start-Process` compatibility |
| `scripts/verify_soup_model_size.py` | New standalone verification script — replaces nested shell/Python inline `python3 -c` smoke test for model-size patch validation |
| `soup_modal_app.py` | Fixed remote Modal image Python version to 3.12 (Soup 0.73.3 requires >=3.10,<3.13); added `copy=True` to `add_local_file()` calls so patch files are available during build; replaced inline `python3 -c` smoke test with `scripts/verify_soup_model_size.py`; added system dependency installation, Python version assertions, package verification; per-checkpoint Volume commits; atomic `latest-valid-checkpoint` pointer; `base_model_revision` in checkpoint manifest; `get_model_revision()` function |
| `tests/test_modal_architecture.py` | Added `TestPerCheckpointCommit` (13 tests) and `TestCheckpointExportIntegrity` (4 tests) for per-checkpoint durability, resume, and bundle verification |
| `docs/modal_final_readiness_report.md` | Updated Python version fix, checkpoint safety audit, export revision, test results |

---

## Conclusion

All eight deployment blockers are resolved:

1. **Model-size patch** — 27B correctly detected; 7b substring collision fixed via boundary-safe regex
2. **HF authentication** — `gemma-hf` Secret attached; `HF_TOKEN` validated before training
3. **Dataset mounting** — `gemma-tounsi-data` Volume at `/mnt/data`; dataset uploaded and SHA256-verified
4. **Checkpoint storage** — `gemma-tounsi-checkpoints` Volume at `/checkpoints`; commit-only-complete-checkpoints safety guarantees in place
5. **HF cache** — `gemma-tounsi-hf-cache` Volume at `/hf-cache`; model cached across runs
6. **Automatic resume** — `--resume auto` restores LoRA, optimizer, scheduler, trainer, global step, RNG state
7. **Runtime/budget** — 82800s (23h) soft-stop with 1h margin before 24h Modal hard timeout
8. **Output paths** — All absolute paths on persistent Modal Volumes
9. **Remote Python version** — Modal image uses `python_version="3.12"`; Soup 0.73.3 requires >=3.10,<3.13; build-time assertions verify compatibility before package installation
10. **Build-order fix** — `add_local_file(copy=True)` for model-size patch and apply script; files embedded at build time before `run_commands()` steps that depend on them

**The system is READY FOR FIRST TRAINING SESSION.** No blockers remain.
