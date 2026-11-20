# Gemma Tounsi 1.0 (27B) — Final GO/NO-GO Report

**Project:** Gemma-Tounsi-1.0-27B  
**Report date:** 2026-11-19  
**Status:** CONDITIONAL GO — cleared to proceed once two cloud-side prerequisites are resolved

---

## 1. Executive Summary

The Blackwell compatibility blocker is CLOSED. The smoke test passed all 7 mandatory checks on real RTX PRO 6000 hardware. The V1 delivery chain (checkpoint → commit → manifest → adapter export → inference) is complete and documented. The budget model has been corrected to an all-in cost model (GPU + CPU + RAM + image build buffer).

**Verdict: CONDITIONAL GO.** Training may launch once:
1. Modal credit balance is confirmed (user to provide)
2. Modal Volumes + `gemma-hf` Secret are created and dataset is uploaded

No billable operation has been performed during this audit. No GPU was allocated. No credits were consumed.

---

## 2. Compatibility — RESOLVED ✅

| Check | Result | Evidence |
|---|---|---|
| GPU detection (RTX PRO 6000, sm_120) | PASS | Smoke test |
| CUDA initialization | PASS | Smoke test |
| PyTorch CUDA operations | PASS | Smoke test |
| NF4 initialization | PASS | Smoke test |
| NF4 forward + backward | PASS | Smoke test |
| AdamW optimizer update | PASS | Smoke test |
| PEFT LoRA attachment + gradient + optimizer | PASS | Smoke test |
| Memory monitoring | PASS | Smoke test |

**Verified environment:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.01 GB VRAM
- Compute capability: 12.0 (sm_120)
- CUDA: 13.2 (container), 13.3 (host driver)
- PyTorch: 2.15.0+cu132
- bitsandbytes: 0.50.2 (NF4 confirmed working)
- PEFT: 0.21.3
- Transformers: 5.23.0
- Accelerate: 1.16.0
- Soup: 0.73.3

**Docs updated:** `docs/training/blackwell_compatibility.md`

---

## 3. Budget — ALL-IN COST MODEL ✅

The previous GPU-only cost model undercounted total spend. The training app now tracks the all-in rate:

| Component | Rate/sec | Rate/hr |
|---|---|---|
| GPU (1× RTX PRO 6000) | $0.00084200 | $3.0312 |
| CPU (1 core) | $0.00001310 | $0.04716 |
| RAM (4 GiB) | $0.00000888 | $0.03197 |
| **All-in** | **$0.00086398** | **$3.11033** |

CPU + RAM add ~2.6% overhead. Image build (~$2.00 buffer) is billed separately.

**Budget allocation (hard ceilings):**
| Purpose | Allocation | Status |
|---|---|---|
| Smoke test | $1.00 | Completed |
| Training (all-in) | $24.00 | Available |
| Finalization | $2.00 | Reserved |
| Emergency reserve | $2.10 | Untouched |
| **Total** | **$29.10** | **Never exceeded** |

**A full epoch is NOT financially realistic** (~0.17–0.66 epoch depending on throughput). V1 must be functional, not complete.

**Docs updated:** `docs/training/modal_budget_plan.md`  
**Code fixed:** `modal_train_custom_75_25_english.py` (all-in rate + budget stop)

---

## 4. Pre-Training Audit — UPDATED ✅

| Blocker | Severity | Status |
|---|---|---|
| B1: bitsandbytes Blackwell 4-bit support | HIGH | ✅ CLEARED (smoke test verified) |
| B2: Modal torch/CUDA sm_120 support | MEDIUM | ✅ CLEARED (smoke test verified) |
| B3: Credit balance confirmation | HIGH | ⏳ PENDING (user to provide) |
| B4: Modal Volumes + Secret + dataset upload | MEDIUM | ⏳ PENDING (manual setup) |
| B5: Validation file limitation | LOW | ✅ ACCEPTED (post-training eval only) |

**Docs updated:** `docs/training/pre_training_audit.md`

---

## 5. Infrastructure — VERIFIED (read-only) ✅

| Resource | Reference | Status |
|---|---|---|
| Volume: gemma-tounsi-data | `modal_train_custom_75_25_english.py:154` | Referenced, not created |
| Volume: gemma-tounsi-checkpoints | `modal_train_custom_75_25_english.py:155` | Referenced, not created |
| Volume: gemma-tounsi-hf-cache | `modal_train_custom_75_25_english.py:156` | Referenced, not created |
| Secret: gemma-hf | `modal_train_custom_75_25_english.py:261` | Referenced, not created |
| Dataset path | `data/experiments/downsampled/custom_75_25_english/train.jsonl` | Exists locally, not uploaded |
| Remote dataset path | `/mnt/data/custom_75_25_english/train.jsonl` | Correct in training config |

**Note:** `scripts/upload_modal_data.ps1` still references `soup_trainval.jsonl` — must be adapted to `custom_75_25_english/train.jsonl` before upload. Not changed per restrictions (no resource creation).

---

## 6. V1 Delivery Chain — VERIFIED ✅

| Stage | Implementation | Status |
|---|---|---|
| Checkpoint saving | HF Trainer, save_steps=100, all required files | ✅ In code |
| Checkpoint commits | Polling loop, only after trainer_state.json | ✅ In code |
| Latest valid pointer | `latest-valid-checkpoint` file | ✅ In code |
| HF cache commit | `commit_hf_cache_volume()` in finally block | ✅ In code |
| Manifest generation | `generate_manifest()` with SHA256 + versions | ✅ In code |
| Resume | `soup train --resume auto` | ✅ In code |
| Adapter export | `scripts/export_checkpoint_bundle.py` | ✅ Exists |
| Inference loading | `scripts/infer_gemma_tounsi.py` | ✅ Exists |
| Release artifacts | `releases/gemma-tounsi-v1/` (structure documented) | ✅ Exists (placeholder files) |

**Note:** `scripts/release.sh` is a placeholder (not implemented). This is a post-training concern.

---

## 7. Production Configuration — UNMODIFIED ✅

| File | target_modules | Status |
|---|---|---|
| `soup.yaml` | `auto` | ✅ Unchanged |
| `soup.modal.yaml` | `auto` | ✅ Unchanged |
| `soup.custom_75_25_english.yaml` | `auto` | ✅ Unchanged |

The toy-model PEFT test in the smoke test uses explicit `q_proj`/`v_proj` target modules — separate from production Gemma config which uses `auto`. No production config was modified.

---

## 8. GO/NO-GO Decision

| Criterion | Decision |
|---|---|
| Blackwell compatibility | GO (verified) |
| V1 delivery chain | GO (complete) |
| Budget model | GO (all-in, corrected) |
| Production configs | GO (unmodified) |
| Credit balance confirmed | NO-GO (pending user input) |
| Modal resources created | NO-GO (pending manual setup) |
| Dataset uploaded | NO-GO (pending manual upload) |

**OVERALL: CONDITIONAL GO**

Training is cleared to proceed **only after**:
1. User confirms Modal credit balance (provide latest dashboard figure)
2. Modal Volumes (`gemma-tounsi-data`, `gemma-tounsi-checkpoints`, `gemma-tounsi-hf-cache`) and Secret (`gemma-hf`) are created
3. Dataset (`custom_75_25_english/train.jsonl`) is uploaded to the data Volume

**Launch command (do not run until authorized):**
```powershell
modal run --detach modal_train_custom_75_25_english.py
```

**Prerequisites (run once, manually, before launch):**
```powershell
modal volume create gemma-tounsi-data
modal volume create gemma-tounsi-checkpoints
modal volume create gemma-tounsi-hf-cache
modal secret create gemma-hf HF_TOKEN=<your_token>
# Adapt upload_modal_data.ps1 to custom_75_25_english/train.jsonl, then:
.\scripts\upload_modal_data.ps1
```

---

## 9. Files Changed

| File | Change |
|---|---|
| `docs/training/blackwell_compatibility.md` | Marked as verified; recorded actual smoke-test versions and results |
| `docs/training/modal_budget_plan.md` | Added all-in cost model (CPU/RAM/image-build/keep-alive); training feasibility scenarios |
| `docs/training/pre_training_audit.md` | Cleared B1/B2; updated dependency table; updated blocker table; updated readiness verdict |
| `modal_train_custom_75_25_english.py` | Added CPU/RAM cost constants; ALL_IN_COST_PER_SECOND; image-build buffer; adjusted BUDGET_STOP_SECONDS; updated monitoring loop and print messages |

**No production configurations were modified. No datasets were modified. No Modal resources were created. No credits were consumed.**
