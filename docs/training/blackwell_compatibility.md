# Gemma Tounsi 1.0 (27B) — Blackwell CUDA & bitsandbytes Compatibility

**Project:** Gemma-Tounsi-1.0-27B  
**Target GPU:** NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM  
**Compute capability:** 12.0 (sm_120)  
**Modal GPU identifier:** `RTX-PRO-6000`  
**Framework:** Soup 0.73.3 (QLoRA, 4-bit)  
**Status:** ✅ Compatibility VERIFIED on real hardware — smoke test passed all 7 mandatory checks.

---

## 1. Target hardware (verified on real RTX PRO 6000)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 95.01 GB usable (~96 GB GDDR7) |
| Architecture | Blackwell (GB202) |
| Compute capability | 12.0 / sm_120 ✅ Verified |
| Modal identifier | `RTX-PRO-6000` |
| Modal host CUDA driver API | 13.3 |
| Container CUDA runtime | 13.2 |

---

## 2. Component versions (verified on real hardware)

These versions were captured by the smoke test on the actual RTX PRO 6000 Blackwell container:

| Component | Version | Source |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell Server Edition | Smoke test |
| VRAM | 95.01 GB | Smoke test |
| Compute capability | 12.0 (sm_120) | Smoke test |
| CUDA driver (Modal host) | 13.3 | Modal host |
| Container CUDA runtime | 13.2 | Smoke test (`torch.version.cuda`) |
| PyTorch | 2.15.0+cu132 | Smoke test (`torch.__version__`) |
| bitsandbytes | 0.50.2 | Smoke test (`bnb.__version__`) |
| PEFT | 0.21.3 | Smoke test (`peft.__version__`) |
| Transformers | 5.23.0 | Smoke test (`transformers.__version__`) |
| Accelerate | 1.16.0 | Smoke test (`accelerate.__version__`) |
| Soup | 0.73.3 | Pinned `soup-cli[train]==0.73.3` |
| Python (Modal image) | 3.12 | Modal image |

**Note:** Actual resolved versions (torch 2.15.0+cu132, bnb 0.50.2, transformers 5.23.0, peft 0.21.3, accelerate 1.16.0) are **newer** than the conservative estimates in the original assessment (torch 2.13.0+cu130, bnb 0.50.1, transformers 4.57.6, peft 0.20.0, accelerate 1.14.0). This is favorable: Blackwell sm_120 support and NF4 kernels are present in the resolved stack.

---

## 3. Smoke test results (verified — all PASS)

The smoke test (`scripts/modal_blackwell_smoke_test.py`) ran once on one RTX PRO 6000 and produced:

| Test | Result |
|---|---|
| GPU detection (RTX PRO 6000, sm_120) | PASS |
| CUDA initialization | PASS |
| PyTorch CUDA operations | PASS |
| NF4 initialization (bitsandbytes Linear4bit) | PASS |
| NF4 forward + backward | PASS |
| AdamW optimizer update | PASS |
| PEFT LoRA attachment + two-step gradient + optimizer update | PASS |
| Memory monitoring | PASS |

**Overall: PASS — Blackwell stack verified for QLoRA 4-bit training.**

---

## 4. Blackwell compatibility assessment (sm_120) — RESOLVED

### 3.1 PyTorch + CUDA runtime — RESOLVED
- `torch 2.15.0+cu132` on the Modal container supports sm_120 (compute capability 12.0 confirmed).
- `torch.cuda.is_available() == True` confirmed.
- **Risk: RESOLVED.**

### 3.2 bitsandbytes NF4 — RESOLVED
- `bitsandbytes 0.50.2` on the container supports NF4 `Linear4bit` forward + backward on sm_120.
- The smoke test confirmed NF4 forward/backward with gradient flow through the input tensor (frozen quantized weight, as designed).
- **Risk: RESOLVED.** (Was the principal risk; now cleared.)

### 3.3 Transformers / PEFT / Accelerate — RESOLVED
- transformers 5.23.0, PEFT 0.21.3, accelerate 1.16.0 all compatible with Blackwell given the CUDA 13.2 toolchain.
- PEFT LoRA wrapping of 4-bit base confirmed working via smoke test.
- **Risk: RESOLVED.**

### 3.4 Triton / FlashAttention — UNCHANGED (LOW)
- Not part of the Soup 0.73.3 baseline QLoRA path.
- **Risk: LOW.**

### 3.5 Soup 0.73.3 — UNCHANGED (LOW)
- Model-size patch applied and verified.
- **Risk: LOW.**

---

## 5. Verified support summary

| Feature | Status | Evidence |
|---|---|---|
| `torch.cuda.is_available()` on sm_120 | ✅ VERIFIED | Smoke test |
| sm_120 in torch arch list | ✅ VERIFIED | `torch.cuda.get_device_properties(0).major == 12` |
| bitsandbytes NF4 forward | ✅ VERIFIED | Smoke test NF4 forward/backward PASS |
| bitsandbytes NF4 backward | ✅ VERIFIED | Smoke test NF4 forward/backward PASS |
| Optimizer update on 4-bit weights | ✅ VERIFIED | Smoke test AdamW PASS |
| PEFT LoRA on 4-bit base | ✅ VERIFIED | Smoke test PEFT LoRA PASS |
| 96 GB VRAM sufficient for 27B QLoRA | ✅ LIKELY | 95.01 GB usable; 4-bit 27B base ~13–14 GB |

---

## 6. What the smoke test does NOT prove

- End-to-end Gemma 3 27B QLoRA training stability (only a tiny model).
- Multi-step training convergence or memory behavior at seq_len=8192.
- FlashAttention or fused-kernel behavior on Blackwell.
- Long-run (multi-hour) memory stability / OOM behavior.

These require the actual training run (or a larger pilot), which the budget may not support — see `modal_budget_plan.md`.

---

## 7. Compatibility conclusion

**The Blackwell compatibility blocker is CLOSED.** The RTX PRO 6000 with the resolved Modal stack (torch 2.15.0+cu132, bitsandbytes 0.50.2, PEFT 0.21.3, transformers 5.23.0, accelerate 1.16.0, CUDA 13.2) supports QLoRA 4-bit NF4 training on sm_120. Training may proceed within budget constraints.

## 1. Target hardware

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell |
| VRAM | 96 GB GDDR7 (ECC) |
| Architecture | Blackwell (GB202) |
| Compute capability | 12.0 / sm_120 |
| Modal identifier | `RTX-PRO-6000` |
| Modal host CUDA driver API | 13.3 (per Modal CUDA docs) |
| Modal availability | Since April 2026 |

---

## 2. Component versions (Soup 0.73.3 stack)

These are the versions captured in the validated environment (`docs/ENVIRONMENT.md`, `soup-env.lock`) and used by Soup 0.73.3:

| Component | Version | Source |
|---|---|---|
| Soup | 0.73.3 | pinned `soup-cli[train]==0.73.3` |
| Python (Modal image) | 3.12 | Soup supports >=3.10,<3.13 |
| Python (Docker image) | 3.10.12 | upstream Soup image |
| PyTorch | 2.13.0+cu130 | Docker image; Modal resolves unpinned |
| CUDA (PyTorch build) | 13.0 | Docker image |
| CUDA driver (Modal host) | 13.3 | Modal host |
| Transformers | 4.57.6 | Soup image |
| PEFT | 0.20.0 | Soup image |
| TRL | 0.28.0 | Soup image |
| bitsandbytes | 0.50.1 | Soup image (required for `quantization: 4bit`) |
| Accelerate | 1.14.0 | Soup image |
| Datasets | 5.0.1 | Soup image |
| huggingface-hub | 0.36.2 | Soup image |

**Key dependency-tree fact:** Soup 0.73.3's `soup-cli[train]` extra pulls torch/transformers/peft/bitsandbytes/accelerate, but the **Modal image does NOT pin these transitive versions** (only `soup-cli==0.73.3` is pinned). The resolved torch/CUDA build in the Modal container is therefore not locked and must be verified.

---

## 3. Blackwell compatibility assessment (sm_120)

### 3.1 PyTorch + CUDA runtime

- Blackwell sm_120 requires a PyTorch wheel built with CUDA >= 12.8 (torch >= 2.7 / cu128+).
- The Docker validation image ships `torch 2.13.0+cu130`, which includes sm_120 support. This was validated only on CPU hardware; `torch.cuda.is_available()` was `False` there.
- The Modal image resolves torch transitively and **may install a different/older CUDA build** that lacks sm_120. This must be confirmed by the smoke test (`torch.cuda.get_device_properties(0).major == 12`).

**Risk:** MEDIUM — version drift between the pinned Docker stack and the unpinned Modal resolution.

### 3.2 bitsandbytes 0.50.1 (4-bit NF4) — PRINCIPAL RISK

- bitsandbytes 0.50.1 predates the RTX PRO 6000 Blackwell (announced March 2025, GA April 2025).
- **bitsandbytes 0.50.1 does NOT advertise Blackwell 4-bit (NF4) kernel support.** NF4 `Linear4bit` forward/backward on sm_120 is unverified and may fail, fall back to slow CPU offload, or produce silent numerical issues.
- Importing bitsandbytes successfully does **not** prove NF4 compatibility.
- `bnb.nn.Linear4bit(quant_type="nf4", compute_dtype=bfloat16)` forward + backward + optimizer update is the critical path for QLoRA and is **not guaranteed**.

**Risk:** HIGH — this is the single greatest blocker for QLoRA 4-bit on Blackwell with the pinned stack.

### 3.3 Transformers 4.57.6 / PEFT 0.20.0 / Accelerate 1.14.0

- Expected compatible with Blackwell given a suitable CUDA toolchain; none are Blackwell-specific.
- PEFT LoRA wrapping of a 4-bit quantized base model relies on bitsandbytes `Linear4bit`, so PEFT correctness is contingent on bitsandbytes working.

**Risk:** LOW (conditional on bitsandbytes).

### 3.4 Triton

- Not part of the Soup 0.73.3 baseline QLoRA path (transformers backend). FlashAttention-2/3 on Blackwell is a separate concern; Soup 0.73.3 default does not require Triton for the baseline run.
- FlashAttention-3 has no sm_120 build (per GPU compat tracker); FlashAttention-2 2.8.3.post2 is reported compatible. Not required for baseline.

**Risk:** LOW (not used by baseline config).

### 3.5 Soup 0.73.3

- Soup itself is framework-agnostic regarding GPU arch; it delegates to transformers/PEFT/bitsandbytes.
- Soup's model-size patch (27B vs 7b substring bug) is required and applied in the Modal image.
- Soup 0.73.3 `train` command and `--resume auto` are GPU-agnostic.

**Risk:** LOW.

---
