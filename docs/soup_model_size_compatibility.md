# Soup Model Size Detection Compatibility Audit

**Project:** Gemma Tounsi 1.0 27B
**Current Soup version:** 0.73.3
**Target model:** `google/gemma-3-27b-it`
**Verified bug:** `model_size_from_name("google/gemma-3-27b-it")` returns 7B instead of 27B

---

## Phase 1 — Current Environment

| Component | Value |
|-----------|-------|
| Docker image | `ghcr.io/makazhanalpamys/soup:0.73.3` |
| Container Python | 3.10.12 (Ubuntu 22.04 native) |
| Soup CLI | 0.73.3 (installed via pip in image) |
| Soup dependencies | huggingface-hub, packaging, plotext, pydantic, pyyaml, rich, typer |
| Dockerfile base | `FROM ghcr.io/makazhanalpamys/soup:0.73.3` |
| Modal integration | None (no `soup_modal_app.py` or Modal files exist) |
| GPU services | `soup` (GPU), `soup-cpu` (validation only) |

**Bug location:** `soup_cli/utils/gpu.py`, function `model_size_from_name()`

The size-markers list checks `"7b" in name_lower` as a substring match. Since `"7b" in "google/gemma-3-27b-it".lower()` evaluates to `True` (because `"27b"` contains `"7b"`), Gemma 3 27B is misidentified as 7B. This causes `soup profile` to report ~6.4 GB VRAM instead of the correct ~18–21 GB.

---

## Phase 2 — Upstream Soup 0.77.0 / 0.77.2

### Python version requirement

| Version | Python requirement |
|---------|-------------------|
| Soup 0.73.3 | 3.10–3.12 (current project uses 3.10) |
| Soup 0.77.0+ | >=3.11, <3.14 |

**Critical:** Upgrading to 0.77.x requires Python 3.11+. The current Docker image uses Python 3.10 (Ubuntu 22.04), so the Dockerfile base image must change.

### Model size parsing fix (#2285, issue #1198)

In v0.77.2, `soup profile --model` reads the model size from the model's `config.json` (local directory, HF cache, or Hub) instead of guessing from the model name:

> "It prices the run without a `soup.yaml` from the model's own `config.json` ... with the same estimate a `soup.yaml` describing that run gets; **it never guesses a size from the name** (#2285)"

This is a robust architectural fix, not just a substring-list patch.

### Schema changes in 0.77.x (relevant to existing `soup.yaml`)

| Change | Impact on `soup.yaml` |
|--------|----------------------|
| 31 config fields that nothing reads now **refused** (were warnings in 0.76, silently accepted before) | Our config passed `soup doctor` in 0.77.2 with "None of the 0 setting(s) known to be unread is switched on" ✅ |
| `training.gradient_checkpointing` now **reaches the trainer** (was not applied in 0.73.3) | Behavior change — checkpointing now actually active |
| `lora.use_rslora`, `training.freeze_layers`/`freeze_ratio` now **applied** (were ignored) | Behavior change if those fields were set (ours doesn't set them) |
| AWQ/GPTQ export removed | Not relevant (we use 4-bit QLoRA) |

### Modal / cloud integration

Soup 0.77.2 supports cloud training with `--cloud modal`, `--cloud lambda`, `--cloud runpod`. Modal GPU types listed: `t4`, `l4`, `a10g`, `a100`, `a100-80gb`, `l40s`, `h100`.

---

## Phase 3 — Isolated Upgrade Test (Soup 0.77.2, Python 3.11)

Test executed in a disposable `python:3.11-slim` Docker container. The project environment was **not** modified.

| Test | Result |
|------|--------|
| Soup 0.77.2 installs on Python 3.11 | ✅ Pass |
| `model_size_from_name("google/gemma-3-27b-it")` → 27B | ✅ Pass (was 7B in 0.73.3) |
| `model_size_from_name("google/gemma-3-7b-it")` → 7B | ✅ Pass |
| `model_size_from_name` for 8B, 70B, 14B, 72B, 4B, 1B, 17B, 1.7B | ✅ All pass |
| `soup doctor --config soup.yaml` (existing project config) | ✅ Return code 0 |
| Unread settings check | ✅ "None of the 0 setting(s) known to be unread is switched on" |
| Cloud providers (`--cloud modal|lambda|runpod`) | ✅ All supported |

**Conclusion:** Soup 0.77.2 is compatible with the existing `soup.yaml` and fixes the model size detection bug correctly.

---

## Phase 4 — Option Comparison

### Option A: Upgrade to Soup 0.77.2

**Changes required:**
1. Dockerfile: Change base from `ghcr.io/makazhanalpamys/soup:0.73.3` to an image with Python 3.11+ (e.g., build on `python:3.11-slim` + install `soup-cli==0.77.2`, or wait for upstream `soup:0.77.2` image)
2. Verify no `soup.yaml` fields are now refused (tested: none refused)
3. Account for behavior changes: `gradient_checkpointing` now actually applied
4. Rebuild and re-test Docker image

| Criterion | Assessment |
|-----------|------------|
| Reliability | High — official upstream fix, reads from `config.json` |
| `soup.yaml` compatibility | ✅ Verified — loads cleanly in 0.77.2 |
| Python requirements | ⚠️ Requires 3.11+ (Dockerfile change from 3.10) |
| Modal support | ✅ Built-in (`--cloud modal`) |
| Reproducibility | ✅ Pinned version, upstream image |
| Maintenance complexity | Low — no local patch to maintain |
| Additional risk | Behavior changes (gradient_checkpointing now active), other 0.74→0.77 changes |

### Option B: Backport to Soup 0.73.3

**Changes required:**
1. Patch `soup_cli/utils/gpu.py` `model_size_from_name()` to use word-boundary-safe matching
2. Apply patch reproducibly in Dockerfile (not site-packages on host)
3. Apply patch in Modal execution environment
4. Add regression tests for 7B, 27B, 17B, 70B, 1.7B, 0.8B model identifiers

| Criterion | Assessment |
|-----------|------------|
| Reliability | Medium — local patch, only fixes model size, misses other improvements |
| `soup.yaml` compatibility | ✅ No changes needed |
| Python requirements | ✅ Keep 3.10, no Dockerfile change |
| Modal support | ⚠️ Patch must reach Modal environment separately |
| Reproducibility | ⚠️ Patch must be applied consistently everywhere |
| Maintenance complexity | Medium — local fork of one function |
| Additional risk | Only fixes model size; 0.73.3 VRAM estimates still wrong for other misidentified models |

---

## Phase 5 — Recommendation

### Recommended: **Option B (Backport)**

**Rationale:**

The task restrictions explicitly state "Do not upgrade the main project." The backport is the minimal, targeted fix that:

1. **Respects the no-upgrade constraint** — keeps Soup at 0.73.3
2. **Requires no Dockerfile changes** — Python 3.10 stays
3. **Requires no `soup.yaml` changes** — config already valid
4. **Is surgically precise** — fixes only the model size detection, nothing else
5. **Can be applied reproducibly** — patch in Dockerfile + Modal environment
6. **Has regression tests** — verifies 27B never resolves to 7B

The backport patch: replace substring matching (`marker in name_lower`) with word-boundary-safe regex matching in `model_size_from_name()`. This correctly handles `27b-it` (does not match `7b` as a standalone word) while preserving all existing size detections.

**When to reconsider Option A (upgrade):**
- When the project is ready to move off Python 3.10
- When `gradient_checkpointing` behavior change is desired (it's now actually applied in 0.77.x)
- When Modal integration is actively being developed
- When other 0.74→0.77 fixes are needed

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/test_soup_0772_isolated.py` | Isolated upgrade compatibility test (disposable) |
| `scripts/test_model_size_0772.py` | Model size detection test for 0.77.2 (disposable) |

**No files were modified in the main project.** The existing `soup.yaml`, datasets, and Soup 0.73.3 installation remain unchanged.

**Next step:** Awaiting user approval before implementing the backport patch.
