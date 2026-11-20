# Gemma Tounsi 1.0 (27B) — Modal Budget Plan

**Project:** Gemma-Tounsi-1.0-27B  
**Target:** ONE NVIDIA RTX PRO 6000 Blackwell (96 GB) on Modal  
**Method:** QLoRA 4-bit via Soup 0.73.3  
**Dataset:** `data/experiments/downsampled/custom_75_25_english/train.jsonl`  
  (384,572 examples, 45,926,282 tokens per epoch; 2 epochs configured = 91,852,564 tokens)  
**Status:** Preparation only — no resources allocated, no credits spent.

---

## 1. Confirmed available balance

**User-confirmed Modal available credit balance: $29.10 USD**  
This is treated as the absolute maximum project budget. It is **not** permission to spend it. If the actual balance is lower than $29.10, stop and report the discrepancy before launching anything.

---

## 2. Current official pricing (Modal, verified Nov 18 2026)

Source: Modal official pricing page (modal.com/pricing), corroborated by ComputePrices and gpueconomy (Nov 18 2026).

| Resource | Rate |
|---|---|
| RTX PRO 6000 GPU | $0.000842 / second = $3.0312 / GPU-hour |
| CPU (per core) | $0.0000131 / second = $0.04716 / CPU-hour |
| Memory (per GiB) | $0.00000222 / second = $0.007992 / GiB-hour |
| Region multiplier | 1.15–1.75x base (avoid; use default region) |
| Non-preemptible | 3x CPU/memory (avoid; default preemptible) |
| Keep-alive window | 60 s after last input (billable) |
| Image build | Billed separately (not included in function run rate) |
| Volume storage | ~$0.15 / GB-month |

Modal bills from application load until the container stops, plus the keep-alive window. GPU is charged for the entire function lifetime. CPU and RAM are charged for the allocated (not just used) amounts per second.

---

## 3. All-in per-second cost model (training function)

The training container requests:
- 1× RTX PRO 6000 GPU
- 1 CPU core
- 4 GiB memory

| Component | Rate/sec | Rate/hr | Share of GPU-only |
|---|---|---|---|
| GPU | $0.00084200 | $3.0312 | 100.0% |
| CPU (1 core) | $0.00001310 | $0.04716 | 1.6% |
| RAM (4 GiB) | $0.00000888 | $0.03197 | 1.1% |
| **All-in per second** | **$0.00086398** | **$3.11033** | **102.6%** |
| **All-in per hour** | — | **$3.11033** | — |

CPU + RAM add ~2.6% overhead on top of the GPU-only rate. Image build, keep-alive, and Volume storage are additional (see section 4).

**Critical:** A GPU-only timer is NOT a guaranteed total-cost cap. The previous budget model (`BUDGET_STOP_SECONDS` based solely on `$0.000842/sec`) underestimates total spend by ~2.6% per hour plus image build and storage costs. The training app has been updated to track the all-in rate.

---

## 4. Additional cost categories (not per-second GPU)

| Category | Estimate | Notes |
|---|---|---|
| Image build (per build) | ~$1.00–$3.00 | Billed separately by Modal; one-time per image change |
| Keep-alive (per function end) | ~$0.00–$0.01 | 60 s window after last input |
| Volume storage | ~$0.15 / GB-month | Checkpoints + HF cache (~10–20 GB for a short pilot) |
| Data transfer | Not documented at this level | Minimal for upload/download of adapter-sized artifacts |

For a single training session, image build is the largest non-per-second cost. The HF cache Volume (persistent across runs) means the 27B base model (~54 GB in 4-bit, or full weights ~54 GB bf16) is downloaded only once; subsequent runs reuse the cache.

---

## 5. All-in available hours (training allocation = $24.00)

Assuming one image build (~$2.00 buffer) and ~2.6% CPU/RAM overhead:

| Item | USD | All-in seconds | All-in hours |
|---|---|---|---|
| Total balance | $29.10 | — | — |
| Smoke test (actual, completed) | ~$0.50–$1.00 | — | — |
| Training (max, all-in) | $24.00 | ~25,400 s | ~7.06 hr |
| Finalization reserve | $2.00 | ~2,100 s | ~0.58 hr |
| Emergency reserve | $2.10 | — | Untouched |

**Comparison with GPU-only figures:** GPU-only said 7.92 hr (28,500 s). All-in reduces this to ~7.06 hr (25,400 s) after CPU/RAM overhead and image-build buffer. The gap is ~0.86 hr (~$2.67) of non-GPU cost.

---

## 6. Required all-in training throughput

Available all-in training seconds: ~25,400 s (after CPU/RAM overhead and image-build buffer).

| Target | Tokens | Required tokens/sec (all-in) |
|---|---|---|
| 1 epoch | 45,926,282 | **1,808** |
| 2 epochs (configured) | 91,852,564 | **3,616** |

One epoch at realistic throughput (likely 300–1,000 tok/sec without packing) requires far more than the available budget. One full epoch is **not** financially realistic at the all-in $24.00 training budget.

---

## 7. Tokens affordable within the $24.00 training budget (all-in)

| Throughput | Tokens affordable | Epochs (of 45.93M) | All-in cost |
|---|---|---|---|
| 300 tok/sec | 7.62M | 0.17 | $22.86 |
| 500 tok/sec | 12.70M | 0.28 | $22.86 |
| 800 tok/sec | 20.32M | 0.44 | $22.86 |
| 1,000 tok/sec | 25.40M | 0.55 | $22.86 |
| 1,200 tok/sec | 30.48M | 0.66 | $22.86 |

---

## 8. Recommended budget allocation (hard ceilings)

| Purpose | Max allocation | Notes |
|---|---|---|
| Blackwell compatibility smoke test | $1.00 | **Completed** — actual spend ~$0.50–$1.00, reconciled against dashboard |
| Actual training (all-in) | $24.00 | Pilot-scale only; full 2-epoch run does not fit |
| Final artifact export & inference verification | $2.00 | Adapter export, inference acceptance test, release artifacts |
| Emergency reserve | $2.10 | **Untouched** unless explicit approval |
| TOTAL | $29.10 | |

---

## 9. Is one full epoch financially realistic?

**No — not at the all-in $24.00 training budget.**

- GPU-only, one epoch needs >=1,611 tok/sec to fit in 7.92 hr ($24).
- All-in, one epoch needs >=1,808 tok/sec to fit in ~7.06 hr ($24).
- With CPU/RAM/image-build/keep-alive overhead, usable GPU time is less, so required throughput is higher.
- 1,500 tok/sec is optimistic for 4-bit QLoRA 27B (batch=1, no packing) on one card; realistic throughput is likely lower.
- The configured run is **2 epochs** (91.85M tokens), which needs ~3,616 tok/sec all-in — clearly impossible within budget.

**Honest conclusion:** $29.10 supports only a **pilot-scale** fine-tune (a small fraction of one epoch, roughly 0.17–0.66 epoch depending on measured throughput), not the configured 2-epoch full run. Do not silently reduce epochs or shorten steps to force a fit; report this and propose alternatives with approval.

---

## 10. Maximum safe runtime

To stay within the $24.00 all-in training ceiling: **~7.06 hours (25,400 s) absolute upper bound**, less if image build costs more than the ~$2.00 buffer.

Recommendation: set `max_runtime_seconds` well below this (e.g., based on measured throughput from a short pilot), rely on `save_steps=100` checkpoints, and stop via `STOP_AFTER_STEPS` when approaching the budget. Never rely on an arbitrary timeout as a dollar cap — monitor cumulative cost in the Modal dashboard.

Add a **cost-aware early stop**: after each checkpoint commit, estimate cumulative spend (elapsed seconds × all-in rate + image build) and request a graceful stop when approaching the $24.00 training ceiling, leaving enough budget for the $2.00 finalization phase. Save the latest valid checkpoint, commit the Volume, then exit before training spend approaches $24.00.

---

## 11. Stopping & budget-control plan

1. **Before launch:** verify balance is still ~$29.10 in the Modal dashboard; confirm Volumes + `gemma-hf` Secret exist; upload dataset once.
2. **Launch:** `modal run --detach modal_train_custom_75_25_english.py` (one RTX-PRO-6000).
3. **Monitor:** `modal app logs gemma-tounsi-custom-train` + Modal dashboard cost view; check cumulative spend after each checkpoint commit.
4. **Stop before limit:** use `STOP_AFTER_STEPS` to halt at a pre-computed step budget (derived from measured tokens/sec and remaining dollars). Do not auto-restart.
5. **Budget-aware early stop:** estimate cumulative spend after each checkpoint commit using the **all-in rate** ($0.00086398/sec + image build amortized). When approaching the $24.00 training ceiling, request a graceful stop, complete the current safe optimization step, save the latest valid checkpoint, and commit the Volume. Leave $2.00 for finalization and $2.10 for emergency reserve.
6. **Preserve checkpoints:** committed checkpoints with `trainer_state.json` persist on the `gemma-tounsi-checkpoints` Volume; resume later with the same command (`--resume auto`).
7. **Finalization phase:** after training stops within the $24.00 budget, use the $2.00 finalization allocation to export the LoRA adapter, run the inference acceptance test, and create release artifacts under `releases/gemma-tounsi-v1/`.
8. **Emergency reserve ($2.10):** never consumed without explicit approval.

---

## 12. Checkpoint-related costs (included in training budget)

- Checkpoint files are LoRA-adapter sized (small; adapter + optimizer + scheduler + trainer state for r=16). Each `checkpoint-*` directory is modest.
- `save_steps=100` → 480 checkpoints/epoch → frequent Modal Volume commits. Commits add latency; storage cost is ~$0.15/GB-month (negligible for a short pilot).
- Consider the commit frequency a runtime/latency factor, not a major cost driver.

---

## 13. Remaining uncertainties

| Uncertainty | Impact |
|---|---|
| Actual tokens/sec on RTX PRO 6000 (no local GPU) | Determines how many tokens fit in $24 |
| Image build cost (billed separately) | Reduces budget available for the run; buffered at ~$2.00 |
| CPU/RAM/keep-alive overhead | ~2.6% per hour; included in all-in model |
| Checkpoint commit frequency / Volume commit cost | Minor latency/cost factor |
| Modal dashboard billing API (none) | Must manually monitor cumulative cost; no programmatic alert |

---

## 15. Training feasibility scenarios (assumptions clearly labeled)

**No training run has been performed.** No tokens/sec measurement exists. All
throughput figures below are **assumptions based on hardware characteristics**,
not measurements. The only measured evidence is that the smoke test confirmed
NF4 4-bit QLoRA works on the RTX PRO 6000 Blackwell (all 7 mandatory checks PASS).

### 15.1 Hardware-derived assumptions

| Factor | Value | Basis |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell, 95.01 GB VRAM | Smoke test verified |
| VRAM for 27B 4-bit base | ~13–14 GB weights + LoRA + activations | 95.01 GB ample; memory not the bottleneck |
| Seq length | 8192 | soup.custom_75_25_english.yaml |
| Avg tokens/example | ~119.4 | Dataset manifest |
| Packing | Not configured | Most 8192-length compute is padding |
| Effective compute per example | 8192 tokens (padded) | Soup/Trainer processes full seq length |
| Throughput assumption | 200–800 tok/sec (effective) | Labeled assumption; not measured |

### 15.2 Scenario table (pilot-scale, $24.00 all-in training budget)

Available all-in seconds: ~25,400 s (~7.06 hr) after image-build buffer.

| Assumed throughput | Seconds affordable | Tokens trained | Epochs (of 45.93M) | Steps (save_steps=100) |
|---|---|---|---|---|
| 200 tok/sec | 25,400 | 5.08M | 0.11 | ~50 |
| 400 tok/sec | 25,400 | 10.16M | 0.22 | ~100 |
| 600 tok/sec | 25,400 | 15.24M | 0.33 | ~150 |
| 800 tok/sec | 25,400 | 20.32M | 0.44 | ~200 |

### 15.3 What "pilot-scale" means for V1

V1 is a **functional proof of concept**, not a fully converged model:
- A small number of real training steps (tens to low hundreds) on real data
- Produces a complete, valid LoRA adapter that loads on `google/gemma-3-27b-it`
- Passes the inference acceptance test (generates non-empty Tunisian responses)
- Can be resumed later if more credits become available

### 15.4 Optimizing the first training run

To maximize value from the first (and possibly only) training run within budget:

1. **Set `STOP_AFTER_STEPS`** based on measured throughput from the first ~50 steps, derived from the all-in budget model.
2. **Save every 100 steps** (already configured) — each checkpoint is a resume point.
3. **Commit after every checkpoint** (already implemented) — no work is lost between commits.
4. **Stop before $24.00** — the budget-aware early stop (using ALL_IN_COST_PER_SECOND) triggers at ~$23.50 estimated cost, leaving room for finalization.
5. **Every step contributes to V1** — even a pilot-scale run produces a usable adapter.

### 15.5 What cannot be promised

- A full epoch (45.93M tokens) — requires ≥1,808 tok/sec all-in; not realistic at $24
- 2 epochs (91.85M tokens) — requires ≥3,616 tok/sec all-in; impossible at $24
- Convergence or linguistic quality — pilot-scale training does not converge
- FlashAttention/fused-kernel performance — not part of baseline Soup path

---

## 16. Bottom line

**CAN I REASONABLY FINE-TUNE GEMMA 3 27B ON 45.93M TOKENS WITH $29.10, ONE RTX PRO 6000, QLORA?**

**No — not the configured 2-epoch (91.85M-token) run, and realistically not even one full epoch.** The all-in budget supports a pilot-scale fine-tune only (~0.17–0.66 epoch depending on measured throughput), assuming bitsandbytes NF4 works on Blackwell (now confirmed via smoke test).

**V1 must be functional, not complete.** A smaller amount of real training that produces a complete, tested, usable Gemma Tounsi V1 adapter is preferred over a larger incomplete training run that consumes all credits and leaves us without a working model.

V1 acceptance requires:
- A complete, valid LoRA adapter that loads on `google/gemma-3-27b-it`
- Correct tokenizer and chat-template configuration
- An inference script that generates actual responses (Tunisian Arabizi and Arabic-script Tunisian)
- The adapter saved to persistent storage and exportable/downloadable
- Reproducible training and inference instructions
- Release artifacts under `releases/gemma-tounsi-v1/`

Proceed only with explicit approval at each spend decision.
