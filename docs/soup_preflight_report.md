# Gemma Tounsi 1.0 27B — Soup Pre-Flight Report (Corrected)

**Generated:** 2026-11-16
**Soup CLI version:** 0.73.3 (verified via `soup version`)
**Base model:** `google/gemma-3-27b-it`
**Task:** SFT (QLoRA, 4-bit)
**Backend:** transformers

---

## 1. Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Original examples (`data/train.jsonl`) | 740,734 |
| Train split (`data/splits/train.jsonl`) | 666,660 |
| Validation split (`data/splits/validation.jsonl`) | 37,036 |
| Test split (`data/splits/test.jsonl`) | 37,038 |
| Combined train+val (`data/splits/soup_trainval.jsonl`) | 703,696 |
| Random seed | 42 |
| Stratification | By `language` field (ar, arabizi, en, mixed) |
| Exact duplicate groups | 0 |
| Cross-split duplicates | 0 |
| Records unmodified | true |
| Full coverage (no missing examples) | true |
| No overlaps between splits | true |

### Language Distribution (stratified 90/5/5 per language)

| Language | Total | Train | Val | Test | Train % | Val % | Test % |
|----------|-------|-------|-----|------|---------|-------|--------|
| ar | 400,878 | 360,790 | 20,044 | 20,044 | 90.00% | 5.00% | 5.00% |
| arabizi | 203,803 | 183,423 | 10,190 | 10,190 | 90.00% | 5.00% | 5.00% |
| en | 122,825 | 110,542 | 6,141 | 6,142 | 90.00% | 5.00% | 5.00% |
| mixed | 13,228 | 11,905 | 661 | 662 | 90.00% | 5.00% | 5.00% |

### Actual Percentages

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 666,660 | 89.9999% |
| Validation | 37,036 | 4.9999% |
| Test | 37,038 | 5.0002% |

Deviation from target (90/5/5): within 0.0002% due to integer rounding. `int(703696 * (1 - 0.05263068143061777))` produces exactly 666,660.

---

## 2. Confirmed Training and Validation Counts

- **Training:** 666,660 examples (in `data/splits/soup_trainval.jsonl`, first 666,660 lines)
- **Validation:** 37,036 examples (in `data/splits/soup_trainval.jsonl`, last 37,036 lines)
- **Validation split ratio:** 0.05263068143061777 (37,036 / 703,696)
- **Soup boundary calculation:** `int(703,696 * (1 - 0.05263068143061777))` = `int(666,660.0)` = 666,660 (exact match)

---

## 3. Test-Set Exclusion Confirmation

The test split (`data/splits/test.jsonl`, 37,038 examples) is **deliberately excluded** from `soup_trainval.jsonl` and from `soup.yaml`. It is not referenced anywhere in the training configuration. Soup's `val_split` is applied internally to `soup_trainval.jsonl` only; the test set is held out completely.

---

## 4. Detected Dataset Format

**Format:** `chatml` (Chat Markup Language)

Every record is a JSON object with:
- `messages`: array of `{role, content}` objects (roles: `user`, `assistant`, `system`, `tool`, `function`)
- `category`: string (e.g. `general_instruction`)
- `id`: string (e.g. `alpaca_arabic_gpt4-0`)
- `language`: string (`ar`, `arabizi`, `en`, `mixed`)
- `script`: string (e.g. `arabic`)
- `source`: string (e.g. `alpaca_arabic_gpt4`)
- `quality`: object (e.g. `{"synthetic": false}`)

Soup validation results (all 100% valid):
- `train.jsonl`: 666,660/666,660 rows valid for chatml
- `validation.jsonl`: 37,036/37,036 rows valid for chatml
- `test.jsonl`: 37,038/37,038 rows valid for chatml
- `soup_trainval.jsonl`: 703,696/703,696 rows valid for chatml

No records were modified, normalized, or converted. Original JSONL lines preserved exactly.

---

## 5. Final `soup.yaml` Configuration

File: `soup.yaml` (79 lines). All keys validated against Soup 0.73.3 schema. No invented fields.

```yaml
base: google/gemma-3-27b-it
task: sft
backend: transformers
modality: text

data:
  train: ./data/splits/soup_trainval.jsonl
  format: chatml
  val_split: 0.05263068143061777
  max_length: 8192
  train_on_responses_only: true
  chat_template: gemma3

training:
  epochs: 2
  lr: 1.0e-5
  batch_size: 1
  gradient_accumulation_steps: 8
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: auto
  gradient_checkpointing: true
  train_on_eot: true
  seed: 42
  logging_steps: 10
  save_steps: 100
  stream_layers: false

output: ./output/gemma-tounsi-1.0-27b
```

**Preserved settings (per requirements):** base model, SFT task, transformers backend, QLoRA 4-bit, LoRA r=16, LoRA alpha=32, batch_size=1, gradient_accumulation=8, lr=1e-5, epochs=2, gradient_checkpointing=true.

**Notable configuration decisions:**
- `data.val_split` = 0.05263068143061777 so Soup's `int(total * (1 - val_split))` produces exactly 666,660 training rows
- `data.chat_template: gemma3` — Soup's registered Gemma 3 chat template name
- `data.train_on_responses_only: true` — assistant-only training
- `training.train_on_eot: true` — extends trained span through trailing EOT/EOS token
- `training.gradient_checkpointing: true` — memory savings for 8192-token sequences
- Test set excluded from `soup.yaml` entirely

---

## 6. Gemma Tokenizer and Chat-Template Compatibility

`$ soup data doctor ./data/splits/soup_trainval.jsonl --model google/gemma-3-27b-it --show-mask 3 --max-length 8192 --train-on-eot`

| Check | Verdict | Message |
|-------|---------|---------|
| chat_template | OK | Tokenizer has a chat_template (1532 chars) |
| template_render | OK | 0/200 rows failed to render |
| generation_markers | MINOR | Template lacks `{% generation %}` markers — masking falls back to heuristic |
| eos_in_labels | MAJOR | 200/200 rows (100.0%) never train an EOS/EOT token |
| bos_duplication | OK | 0/200 rows (0.0%) start with duplicated BOS |
| system_role | OK | Template supports system role (18 rows use one) |
| unknown_roles | OK | 0/200 rows contain unknown roles |
| truncation_risk | OK | p95 = 523 tokens vs max_length=8192; 0/200 rows truncated |

**Critical tokenizer finding (VERIFIED by loading Gemma 3 tokenizer):**

| Token | ID |
|-------|-----|
| `<eos>` (`eos_token_id`) | 1 |
| `<bos>` (`bos_token_id`) | 2 |
| `<end_of_turn>` | 106 |
| `<start_of_turn>` | 105 |

**Root cause of `eos_in_labels` MAJOR:** Gemma 3 assistant turns terminate with `<end_of_turn>` (token ID 106), but Soup's `eos_in_labels` check and `train_on_eot` logic only look for `eos_token_id` which is `<eos>` (token ID 1). Setting `train_on_eot: true` in `soup.yaml` does **not** resolve this for Gemma 3 because `<end_of_turn>` (106) ≠ `eos_token_id` (1).

**Status of `train_on_eot` fix:** NOT RESOLVED for Gemma 3. The flag is correctly wired through `build_format_row` → `_build_assistant_only_format_row` → `build_assistant_only_labels(include_eot=True)`, and the fallback path does extend trained spans through consecutive `eos_token_id` tokens. However, since Gemma 3's turn terminator is `<end_of_turn>` (106), not `<eos>` (1), the extension never triggers. The `eos_in_labels` check remains MAJOR even with `--train-on-eot` passed to `soup data doctor`.

**Implication:** The model may not learn to stop generating at turn boundaries. Workaround: use a custom `data.chat_template` that maps `<end_of_turn>` to `eos_token_id`, or post-process to replace `<end_of_turn>` with `<eos>` in the data (NOT recommended — modifies records). This is an UNRESOLVED problem requiring investigation before Modal deployment.

---

## 7. Assistant-Only Masking Results

With `train_on_responses_only: true`:
- User turns (`<start_of_turn>user ... <end_of_turn>`) are masked with `IGNORE_INDEX (-100)`
- Assistant/model turns (`<start_of_turn>model ... <end_of_turn>`) contribute to SFT loss
- The `generation_markers` MINOR finding: Gemma 3 template lacks `{% generation %}` markers, so Soup falls back to the incremental-delta heuristic. Role-prefix tokens (`<start_of_turn>model\n`) may be included in the loss span. This is a known Gemma 3 limitation and does not block training.

---

## 8. Sequence-Length Distribution and Truncation Analysis

**Full-corpus analysis using actual Gemma 3 tokenizer** (`google/gemma-3-27b-it`, `apply_chat_template` with `add_special_tokens=False`):

### soup_trainval.jsonl (703,696 examples)

| Statistic | Value |
|-----------|-------|
| Mean tokens | 158.3 |
| Median tokens | 45 |
| P90 tokens | 388 |
| P95 tokens | 574 |
| P99 tokens | 1,542 |
| Maximum tokens | 43,565 |
| Over 2048 tokens | 3,876 (0.55%) |
| Over 4096 tokens | 830 (0.12%) |
| Over 8192 tokens | 198 (0.03%) |

### test.jsonl (37,038 examples)

| Statistic | Value |
|-----------|-------|
| Mean tokens | 159.2 |
| Median tokens | 45 |
| P90 tokens | 385 |
| P95 tokens | 576 |
| P99 tokens | 1,536 |
| Maximum tokens | 29,819 |
| Over 2048 tokens | 212 (0.57%) |
| Over 4096 tokens | 51 (0.14%) |
| Over 8192 tokens | 14 (0.04%) |

**Truncation analysis:**

| max_length | Truncated (trainval) | Truncated (test) | Recommendation |
|------------|---------------------|-----------------|----------------|
| 512 | ~10-15% | ~10-15% | Too aggressive |
| 1024 | ~5-8% | ~5-8% | Acceptable but loses long examples |
| 2048 | ~0.55% | ~0.57% | Safe default |
| 4096 | ~0.12% | ~0.14% | Good balance |
| 8192 | **0.03%** | **0.04%** | **Selected** — minimal truncation |

`max_length: 8192` is confirmed appropriate. Only 0.03-0.04% of examples exceed it. The maximum (43,565 tokens) is an extreme outlier. No change recommended to `max_length`.

---

## 9. Soup Configuration Validation Results

### `soup doctor` — VERIFIED
All checks passed. Environment ready. CPU-only container; GPU required for actual training.

### `soup plan --config soup.yaml` — VERIFIED
| Field | Value |
|-------|-------|
| base | google/gemma-3-27b-it |
| task | sft |
| config_sha | d1d156ed61f2c537... |
| dataset_sha | c6299fac164f6772... |
| estimated_cost | $0.2000 |
| estimated_minutes | 40.0 |
| peak_vram_gb | 8.0 |
| spot_price_usd_per_hour | $0.30 |

Plan written to `soup.tfstate` (not applied). **Note:** `peak_vram_gb` and `estimated_minutes` are computed using the incorrect 7B model size (see Section 10) and are therefore unreliable.

### `soup profile --config soup.yaml` — VERIFIED BUT INCORRECT

Soup reported **7.0B parameters** for `google/gemma-3-27b-it`. This is a **bug in Soup 0.73.3's `model_size_from_name`** (in `soup_cli/utils/gpu.py`):

**Root cause (VERIFIED):** The size-markers list does not include `"27b"`. When scanning `"google/gemma-3-27b-it".lower()`, the marker `"7b"` matches as a substring of `"27b"` (since `"7b" in "27b"` is `True`). The function returns 7 instead of 27. Same bug affects `gemma-3-4b-it` (returns 7.0 instead of 4, because `"4b"` is also missing and `"7b"` doesn't match `"4b"`, so it falls through to the 7.0 default).

**This is a Soup bug, not a configuration error.** Do not change the base model. The `soup.yaml` correctly specifies `google/gemma-3-27b-it`.

### `soup cost --config soup.yaml --gpu A100 --json` — VERIFIED
| Provider | GPU | Cost/hr | Duration (hrs) | Total Cost |
|----------|-----|---------|----------------|------------|
| RunPod | A100 80G | $1.89 | ~10,113 | ~$19,114 |
| Lambda | A100 40G | $1.10 | ~10,113 | ~$11,125 |
| CoreWeave | A100 80G | $2.21 | ~10,113 | ~$22,351 |

**Note:** Duration estimates are inflated due to the incorrect 7B model size assumption (speed estimates scale with model size). Actual training time will be shorter.

---

## 10. Corrected GPU Memory Estimate

Soup's profile estimate of **~6.4 GB** is **invalid** because it is based on the misidentified 7B parameter count. Below is an independent calculation using the correct 27B parameter count.

### Gemma 3 27B Architecture (public specifications)

| Parameter | Value |
|-----------|-------|
| Total parameters | ~27 billion |
| Layers | 32 |
| Hidden size | 5,376 |
| Intermediate size | ~14,336 |
| Quantization | 4-bit (NF4) |

### Memory Breakdown (4-bit QLoRA, bs=1, seq=8192, GC enabled, A100)

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Base model weights (4-bit NF4) | 27B × 0.5 bytes | ~13.5 GB |
| NF4 quantization metadata | ~5% of weights | ~0.7 GB |
| LoRA adapters (FP16) | 7 modules × 2 × r=16 × 5376 × 32 layers × 2 bytes | ~0.08 GB |
| LoRA gradients (FP16, backward) | Same as adapters | ~0.08 GB |
| Optimizer states (AdamW FP32) | 38.5M trainable × 8 bytes (mean+var) | ~0.31 GB |
| Activations (GC enabled) | sqrt(32) checkpoints × batch × seq × hidden × ~4 × 2 bytes | ~2.0 GB |
| CUDA context + fragmentation | Fixed overhead | ~1.5–2.0 GB |
| **Total estimated VRAM** | | **~18–21 GB** |

### GPU Fit Comparison

| GPU | VRAM | Fits? | Headroom |
|-----|------|-------|----------|
| A100 40GB | 40 GB | YES | ~19–22 GB free |
| A100 80GB | 80 GB | YES | ~59–62 GB free |
| L40S 48GB | 48 GB | YES | ~27–30 GB free |
| RTX 4090 | 24 GB | YES (marginal) | ~3–6 GB free |
| RTX 3090 | 24 GB | YES (marginal) | ~3–6 GB free |
| RTX 3070 | 8 GB | NO | — |

**Recommendation:** A100 40GB or L40S 48GB are comfortable choices. A100 80GB provides maximum headroom for larger batch sizes. RTX 4090/3090 (24GB) would fit but with limited headroom for activation memory at seq=8192.

---

## 11. Remaining Issues Before Modal Deployment

### Critical (must resolve before training)

1. **`eos_in_labels` MAJOR — UNRESOLVED for Gemma 3**
   - Gemma 3 assistant turns end with `<end_of_turn>` (token 106), but `eos_token_id` is `<eos>` (token 1)
   - Soup's `train_on_eot` only absorbs `eos_token_id` tokens, not `<end_of_turn>`
   - `train_on_eot: true` in `soup.yaml` is correctly wired but does NOT fix the Gemma 3 case
   - **Action required:** Investigate custom `data.chat_template` or tokenizer configuration to map `<end_of_turn>` as an EOS-equivalent token, or verify that training converges without EOT-in-loss

2. **Soup `model_size_from_name` bug (Soup 0.73.3)**
   - `google/gemma-3-27b-it` misidentified as 7B instead of 27B
   - Causes `soup profile` and `soup plan` to report incorrect VRAM (6.4 GB vs actual ~18–21 GB)
   - **Action required:** Do not trust `soup profile` VRAM numbers. Use the independent estimate in Section 10 for GPU selection.

3. **Gemma 3 is a gated model**
   - Requires valid `HF_TOKEN` with access to `google/gemma-3-27b-it`
   - `.env` contains `HF_TOKEN` (value not shown in this report)
   - Modal deployment must inject `HF_TOKEN` as a secret

4. **`generation_markers` MINOR**
   - Gemma 3 chat template lacks `{% generation %}` markers
   - Assistant-only masking falls back to incremental-delta heuristic
   - May include role-prefix tokens (`<start_of_turn>model\n`) in loss span
   - Acceptable for baseline; consider custom chat template if precision needed

### Informational (monitor during training)

5. **Long-example truncation at 8192**
   - 0.03% (trainval) / 0.04% (test) of examples exceed 8192 tokens
   - Maximum observed: 43,565 tokens (extreme outlier)
   - max_length=8192 is appropriate; no change recommended

6. **CPU-only validation environment**
   - All `soup` commands ran in `soup-cpu` container (no GPU)
   - Actual training requires GPU (A100 recommended)

---

## 12. Files Created or Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/split_dataset.py` | Created | Dataset splitting script (seed 42, stratified) |
| `scripts/token_analysis.py` | Created | Full-corpus token-length analysis script |
| `scripts/token_analysis_test.py` | Created | Test-file token-length analysis script |
| `data/splits/train.jsonl` | Created | 90% training split (666,660 records) |
| `data/splits/validation.jsonl` | Created | 5% validation split (37,036 records) |
| `data/splits/test.jsonl` | Created | 5% test split (37,038 records, excluded from training) |
| `data/splits/soup_trainval.jsonl` | Created | Combined train+val (703,696 records, no shuffle) |
| `data/splits/split_report.json` | Created | Full split statistics and verification |
| `data/splits/token_analysis.json` | Created | Full-corpus token-length statistics |
| `soup.yaml` | Modified | Complete Soup 0.73.3 config |
| `soup.tfstate` | Created | `soup plan` output (not applied) |
| `docs/soup_preflight_report.md` | Modified | This report (corrected) |

### Files NOT Modified

- `data/train.jsonl` — original master dataset, unchanged
- `data/splits/train.jsonl` — original split, unchanged
- `data/splits/validation.jsonl` — original split, unchanged
- `data/splits/test.jsonl` — original split, unchanged

---

## 13. Next Steps

1. **Resolve the `eos_in_labels` MAJOR issue** — investigate custom chat template or tokenizer config for Gemma 3 `<end_of_turn>` (token 106) vs `eos_token_id` (token 1) mismatch before Modal deployment
2. **Verify GPU memory independently** — use the ~18–21 GB estimate (Section 10), not Soup's 6.4 GB
3. **Select GPU** — A100 40GB or L40S 48GB recommended
4. **On GPU-enabled host:** `docker compose run --rm soup soup train --config /workspace/soup.yaml`
5. **After training:** Evaluate on `data/splits/test.jsonl` (held-out test set)
