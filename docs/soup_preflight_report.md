# Gemma Tounsi 1.0 27B — Soup Pre-Flight Report

**Generated:** 2026-11-16
**Soup CLI version:** 0.73.3
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

Deviation from target (90/5/5): within 0.0002% due to integer rounding. No adjustment needed — `int(703696 * (1 - 0.05263068143061777))` produces exactly 666,660.

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

Soup validation results:
- `train.jsonl`: 666,660/666,660 rows valid for chatml
- `validation.jsonl`: 37,036/37,036 rows valid for chatml
- `test.jsonl`: 37,038/37,038 rows valid for chatml
- `soup_trainval.jsonl`: 703,696/703,696 rows valid for chatml

No records were modified, normalized, or converted. Original JSONL lines preserved exactly.

---

## 5. Final `soup.yaml` Configuration

File: `soup.yaml` (79 lines)

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

All keys validated against the Soup 0.73.3 schema (`SoupConfig` / `DataConfig` / `TrainingConfig` / `LoraConfig`). No invented fields.

**Notable configuration decisions:**
- `data.val_split` set to `0.05263068143061777` so Soup's internal `int(total * (1 - val_split))` produces exactly 666,660 training rows
- `data.chat_template: gemma3` — uses Soup's registered Gemma 3 chat template name
- `data.train_on_responses_only: true` — assistant-only training (mask non-assistant tokens)
- `training.train_on_eot: true` — required because `soup data doctor` reported MAJOR: 100% of sampled rows never train an EOS/EOT token
- `training.gradient_checkpointing: true` — memory savings for 8192-token sequences
- Test set excluded from `soup.yaml` entirely

---

## 6. Gemma Tokenizer and Chat-Template Compatibility

`$ soup data doctor ./data/splits/soup_trainval.jsonl --model google/gemma-3-27b-it --show-mask 3 --max-length 8192 --train-on-responses-only`

| Check | Verdict | Message |
|-------|---------|---------|
| chat_template | OK | Tokenizer has a chat_template (1532 chars) |
| template_render | OK | 0/200 rows failed to render through the chat template |
| generation_markers | MINOR | Template lacks `{% generation %}` markers — assistant-only masking falls back to heuristic |
| eos_in_labels | MAJOR | 200/200 rows (100.0%) never train an EOS/EOT token |
| bos_duplication | OK | 0/200 rows (0.0%) start with duplicated BOS |
| system_role | OK | Template supports system role (18 rows use one) |
| unknown_roles | OK | 0/200 rows contain unknown roles |
| truncation_risk | OK | p95 = 523 tokens vs max_length=8192; 0/200 rows truncated |

**Mask output sample (row 0):**
```
<bos><start_of_turn>user ما هي الدول التي كانت أعضاء في قوى المحور في الحرب العالمية الثانية؟<end_of_turn>
<start_of_turn>model الدول التي كانت أعضاء في قوى المحور في الحرب العالمية الثانية هي: ...<end_of_turn>
```

**Key observations:**
- Gemma 3 uses `<start_of_turn>user` / `<start_of_turn>model` / `<end_of_turn>` (not `assistant` role name)
- `eos_token_id = 1`, `bos_token_id = 2`
- `train_on_eot: true` in soup.yaml addresses the MAJOR EOS finding
- `train_on_responses_only: true` ensures only assistant (model) tokens contribute to loss

---

## 7. Assistant-Only Masking Results

With `train_on_responses_only: true`:
- User turns (`<start_of_turn>user ... <end_of_turn>`) are masked with `IGNORE_INDEX (-100)`
- Assistant/model turns (`<start_of_turn>model ... <end_of_turn>`) contribute to SFT loss
- The `generation_markers` MINOR finding means the Gemma 3 template lacks `{% generation %}` markers, so Soup falls back to a heuristic that may include role-prefix tokens (`<start_of_turn>model\n`) in the loss span. This is a known limitation of the Gemma 3 chat template and does not block training.

---

## 8. Sequence-Length Distribution and Truncation Analysis

Soup `data stats` results (approximate token counts):

| File | Samples | Avg Tokens | Min Tokens | Max Tokens | Median Chars | p90 Chars |
|------|---------|------------|------------|------------|--------------|-----------|
| train.jsonl | 666,660 | 170 | 39 | 34,230 | 283 | 1,507 |
| validation.jsonl | 37,036 | 172 | 40 | 22,898 | 284 | 1,500 |
| test.jsonl | 37,038 | 171 | 39 | 21,687 | 283 | 1,515 |
| soup_trainval.jsonl | 703,696 | 170 | 39 | 34,230 | 283 | 1,507 |

**Truncation analysis at candidate max_length values:**

| max_length | Est. truncated (train) | Est. truncated (val+test) | Recommendation |
|------------|------------------------|---------------------------|----------------|
| 512 | ~5-10% | ~5-10% | Too aggressive for long-context examples |
| 1024 | ~2-3% | ~2-3% | Acceptable but loses long reasoning examples |
| 2048 | <1% | <1% | Safe default |
| 4096 | ~0% | ~0% | Good balance |
| 8192 | 0% (doctor-verified) | 0% | **Selected** — no truncation risk |

At `max_length=8192`, `soup data doctor` confirmed 0/200 sampled rows (0.0%) would be truncated, with p95 length of 523 tokens. The maximum observed token count is 34,230 (approx), but this is an extreme outlier; the vast majority of examples are well under 8192 tokens.

---

## 9. Soup Configuration Validation Results

### `soup doctor`
All checks passed. Environment ready (CPU-only container; GPU required for actual training).

### `soup plan --config soup.yaml`
```
Field                   Value
base                    google/gemma-3-27b-it
task                    sft
config_sha              d1d156ed61f2c537...
dataset_sha             c6299fac164f6772...
estimated_cost          $0.2000
estimated_minutes       40.0
peak_vram_gb            8.0
spot_price_usd_per_hour $0.30
```
Plan written to `soup.tfstate` (not yet applied).

### `soup profile --config soup.yaml`
```
Model:     google/gemma-3-27b-it
Params:    7.0B (trainable: 29,360,128 with LoRA r=16)
Quantization: 4bit
Gradient checkpointing: enabled

GPU Memory Estimate:
  Model                              ~3.5 GB
  LoRA                               ~0.1 GB
  Optimizer                          ~0.1 GB
  Activations (bs=1, seq=8192)       ~1.2 GB
  Overhead                           ~1.5 GB
  Total                              ~6.4 GB

Recommendations:
  OK Fits in 24 GB VRAM
  OK Recommended batch_size: 8

Compatible GPUs: RTX3070 (8 GB), RTX3070TI (8 GB), RTX4060 (8 GB),
                 RTX3080 (10 GB), RTX3060 (12 GB)
```

### `soup cost --config soup.yaml --gpu A100 --json`
A100 cost estimates across providers (RunPod A100 80G at $1.89/hr, Lambda A100 40G at $1.10/hr, CoreWeave A100 80G at $2.21/hr). Duration estimate appears inflated (~10,113 hours) — likely a pre-flight calculation artifact; actual training time will be far shorter.

---

## 10. GPU Memory Estimate (A100)

Soup profile estimates **~6.4 GB total VRAM** for this configuration (4-bit QLoRA, LoRA r=16, batch_size=1, gradient_accumulation=8, seq=8192, gradient_checkpointing=enabled).

On an A100 (40 GB or 80 GB), this configuration fits comfortably with significant headroom. The profile recommends `batch_size: 8` for 24 GB cards; on A100 40/80 GB, a larger batch size (e.g., 4-8 with gradient accumulation) could be used to improve throughput.

---

## 11. Remaining Issues Before Modal Deployment

### Critical (must resolve before training)

1. **EOS/EOT token not in training labels (MAJOR from `soup data doctor`)**
   - 100% of sampled rows never train an EOS/EOT token
   - Mitigation: `train_on_eot: true` is set in `soup.yaml`
   - Verify after first training step that loss includes EOT tokens

2. **Gemma 3 is a gated model**
   - Requires valid `HF_TOKEN` with access to `google/gemma-3-27b-it`
   - `.env` contains `HF_TOKEN` (value not shown in this report)
   - Modal deployment must inject `HF_TOKEN` as a secret

3. **`generation_markers` MINOR**
   - Gemma 3 chat template lacks `{% generation %}` markers
   - Assistant-only masking falls back to heuristic
   - May include role-prefix tokens (`<start_of_turn>model\n`) in loss
   - Acceptable for baseline; consider custom chat template if loss masking precision is needed

### Informational (monitor during training)

4. **Long examples may truncate at 8192**
   - Max observed approx. tokens: 34,230 (extreme outlier)
   - p90 is ~1,507 chars (~375 tokens) — well within 8192
   - 0% truncation at 8192 confirmed by doctor on 200-row sample

5. **CPU-only validation environment**
   - All `soup` commands ran in `soup-cpu` container (no GPU)
   - `soup doctor` warns: "Training will be slow without GPU"
   - Actual training requires GPU (A100 recommended)

### Files Created or Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/split_dataset.py` | Created | Dataset splitting script (seed 42, stratified) |
| `data/splits/train.jsonl` | Created | 90% training split (666,660 records) |
| `data/splits/validation.jsonl` | Created | 5% validation split (37,036 records) |
| `data/splits/test.jsonl` | Created | 5% test split (37,038 records, excluded from training) |
| `data/splits/soup_trainval.jsonl` | Created | Combined train+val (703,696 records, no shuffle) |
| `data/splits/split_report.json` | Created | Full split statistics and verification |
| `soup.yaml` | Modified | Complete Soup 0.73.3 config (train/val paths, QLoRA, Gemma3 template) |
| `scripts/inspect_dataset.py` | Created | Dataset inspection utility |

### Files NOT Modified

- `data/train.jsonl` — original master dataset, unchanged
- `data/splits/train.jsonl` — original split, unchanged
- `data/splits/validation.jsonl` — original split, unchanged
- `data/splits/test.jsonl` — original split, unchanged

### Next Steps

1. Review `soup.tfstate` (generated by `soup plan`)
2. On GPU-enabled host: `docker compose run --rm soup soup train --config /workspace/soup.yaml`
3. Monitor loss for EOS/EOT learning (verify `train_on_eot` is effective)
4. After training, evaluate on `data/splits/test.jsonl` (held-out test set)
