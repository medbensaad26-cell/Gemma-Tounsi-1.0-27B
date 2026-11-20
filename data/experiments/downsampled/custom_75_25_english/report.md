# Custom Dataset Report — Gemma Tounsi 1.0 27B

## Variant: `custom_75_25_english`

- Source (only): `data/experiments/downsampled/75_percent/train.jsonl`
- Output: `data\experiments\downsampled\custom_75_25_english\train.jsonl`
- Seed: 42 (established project seed)
- English retention target: ~25% of English example count (27,407 of 109,626)

> The 25% English target refers to EXAMPLE COUNT, not token count.

## 1. Source dataset statistics (75% variant)

- Total examples: 466,791
- Total tokens: 79,092,862
- Disk size: 447,488,314 bytes (426.8 MB)
- SHA-256: 33bddcd738e1ff72cccac0c72026356d45cd45575523678018bd9db8dea7de73

| Category | Examples | Tokens | % Examples | % Tokens |
|----------|----------|--------|------------|----------|
| arabizi | 183,423 | 8,940,111 | 39.29% | 11.30% |
| ar | 161,837 | 25,475,341 | 34.67% | 32.21% |
| mixed | 11,905 | 541,632 | 2.55% | 0.68% |
| en | 109,626 | 44,135,778 | 23.49% | 55.80% |

## 2. New dataset statistics

- Total examples: 384,572
- Total tokens: 45,926,282
- Disk size: 293,746,534 bytes (280.1 MB)
- SHA-256: b6369becf9d536495df59a0849065f5117903596038eff7a9945168ca4672ff0
- Average tokens per example: 119.42

## 3. Exact examples retained per category

| Category | Source examples | Target examples | Retained | % Retained |
|----------|-----------------|-----------------|----------|------------|
| arabizi | 183,423 | 183,423 | 183,423 | 100.00% |
| ar | 161,837 | 161,837 | 161,837 | 100.00% |
| mixed | 11,905 | 11,905 | 11,905 | 100.00% |
| en | 109,626 | 27,407 | 27,407 | 25.00% |
| TOTAL | 466,791 | 384,572 | 384,572 | 82.39% |

## 4. Exact tokens retained per category

> Tokens computed with the Gemma 3 27B tokenizer via apply_chat_template (add_special_tokens=False). Preserved-category token totals are reused from the downsampled manifest (identical tokenizer/method); only the selected English examples were tokenized.

| Category | Source tokens | Retained tokens | % Retained |
|----------|---------------|-----------------|------------|
| arabizi | 8,940,111 | 8,940,111 | 100.00% |
| ar | 25,475,341 | 25,475,341 | 100.00% |
| mixed | 541,632 | 541,632 | 100.00% |
| en | 44,135,778 | 10,969,198 | 24.85% |
| TOTAL | 79,092,862 | 45,926,282 | 58.07% |

## 5. New category distribution by examples

| Category | Examples | % Examples |
|----------|----------|------------|
| arabizi | 183,423 | 47.70% |
| ar | 161,837 | 42.08% |
| mixed | 11,905 | 3.10% |
| en | 27,407 | 7.13% |
| TOTAL | 384,572 | 100.00% |

## 6. New category distribution by tokens

| Category | Tokens | % Tokens |
|----------|--------|----------|
| arabizi | 8,940,111 | 19.47% |
| ar | 25,475,341 | 55.47% |
| mixed | 541,632 | 1.18% |
| en | 10,969,198 | 23.88% |
| TOTAL | 45,926,282 | 100.00% |

## 7. Total token reduction (vs 75% variant)

- Source examples: 466,791
- New examples: 384,572
- Example reduction: 82,219 (17.61%)
- Source tokens: 79,092,862
- New tokens: 45,926,282
- Token reduction: 33,166,580 (41.93%)
- Disk size reduction: 153,741,780 bytes
> Note: 25% of English examples does NOT equal 25% of English tokens; English examples carry a higher-than-average token load, so English token retention differs from its example retention.

## 8. Integrity verification results

- Every Output Example Exists In Source: PASS
- No Duplicates In Output: PASS
- Preserved Categories Complete: PASS
- English Approximately 25 Percent: PASS
- No Validation Test Leakage: PASS
- All Source Datasets Unchanged: PASS
- English ratio: 0.2500 (27407/109626)
- Preserved categories: {'arabizi': {'source': 183423, 'kept': 183423, 'complete': True}, 'ar': {'source': 161837, 'kept': 161837, 'complete': True}, 'mixed': {'source': 11905, 'kept': 11905, 'complete': True}}

## 9. SHA-256 checksums

### Output files
- `train.jsonl`: b6369becf9d536495df59a0849065f5117903596038eff7a9945168ca4672ff0  (293,746,534 bytes)
- `manifest.json`: 5dc22e7e9c72b6be1923794142e9e73999aab8fd8f077f72e8e800183f7cbd13  (6,115 bytes)
- `report.md`: SHA-256 printed at the end of this run (self-checksum omitted to keep the report self-consistent).

### Source / original files (verified unchanged)
- `data\splits\train.jsonl`: 86fd39254eda303c7e2f8dc4adaa11bfb5fa7e6a542fbf4fa0675548765b3a14  (626,458,876 bytes)
- `data\splits\validation.jsonl`: b57b3bb87999c6f58c656d1b25db747267ef26913b32be0f6610f716451060d4  (35,166,425 bytes)
- `data\splits\test.jsonl`: 95f43a8fba2f5e0f7f309a00d8a52b705952e4bfa62158aff029a1e0ac79e53a  (34,850,410 bytes)
- `data\splits\soup_trainval.jsonl`: c6299fac164f6772175342b986856927a3c1c5a7496cd273f937206361f5cfe7  (661,625,301 bytes)
- `data\experiments\downsampled\75_percent\train.jsonl`: 33bddcd738e1ff72cccac0c72026356d45cd45575523678018bd9db8dea7de73  (447,488,314 bytes)

## 10. Limitations and deviations

- The dataset is a strict subset of the 75% variant; no examples were generated, translated, augmented, or duplicated.
- No content or metadata of any selected example was modified.
- The 25% English target is approximate at the example-count level (actual: 25.00% of English examples).
- English tokens retained differ from 25% of English tokens because English examples are longer on average; this is expected and documented.
- The `language` field is coarse: `ar` combines Arabic-script Tunisian Derja and MSA formal; `en` maps to capability retention but may include general English content. No finer-grained breakdown was invented.
- English sampling is stratified by task `category` to preserve topic/task diversity, but is otherwise a deterministic random sample (seed 42).
- Token counts for preserved categories (arabizi, ar, mixed) are reused from the existing downsampled manifest rather than re-tokenized, since the tokenizer and method are identical.

## Output files

| File | Path |
|------|------|
| train.jsonl | `data\experiments\downsampled\custom_75_25_english\train.jsonl` |
| manifest.json | `data\experiments\downsampled\custom_75_25_english\manifest.json` |
| report.md | `data\experiments\downsampled\custom_75_25_english\report.md` |

## Using this dataset with Soup

Do NOT modify the production training configuration (`soup.yaml`).
Instead, create a variant config that points `data.train` at the new file:

```yaml
# soup.custom_75_25_english.yaml
base: google/gemma-3-27b-it
task: sft
backend: transformers
modality: text
data:
  train: ./data/experiments/downsampled/custom_75_25_english/train.jsonl
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
output: ./output/gemma-tounsi-1.0-27b-custom-75-25-english
```

Run it with:

```bash
docker compose run --rm soup soup train --config soup.custom_75_25_english.yaml
```

The variant config only changes `data.train` (and the output directory); all other settings mirror `soup.yaml`. Validation is carved internally from the train file via `val_split`, consistent with Soup 0.73.3.