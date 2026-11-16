# Gemma 3 EOT (End-of-Turn) Verification Report

**Project:** Gemma Tounsi 1.0 27B
**Soup version:** 0.73.3
**Base model:** `google/gemma-3-27b-it`
**Verified via:** Docker container `soup-cpu` (Soup 0.73.3 installed)

---

## 1. Gemma 3 Tokenizer Verification

Loaded the actual tokenizer via `transformers.AutoTokenizer.from_pretrained("google/gemma-3-27b-it")`.

| Token | ID | Source |
|-------|-----|--------|
| `<eos>` | 1 | `tokenizer.eos_token_id` |
| `<bos>` | 2 | `tokenizer.bos_token_id` |
| `<end_of_turn>` | 106 | `tokenizer.encode("<end_of_turn>", add_special_tokens=False)` |
| `<start_of_turn>` | 105 | `tokenizer.encode("<start_of_turn>", add_special_tokens=False)` |

**Confirmed:** Gemma 3 uses `<end_of_turn>` (token **106**) as the turn terminator, NOT `<eos>` (token **1**).

**Chat template:**
- Template length: 1532 characters
- `{% generation %}` markers: **ABSENT** → preferred masking path (`return_assistant_tokens_mask`) is NOT usable
- Roles alternate user/assistant (consecutive assistant roles rejected by template)

---

## 2. Soup Source Code Inspected

All paths below are in Soup 0.73.3 site-packages:

| File | Line(s) | Function | Role |
|------|---------|----------|------|
| `soup_cli/data/loss_mask.py` | 90–126 | `_apply_template_with_mask` | Preferred path: `apply_chat_template(..., return_assistant_tokens_mask=True, return_dict=True)` |
| `soup_cli/data/loss_mask.py` | 122–126 | — | Rejects all-zero assistant masks (no `{% generation %}` markers → returns `None`) |
| `soup_cli/data/loss_mask.py` | 128–163 | `_tokenize_only` | Fallback: renders messages without auto-prepending BOS |
| `soup_cli/data/loss_mask.py` | 165–240 | `build_assistant_only_labels` | Main masking function — dispatch between preferred/fallback |
| `soup_cli/data/loss_mask.py` | 202 | — | `eos_token_id = _resolve_eos_token_id(tokenizer) if include_eot else None` → resolves to **1** |
| `soup_cli/data/loss_mask.py` | 207–208 | — | `_extend_mask_to_eot(input_ids, mask, eos_token_id)` — absorbs trailing token 1s |
| `soup_cli/data/loss_mask.py` | 237–255 | `_resolve_eos_token_id` | Returns `tokenizer.eos_token_id` (= 1 for Gemma 3) |
| `soup_cli/data/loss_mask.py` | 256–277 | `_extend_mask_to_eot` | Walks kept spans, absorbs consecutive `eos_token_id` tokens |
| `soup_cli/data/sft_format.py` | 62 | — | `include_eot = bool(getattr(training_cfg, "train_on_eot", False))` |
| `soup_cli/data/sft_format.py` | 88–89 | — | Passes `include_eot` to `_build_assistant_only_format_row` |
| `soup_cli/data/sft_format.py` | 152–160 | `_build_assistant_only_format_row` | Calls `build_assistant_only_labels(..., include_eot=include_eot)` |

**Data doctor check** (`soup_cli/utils/data_doctor.py`, `check_eos_in_labels`):
- Looks for `eos_token_id` (token 1) within trained spans
- Reports MAJOR when ≥50% of rows lack token 1 in any trained span
- **False positive for Gemma 3** — Gemma 3 uses token 106 (`<end_of_turn>`), not token 1

---

## 3. Pipeline Behavior for Gemma 3

### Masking path taken

Gemma 3's chat template lacks `{% generation %}` markers, so:

1. **Preferred path** (`_apply_template_with_mask`): `return_assistant_tokens_mask=True` returns an all-zero or missing `assistant_masks` → function returns `None` → fallback used.
2. **Fallback path** (incremental delta in `build_assistant_only_labels`): Renders cumulative messages via `_tokenize_only`, computes per-turn token deltas, and trains only assistant-turn deltas.

### How assistant `<end_of_turn>` gets trained

In the fallback path, for each assistant turn:
```python
cumulative.append(msg)           # add assistant turn
rendered = _tokenize_only(cumulative)  # full chat through this assistant turn
new_len = len(rendered)
if msg.get("role") == "assistant":
    end = min(new_len, len(full_ids))   # end includes assistant's <end_of_turn>
    labels[prev_len:end] = full_ids[prev_len:end]  # train everything including <end_of_turn>
```

The assistant span (`prev_len` to `end`) includes **all** tokens from `<start_of_turn>model` through `<end_of_turn>` and the trailing `\n`. So `<end_of_turn>` (106) **is** trained.

### `train_on_eot` is a no-op for Gemma 3

`include_eot=True` resolves `eos_token_id` = 1 (`<eos>`), then `_extend_mask_to_eot` absorbs consecutive token-1 tokens after each trained span. Gemma 3 turns end with token 106 (`<end_of_turn>`), not token 1 (`<eos>`), so the `while full_ids[extra] == eos_token_id` loop never matches. **Setting `train_on_eot: true` produces identical labels to `train_on_eot: false` for Gemma 3.**

---

## 4. Controlled Test Results

Test conversations run through Soup's actual `build_assistant_only_labels`:

### Test 1 — Single turn
```
Messages: [user: "What is the capital of France?", assistant: "Paris"]
Decoded: <bos><start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
Paris<end_of_turn>
```
- Trained span: positions [13,18] → `<start_of_turn>`, `model`, `\n`, `Paris`, `<end_of_turn>`, `\n`
- Contains `<end_of_turn>` (106): **Yes**
- Contains `<eos>` (1): **No**
- All assertions: **PASSED**

### Test 2 — Multiple turns (2 user/assistant pairs)
```
Decoded: <bos><start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
4<end_of_turn>
<start_of_turn>user
What is 3+3?<end_of_turn>
<start_of_turn>model
6<end_of_turn>
```
- Trained spans: [(13,18), (31,36)] — both contain `<end_of_turn>` (106)
- All assertions: **PASSED**

### Test 3 — With system message
```
Decoded: <bos><start_of_turn>system
You are a helpful assistant.<end_of_turn>
<start_of_turn>user
Hello<end_of_turn>
<start_of_turn>model
Hi there!<end_of_turn>
```
- System + user positions all masked (-100)
- Assistant span contains `<end_of_turn>` (106)
- All assertions: **PASSED**

### Test 4 — Alternating multi-turn (4 turns)
- Two assistant spans, both contain `<end_of_turn>` (106)
- All assertions: **PASSED**

### Test 5 — `include_eot=True` vs `False`
- **Identical labels** — confirms `train_on_eot` is a no-op for Gemma 3

---

## 5. Required Assertions — Results

| Assertion | Result |
|-----------|--------|
| User content labels = -100 | **Pass** — all user positions masked |
| User `<end_of_turn>` remains masked | **Pass** — user EOT (106) has label -100 |
| Assistant response content contributes to loss | **Pass** — assistant content tokens trained |
| Assistant `<end_of_turn>` contributes to loss | **Pass** — token 106 in trained span |
| Assistant-only masking enabled | **Pass** — `train_on_responses_only: true` |
| User/system tokens not accidentally unmasked | **Pass** — all non-assistant positions are -100 |

---

## 6. Fix Required?

**No fix is necessary.**

The existing Soup 0.73.3 implementation already trains assistant `<end_of_turn>` (106) correctly via the incremental-delta fallback path. The `train_on_eot: true` setting in `soup.yaml` is harmless but unnecessary for Gemma 3 — it looks for `eos_token_id` (1), which Gemma 3 doesn't use as a turn terminator.

**What was verified:**
- Assistant `<end_of_turn>` IS in the trained span (confirmed in all 4 test conversations)
- User `<end_of_turn>` IS masked (label = -100)
- `include_eot=True` vs `False` produces identical output for Gemma 3

**What is a false positive (not a bug):**
- `soup data doctor` reports `eos_in_labels: MAJOR` because it looks for `eos_token_id` (1), not `<end_of_turn>` (106). Gemma 3's turn terminator is token 106, which IS trained — just not token 1.

---

## 7. Files Modified

| File | Action |
|------|--------|
| `tests/test_data_pipeline.py` | Added `TestGemma3EOTMasking` class with 7 regression tests |
| `scripts/verify_gemma3_eot.py` | Created standalone verification script |

No changes to `soup.yaml`, `soup_cli` site-packages, dataset files, or splits.

---

## 8. Regression Test Results

```
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_token_ids                    PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_single_turn_assistant_eot_trained PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_single_turn_user_eot_masked   PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_multi_turn_all_assistant_eots_trained PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_system_message_masked         PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_train_on_eot_is_noop_for_gemma3 PASSED
tests/test_data_pipeline.py::TestGemma3EOTMasking::test_no_eos_token_id_in_trained_spans PASSED

7 passed
```

Tests use `pytest.skip()` if `HF_TOKEN` is not set or the Gemma 3 tokenizer cannot be loaded (no network), making them safe for CI environments without HF access.

---

## 9. Remaining Limitations

1. **`soup data doctor` `eos_in_labels` MAJOR verdict** — This is a false positive for Gemma 3. The check looks for `eos_token_id` (1) in trained spans, but Gemma 3's turn terminator is `<end_of_turn>` (106). The assistant EOT IS trained; the check just looks for the wrong token. To resolve, the data doctor check would need to also consider `<end_of_turn>` or the template's actual turn-closing token.

2. **`train_on_eot` no-op** — Setting `train_on_eot: true` in `soup.yaml` has no effect for Gemma 3 because it absorbs `eos_token_id` (1) tokens, not `<end_of_turn>` (106). This is harmless but provides no benefit. Remove or keep — no training impact either way.

3. **Trailing `\n` after `<end_of_turn>`** — Gemma 3's chat template appends `\n` (token 107) after `<end_of_turn>`. The trained span includes this `\n`, which is slightly broader than strictly necessary but harmless.

4. **Gemma 3 is a gated model** — Requires valid `HF_TOKEN` with access to `google/gemma-3-27b-it`. Tests skip gracefully without it.

---

## Conclusion

**Assistant `<end_of_turn>` (token 106) IS trained** by Soup 0.73.3's existing assistant-only loss masking pipeline for Gemma 3, via the incremental-delta fallback path. No code changes are required. The `train_on_eot: true` setting in `soup.yaml` is a no-op for Gemma 3 but does not cause any harm. The `soup data doctor` `eos_in_labels` MAJOR verdict is a false positive specific to Gemma 3's use of token 106 instead of token 1 as the turn terminator.
