# Installation & Usage — Gemma Tounsi 1.0 (v1.0)

This document explains how to install dependencies, load the Gemma Tounsi V1
LoRA adapter for inference, and resume training from the saved checkpoint.

---

## 1. Prerequisites

- Python 3.10–3.12 (Soup 0.73.3 requires `>=3.10,<3.13`)
- A GPU with sufficient VRAM for Gemma 3 27B QLoRA inference (RTX-PRO-6000 /
  L40S or equivalent recommended)
- Hugging Face token with Gemma 3 license accepted (`HF_TOKEN`)
- ~96 GB system RAM recommended for 4-bit QLoRA loading

---

## 2. Install dependencies

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install Soup and training dependencies
pip install soup-cli[train]==0.73.3
pip install torch transformers peft bitsandbytes accelerate

# Verify installation
python -c "import torch, transformers, peft, bitsandbytes, accelerate; print('deps OK')"
```

Soup is an external dependency — it is not vendored in this repository.

---

## 3. Set up Hugging Face authentication

Gemma 3 27B is a gated model. You need a Hugging Face User Access Token with
the Gemma license accepted.

```bash
export HF_TOKEN=hf_your_token_here   # Linux/macOS
# or: set HF_TOKEN=hf_your_token_here  # Windows CMD
# or: $env:HF_TOKEN="hf_your_token_here"  # Windows PowerShell
```

Never commit the token to Git. Do not share it.

---

## 4. Run inference on Gemma Tounsi V1

### Option A: Using the acceptance-test script (recommended)

```bash
python scripts/infer_gemma_tounsi.py \
    --adapter-path releases/gemma-tounsi-v1/adapter \
    --output-file inference_acceptance.json \
    --base-model google/gemma-3-27b-it \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.9 \
    --load-in-4bit
```

The script loads `google/gemma-3-27b-it`, applies the LoRA adapter from
`releases/gemma-tounsi-v1/adapter/`, loads the tokenizer + gemma3 chat
template, and generates responses on held-out Tunisian Arabizi and
Arabic-script Tunisian prompts. Results are written to `inference_acceptance.json`.

Exit code 0 = all generations produced non-empty output (technically functional).
Exit code 1 = one or more generations failed or produced empty output.

### Option B: Manual loading in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "google/gemma-3-27b-it"
adapter_path = "releases/gemma-tounsi-v1/adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt = "أهلاً، كيفاش الحال؟ شو الأخبار بتونس اليوم؟"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

input_length = inputs["input_ids"].shape[1]
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response.strip())
```

---

## 5. Resume training from the V1 checkpoint

The `checkpoint/` directory contains the full resumable training state. To
continue training on Modal:

```powershell
modal run --detach modal_train_custom_75_25_english.py
```

Soup's `--resume auto` finds the latest `checkpoint-*` directory containing
`trainer_state.json` and resumes from that global step, restoring LoRA weights,
optimizer state, scheduler state, trainer state, and RNG state.

To migrate to another provider:

1. Download the checkpoint from the Modal Volume:
   ```powershell
   modal volume get gemma-tounsi-checkpoints ./local-checkpoints/
   ```
2. Export a portable bundle:
   ```powershell
   python scripts/export_checkpoint_bundle.py \
       --checkpoint-dir ./local-checkpoints/checkpoint-<step> \
       --output-dir ./checkpoint_bundles \
       --dataset-fingerprint ./dataset.sha256 \
       --manifest ./checkpoint_manifest.json
   ```
3. Transfer the bundle to the target provider and run:
   ```bash
   soup train --config soup.custom_75_25_english.yaml --yes --resume auto
   ```

See `docs/training/checkpoint_and_recovery.md` for details.

---

## 6. Directory reference

| Path | Contents |
|---|---|
| `adapter/adapter_model.safetensors` | LoRA adapter weights |
| `adapter/adapter_config.json` | LoRA configuration (r=16, alpha=32, dropout=0.05) |
| `adapter/generation_config.json` | Generation defaults |
| `tokenizer/tokenizer.json` | Tokenizer vocabulary + merges |
| `tokenizer/tokenizer_config.json` | Tokenizer configuration |
| `tokenizer/special_tokens_map.json` | Special tokens |
| `tokenizer/chat_template.jinja` | gemma3 chat template |
| `checkpoint/checkpoint-<step>/` | Full resumable training state |
| `training_metadata.json` | Run hyperparameters + metadata |
| `training_metrics.json` | Per-step loss, throughput, cost |
| `dataset_manifest.json` | Dataset path, SHA256, record count |
| `checkpoint_info.json` | Base model revision, config fingerprint |
| `sha256_checksums.txt` | SHA-256 for every artifact |
| `inference_acceptance.json` | Functional acceptance test results |

---

## 7. Support

- Repository: `Gemma-Tounsi-1.0-27B`
- Training engine: Soup 0.73.3 (external, pinned)
- Base model: `google/gemma-3-27b-it` (gated, Hugging Face Hub)
