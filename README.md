<p align="center">
  <img src="assets/Gemma Tounsi Logo.png" alt="Gemma Tounsi Alpha" width="300">
</p>

<h1 align="center">Gemma Tounsi Alpha</h1>

<p align="center">
  <strong>Bringing Gemma closer to Tunisian Arabic.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/med-bsd/gemma-tounsi-alpha">
    <img src="https://img.shields.io/badge/HuggingFace-gemma--tounsi--alpha-1f6feb?style=flat-square" alt="Hugging Face">
  </a>
  <a href="https://github.com/MakazhanAlpamys/Soup">
    <img src="https://img.shields.io/badge/Training-Soup%200.73.3-1f6feb?style=flat-square" alt="Soup 0.73.3">
  </a>
  <a href="https://github.com/google/gemma_pytorch">
    <img src="https://img.shields.io/badge/Base%20Model-Gemma%203%2027B-1f6feb?style=flat-square" alt="Gemma 3 27B">
  </a>
  <img src="https://img.shields.io/badge/Stage-Alpha%20Release-f59e0b?style=flat-square" alt="Alpha Release">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 (source code)">
</p>

<p align="center">
  Tunisian Derja &bull; Arabizi &bull; Arabic script &bull; QLoRA &bull; PEFT adapter
</p>

---

## Overview

**Gemma Tounsi Alpha** is a QLoRA adaptation of [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) for Tunisian Arabic. The project aims to improve the base model's ability to understand and generate Tunisian Derja in both Arabic script and Arabizi (Latin-script orthography using digits such as `3`, `7`, and `9`), including the natural code-switching and informal variation found in Tunisian online communication.

The project is an engineering effort focused on dialect adaptation of a large open-weight foundation model, not a new foundation model trained from scratch. The released artifact is a **PEFT LoRA adapter** trained with QLoRA (4-bit quantization). To run inference, the original Gemma 3 27B IT base model is required — the adapter does not contain base model weights.

This repository contains the dataset preparation pipeline, training configurations, evaluation track definitions, release orchestration scripts, and inference tooling. Training itself is delegated to the external, pinned **[Soup](https://github.com/MakazhanAlpamys/Soup)** 0.73.3 engine; Soup is a dependency, not vendored or reimplemented here.

> **Alpha release.** A trained QLoRA adapter has been produced and its artifacts have been uploaded to the public Hugging Face repository [`med-bsd/gemma-tounsi-alpha`](https://huggingface.co/med-bsd/gemma-tounsi-alpha). Comprehensive benchmark evaluation and linguistic-quality review remain in progress. No performance benchmarks are claimed in this release.

---

## Project Status

| Area | Status | Notes |
|---|---|---|
| Release stage | Alpha | First functional adapter release |
| Trained adapter | Produced locally | QLoRA PEFT adapter; pilot-scale training on Modal |
| Adapter artifacts | Prepared | `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, `chat_template.jinja` |
| Hugging Face repo | Uploaded | Adapter artifacts at `med-bsd/gemma-tounsi-alpha`; independent verification of public access and adapter loading not yet performed from this repository |
| Base model | Gated | `google/gemma-3-27b-it`; requires HF_TOKEN with Gemma license accepted |
| Training framework | Soup 0.73.3 | Pinned external dependency |
| Data pipeline | Implemented | Validated end-to-end on synthetic fixtures; real-corpus selection pending |
| Evaluation tracks | Defined | TounsiBench, Arabizi, Retention — structure exists, no benchmark results yet |
| Inference tooling | Implemented | `scripts/infer_gemma_tounsi.py` — functional batch inference on GPU |
| Formal evaluation | Not completed | No benchmark scores or linguistic-quality results available |
| Training from scratch reproducibility | Not claimed | Requires Modal cloud resources and dataset upload not present in this repo |

The adapter is **technically functional** — it loads, attaches to the base model, and generates non-empty responses on Tunisian prompts. Linguistic quality of Tunisian dialect generation has not been independently evaluated in this release.

---

## Model Architecture and Technical Details

| Component | Value |
|---|---|
| Base model | `google/gemma-3-27b-it` (27B parameters, instruct-tuned, gated) |
| Base model revision | `005ad3404e59d6023443cb575daa05336842228a` |
| Adaptation method | QLoRA — 4-bit NF4 quantization + LoRA adapter |
| Adapter format | PEFT LoRA (`peft_type: LORA`) |
| Training framework | Soup 0.73.3 (external, pinned container image) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Quantization | 4-bit (bnb_4bit_quant_type: nf4, double quant enabled) |
| Compute dtype | bfloat16 |
| Max sequence length | 8192 tokens |
| Gradient checkpointing | Enabled |
| Training objective | Supervised fine-tuning (SFT), responses only |
| Chat template | gemma3 |

### Foundation model vs. fine-tuned adapter

Gemma Tounsi Alpha is **not** a standalone model. It is a LoRA adapter that modifies the behavior of `google/gemma-3-27b-it` at inference time. The adapter weights are small relative to the 27B base model and are applied on top of the frozen base weights. You must have access to the base model separately (via a Hugging Face token with the Gemma license accepted) before the adapter can be used for inference.

### Training scope

Training was performed on Modal cloud infrastructure using one RTX PRO 6000 GPU within a $24.00 pilot budget. The run was a partial-epoch pilot — not full coverage of the complete dataset or the originally planned 2 epochs. Training can be resumed from the saved checkpoint using `--resume auto`.

---

## Language and Project Objectives

Gemma Tounsi targets Tunisian Arabic across the forms in which Tunisians actually communicate online:

- **Tunisian Derja (Arabic script)** — the native Arabic-script written form of Tunisian dialect.
- **Tunisian Arabizi (Latin script)** — Latin-script orthography using digits (`3`, `7`, `9`) and Latin letters to represent Tunisian phonemes, including informal spelling variation.
- **Code-switching** — natural mixing of Tunisian with French and English, as found in everyday Tunisian online communication.

These are **development objectives and target capabilities**. They are not backed by published benchmark scores in this release. Systematic evaluation of Tunisian dialect understanding and generation quality remains planned work.

---

## Getting Started and Inference

### Prerequisites

- Python 3.10–3.12 (Soup 0.73.3 requires `>=3.10,<3.13`)
- A GPU with sufficient VRAM for Gemma 3 27B QLoRA inference (RTX PRO 6000 / L40S or equivalent recommended)
- A Hugging Face user access token with the Gemma 3 license accepted (`HF_TOKEN`)
- Approximately 96 GB system RAM recommended for 4-bit QLoRA loading

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install soup-cli[train]==0.73.3
pip install torch transformers peft bitsandbytes accelerate safetensors

# Verify
python -c "import torch, transformers, peft, bitsandbytes, accelerate; print('deps OK')"
```

Soup is an external dependency — it is not vendored in this repository.

### Hugging Face authentication

Gemma 3 27B is a **gated** model. You need a Hugging Face token with the Gemma license accepted:

```bash
export HF_TOKEN=hf_your_token_here    # Linux/macOS
# or:
set HF_TOKEN=hf_your_token_here       # Windows CMD
# or:
$env:HF_TOKEN="hf_your_token_here"    # Windows PowerShell
```

Never commit your token to Git. Do not share it.

### Load the adapter for inference (manual Python example)

Load the adapter from the public Hugging Face repository [`med-bsd/gemma-tounsi-alpha`](https://huggingface.co/med-bsd/gemma-tounsi-alpha). The base model `google/gemma-3-27b-it` must also be accessible via a Hugging Face token with the Gemma license accepted.

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "google/gemma-3-27b-it"
adapter_id = "med-bsd/gemma-tounsi-alpha"   # public HF repository
hf_token = os.environ.get("HF_TOKEN")        # set via environment variable

if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable is required")

# 1. Load the base model in 4-bit NF4 quantization
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
    token=hf_token,
)

# 2. Attach the trained LoRA adapter from Hugging Face
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

# 4. Generate on a Tunisian prompt
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

### Run the bundled inference evaluation script

The repository includes `scripts/infer_gemma_tounsi.py`, a functional batch-inference tool that loads the base model plus the saved LoRA adapter and generates responses on a list of Tunisian prompts:

```bash
# Run inference with the adapter downloaded from Hugging Face (requires GPU + HF_TOKEN):
python scripts/infer_gemma_tounsi.py \
    --adapter-path med-bsd/gemma-tounsi-alpha \
    --prompts-file eval/gemma_tounsi_eval_150.json \
    --output-file inference_results.json \
    --max-new-tokens 128 \
    --load-in-4bit
```

The `--no-generate` local metadata check expects a local adapter directory; it is not required when loading from Hugging Face.

The script reports whether each prompt produced a non-empty response. A non-empty response is a **functional check only** — it does not certify Tunisian linguistic quality. Human review and comparison against the unmodified base model are still required.

---

## Training and Reproducibility

Training is performed with **Soup 0.73.3**, an external open-source training engine. This repository owns the configuration, datasets, and release orchestration; Soup owns the training loop, QLoRA implementation, model loading, quantization, and checkpointing.

### Training configuration

The active training configuration for the custom experiment is `soup.custom_75_25_english.yaml` (repository root). It specifies:

- Base model: `google/gemma-3-27b-it`
- QLoRA: 4-bit NF4 quantization, LoRA r=16, alpha=32, dropout=0.05
- Data format: ChatML with the gemma3 chat template
- Max sequence length: 8192 tokens
- Training on responses only, with EOT/EOS tokens included in the loss

The configuration is embedded (base64-encoded) into `modal_train_custom_75_25_english.py` for Modal cloud execution.

### Launching training (Modal)

Training was executed on Modal using `modal_train_custom_75_25_english.py`:

```powershell
modal run --detach modal_train_custom_75_25_english.py
```

Soup's `--resume auto` continues from the latest committed checkpoint containing `trainer_state.json` on the `gemma-tounsi-checkpoints` Volume.

### Where adapter artifacts are saved

The trained adapter and supporting artifacts are prepared under `releases/gemma-tounsi-v1/`:

| Artifact | Purpose |
|---|---|
| `adapter/adapter_model.safetensors` | LoRA adapter weights |
| `adapter/adapter_config.json` | LoRA configuration (r=16, alpha=32, target modules) |
| `tokenizer/tokenizer.json` | Tokenizer vocabulary and merges |
| `tokenizer/tokenizer_config.json` | Tokenizer configuration |
| `tokenizer/special_tokens_map.json` | Special tokens |
| `tokenizer/chat_template.jinja` | gemma3 chat template |
| `checkpoint/checkpoint-<step>/` | Full resumable training state |
| `training_metadata.json` | Run hyperparameters and metadata |
| `dataset_manifest.json` | Dataset path, SHA-256, record count |
| `inference_acceptance.json` | Functional acceptance test results |

The full local release directory is `releases/gemma-tounsi-v1/`. Model weights and checkpoints are **not stored in Git** — they are large binary artifacts distributed outside version control.

### Reproducibility claims

The repository records the base model revision, dependency versions, training configuration, dataset fingerprints, and config checksums. However, training from scratch requires Modal cloud resources (one RTX PRO 6000, HF Secret, dataset Volume upload) and the actual training corpora, which are not included in this repository. Reproducibility from scratch is therefore **not claimed** without those external resources.

---

## Dataset and Data Strategy

The tracked data pipeline (`scripts/prepare_data.sh`, validated in `data/README.md`) is implemented and validated end-to-end against **synthetic fixtures** only. It has **not** been used to generate the actual training dataset.

The training dataset used for the pilot run — a custom 75/25 English-Tunisian mixture — was prepared outside the tracked pipeline and is **not** present in this repository. The tracked pipeline's manifests under `data/manifests/` declare the intended data strategy (Tunisian adaptation slices, English retention/replay, MSA formal register), but production-scale selection and packing for real corpora remain pending. The pipeline's stage order is frozen; only the input sources will change when real corpora are wired in.

Key principles documented in `data/README.md`:

- No undeclared data enters a training file; licensing is checked before use.
- Raw data is immutable; cleaning happens downstream.
- Evaluation data is never training data — TounsiBench, the Arabizi benchmark, and the retention evaluation set are evaluation-only.
- Holdout is reserved before selection; train/holdout contamination aborts the pipeline.
- Outputs are deterministic given the same manifests, pinned sources, and fixed seeds.

Detailed dataset provenance, source licenses, and permitted use are still being developed in `docs/DATA.md` and `data/manifests/`. Dataset sizes, composition, and licenses are **not documented** in this release and are intentionally excluded from this README.

---

## Evaluation and Limitations

### What has been tested

- A **functional inference pilot** was run on 5 Tunisian prompts (Arabizi and Arabic script) using the trained adapter. All 5 prompts produced non-empty responses. Results are recorded in `eval/results/inference_pilot_5.json`.
- The adapter loads and attaches to `google/gemma-3-27b-it` in 4-bit NF4 quantization.
- The tokenizer and gemma3 chat template load correctly with the adapter.

### What remains untested

- **No formal benchmark results exist.** `eval/README.md` explicitly states: "No benchmarks defined, no runs performed, no results exist."
- **No linguistic-quality evaluation** of Tunisian dialect understanding or generation has been completed.
- **No retention/regression benchmarks** have been run against the base model.
- **No comparison** with other Tunisian Arabic models has been performed.

### Known limitations

- **Pilot-scale training only.** V1 was a partial-epoch pilot, not full coverage of the complete dataset or the originally planned 2 epochs.
- **Base model is gated.** `google/gemma-3-27b-it` requires a Hugging Face token with the Gemma license accepted.
- **Adapter is not merged.** Inference requires loading both the base model and the LoRA adapter.
- **Technical functionality ≠ linguistic quality.** The model loads and generates Tunisian responses, but dialect quality is not independently evaluated in this release.
- **Public HF availability not verified.** The repository `med-bsd/gemma-tounsi-alpha` has been created as the intended distribution channel, but its public accessibility and the availability of the adapter files have not been independently confirmed in this release.

No benchmark scores, performance claims, or comparisons with other models are made in this Alpha release.

---

## Roadmap

| Milestone | Status |
|---|---|
| Produce the first Gemma Tounsi QLoRA adapter | ✅ Complete (pilot-scale, V1) |
| Prepare Alpha release artifacts | ✅ Complete (releases/gemma-tounsi-v1/) |
| Verify public Hugging Face availability and adapter loading | ⬜ Pending (artifacts uploaded; independent verification of public access and loading not yet performed from this repository) |
| Perform systematic inference testing | ⬜ Pending |
| Build a documented Tunisian Arabic evaluation suite | ⬜ Pending |
| Improve data quality and language coverage | ⬜ Pending |
| Refine future adapter versions based on evaluation results | ⬜ Pending |

Checkboxes reflect what can be verified from repository contents and the current release; they are not guarantees of future delivery dates.

---

## Licensing and Attribution

### Repository source code

The source code, configuration, scripts, and documentation in this repository are licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

### Base model

`google/gemma-3-27b-it` is subject to **Google's Gemma Terms of Use** and the **Gemma Prohibited Use Policy**. The model is gated on Hugging Face and requires acceptance of its license terms before use. The Apache 2.0 license of this repository does **not** apply to the base model or any derivative model weights.

### Trained adapter

The Gemma Tounsi Alpha LoRA adapter is a derivative work of the base model and inherits the restrictions of the Gemma Terms of Use and of the training data. It is distributed as a release artifact outside Git, not as open-source software under Apache 2.0.

### Training engine

Soup 0.73.3 is an external open-source dependency consumed by this repository. It is not vendored, copied, or redistributed here and remains under its own license.

### Datasets

Training, replay, and evaluation datasets are **not distributed** in this repository. Each dataset remains under its own license and terms. Provenance and licensing will be recorded in `docs/DATA.md` and `data/manifests/` as that work progresses.

---

## Acknowledgments and Contact

Gemma Tounsi builds on [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) by Google and the Gemma team. Training is performed with the external open-source [Soup](https://github.com/MakazhanAlpamys/Soup) training engine (v0.73.3), alongside the broader Hugging Face ecosystem (transformers, PEFT, bitsandbytes, accelerate, safetensors).

**Repository:** https://github.com/medbensaad26-cell/Gemma-Tounsi-1.0-27B  
**Intended model distribution:** https://huggingface.co/med-bsd/gemma-tounsi-alpha  
**Citation:** See [`CITATION.cff`](CITATION.cff).

---

<p align="center">
  <strong>Gemma Tounsi Alpha</strong><br>
  <sub>Bringing Gemma closer to Tunisian Arabic.</sub><br>
  <sub>Alpha release · QLoRA PEFT adapter · Base model: google/gemma-3-27b-it required</sub>
</p>
