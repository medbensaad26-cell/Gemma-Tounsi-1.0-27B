#!/usr/bin/env python3
"""Gemma Tounsi 1.0 (27B) batch inference and evaluation runner.

Loads the pinned Gemma 3 27B base model plus the saved LoRA adapter, reads a
UTF-8 JSON prompt file, generates one response per prompt, and writes a JSON
report to the persistent Modal checkpoint Volume. Progress is written after
every prompt and the Volume is committed every N prompts.

This is functional inference/evaluation infrastructure, not an automatic
linguistic-quality judge. Human review and comparison against the unmodified
base model are still required.

Typical usage from the repository root:

  # Validate the local adapter/tokenizer without loading the 27B model or GPU:
  modal run scripts/infer_gemma_tounsi.py \\
      --no-generate \\
      --adapter-path ./downloads/Gemma-Tounsi-1.0-27B/checkpoint-7100/checkpoint-7100

  # Run a five-question remote pilot first:
  modal run --detach scripts/infer_gemma_tounsi.py \\
      --prompts-file eval/gemma_tounsi_eval_150.json \\
      --max-prompts 5 \\
      --output-file /checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/inference_pilot_5.json

  # Run all 150 questions after the pilot succeeds and GPU budget is confirmed:
  modal run --detach scripts/infer_gemma_tounsi.py \\
      --prompts-file eval/gemma_tounsi_eval_150.json \\
      --output-file /checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english/inference_eval_150.json

Prompt JSON schema (extra metadata fields are retained in the output):
  [{"id":"q001", "language":"arabizi", "category":"...",
    "prompt":"...", "description":"...", "expected_behavior":"..."}]
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import modal
except ImportError:  # Permit local tokenizer/JSON validation without Modal.
    modal = None


class _ModalStub:
    """Minimal Modal stand-in for local helper testing."""

    class App:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def function(self, *args: Any, **kwargs: Any):
            def decorator(fn):
                return fn
            return decorator

        def local_entrypoint(self, *args: Any, **kwargs: Any):
            def decorator(fn):
                return fn
            return decorator

    class Image:
        @staticmethod
        def debian_slim(python_version: Optional[str] = None):
            return _ModalStub.Image()

        def pip_install(self, *args: Any, **kwargs: Any):
            return self

        def run_commands(self, *args: Any, **kwargs: Any):
            return self

    class Volume:
        @staticmethod
        def from_name(*args: Any, **kwargs: Any):
            return _ModalStub.Volume()

        def commit(self) -> None:
            pass

    class Secret:
        @staticmethod
        def from_name(*args: Any, **kwargs: Any):
            return _ModalStub.Secret()


if modal is None:
    modal = _ModalStub()


# -----------------------------------------------------------------------------
# Pinned model, library versions and Modal resources
# -----------------------------------------------------------------------------
BASE_MODEL = "google/gemma-3-27b-it"
BASE_MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"

# Matched to the versions recorded in the completed training manifest.
TORCH_VERSION = "2.15.0"
TRANSFORMERS_VERSION = "4.57.6"
PEFT_VERSION = "0.21.3"
BITSANDBYTES_VERSION = "0.50.2"

GPU_IDENTIFIER = "RTX-PRO-6000"
INFERENCE_TIMEOUT_SECONDS = 3600  # 1 hour; start with a small pilot.
CHECKPOINT_COMMIT_EVERY = 10

CHECKPOINT_VOLUME_NAME = "gemma-tounsi-checkpoints"
HF_CACHE_VOLUME_NAME = "gemma-tounsi-hf-cache"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/hf-cache"
EXPERIMENT_DIR = (
    f"{CHECKPOINT_MOUNT}/gemma-tounsi-1.0-27b-custom-75-25-english"
)
DEFAULT_ADAPTER_PATH = f"{EXPERIMENT_DIR}/checkpoint-7100"
DEFAULT_OUTPUT_FILE = f"{EXPERIMENT_DIR}/inference_eval_results.json"

app = modal.App("gemma-tounsi-infer")

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"bitsandbytes=={BITSANDBYTES_VERSION}",
        "accelerate",
        "safetensors",
    )
    .run_commands("python3 -m pip check")
    .run_commands(
        "python3 -c 'import torch, transformers, peft, bitsandbytes, accelerate; "
        "print(\"Inference dependencies imported successfully\")'"
    )
)

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=False
)
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME, create_if_missing=False
)
hf_secret = modal.Secret.from_name("gemma-hf")


# Small default smoke test. The full 150-question evaluation is supplied via
# --prompts-file; keeping the default small avoids accidental GPU expenditure.
DEFAULT_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "smoke_arabizi_01",
        "language": "arabizi",
        "category": "tunisian_conversation",
        "prompt": "Jawibni b derja tounsia: chnowa a7welek? Kifech 3addit nharek?",
        "description": "Short Tunisian Arabizi greeting",
    },
    {
        "id": "smoke_arabic_01",
        "language": "arabic_script_derja",
        "category": "tunisian_conversation",
        "prompt": "جاوبني بالتونسي: شنوّة أحوالك اليوم؟",
        "description": "Short Tunisian Arabic-script greeting",
    },
    {
        "id": "smoke_mixed_01",
        "language": "mixed_tunisian",
        "category": "code_switching",
        "prompt": "Fasserli en tounsi chnowa ya3ni machine learning w a3tini exemple simple.",
        "description": "Tunisian/English code-switching",
    },
    {
        "id": "smoke_english_01",
        "language": "english",
        "category": "english_retention",
        "prompt": "Explain the difference between a Python list and a tuple in two or three sentences.",
        "description": "English capability smoke test",
    },
]


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_prompts_file(path: str) -> List[Dict[str, Any]]:
    """Read and validate the user's UTF-8 JSON question list."""
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt JSON file not found: {prompt_path}")

    try:
        data = json.loads(prompt_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {prompt_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError("Prompt JSON must contain a non-empty array of objects.")

    seen_ids = set()
    cleaned: List[Dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt #{index} must be a JSON object.")

        for required in ("id", "language", "prompt"):
            if not isinstance(item.get(required), str) or not item[required].strip():
                raise ValueError(
                    f"Prompt #{index} must contain a non-empty string field {required!r}."
                )

        prompt_id = item["id"].strip()
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)

        normalized = dict(item)
        normalized["id"] = prompt_id
        normalized["language"] = item["language"].strip()
        normalized["prompt"] = item["prompt"].strip()
        normalized.setdefault("category", "uncategorized")
        normalized.setdefault("description", "")
        normalized.setdefault("expected_behavior", "")
        cleaned.append(normalized)

    return cleaned


def validate_adapter_directory(adapter_path: str) -> Dict[str, Any]:
    """Validate the minimum LoRA artifact structure before model loading."""
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {path}")

    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Adapter directory is missing required files: {', '.join(missing)}"
        )
    empty = [name for name in required if (path / name).stat().st_size <= 0]
    if empty:
        raise ValueError(f"Adapter directory contains empty files: {', '.join(empty)}")

    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    recorded_base = config.get("base_model_name_or_path")
    if recorded_base != BASE_MODEL:
        raise ValueError(
            f"Adapter base model is {recorded_base!r}; expected {BASE_MODEL!r}."
        )
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Expected a LoRA adapter, got {config.get('peft_type')!r}.")

    return {
        "adapter_path": str(path),
        "adapter_size_bytes": (path / "adapter_model.safetensors").stat().st_size,
        "base_model_name_or_path": recorded_base,
        "peft_type": config.get("peft_type"),
        "peft_version": config.get("peft_version"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": config.get("target_modules", []),
    }


def load_tokenizer(
    base_model: str = BASE_MODEL,
    adapter_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    revision: str = BASE_MODEL_REVISION,
):
    """Prefer the tokenizer saved with the trained checkpoint."""
    from transformers import AutoTokenizer

    tokenizer = None
    adapter_dir = Path(adapter_path) if adapter_path else None

    if adapter_dir and adapter_dir.is_dir() and (adapter_dir / "tokenizer.json").is_file():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir),
                local_files_only=True,
                use_fast=True,
            )
        except Exception as exc:  # Fall back to the pinned Hub revision.
            print(
                f"[tokenizer] Local checkpoint tokenizer failed: {exc}. "
                "Falling back to the pinned base tokenizer.",
                flush=True,
            )

    if tokenizer is None:
        kwargs: Dict[str, Any] = {
            "revision": revision,
            "use_fast": True,
        }
        if hf_token:
            kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)

    # The training checkpoint saved this chat template as a separate file.
    if adapter_dir and (adapter_dir / "chat_template.jinja").is_file():
        template = (adapter_dir / "chat_template.jinja").read_text(encoding="utf-8")
        if template.strip():
            tokenizer.chat_template = template

    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "Tokenizer has no chat template. Check tokenizer_config.json and "
            "chat_template.jinja in the adapter directory."
        )
    return tokenizer


def load_model_and_adapter(
    base_model: str = BASE_MODEL,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
    revision: str = BASE_MODEL_REVISION,
):
    """Load the pinned base model in NF4 and attach the trained LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    model_kwargs: Dict[str, Any] = {
        "revision": revision,
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
    model.eval()
    return model


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_device(model):
    """Get the input embedding device for a potentially sharded model."""
    device = model.get_input_embeddings().weight.device
    if str(device) == "meta":
        raise RuntimeError("Input embeddings are on the meta device; check device_map.")
    return device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 192,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Generate one response; greedy decoding is the reproducible default."""
    import torch

    rendered = format_prompt(tokenizer, prompt)
    tokens = tokenizer(rendered, return_tensors="pt")
    device = _input_device(model)
    tokens = {key: value.to(device) for key, value in tokens.items()}

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    start = time.monotonic()
    with torch.inference_mode():
        output_ids = model.generate(**tokens, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start

    input_length = int(tokens["input_ids"].shape[-1])
    generated_ids = output_ids[0, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    output_count = int(generated_ids.numel())

    return {
        "response": response,
        "non_empty": bool(response),
        "generation_seconds": round(elapsed, 3),
        "input_tokens": input_length,
        "output_tokens": output_count,
        "tokens_per_second": round(output_count / elapsed, 3) if elapsed > 0 else None,
    }


def run_inference_batch(
    adapter_path: str,
    prompts: List[Dict[str, Any]],
    base_model: str = BASE_MODEL,
    base_model_revision: str = BASE_MODEL_REVISION,
    max_new_tokens: int = 192,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run all prompts and optionally persist a progressive report."""
    result: Dict[str, Any] = {
        "project": "gemma-tounsi-1.0-27b-custom-75-25-english",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "adapter_path": adapter_path,
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "load_in_4bit": load_in_4bit,
        "device_map": "auto",
        "decoding": "greedy",
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(prompts),
        "completed_prompts": 0,
        "results": [],
        "errors": [],
        "model_loaded": False,
        "adapter_loaded": False,
        "all_pass": False,
        "test_scope": (
            "Functional batch inference. Non-empty responses do not prove "
            "Tunisian linguistic quality; use human review and base-model comparison."
        ),
        "versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "bitsandbytes": package_version("bitsandbytes"),
        },
    }

    try:
        adapter_info = validate_adapter_directory(adapter_path)
        result["adapter_metadata"] = adapter_info
        tokenizer = load_tokenizer(
            base_model=base_model,
            adapter_path=adapter_path,
            hf_token=hf_token,
            revision=base_model_revision,
        )
        result["tokenizer_loaded"] = True
        result["tokenizer_vocab_size"] = len(tokenizer)
        result["chat_template_loaded"] = bool(tokenizer.chat_template)

        model = load_model_and_adapter(
            base_model=base_model,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            hf_token=hf_token,
            revision=base_model_revision,
        )
        result["model_loaded"] = True
        result["adapter_loaded"] = True
        result["model_class"] = type(model).__name__
    except Exception as exc:  # Record loading failures in the report.
        result["errors"].append(
            {"stage": "model_or_adapter_load", "error": f"{type(exc).__name__}: {exc}"}
        )
        result["message"] = "Model/tokenizer/adapter loading failed; inspect errors."
        if progress_callback:
            progress_callback(result)
        return result

    for index, item in enumerate(prompts, start=1):
        item_result: Dict[str, Any] = {
            "id": item["id"],
            "language": item["language"],
            "category": item.get("category", "uncategorized"),
            "description": item.get("description", ""),
            "expected_behavior": item.get("expected_behavior", ""),
            "prompt": item["prompt"],
        }
        try:
            generated = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            item_result.update(generated)
            if not generated["non_empty"]:
                item_result["error"] = "The model returned an empty response."
        except Exception as exc:  # Continue testing remaining prompts.
            item_result.update(
                {
                    "response": "",
                    "non_empty": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        result["results"].append(item_result)
        result["completed_prompts"] = index
        if progress_callback:
            progress_callback(result)

        print(
            f"[{index}/{len(prompts)}] "
            f"{'PASS' if item_result.get('non_empty') else 'FAIL'} "
            f"{item_result['id']} ({item_result['language']})",
            flush=True,
        )

    non_empty_count = sum(1 for item in result["results"] if item.get("non_empty"))
    result["non_empty_responses"] = non_empty_count
    result["failed_responses"] = len(result["results"]) - non_empty_count
    result["all_pass"] = (
        result["model_loaded"]
        and result["adapter_loaded"]
        and result["completed_prompts"] == result["num_prompts"]
        and result["failed_responses"] == 0
        and not result["errors"]
    )
    result["message"] = (
        "Every prompt produced a non-empty response. This is a functional "
        "inference pass, not a linguistic-quality certification."
        if result["all_pass"]
        else "Some prompts failed or returned empty output; inspect individual results."
    )
    return result


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


@app.function(
    image=infer_image,
    gpu=GPU_IDENTIFIER,
    timeout=INFERENCE_TIMEOUT_SECONDS,
    secrets=[hf_secret],
    volumes={
        CHECKPOINT_MOUNT: checkpoint_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def run_infer_remote(
    adapter_path: str,
    output_file: str,
    prompts: List[Dict[str, Any]],
    base_model: str = BASE_MODEL,
    base_model_revision: str = BASE_MODEL_REVISION,
    max_new_tokens: int = 192,
    load_in_4bit: bool = True,
) -> Dict[str, Any]:
    """Remote Modal GPU function; persist incremental and final reports."""
    os.environ["HF_HOME"] = HF_CACHE_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_MOUNT}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_MOUNT}/transformers"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is missing. Check that Modal Secret 'gemma-hf' contains HF_TOKEN."
        )

    out_path = Path(output_file)
    if not str(out_path).startswith(CHECKPOINT_MOUNT + "/"):
        raise ValueError(
            f"output_file must be inside the persistent {CHECKPOINT_MOUNT} Volume."
        )

    progress_counter = {"last_commit": 0}

    def persist_progress(report: Dict[str, Any]) -> None:
        _write_json_atomic(out_path, report)
        completed = int(report.get("completed_prompts", 0))
        if completed and completed - progress_counter["last_commit"] >= CHECKPOINT_COMMIT_EVERY:
            checkpoint_volume.commit()
            progress_counter["last_commit"] = completed
            print(f"[persistence] Committed results through prompt {completed}.", flush=True)

    print(f"[inference] Adapter: {adapter_path}", flush=True)
    print(f"[inference] Prompt count: {len(prompts)}", flush=True)
    print(f"[inference] GPU: {GPU_IDENTIFIER}", flush=True)
    print(f"[inference] Output: {out_path}", flush=True)

    report = run_inference_batch(
        adapter_path=adapter_path,
        prompts=prompts,
        base_model=base_model,
        base_model_revision=base_model_revision,
        max_new_tokens=max_new_tokens,
        load_in_4bit=load_in_4bit,
        hf_token=hf_token,
        progress_callback=persist_progress,
    )

    _write_json_atomic(out_path, report)
    checkpoint_volume.commit()
    print(f"[output] Final report committed: {out_path}", flush=True)
    print(
        f"[summary] all_pass={report.get('all_pass')} "
        f"completed={report.get('completed_prompts')}/{report.get('num_prompts')} "
        f"errors={len(report.get('errors', []))}",
        flush=True,
    )
    return report


def _local_tokenizer_check(adapter_path: str) -> Dict[str, Any]:
    """Validate local checkpoint metadata and tokenizer without loading 27B."""
    adapter_info = validate_adapter_directory(adapter_path)
    tokenizer = load_tokenizer(adapter_path=adapter_path)
    return {
        "adapter_metadata": adapter_info,
        "tokenizer_loaded": True,
        "chat_template_loaded": bool(tokenizer.chat_template),
        "tokenizer_vocab_size": len(tokenizer),
        "model_loaded": False,
        "gpu_allocated": False,
        "message": "Local tokenizer/adapter metadata check passed; 27B model not loaded.",
    }


@app.local_entrypoint()
def main(
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    output_file: str = DEFAULT_OUTPUT_FILE,
    base_model: str = BASE_MODEL,
    base_model_revision: str = BASE_MODEL_REVISION,
    max_new_tokens: int = 192,
    load_in_4bit: bool = True,
    prompts_file: Optional[str] = None,
    max_prompts: Optional[int] = None,
    start_index: int = 0,
    no_generate: bool = False,
) -> None:
    """Read prompts locally, then send them to one remote GPU batch."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be a positive integer")

    if prompts_file:
        prompts = load_prompts_file(prompts_file)
        print(f"[prompts] Loaded {len(prompts)} questions from {prompts_file}")
    else:
        prompts = list(DEFAULT_PROMPTS)
        print(f"[prompts] Using {len(prompts)} built-in smoke-test questions")

    total = len(prompts)
    prompts = prompts[start_index:]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(
            f"No prompts selected: total={total}, start_index={start_index}, "
            f"max_prompts={max_prompts}."
        )
    print(
        f"[prompts] Selected {len(prompts)} questions "
        f"(start_index={start_index}, source_total={total})",
        flush=True,
    )

    if no_generate:
        local_path = Path(adapter_path)
        if not local_path.is_dir():
            raise FileNotFoundError(
                "--no-generate is a LOCAL tokenizer check. Pass the local adapter "
                "folder with --adapter-path, for example: "
                "./downloads/Gemma-Tounsi-1.0-27B/checkpoint-7100/checkpoint-7100"
            )
        report = _local_tokenizer_check(str(local_path))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print("=" * 68, flush=True)
    print("Gemma Tounsi 1.0 27B — Batch Inference Evaluation", flush=True)
    print(f"Base model: {base_model}", flush=True)
    print(f"Base revision: {base_model_revision}", flush=True)
    print(f"Adapter: {adapter_path}", flush=True)
    print(f"Questions: {len(prompts)}", flush=True)
    print(f"Max new tokens per question: {max_new_tokens}", flush=True)
    print(f"GPU: {GPU_IDENTIFIER}", flush=True)
    print(f"Output: {output_file}", flush=True)
    print("Decoding: greedy (deterministic)", flush=True)
    print("=" * 68, flush=True)

    report = run_infer_remote.remote(
        adapter_path=adapter_path,
        output_file=output_file,
        prompts=prompts,
        base_model=base_model,
        base_model_revision=base_model_revision,
        max_new_tokens=max_new_tokens,
        load_in_4bit=load_in_4bit,
    )

    if not report.get("all_pass"):
        raise RuntimeError(
            "Inference batch did not fully pass. Inspect the JSON report in the "
            f"checkpoint Volume: {output_file}"
        )
    print("INFERENCE BATCH: PASS (functional check only; quality requires review).", flush=True)
