#!/usr/bin/env python3
"""
scripts/verify_lora_target_modules.py — Gemma Tounsi 1.0 (27B)

CPU-only validation that the Soup LoRA target-module configuration resolves
to an explicit, non-empty list and that PEFT can consume it.

This script:
  1. Loads the actual production experimental YAML (soup.custom_75_25_english.yaml).
  2. Verifies training.lora.target_modules is a non-empty list.
  3. Verifies all four intended module names are present (q_proj, k_proj,
     v_proj, o_proj).
  4. Verifies the final Modal-generated YAML (base64-embedded in
     modal_train_custom_75_25_english.py) contains the same list.
  5. Verifies the compatibility patch does not overwrite the field
     (apply_soup_compat_patch.py only patches soup_cli.utils.gpu.model_size_from_name;
      it never touches YAML config).
  6. Constructs a PEFT LoraConfig using the installed PEFT version.
  7. Attaches LoRA to a tiny CPU-only Llama model with matching module names.
  8. Verifies trainable parameters exist.
  9. Verifies a forward/backward pass and optimizer update.

No GPU. No model downloads (tiny model initialized from config). No Modal.
Exits nonzero if any check fails.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - peft must be installed locally
    PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = exc

REPO = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO / "soup.custom_75_25_english.yaml"
TRAIN_SCRIPT = REPO / "modal_train_custom_75_25_english.py"
PATCH_SCRIPT = REPO / "scripts" / "apply_soup_compat_patch.py"

REQUIRED_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def fail(message: str) -> None:
    print(f"FAIL: {message}", flush=True)
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}", flush=True)
    if not condition:
        fail(message)


# --- 1 & 2: production experimental YAML -------------------------------------


def verify_production_yaml() -> List[str]:
    print(f"\n== 1. Production YAML: {CONFIG_YAML} ==")
    cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    check(isinstance(cfg, dict), "YAML parses to a mapping")
    training = cfg.get("training")
    check(isinstance(training, dict), "training section present")
    lora = training.get("lora")
    check(isinstance(lora, dict), "training.lora section present")
    target_modules = lora.get("target_modules")
    check(target_modules is not None, "training.lora.target_modules is not None")
    check(isinstance(target_modules, list), "target_modules is a list")
    check(len(target_modules) > 0, "target_modules is non-empty")
    missing = [m for m in REQUIRED_TARGETS if m not in target_modules]
    check(not missing, f"all four targets present ({REQUIRED_TARGETS})")
    check(
        lora.get("r") == 16 and lora.get("alpha") == 32,
        "rank unchanged (r=16) and alpha unchanged (alpha=32)",
    )
    print(f"    target_modules = {target_modules}", flush=True)
    return target_modules


# --- 4: embedded Modal-generated YAML ----------------------------------------


def _extract_config_b64(source: str) -> str:
    m = re.search(r"_CONFIG_B64\s*=\s*\(([^)]*)\)", source, re.S)
    check(m is not None, "_CONFIG_B64 block found in training script")
    chunks = re.findall(r'"([A-Za-z0-9+/=]+)"', m.group(1))
    return "".join(chunks)


def verify_embedded_modal_yaml(prod_targets: List[str]) -> None:
    print(f"\n== 2. Embedded Modal YAML: {TRAIN_SCRIPT} ==")
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    b64 = _extract_config_b64(source)
    yaml_text = base64.b64decode(b64).decode(encoding="utf-8")
    cfg = yaml.safe_load(yaml_text)
    check(isinstance(cfg, dict), "Modal config parses to a mapping")
    training = cfg.get("training")
    check(isinstance(training, dict), "Modal training section present")
    lora = training.get("lora")
    check(isinstance(lora, dict), "Modal training.lora section present")
    targets = lora.get("target_modules")
    check(targets is not None, "Modal target_modules is not None")
    check(isinstance(targets, list), "Modal target_modules is a list")
    check(len(targets) > 0, "Modal target_modules is non-empty")
    check(targets == prod_targets, "Modal list matches production YAML list")
    missing = [m for m in REQUIRED_TARGETS if m not in targets]
    check(not missing, f"all four targets present in Modal config ({REQUIRED_TARGETS})")
    check("target_modules: auto" not in yaml_text, "no 'target_modules: auto' in Modal config")
    print(f"    Modal target_modules = {targets}", flush=True)


# --- 5: compatibility patch does not overwrite the field --------------------


def verify_patch_does_not_overwrite() -> None:
    print(f"\n== 3. Compatibility patch: {PATCH_SCRIPT} ==")
    patch_src = PATCH_SCRIPT.read_text(encoding="utf-8")
    check("target_modules" not in patch_src, "patch does not reference target_modules")
    check("training.lora" not in patch_src, "patch does not reference training.lora")
    check("soup_cli.utils.gpu" in patch_src, "patch targets soup_cli.utils.gpu")
    check("model_size_from_name" in patch_src, "patch targets model_size_from_name")
    print("    patch only fixes model-size detection; it never touches YAML config", flush=True)


# --- 6: construct PEFT LoraConfig -------------------------------------------


def verify_peft_config(targets: List[str]) -> Any:
    print("\n== 4. PEFT LoraConfig (installed PEFT) ==")
    check(PEFT_AVAILABLE, f"peft importable ({'OK' if PEFT_AVAILABLE else 'MISSING'})")
    if not PEFT_AVAILABLE:
        print(f"    peft import error: {_PEFT_IMPORT_ERROR}", flush=True)
        return None
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    check(set(lora_config.target_modules) == set(targets), "PEFT config target_modules set")
    check(lora_config.r == 16, "PEFT config r=16")
    check(lora_config.lora_alpha == 32, "PEFT config alpha=32")
    print(f"    PEFT LoraConfig target_modules = {lora_config.target_modules}", flush=True)
    return lora_config


# --- 7/8/9: attach to tiny CPU model, trainable params, fwd/bwd + optim -----


def verify_peft_on_tiny_cpu_model(lora_config: Any) -> None:
    print("\n== 5. LoRA on tiny CPU-only Llama model ==")
    check(PEFT_AVAILABLE, "peft available for model test")
    if not PEFT_AVAILABLE:
        return

    # Tiny Llama config (random init — no download, CPU only).
    # Llama uses self_attn.{q,k,v,o}_proj matching our target names.
    tiny_config = LlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        pretraining_tp=1,
    )
    model = LlamaForCausalLM(tiny_config)
    model.to("cpu")
    model.train()

    peft_model = get_peft_model(model, lora_config)
    peft_model.to("cpu")
    peft_model.train()

    trainable = [
        (n, p) for n, p in peft_model.named_parameters() if p.requires_grad
    ]
    total_trainable = sum(p.numel() for _, p in trainable)
    check(total_trainable > 0, f"trainable parameters exist ({total_trainable})")
    lora_param_names = [n for n, _ in trainable]
    check(any("lora" in n.lower() for n in lora_param_names), "LoRA params attached")
    print(f"    trainable params: {total_trainable}", flush=True)

    # Forward + backward + optimizer update.
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 16))
    labels = input_ids.clone()
    outputs = peft_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    check(loss is not None, "forward pass produces a loss")
    loss.backward()
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-3)
    opt.step()
    opt.zero_grad()
    check(True, "forward/backward + optimizer update ran on CPU")
    print(f"    initial loss: {float(loss.item()):.4f}", flush=True)
    print("    CPU forward/backward + optimizer update: OK", flush=True)


def main() -> None:
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — LoRA target-module CPU prevalidation")
    print("=" * 60)

    prod_targets = verify_production_yaml()
    verify_embedded_modal_yaml(prod_targets)
    verify_patch_does_not_overwrite()
    lora_config = verify_peft_config(prod_targets)
    verify_peft_on_tiny_cpu_model(lora_config)

    print("\n" + "=" * 60)
    print("LORA TARGET-MODULE PREVALIDATION: PASS")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
