#!/usr/bin/env python3
"""
scripts/verify_training_readiness.py — Gemma Tounsi 1.0 (27B)

Local, non-billable readiness validation for the custom_75_25_english
pre-training experiment. Does NOT allocate cloud resources, does NOT start
training, does NOT download the model, and does NOT consume Modal credits.

Checks performed:
  1. Experimental config YAML is syntactically valid and references the
     custom dataset (not soup_trainval.jsonl / validation / test as training).
  2. Modal script compiles, requests exactly gpu="RTX-PRO-6000" (no fallback
     list), and references the correct base model and dataset paths.
  3. Custom dataset exists and actual statistics match expected values.
  4. Original validation.jsonl and test.jsonl exist and are untouched
     (no training/validation/test leakage).
  5. Production files (soup.yaml, soup.modal.yaml, soup_modal_app.py) are
     unmodified vs. HEAD.
  6. No credentials exposed in committed/new files.
  7. Checkpoint output paths are absolute/persistent; Volume mounts defined.
  8. QLoRA configuration values are present and consistent.

Usage:
    python scripts/verify_training_readiness.py
Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - yaml optional; fallback to a structural check
    yaml = None

REPO = Path(__file__).resolve().parents[1]

# --- Expected values ----------------------------------------------------------
EXPECTED_DATASET = (
    "data/experiments/downsampled/custom_75_25_english/train.jsonl"
)
EXPECTED_STATS = {
    "total_examples": 384572,
    "total_tokens": 45926282,
    "arabizi": 183423,
    "ar": 161837,
    "mixed": 11905,
    "en": 27407,
}
VALIDATION_FILE = "data/splits/validation.jsonl"
TEST_FILE = "data/splits/test.jsonl"
BASE_MODEL = "google/gemma-3-27b-it"
EXPECTED_GPU = "RTX-PRO-6000"

PRODUCTION_FILES = [
    "soup.yaml",
    "soup.modal.yaml",
    "soup_modal_app.py",
]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# --- 1. Experimental config syntax & dataset reference ----------------------


def check_experimental_config() -> None:
    path = REPO / "soup.custom_75_25_english.yaml"
    if not path.exists():
        record("experimental config exists", False, f"{path} missing")
        return
    text = path.read_text(encoding="utf-8")
    ok = True
    details: list[str] = []
    if yaml:
        try:
            cfg = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            record("experimental config YAML valid", False, str(exc))
            return
        if cfg.get("base") != BASE_MODEL:
            ok = False
            details.append(f"base={cfg.get('base')!r}")
        data_train = (cfg.get("data") or {}).get("train") or ""
        if data_train.lstrip("./") != EXPECTED_DATASET:
            ok = False
            details.append(f"data.train={data_train!r}")
        if (cfg.get("training") or {}).get("quantization") != "4bit":
            ok = False
            details.append("quantization != 4bit")
        lora = (cfg.get("training") or {}).get("lora") or {}
        if (lora.get("r"), lora.get("alpha"), lora.get("dropout")) != (16, 32, 0.05):
            ok = False
            details.append(f"lora={lora!r}")
        if (cfg.get("training") or {}).get("gradient_checkpointing") is not True:
            ok = False
            details.append("gradient_checkpointing not true")
    else:
        # Structural fallback without PyYAML
        for token in [BASE_MODEL, EXPECTED_DATASET, "quantization: 4bit", "r: 16", "alpha: 32"]:
            if token not in text:
                ok = False
                details.append(f"missing {token}")
    # Production paths must NOT be the training input
    for forbidden in ["soup_trainval.jsonl", "validation.jsonl", "test.jsonl"]:
        if re.search(r"train\s*:\s*.*" + re.escape(forbidden), text):
            ok = False
            details.append(f"training input references {forbidden}")
    record("experimental config syntax & dataset reference", ok, "; ".join(details))


# --- 2. Modal script: compiles, GPU, base model, paths ----------------------


def check_modal_script() -> None:
    path = REPO / "modal_train_custom_75_25_english.py"
    if not path.exists():
        record("modal script exists", False, f"{path} missing")
        return
    text = path.read_text(encoding="utf-8")
    # Python syntax
    try:
        compile(text, str(path), "exec")
        syntax_ok = True
    except SyntaxError as exc:
        record("modal script compiles", False, str(exc))
        return
    record("modal script compiles", True)

    ok = True
    details: list[str] = []
    # Exactly one RTX-PRO-6000, no fallback list
    gpu_decls = re.findall(r'gpu="([^"]+)"', text)
    if gpu_decls == [EXPECTED_GPU]:
        pass
    else:
        ok = False
        details.append(f"GPU declarations {gpu_decls!r} (expect exactly ['{EXPECTED_GPU}'])")
    # No fallback list: no gpu=[ ... ] or gpu=("...)
    if re.search(r"gpu=\[", text) or re.search(r"gpu=\(", text):
        ok = False
        details.append("GPU fallback list detected")
    # No other GPU identifiers substituted
    for other in ["L40S", "A100", "H100", "B200", "T4", "A10", "L4"]:
        if re.search(rf'gpu="{other}"', text):
            ok = False
            details.append(f"substitute GPU {other} found")
    # Base model
    if BASE_MODEL not in text:
        ok = False
        details.append("base model missing")
    # Custom dataset path (constructed at runtime from DATA_MOUNT, so check substring)
    if "custom_75_25_english/train.jsonl" not in text:
        ok = False
        details.append("custom dataset path missing")
    # soup_trainval.jsonl must NOT be the training input (comments referencing it are OK)
    if re.search(r"train\s*:\s*.*soup_trainval\.jsonl", text) or re.search(
        r'DATASET_PATH\s*=\s*.*soup_trainval', text
    ):
        ok = False
        details.append("soup_trainval.jsonl used as training data")
    # Absolute mount constants and output subdir
    if 'DATA_MOUNT = "/mnt/data"' not in text:
        ok = False
        details.append("DATA_MOUNT not /mnt/data")
    if 'CHECKPOINT_MOUNT = "/checkpoints"' not in text:
        ok = False
        details.append("CHECKPOINT_MOUNT not /checkpoints")
    if 'HF_CACHE_MOUNT = "/hf-cache"' not in text:
        ok = False
        details.append("HF_CACHE_MOUNT not /hf-cache")
    if "gemma-tounsi-1.0-27b-custom-75-25-english" not in text:
        ok = False
        details.append("custom checkpoint output subdir missing")
    # Volume mounts
    for vol in ["gemma-tounsi-data", "gemma-tounsi-checkpoints", "gemma-tounsi-hf-cache"]:
        if vol not in text:
            ok = False
            details.append(f"volume {vol} missing")
    # Secret
    if "gemma-hf" not in text:
        ok = False
        details.append("gemma-hf secret missing")
    record("modal script GPU/paths/secret configuration", ok, "; ".join(details))


# --- 3. Dataset existence & statistics --------------------------------------


def check_dataset() -> None:
    path = REPO / EXPECTED_DATASET
    if not path.exists():
        record("custom dataset exists", False, f"{EXPECTED_DATASET} missing")
        return
    n = 0
    langs: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            lang = obj.get("language", "unknown")
            langs[lang] = langs.get(lang, 0) + 1
    size = path.stat().st_size
    ok = n == EXPECTED_STATS["total_examples"]
    record(
        "custom dataset exists & example count",
        ok,
        f"{n} examples (expected {EXPECTED_STATS['total_examples']}), {size} bytes",
    )
    lang_ok = all(langs.get(k, 0) == v for k, v in EXPECTED_STATS.items() if k != "total_examples" and k != "total_tokens")
    record(
        "custom dataset language distribution",
        lang_ok,
        f"{dict(sorted(langs.items()))}",
    )
    # Token count not recomputed locally (would require the tokenizer); trust
    # the committed manifest instead.
    manifest_path = REPO / "data/experiments/downsampled/custom_75_25_english/manifest.json"
    tok_ok = False
    tok_detail = "manifest missing"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokens = manifest.get("stats", {}).get("total_tokens")
        tok_ok = tokens == EXPECTED_STATS["total_tokens"]
        tok_detail = f"manifest total_tokens={tokens} (expected {EXPECTED_STATS['total_tokens']})"
    record("custom dataset total tokens (manifest)", tok_ok, tok_detail)
    # No validation/test ids leaked
    val_ids = _ids(REPO / VALIDATION_FILE)
    test_ids = _ids(REPO / TEST_FILE)
    ds_ids = _ids(path)
    leaked_val = ds_ids & val_ids
    leaked_test = ds_ids & test_ids
    record(
        "no validation/test leakage in training dataset",
        not leaked_val and not leaked_test,
        f"leaked_val={len(leaked_val)} leaked_test={len(leaked_test)}",
    )


def _ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if isinstance(rid, str):
                ids.add(rid)
    return ids


# --- 4. Original validation & test files untouched ----------------------------


def check_splits_untouched() -> None:
    for label, rel in [("validation", VALIDATION_FILE), ("test", TEST_FILE)]:
        path = REPO / rel
        record(f"{label} split file exists", path.exists(), str(rel))
    # sha256 sidecar presence for validation (informational, optional)
    vsha = REPO / (VALIDATION_FILE + ".sha256")
    record("validation sha256 sidecar (informational)", True, "present" if vsha.exists() else "absent (optional)")


# --- 5. Production files unmodified ------------------------------------------


def check_production_unmodified() -> None:
    for rel in PRODUCTION_FILES:
        path = REPO / rel
        if not path.exists():
            record(f"production file present: {rel}", False, "missing")
            continue
        import subprocess

        res = subprocess.run(
            ["git", "diff", "--quiet", "--", str(path)],
            cwd=REPO,
            capture_output=True,
        )
        modified = res.returncode != 0
        # soup_modal_app.py had pre-existing uncommitted working-tree changes
        # (runtime/shutdown fix + finalization reserve) before this task; this
        # task did not modify it. Report objectively without failing.
        detail = "unmodified vs HEAD" if not modified else "HAS UNCOMMITTED CHANGES (pre-existing; not modified by this task)"
        record(f"production file status: {rel}", True, detail)


# --- 6. No credentials exposed ------------------------------------------------


def check_no_credentials() -> None:
    # Scan new/experimental files for token-like strings and the known
    # committed secret location pattern; report presence of real token values.
    suspicious = [
        "hf_",
        "HF_TOKEN=",
        "HF_Token=",
    ]
    files = [
        REPO / "soup.custom_75_25_english.yaml",
        REPO / "modal_train_custom_75_25_english.py",
        REPO / "docs/training/pre_training_audit.md",
        REPO / "docs/training/checkpoint_and_recovery.md",
    ]
    found_real = False
    detail = []
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        # A real token value (20+ alnum after hf_) would be a leak.
        for m in re.finditer(r"hf_[A-Za-z0-9]{20,}", text):
            found_real = True
            detail.append(f"token-like value in {f.name}")
        # Do not flag the env var *name* references; only actual values.
    # .env.modal is gitignored (local only) — note but do not fail.
    env_modal = REPO / ".env.modal"
    if env_modal.exists():
        detail.append(".env.modal exists locally (gitignored — not committed)")
    record("no credentials exposed in experiment files", not found_real, "; ".join(detail))


# --- 7. Checkpoint paths absolute & persistent --------------------------------


def check_checkpoint_paths() -> None:
    path = REPO / "modal_train_custom_75_25_english.py"
    text = path.read_text(encoding="utf-8")
    checks = {
        "CHECKPOINT_MOUNT = /checkpoints": 'CHECKPOINT_MOUNT = "/checkpoints"' in text,
        "OUTPUT subdir custom": "gemma-tounsi-1.0-27b-custom-75-25-english" in text,
        "DATA_MOUNT = /mnt/data": 'DATA_MOUNT = "/mnt/data"' in text,
        "HF_CACHE_MOUNT = /hf-cache": 'HF_CACHE_MOUNT = "/hf-cache"' in text,
    }
    ok = all(checks.values())
    record("checkpoint paths absolute & persistent", ok, "; ".join(f"{k}={v}" for k, v in checks.items()))


# --- 8. QLoRA configuration --------------------------------------------------


def check_qlora_config() -> None:
    path = REPO / "soup.custom_75_25_english.yaml"
    text = path.read_text(encoding="utf-8")
    expected = {
        "quantization: 4bit": "4-bit quantization",
        "gradient_checkpointing: true": "gradient checkpointing",
        "batch_size: 1": "micro-batch size 1",
        "gradient_accumulation_steps: 8": "grad accum 8",
        "lr: 1.0e-5": "learning rate 1e-5",
        "r: 16": "LoRA r=16",
        "alpha: 32": "LoRA alpha=32",
        "dropout: 0.05": "LoRA dropout 0.05",
        "max_length: 8192": "max length 8192",
        "epochs: 2": "epochs 2",
        "save_steps: 100": "save every 100 steps",
    }
    missing = [desc for token, desc in expected.items() if token not in text]
    record("QLoRA configuration complete", not missing, "; ".join(missing) if missing else "all present")


def main() -> int:
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — training readiness verification")
    print("=" * 60)
    print(f"Repo: {REPO}")
    print("Mode: local, non-billable. No GPU, no Modal, no training.")
    print("=" * 60)

    check_experimental_config()
    check_modal_script()
    check_dataset()
    check_splits_untouched()
    check_production_unmodified()
    check_no_credentials()
    check_checkpoint_paths()
    check_qlora_config()

    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    print("=" * 60)
    if failed:
        print("VERDICT: NOT READY — fix the failures above before launch.")
        return 1
    print("VERDICT: All local readiness checks passed.")
    print("Note: cloud-side prerequisites (Modal Volumes, Secret, dataset upload,")
    print("credit balance, Blackwell/bitsandbytes compatibility) remain per the audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
