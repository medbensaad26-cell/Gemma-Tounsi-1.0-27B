#!/usr/bin/env python3
"""
scripts/modal_volume_mount_preflight.py — Gemma Tounsi 1.0 (27B)

Standalone CPU-only Modal Volume mount verification.

Verifies that the actual Modal container can access the Gemma Tounsi
training and validation datasets through the exact mount paths used by
the real training function.

This script:
- Requests NO GPU (CPU only: 0.25 cores, 512 MiB RAM, 180 s timeout).
- Mounts gemma-tounsi-data read-only at /mnt/data.
- Verifies /mnt/data/custom_75_25_english/train.jsonl and
  /mnt/data/validation.jsonl exist, are files, and match expected
  sizes and SHA-256 checksums.
- Returns a JSON summary and exits nonzero on any mandatory failure.

No model downloads, no HF Secret, no GPU, no writes to the Volume.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import modal
except ImportError:  # Allow importing pure functions for local linting
    modal = None


class _ModalStub:
    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    class Image:
        @staticmethod
        def debian_slim(python_version=None):
            return _ModalStub.Image()

        def pip_install(self, *args, **kwargs):
            return self

    class Volume:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Volume()

        def with_mount_options(self, **kwargs):
            return self

    class Secret:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Secret()


if modal is None:
    modal = _ModalStub()

# --- Modal resources ---------------------------------------------------------
VOLUME_NAME = "gemma-tounsi-data"
MOUNT_PATH = "/mnt/data"

app = modal.App("gemma-tounsi-volume-preflight")

preflight_image = modal.Image.debian_slim(python_version="3.12")

# Expected dataset fingerprints (verified locally before upload).
EXPECTED_TRAIN_SIZE: int = 293746534
EXPECTED_TRAIN_SHA256: str = (
    "b6369becf9d536495df59a0849065f5117903596038eff7a9945168ca4672ff0"
)
EXPECTED_VAL_SIZE: int = 35166425
EXPECTED_VAL_SHA256: str = (
    "b57b3bb87999c6f58c656d1b25db747267ef26913b32be0f6610f716451060d4"
)

# Container paths (exactly as used by the real training function).
TRAIN_PATH = f"{MOUNT_PATH}/custom_75_25_english/train.jsonl"
VAL_PATH = f"{MOUNT_PATH}/validation.jsonl"


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 incrementally in small chunks (no full file in RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=preflight_image,
    cpu=0.25,
    memory=512,  # MiB
    timeout=180,
    volumes={
        MOUNT_PATH: modal.Volume.from_name(VOLUME_NAME).with_mount_options(
            read_only=True
        ),
    },
)
def run_preflight_remote() -> Dict[str, Any]:
    """Remote entry point: verify Volume mount and dataset files."""
    results: Dict[str, Any] = {
        "gpu_requested": False,
        "cpu_requested": 0.25,
        "ram_requested_mb": 512,
        "volume_name": VOLUME_NAME,
        "mount_path": MOUNT_PATH,
    }

    # --- Mount root ---
    mount_root = MOUNT_PATH
    results["mount_root"] = mount_root

    # --- Directory listing of /mnt/data ---
    try:
        mnt_entries = sorted(os.listdir(mount_root))
    except FileNotFoundError:
        mnt_entries = []
    results["mnt_data_contents"] = mnt_entries

    # --- Directory listing of /mnt/data/custom_75_25_english ---
    custom_dir = f"{mount_root}/custom_75_25_english"
    try:
        custom_entries = sorted(os.listdir(custom_dir))
    except FileNotFoundError:
        custom_entries = []
    results["custom_75_25_english_contents"] = custom_entries

    # --- Training dataset ---
    train_exists = os.path.exists(TRAIN_PATH)
    train_isfile = os.path.isfile(TRAIN_PATH) if train_exists else False
    train_size = os.path.getsize(TRAIN_PATH) if train_isfile else None
    train_sha256 = (
        sha256_file(TRAIN_PATH) if train_isfile else None
    )
    train_hash_match = (
        train_sha256 == EXPECTED_TRAIN_SHA256 if train_sha256 else False
    )
    results["training_exists"] = train_exists
    results["training_isfile"] = train_isfile
    results["training_size"] = train_size
    results["training_sha256"] = train_sha256
    results["training_hash_match"] = train_hash_match

    # --- Validation dataset ---
    val_exists = os.path.exists(VAL_PATH)
    val_isfile = os.path.isfile(VAL_PATH) if val_exists else False
    val_size = os.path.getsize(VAL_PATH) if val_isfile else None
    val_sha256 = sha256_file(VAL_PATH) if val_isfile else None
    val_hash_match = val_sha256 == EXPECTED_VAL_SHA256 if val_sha256 else False
    results["validation_exists"] = val_exists
    results["validation_isfile"] = val_isfile
    results["validation_size"] = val_size
    results["validation_sha256"] = val_sha256
    results["validation_hash_match"] = val_hash_match

    # --- Overall result ---
    mandatory_ok = (
        train_exists
        and train_isfile
        and train_size == EXPECTED_TRAIN_SIZE
        and train_hash_match
        and val_exists
        and val_isfile
        and val_size == EXPECTED_VAL_SIZE
        and val_hash_match
    )
    results["overall"] = "PASS" if mandatory_ok else "FAIL"
    return results


@app.local_entrypoint()
def main() -> None:
    """Entry point: `modal run scripts/modal_volume_mount_preflight.py`.

    Runs CPU-only (no GPU), prints the JSON summary, and exits nonzero if
    any mandatory check fails.
    """
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — Modal Volume mount preflight")
    print("=" * 60)
    print(f"Volume: {VOLUME_NAME}")
    print(f"Mount path: {MOUNT_PATH}")
    print(f"CPU: 0.25 cores, RAM: 512 MiB, Timeout: 180 s")
    print("GPU: none")
    print("=" * 60)

    result = run_preflight_remote.remote()

    print(json.dumps(result, indent=2))

    print("-" * 60)
    print(f"Mount root: {result['mount_root']}")
    print(f"/mnt/data contents: {result['mnt_data_contents']}")
    print(f"/mnt/data/custom_75_25_english contents: {result['custom_75_25_english_contents']}")
    print(f"Training file: {result['training_exists']} (size={result['training_size']}, hash_match={result['training_hash_match']})")
    print(f"Validation file: {result['validation_exists']} (size={result['validation_size']}, hash_match={result['validation_hash_match']})")
    print("-" * 60)

    if result["overall"] == "PASS":
        print("VOLUME MOUNT PREFLIGHT: PASS — datasets accessible at expected paths.")
        sys.exit(0)
    else:
        print("VOLUME MOUNT PREFLIGHT: FAIL — dataset path or checksum mismatch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
