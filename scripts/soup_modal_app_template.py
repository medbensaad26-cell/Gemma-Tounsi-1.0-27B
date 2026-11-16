"""Modal application for Gemma Tounsi 1.0 (27B) SFT training.

All 8 deployment blockers resolved:
  1. Model-size patch applied in remote Modal image (soup-cli 0.73.3)
  2. HF Secret "gemma-hf" attached for HF_TOKEN
  3. Persistent dataset Volume "gemma-tounsi-data" mounted at /mnt/data
  4. Persistent checkpoint Volume "gemma-tounsi-checkpoints" mounted at /checkpoints
     with background commit thread for durability
  5. Persistent HF cache Volume "gemma-tounsi-hf-cache" mounted at /hf-cache
  6. Automatic checkpoint resumption via "soup train --resume auto" + manifest
  7. Runtime timeout and budget protection (MAX_RUNTIME_SECONDS, STOP_AFTER_STEPS)
  8. All paths are absolute persistent paths

Run with: modal run --detach soup_modal_app.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import modal

# --- Embedded training config (soup.modal.yaml) ------------------------------
# Base64-encoded Modal-specific configuration. Uses absolute persistent paths:
#   data.train  -> /mnt/data/soup_trainval.jsonl  (Volume: gemma-tounsi-data)
#   output      -> /checkpoints/gemma-tounsi-1.0-27b  (Volume: gemma-tounsi-checkpoints)
__CONFIG_B64_PLACEHOLDER__

# --- Runtime limits ----------------------------------------------------------
MAX_RUNTIME_SECONDS: int = 86400  # 24 hours (Modal max per function attempt)
STOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps
COMMIT_INTERVAL_SECONDS: int = 30  # Modal Volume commit interval

# --- Volume names and mount paths --------------------------------------------
DATA_VOLUME_NAME = "gemma-tounsi-data"
CHECKPOINT_VOLUME_NAME = "gemma-tounsi-checkpoints"
HF_CACHE_VOLUME_NAME = "gemma-tounsi-hf-cache"

DATA_MOUNT = "/mnt/data"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/hf-cache"

DATASET_PATH = f"{DATA_MOUNT}/soup_trainval.jsonl"
DATASET_SHA256_FILE = f"{DATA_MOUNT}/soup_trainval.jsonl.sha256"
OUTPUT_DIR = f"{CHECKPOINT_MOUNT}/gemma-tounsi-1.0-27b"
CHECKPOINT_MANIFEST_PATH = f"{OUTPUT_DIR}/checkpoint_manifest.json"

# --- Modal resources ---------------------------------------------------------
app = modal.App("gemma-tounsi-train")

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME, create_if_missing=True
)

# BLOCKER 1: Apply model-size compatibility patch in the remote Modal image.
# The patch copies the repository-managed patch file and apply script into the
# image, runs them after soup-cli installation, and smoke-tests that
# model_size_from_name("google/gemma-3-27b-it") returns 27 (not 7).
image = (
    modal.Image.debian_slim()
    .pip_install("soup-cli[train]==0.73.3")
    .add_local_file(
        "patches/soup-0.73.3-model-size.patch",
        remote_path="/tmp/soup-0.73.3-model-size.patch",
    )
    .add_local_file(
        "scripts/apply_soup_compat_patch.py",
        remote_path="/tmp/apply_soup_compat_patch.py",
    )
    .run_commands("python3 /tmp/apply_soup_compat_patch.py")
    .run_commands(
        'python3 -c "from soup_cli.utils.gpu import model_size_from_name as m; '
        'assert m(\"google/gemma-3-27b-it\") == 27, f\"Expected 27B, got {m(\"google/gemma-3-27b-it\")}B\"; '
        'print(\"Patch smoke test passed: Gemma 3 27B confirmed\")"'
    )
)

# BLOCKER 2: HF Secret "gemma-hf" provides HF_TOKEN. Attached to the function.
# The secret must be created manually: modal secret create gemma-hf HF_TOKEN=...
hf_secret = modal.Secret.from_name("gemma-hf")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset() -> str:
    """Validate dataset file exists and optionally verify SHA256.

    Returns the actual SHA256 of the dataset file.
    Raises FileNotFoundError if the dataset is missing.
    """
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Upload data/splits/soup_trainval.jsonl to the "
            f"{DATA_VOLUME_NAME} Volume first (scripts/upload_modal_data.ps1)."
        )

    actual_sha256 = compute_sha256(dataset_path)

    # Optional SHA256 validation against precomputed fingerprint
    sha256_file = Path(DATASET_SHA256_FILE)
    if sha256_file.exists():
        expected = sha256_file.read_text(encoding="utf-8").strip()
        if actual_sha256 != expected:
            raise ValueError(
                f"Dataset SHA256 mismatch: expected {expected}, got {actual_sha256}"
            )
        print(f"Dataset SHA256 validated: {actual_sha256}")
    else:
        # No precomputed fingerprint yet; compute and store for future validation
        sha256_file.write_text(actual_sha256 + "\n", encoding="utf-8")
        print(f"Dataset SHA256 computed and stored: {actual_sha256}")

    return actual_sha256


def validate_hf_token() -> None:
    """Validate HF_TOKEN is available from the attached Secret.

    Raises RuntimeError if the token is missing.
    """
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Ensure the 'gemma-hf' Modal Secret is attached to this function "
            "(modal secret create gemma-hf HF_TOKEN=...)."
        )


def find_latest_checkpoint_step(output_dir: Path) -> Optional[int]:
    """Find the latest completed checkpoint step in the output directory.

    Returns None if no checkpoints exist.
    Only returns checkpoints that appear complete (contain trainer_state.json).
    """
    if not output_dir.exists():
        return None

    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[-1])
            except (ValueError, IndexError):
                continue
            # Only consider checkpoints that have a trainer_state.json (complete)
            if (d / "trainer_state.json").exists():
                checkpoints.append((step, d))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][0]


def periodic_commit(stop_event: threading.Event, interval: int) -> None:
    """Background thread that commits all Modal Volumes periodically.

    Ensures checkpoint writes are durable even if the container is terminated
    unexpectedly. Modal Volumes require explicit .commit() calls.
    """
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        try:
            checkpoint_volume.commit()
            hf_cache_volume.commit()
            data_volume.commit()
        except Exception as exc:  # noqa: BLE001 — never crash training
            print(f"[warn] Periodic Volume commit failed: {exc}")


def commit_all_volumes() -> None:
    """Commit all Modal Volumes (final durability flush)."""
    for vol in (checkpoint_volume, hf_cache_volume, data_volume):
        try:
            vol.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Final Volume commit failed for {vol}: {exc}")


def generate_manifest() -> None:
    """Generate a checkpoint manifest with metadata and SHA256 checksums.

    Written to OUTPUT_DIR/checkpoint_manifest.json.
    Includes versions, training config fingerprint, dataset fingerprint,
    checkpoint file list, and checksums.
    """
    import importlib.metadata

    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        return

    # Collect dependency versions
    versions = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "soup-cli"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"

    # Collect checkpoint information
    checkpoints = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[-1])
            except (ValueError, IndexError):
                continue

            files = []
            for f in sorted(d.iterdir()):
                if f.is_file():
                    files.append(
                        {
                            "name": f.name,
                            "size": f.stat().st_size,
                            "sha256": compute_sha256(f),
                        }
                    )

            checkpoints.append(
                {
                    "step": step,
                    "path": d.name,
                    "files": files,
                }
            )

    # Dataset fingerprint
    dataset_sha256 = ""
    dataset_path = Path(DATASET_PATH)
    if dataset_path.exists():
        dataset_sha256 = compute_sha256(dataset_path)

    # Config fingerprint (hash of soup.modal.yaml)
    config_sha256 = ""
    config_path = Path("/root/soup.modal.yaml")
    if config_path.exists():
        config_sha256 = compute_sha256(config_path)

    manifest = {
        "project": "gemma-tounsi-1.0-27b",
        "base_model": "google/gemma-3-27b-it",
        "soup_version": "0.73.3",
        "python_version": sys.version,
        "dependencies": versions,
        "training_config_fingerprint": config_sha256,
        "dataset_fingerprint": dataset_sha256,
        "checkpoints": checkpoints,
    }

    manifest_path = Path(CHECKPOINT_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Checkpoint manifest written: {manifest_path}")


@app.function(
    image=image,
    gpu="L40S",
    timeout=MAX_RUNTIME_SECONDS,
    secrets=[hf_secret],
    volumes={
        DATA_MOUNT: data_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def train(
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
    stop_after_steps: Optional[int] = STOP_AFTER_STEPS,
) -> None:
    """Run Soup SFT training with checkpoint persistence and resume support.

    BLOCKER 6: Automatic checkpoint resumption via --resume auto.
    BLOCKER 7: Runtime timeout and soft-stop protection.
    BLOCKER 4: Background commit thread for Volume durability.
    """
    # Set HF cache environment variables (BLOCKER 5)
    os.environ["HF_HOME"] = HF_CACHE_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_MOUNT}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_MOUNT}/transformers"

    # Write embedded Modal-specific config
    config_path = "/root/soup.modal.yaml"
    config_bytes = base64.b64decode(_CONFIG_B64)
    Path(config_path).write_bytes(config_bytes)

    # BLOCKER 3: Validate dataset exists and (optionally) SHA256
    dataset_sha256 = validate_dataset()

    # BLOCKER 2: Validate HF_TOKEN is available from the Secret
    validate_hf_token()

    # BLOCKER 7: Check if STOP_AFTER_STEPS already reached by existing checkpoint
    output_dir = Path(OUTPUT_DIR)
    if stop_after_steps is not None:
        latest_step = find_latest_checkpoint_step(output_dir)
        if latest_step is not None and latest_step >= stop_after_steps:
            print(
                f"Checkpoint at step {latest_step} >= STOP_AFTER_STEPS="
                f"{stop_after_steps}. Training already complete."
            )
            return

    # Start background commit thread for Volume durability (BLOCKER 4)
    stop_event = threading.Event()
    commit_thread = threading.Thread(
        target=periodic_commit,
        args=(stop_event, COMMIT_INTERVAL_SECONDS),
        daemon=True,
    )
    commit_thread.start()

    try:
        # Build soup train command with --resume auto (BLOCKER 6)
        cmd = [
            "soup",
            "train",
            "--config",
            config_path,
            "--yes",
            "--resume",
            "auto",
        ]

        print(f"Starting training: {' '.join(cmd)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Max runtime: {max_runtime_seconds}s")
        if stop_after_steps:
            print(f"Stop after steps: {stop_after_steps}")

        start_time = time.time()
        process = subprocess.Popen(cmd)

        # Monitor process for soft-stop conditions (BLOCKER 7)
        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed >= max_runtime_seconds:
                print(
                    f"MAX_RUNTIME_SECONDS ({max_runtime_seconds}) reached. "
                    "Sending SIGTERM for graceful stop at next checkpoint."
                )
                process.send_signal(signal.SIGTERM)
                break

            if stop_after_steps is not None:
                latest_step = find_latest_checkpoint_step(output_dir)
                if latest_step is not None and latest_step >= stop_after_steps:
                    print(
                        f"STOP_AFTER_STEPS ({stop_after_steps}) reached at "
                        f"step {latest_step}. Sending SIGTERM for graceful stop."
                    )
                    process.send_signal(signal.SIGTERM)
                    break

            time.sleep(10)

        process.wait()

        if process.returncode not in (0, -signal.SIGTERM):
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print(f"Training completed with return code {process.returncode}")

    finally:
        stop_event.set()
        commit_thread.join(timeout=10)
        # Final commit of all Volumes (BLOCKER 4)
        commit_all_volumes()
        # Generate checkpoint manifest (BLOCKER 6)
        generate_manifest()


@app.local_entrypoint()
def main(
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
    stop_after_steps: Optional[int] = None,
) -> None:
    """Entry point for `modal run --detach soup_modal_app.py`.

    Args:
        max_runtime_seconds: Maximum runtime per session (default 86400s = 24h).
        stop_after_steps: Stop training after this many completed steps.
    """
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — Modal SFT Training")
    print("=" * 60)
    print(f"App: gemma-tounsi-train")
    print(f"GPU: L40S (48 GB)")
    print(f"Soup: 0.73.3 (patched for 27B model-size detection)")
    print(f"Dataset: {DATASET_PATH} (Volume: {DATA_VOLUME_NAME})")
    print(f"Output: {OUTPUT_DIR} (Volume: {CHECKPOINT_VOLUME_NAME})")
    print(f"HF Cache: {HF_CACHE_MOUNT} (Volume: {HF_CACHE_VOLUME_NAME})")
    print(f"HF Secret: gemma-hf (provides HF_TOKEN)")
    print(f"Max runtime: {max_runtime_seconds}s")
    if stop_after_steps:
        print(f"Stop after steps: {stop_after_steps}")
    print("=" * 60)

    train(
        max_runtime_seconds=max_runtime_seconds,
        stop_after_steps=stop_after_steps,
    )

    print("=" * 60)
    print("Training session complete.")
    print(f"Download checkpoints: modal volume get {CHECKPOINT_VOLUME_NAME} .")
    print(f"Resume training: modal run --detach soup_modal_app.py")
    print("=" * 60)
