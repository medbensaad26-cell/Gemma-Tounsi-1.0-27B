#!/usr/bin/env python3
"""
scripts/export_checkpoint_bundle.py

Exports a complete, resumable checkpoint bundle from a local directory to a
portable bundle directory. The bundle preserves all components needed for
exact training resumption, not just adapter weights.

Usage:
    python scripts/export_checkpoint_bundle.py \
        --checkpoint-dir /path/to/checkpoint-500 \
        --output-dir ./checkpoint_bundle \
        [--dataset-fingerprint /path/to/dataset.sha256] \
        [--manifest /path/to/checkpoint_manifest.json]

The bundle contains:
    adapter_model.safetensors      LoRA adapter weights
    adapter_config.json            LoRA configuration
    optimizer.pt                   Optimizer state (AdamW)
    scheduler.pt                   Learning-rate scheduler state
    trainer_state.json             Trainer state (global_step, epoch, etc.)
    rng_state.pth                  RNG state
    training_args.bin              Training arguments
    tokenizer.json                 Tokenizer (if available)
    tokenizer_config.json          Tokenizer config (if available)
    special_tokens_map.json        Special tokens map (if available)
    bundle_manifest.json           Bundle metadata with checksums
    dependency_versions.txt        Python package versions

The bundle can be transferred to Alibaba Cloud or any other provider and
resumed with `soup train --resume auto` in a compatible Soup 0.73.3
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Files expected in a complete HF Trainer + PEFT checkpoint
CHECKPOINT_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
]

# Optional tokenizer files
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_dependency_versions() -> dict:
    """Get versions of key Python packages."""
    packages = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "soup-cli",
        "accelerate",
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def verify_checkpoint_complete(checkpoint_dir: Path) -> list:
    """Verify a checkpoint directory is complete.

    Returns a list of missing files (empty if complete).
    """
    missing = []
    for fname in CHECKPOINT_FILES:
        if not (checkpoint_dir / fname).exists():
            missing.append(fname)

    # At minimum, adapter_model and trainer_state must be present
    critical = ["adapter_model.safetensors", "trainer_state.json"]
    critical_missing = [f for f in critical if f not in missing and not (checkpoint_dir / f).exists()]
    # Recompute: actually just check the critical ones directly
    critical_missing = [f for f in critical if not (checkpoint_dir / f).exists()]
    if critical_missing:
        missing.extend(critical_missing)

    return list(set(missing))


def export_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    dataset_fingerprint: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> Path:
    """Export a complete checkpoint bundle.

    Returns the path to the bundle directory.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Verify completeness
    missing = verify_checkpoint_complete(checkpoint_dir)
    if missing:
        print(f"[WARN] Checkpoint {checkpoint_dir.name} is missing: {missing}")
        print("       Bundle will be incomplete — resumption may not be exact.")

    # Create bundle directory
    bundle_name = f"{checkpoint_dir.name}_bundle"
    bundle_dir = output_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint files
    checkpoint_files_dir = bundle_dir / "checkpoint"
    checkpoint_files_dir.mkdir(exist_ok=True)

    copied_files = []
    for fname in CHECKPOINT_FILES:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, checkpoint_files_dir / fname)
            copied_files.append(fname)

    # Copy optional tokenizer files
    for fname in TOKENIZER_FILES:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, checkpoint_files_dir / fname)
            copied_files.append(fname)

    # Read trainer_state for metadata
    trainer_state = {}
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))

    # Build bundle manifest
    bundle_files = []
    for f in sorted(checkpoint_files_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(bundle_dir)
            bundle_files.append(
                {
                    "name": str(rel),
                    "size": f.stat().st_size,
                    "sha256": compute_sha256(f),
                }
            )

    # Dataset fingerprint
    if dataset_fingerprint is None:
        # Try to read from a .sha256 file next to the checkpoint
        sha256_file = checkpoint_dir.parent / "soup_trainval.jsonl.sha256"
        if sha256_file.exists():
            dataset_fingerprint = sha256_file.read_text(encoding="utf-8").strip()

    # External manifest (from training run)
    external_manifest = {}
    if manifest_path and manifest_path.exists():
        external_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bundle_manifest = {
        "bundle_created": datetime.now(timezone.utc).isoformat(),
        "project": "gemma-tounsi-1.0-27b",
        "checkpoint_source": str(checkpoint_dir),
        "checkpoint_name": checkpoint_dir.name,
        "global_step": trainer_state.get("global_step"),
        "epoch": trainer_state.get("epoch"),
        "base_model": "google/gemma-3-27b-it",
        "soup_version": "0.73.3",
        "python_version": sys.version,
        "dependencies": get_dependency_versions(),
        "dataset_fingerprint": dataset_fingerprint,
        "training_config_fingerprint": (
            external_manifest.get("training_config_fingerprint", "")
            if external_manifest
            else ""
        ),
        "checkpoint_files": bundle_files,
        "is_complete": len(missing) == 0,
        "missing_files": missing,
    }

    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2), encoding="utf-8"
    )

    # Write dependency versions file
    dep_file = bundle_dir / "dependency_versions.txt"
    dep_lines = [
        f"# Bundle created: {bundle_manifest['bundle_created']}",
        f"# Checkpoint: {checkpoint_dir.name}",
        f"# Global step: {bundle_manifest['global_step']}",
        "",
    ]
    for pkg, ver in sorted(bundle_manifest["dependencies"].items()):
        dep_lines.append(f"{pkg}=={ver}")
    dep_file.write_text("\n".join(dep_lines) + "\n", encoding="utf-8")

    print(f"Checkpoint bundle exported: {bundle_dir}")
    print(f"  Checkpoint: {checkpoint_dir.name}")
    print(f"  Global step: {bundle_manifest['global_step']}")
    print(f"  Files: {len(bundle_files)}")
    print(f"  Complete: {bundle_manifest['is_complete']}")

    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a complete, resumable checkpoint bundle."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the checkpoint-* directory to export.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the bundle will be created.",
    )
    parser.add_argument(
        "--dataset-fingerprint",
        default=None,
        help="SHA256 fingerprint of the training dataset (optional).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to checkpoint_manifest.json (optional, for config fingerprint).",
    )

    args = parser.parse_args()

    export_checkpoint(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        dataset_fingerprint=args.dataset_fingerprint,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )


if __name__ == "__main__":
    main()
