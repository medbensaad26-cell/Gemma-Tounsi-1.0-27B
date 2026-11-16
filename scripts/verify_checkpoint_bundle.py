#!/usr/bin/env python3
"""
scripts/verify_checkpoint_bundle.py

Verifies the integrity of a checkpoint bundle created by
export_checkpoint_bundle.py.

Checks:
    - Bundle manifest exists and is valid JSON
    - All listed files exist in the bundle
    - SHA256 checksums match for every file
    - Critical checkpoint files are present (adapter weights, trainer state)
    - Dependency versions are recorded

Usage:
    python scripts/verify_checkpoint_bundle.py \
        --bundle-dir /path/to/checkpoint-500_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Critical files that must be present for exact resumption
CRITICAL_FILES = [
    "checkpoint/adapter_model.safetensors",
    "checkpoint/trainer_state.json",
    "bundle_manifest.json",
    "dependency_versions.txt",
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(bundle_dir: Path) -> bool:
    """Verify a checkpoint bundle. Returns True if valid."""
    bundle_dir = bundle_dir.resolve()

    if not bundle_dir.is_dir():
        print(f"[FAIL] Bundle directory not found: {bundle_dir}")
        return False

    errors = []
    warnings = []

    # Load manifest
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] bundle_manifest.json not found in {bundle_dir}")
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Check manifest fields
    required_fields = [
        "bundle_created",
        "project",
        "checkpoint_name",
        "global_step",
        "base_model",
        "soup_version",
        "dependencies",
        "checkpoint_files",
        "is_complete",
    ]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing manifest field: {field}")

    # Verify each file in the manifest
    for file_info in manifest.get("checkpoint_files", []):
        fname = file_info["name"]
        expected_sha256 = file_info["sha256"]
        expected_size = file_info["size"]

        fpath = bundle_dir / fname
        if not fpath.exists():
            errors.append(f"Missing file: {fname}")
            continue

        actual_sha256 = compute_sha256(fpath)
        actual_size = fpath.stat().st_size

        if actual_sha256 != expected_sha256:
            errors.append(
                f"SHA256 mismatch: {fname} (expected {expected_sha256}, got {actual_sha256})"
            )
        if actual_size != expected_size:
            errors.append(
                f"Size mismatch: {fname} (expected {expected_size}, got {actual_size})"
            )

    # Check critical files
    for critical in CRITICAL_FILES:
        if not (bundle_dir / critical).exists():
            errors.append(f"Critical file missing: {critical}")

    # Check checkpoint completeness
    if not manifest.get("is_complete", False):
        warnings.append(
            "Bundle is marked incomplete — some checkpoint files were missing "
            "at export time. Resumption may not be exact."
        )
    missing_files = manifest.get("missing_files", [])
    if missing_files:
        warnings.append(f"Missing checkpoint files: {missing_files}")

    # Report
    print(f"Bundle: {bundle_dir}")
    print(f"Checkpoint: {manifest.get('checkpoint_name', 'unknown')}")
    print(f"Global step: {manifest.get('global_step', 'unknown')}")
    print(f"Base model: {manifest.get('base_model', 'unknown')}")
    print(f"Soup version: {manifest.get('soup_version', 'unknown')}")
    print(f"Complete: {manifest.get('is_complete', False)}")
    print()

    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        print()
        return False

    if warnings:
        print(f"[WARN] {len(warnings)} warning(s):")
        for warn in warnings:
            print(f"  - {warn}")
        print()

    print("[PASS] Bundle integrity verified.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the integrity of a checkpoint bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to the checkpoint bundle directory to verify.",
    )

    args = parser.parse_args()

    success = verify_bundle(Path(args.bundle_dir))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
