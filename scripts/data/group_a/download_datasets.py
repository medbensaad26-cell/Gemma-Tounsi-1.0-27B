"""Download Group A Tunisian Arabizi candidate datasets.

Downloads only APPROVED/CONDITIONAL datasets into data/raw/group_a/.
REJECTED/HOLD datasets are NOT downloaded into the training-ready corpus.

Usage:
    python scripts/data/group_a/download_datasets.py
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "group_a"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# Load dataset config
import yaml

CONFIG_PATH = ROOT / "configs" / "data" / "group_a.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DOWNLOADABLE_STATUSES = {"APPROVED", "CONDITIONAL"}

try:
    from huggingface_hub import HfApi, snapshot_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("WARNING: huggingface_hub not installed — HF datasets cannot be downloaded")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_files(dest: Path) -> list[Path]:
    """Original data files, excluding provenance sidecar and HF cache."""
    return [
        f
        for f in dest.rglob("*")
        if f.is_file()
        and f.name != "_provenance.json"
        and ".cache" not in f.parts
        and not f.name.endswith(".incomplete")
    ]


def _write_provenance(dest: Path, dataset_name: str, meta: dict, revision: str | None) -> None:
    files = _data_files(dest)
    largest = max(files, key=lambda f: f.stat().st_size) if files else None
    prov = dest / "_provenance.json"
    prov.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "source": meta.get("source"),
                "source_url": meta.get("source_url"),
                "license": meta.get("license"),
                "license_status": meta.get("license_status"),
                "revision": revision,
                "expected_rows": meta.get("expected_rows"),
                "file": largest.name if largest else None,
                "file_size_bytes": largest.stat().st_size if largest else None,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "note": "original download, immutable, read-only",
                "sha256": {f.name: _sha256(f) for f in files},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def download_hf(repo_id: str, dest: Path, meta: dict) -> None:
    """Download from Hugging Face Hub via snapshot_download."""
    dest.mkdir(parents=True, exist_ok=True)
    revision = None
    try:
        api = HfApi()
        info = api.dataset_info(repo_id, files_metadata=True)
        revision = info.sha
        print(f"  revision: {revision}")
    except Exception as exc:
        print(f"  WARNING: could not read dataset info: {exc}")
        revision = "unknown"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[
            "*.json",
            "*.jsonl",
            "*.parquet",
            "*.txt",
            "*.csv",
            "*.tsv",
            "*.md",
            "README*",
            "LICENSE*",
            "*.yaml",
            "*.yml",
        ],
    )
    _write_provenance(dest, repo_id, meta, revision)
    files = _data_files(dest)
    for f in files:
        os.chmod(f, stat.S_IREAD)
    print(f"  Downloaded {len(files)} file(s) to {dest}")


def download_github_raw(repo_id: str, dest: Path, meta: dict) -> None:
    """Download raw files from GitHub repository."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    org, name = repo_id.split("/")
    # Try to fetch the repository file list via GitHub API
    api_url = f"https://api.github.com/repos/{org}/{name}/contents"
    try:
        import urllib.request as req

        request = req.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with req.urlopen(request, timeout=30) as resp:
            contents = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ERROR: could not list GitHub contents: {exc}")
        return

    files_downloaded = 0
    for item in contents:
        if item["type"] != "file":
            continue
        name = item["name"]
        # Only download text/data files
        if not any(name.endswith(ext) for ext in (".txt", ".json", ".jsonl", ".csv", ".tsv", ".md", ".LICENSE", "LICENSE")):
            continue
        file_url = item["download_url"]
        dest_file = dest / name
        try:
            with req.urlopen(file_url, timeout=60) as resp, open(dest_file, "wb") as out:
                out.write(resp.read())
            files_downloaded += 1
            print(f"  Downloaded {name} ({resp.length} bytes)")
        except Exception as exc:
            print(f"  ERROR downloading {name}: {exc}")

    # Write provenance
    prov = {
        "dataset": repo_id,
        "source": meta.get("source"),
        "source_url": meta.get("source_url"),
        "license": meta.get("license"),
        "license_status": meta.get("license_status"),
        "revision": "github_main",
        "expected_rows": meta.get("expected_rows"),
        "file": None,
        "file_size_bytes": None,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "note": "GitHub raw download, immutable, read-only",
        "sha256": {f.name: _sha256(f) for f in dest.iterdir() if f.is_file() and f.name != "_provenance.json"},
    }
    (dest / "_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for f in dest.iterdir():
        if f.is_file() and f.name != "_provenance.json":
            os.chmod(f, stat.S_IREAD)
    print(f"  Downloaded {files_downloaded} file(s) to {dest}")


def download_zenodo(record_id: str, dest: Path, meta: dict) -> None:
    """Download files from a Zenodo record."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        import urllib.request as req

        with req.urlopen(api_url, timeout=30) as resp:
            record = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ERROR: could not query Zenodo API: {exc}")
        return

    files = record.get("files", [])
    revision = record.get("metadata", {}).get("version", "unknown")

    files_downloaded = 0
    for file_info in files:
        file_url = file_info["links"]["self"]
        file_name = file_info["key"]
        file_size = file_info.get("size", 0)
        dest_file = dest / file_name
        try:
            print(f"  Downloading {file_name} ({file_size} bytes)...")
            with req.urlopen(file_url, timeout=120) as resp, open(dest_file, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            files_downloaded += 1
        except Exception as exc:
            print(f"  ERROR downloading {file_name}: {exc}")

    prov = {
        "dataset": record_id,
        "source": meta.get("source"),
        "source_url": meta.get("source_url"),
        "license": meta.get("license"),
        "license_status": meta.get("license_status"),
        "revision": revision,
        "expected_rows": meta.get("expected_rows"),
        "file": None,
        "file_size_bytes": None,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "note": "Zenodo download, immutable, read-only",
        "sha256": {f.name: _sha256(f) for f in dest.iterdir() if f.is_file() and f.name != "_provenance.json"},
    }
    (dest / "_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for f in dest.iterdir():
        if f.is_file() and f.name != "_provenance.json":
            os.chmod(f, stat.S_IREAD)
    print(f"  Downloaded {files_downloaded} file(s) to {dest}")


def main() -> int:
    print("=" * 70)
    print("GROUP A DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Target directory: {RAW_ROOT}")
    print()

    datasets = CONFIG["datasets"]
    downloaded = 0
    skipped = 0

    for name, meta in datasets.items():
        status = meta.get("license_status", "UNKNOWN")
        if status not in DOWNLOADABLE_STATUSES:
            print(f"SKIP {name}: license_status={status} (not downloadable)")
            skipped += 1
            continue

        dest = RAW_ROOT / name
        print(f"\n=== {name} ===")
        print(f"  Source: {meta.get('source_url')}")
        print(f"  License: {meta.get('license')} ({status})")

        method = meta.get("download_method", "")
        try:
            if method == "huggingface_snapshot":
                if not HAS_HF:
                    print("  SKIP: huggingface_hub not installed")
                    skipped += 1
                    continue
                download_hf(meta["source"], dest, meta)
            elif method == "github_raw":
                download_github_raw(meta["source"], dest, meta)
            elif method == "zenodo":
                record_id = meta["source_url"].split("/")[-1]
                download_zenodo(record_id, dest, meta)
            else:
                print(f"  ERROR: unknown download method '{method}'")
                skipped += 1
                continue
            downloaded += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            skipped += 1

    print(f"\n{'=' * 70}")
    print(f"Downloaded: {downloaded}, Skipped: {skipped}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
