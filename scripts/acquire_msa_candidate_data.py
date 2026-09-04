"""Acquire MSA (Modern Standard Arabic) candidate data for the msa_formal slice.

Downloads CIDAR and the Arabic QA Dataset - SIGIR 2024 into data/raw/ as
original, untouched files (immutable once fetched). Records revision data
for later manifest creation, and marks files read-only on disk so they
can't be accidentally modified after acquisition.

This script is SEPARATE from acquire_candidate_data.py (retention data):
it never downloads, touches, or rewrites the English retention pools
(MetaMathQA, Code-Feedback, SlimOrca).

Datasets:
  - arbml/CIDAR                          (3.6 MB, Parquet)
  - bobez999/arabic-qa-dataset-sigir2024 (10.3 MB, JSONL)
"""
import json
import os, stat
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RAW_ROOT = Path("data/raw")
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# Arabic/MSA candidate pools for the msa_formal slice (see configs/data/msa.yaml).
# NOT retention data: retention is English capability preservation and lives in
# data/manifests/retention.yaml + scripts/acquire_candidate_data.py.
DATASETS = [
    "arbml/CIDAR",
    "bobez999/arabic-qa-dataset-sigir2024",
]
DATASET_META = {
    "arbml/CIDAR": {
        "dataset": "CIDAR",
        "license": "Apache-2.0",
        "row_count": 10000,
    },
    "bobez999/arabic-qa-dataset-sigir2024": {
        "dataset": "Arabic QA Dataset - SIGIR 2024",
        "license": "MIT",
        "row_count": 10000,
    },
}

api = HfApi()


def _data_files(dest: Path):
    """Original data files, excluding the provenance sidecar and HF cache."""
    return [
        f
        for f in dest.rglob("*")
        if f.is_file()
        and f.name != "_provenance.json"
        and ".cache" not in f.parts
        and not f.name.endswith(".incomplete")
    ]


def main() -> None:
    for repo_id in DATASETS:
        org, name = repo_id.split("/")
        dest = RAW_ROOT / f"{org}__{name}"
        dest.mkdir(parents=True, exist_ok=True)

        print(f"=== Downloading {repo_id} -> {dest} ===")

        # Record revision before download for provenance.
        try:
            info = api.dataset_info(repo_id, files_metadata=True)
            revision = info.sha
            print(f"    revision: {revision}")
        except Exception as exc:
            print(f"    WARNING: could not read dataset info: {exc}")
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
                "*.md",
                "README*",
                "LICENSE*",
                "*.yaml",
                "*.yml",
            ],
        )

        files = _data_files(dest)
        for f in files:
            os.chmod(f, stat.S_IREAD)

        # Provenance sidecar (the manifest lives in data/manifests/msa.yaml).
        meta = DATASET_META.get(repo_id, {})
        largest = max(files, key=lambda f: f.stat().st_size) if files else None
        prov = dest / "_provenance.json"
        prov.write_text(
            json.dumps(
                {
                    "source": repo_id,
                    "dataset": meta.get("dataset"),
                    "revision": revision,
                    "license": meta.get("license", "unknown"),
                    "row_count": meta.get("row_count"),
                    "file": largest.name if largest else None,
                    "file_size_bytes": largest.stat().st_size if largest else None,
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                    "note": "original download, immutable, read-only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"=== Done {repo_id} ===")

    print("ALL MSA CANDIDATE DATA ACQUIRED")


if __name__ == "__main__":
    main()