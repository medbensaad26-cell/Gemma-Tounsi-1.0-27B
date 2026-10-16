"""Acquire French/Tunisian code-switching candidate data for the franco_tunisian slice.

Downloads the five franco_tunisian candidate pools into data/raw/ as
original, untouched files (immutable once fetched). Records revision data
for later manifest creation, and marks files read-only on disk so they
can't be accidentally modified after acquisition.

This script is SEPARATE from acquire_msa_candidate_data.py (MSA data) and
acquire_candidate_data.py (retention data): it never downloads, touches,
or rewrites those pools.

Datasets:
  - Farah21/reddit-tunisia-qa                      (Reddit QA, dialect labels)
  - Syrinesmati/tunisian-question-response-dataset   (~31K Tunisian QA pairs)
  - linagora/Tunisian_Derja_Dataset                  (TunSwitchCodeSwitching config)
  - hamzabouajila/tunisian-derja-unified-raw-corpus  (~802K Derja examples)
  - linagora/linto-dataset-text-ar-tn                (~4.5M Tunisian text lines)
"""
import json
import os, stat
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RAW_ROOT = Path("data/raw")
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# French/Tunisian code-switching candidate pools for the franco_tunisian slice
# (see data/manifests/franco_tunisian.yaml).
# NOT retention data: retention is English capability preservation and lives in
# data/manifests/retention.yaml + scripts/acquire_candidate_data.py.
# NOT MSA data: MSA is formal-register coverage and lives in
# data/manifests/msa.yaml + scripts/acquire_msa_candidate_data.py.
DATASETS = [
    "Farah21/reddit-tunisia-qa",
    "Syrinesmati/tunisian-question-response-dataset",
    "linagora/Tunisian_Derja_Dataset",
    "hamzabouajila/tunisian-derja-unified-raw-corpus",
    "linagora/linto-dataset-text-ar-tn",
]
DATASET_META = {
    "Farah21/reddit-tunisia-qa": {
        "dataset": "Reddit Tunisia Tech QA",
        "license": "TBD",  # verify SPDX id before use
        "row_count": None,  # ~4,541–5,500 QA pairs reported
    },
    "Syrinesmati/tunisian-question-response-dataset": {
        "dataset": "Tunisian Question-Response Dataset",
        "license": "CC-BY-4.0",
        "row_count": None,  # ~31,000 QA pairs reported
    },
    "linagora/Tunisian_Derja_Dataset": {
        "dataset": "Tunisian Derja Dataset (TunSwitchCodeSwitching config)",
        "license": "CC-BY-SA-4.0",  # share-alike — legal review required
        "row_count": None,  # TunSwitchCodeSwitching: 394,160 lines reported
    },
    "hamzabouajila/tunisian-derja-unified-raw-corpus": {
        "dataset": "Tunisian Derja Unified Raw Corpus",
        "license": "CC-BY-SA-4.0",  # share-alike — legal review required
        "row_count": None,  # ~802,659 examples reported
    },
    "linagora/linto-dataset-text-ar-tn": {
        "dataset": "LinTO Textual Dataset in Arabic Tunisian",
        "license": "CC-BY-4.0",
        "row_count": None,  # ~4.5M lines reported
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

        # Provenance sidecar (the manifest lives in data/manifests/franco_tunisian.yaml).
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

    print("ALL FRANCO-TUNISIAN CANDIDATE DATA ACQUIRED")


if __name__ == "__main__":
    main()
