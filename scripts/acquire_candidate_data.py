"""Step 1: Acquire candidate data for the retention slice.

Downloads MetaMathQA and Code-Feedback into data/raw/ as original,
untouched files (immutable once fetched). Records revision data for
later manifest creation.

Datasets:
  - meta-math/MetaMathQA   (377 MB, JSON)
  - m-a-p/Code-Feedback    (394 MB, JSONL)
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RAW_ROOT = Path("data/raw")
RAW_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "meta-math/MetaMathQA",
    "m-a-p/Code-Feedback",
    "Open-Orca/SlimOrca",
]

api = HfApi()


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

        # Small provenance sidecar (the manifest is task 5).
        prov = dest / "_provenance.json"
        prov.write_text(
            json.dumps(
                {
                    "source": repo_id,
                    "revision": revision,
                    "note": "original download, immutable",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"=== Done {repo_id} ===")

    print("ALL CANDIDATE DATA ACQUIRED")


if __name__ == "__main__":
    main()