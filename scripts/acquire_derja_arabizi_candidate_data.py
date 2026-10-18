"""Acquire Derja/Arabizi candidate data declared in data/manifests/train.yaml.

Downloads ONLY what train.yaml declares, at the pinned revisions, into
data/raw/ as original files (read-only once fetched). Excluded/quarantined
sources (see docs/DATA.md 2.1) are never downloaded. TunSwitch subsets belong
to franco_tunisian.yaml and are not fetched here.

Sources:
  - linagora/linto-dataset-text-ar-tn  (approved subsets only)
  - Farah21/Tunisia_Tech               (train split only; test split reserved)
  - CTAB, Zenodo record 4781769        (md5-verified)

Idempotent: a source whose _provenance.json already exists is skipped.
"""
import hashlib
import json
import os
import stat
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

RAW_ROOT = Path("data/raw")
RAW_ROOT.mkdir(parents=True, exist_ok=True)

LINTO_SUBSETS = [
    "Derja_tunsi", "HkayetTounsia", "KisatiAna", "TunBERT", "TuDiCOI",
    "HkayetErwi", "TunHistoires", "Chakhabitt", "TRCD", "TSAC", "Tweet_TN",
]

HF_SOURCES = [
    {
        "id": "linto-text-ar-tn-derja",
        "repo_id": "linagora/linto-dataset-text-ar-tn",
        "revision": "25dbb0aff3e5d4db1db5a69f7a4077280b442f4b",
        "dataset": "LinTO Textual Dataset in Arabic Tunisian (approved subsets)",
        "license": "Apache-2.0",
        "allow_patterns": ["README.md"] + [f"data/{s}/*" for s in LINTO_SUBSETS],
        "note": "PROVISIONAL license (card says Apache-2.0; same subsets appear in a "
                "CC-BY-SA repo). Approved subsets only. Original download, read-only.",
    },
    {
        "id": "farah21-tunisia-tech-arabizi",
        "repo_id": "Farah21/Tunisia_Tech",
        "revision": "be5ca6c51c2c6c7557da73672be030eeb64c68be",
        "dataset": "Reddit Tunisia Tech QA (train split)",
        "license": "CC-BY-4.0",
        "allow_patterns": ["README.md", "data/train-*"],
        "note": "Train split only; only dialect == tunisian_arabizi rows may be used. "
                "Original download, read-only.",
    },
]

CTAB_RECORD = "https://zenodo.org/records/4781769"
CTAB_FILES = {
    "CTAB-SAMPLE0001.txt": "dc42ee73cf7e5b125188785d0e26306f",
    "CTAB-SAMPLE0002.txt": "28a1c0d0bae0ccb61427446748e714bb",
    "CTAB-SAMPLE0003.txt": "ea1a3a417892d9935db8ba70d0fcf03e",
    "CTAB-SAMPLE0004.txt": "db569c9391eadb4b5f796facdf853802",
    "CTAB-SAMPLE0005.txt": "3b8f8919a844931bee9070d12f5de8a7",
    "CTAB-SAMPLE0006.txt": "b01d16e95e674e3ded630e92589cb33e",
    "CTAB-SAMPLE0007.txt": "6f3a59e99c2330a45f520569c38bb18d",
    "CTAB-SAMPLE0008.txt": "17e4180b6af6965217a5a0638353d0a8",
    "CTAB-SAMPLE0009.txt": "2d51ebbeeaa5431c36765f76e438f3c8",
    "CTAB-SAMPLE0010.txt": "7424d805e2502b222805c03bf956a1e5",
    "CTAB-SAMPLE0011.txt": "89c64af867793fe690531722348edc1d",
    "CTAB-SAMPLE0012.txt": "263ac4be6e096a3078e08ae8a5c4fb3f",
    "CTAB-SAMPLE0013.txt": "a4f6b25fe7a97df601750f538046bb4c",
    "CTAB.pdf": "5dee23175faaaf83fffea0d4bcda78de",
}


def _data_files(dest: Path):
    return [
        f for f in dest.rglob("*")
        if f.is_file()
        and f.name != "_provenance.json"
        and ".cache" not in f.parts
        and not f.name.endswith((".incomplete", ".lock"))
    ]


def _count_rows(f: Path):
    if f.suffix == ".parquet":
        return pq.ParquetFile(f).metadata.num_rows
    if f.suffix == ".txt":
        with open(f, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    return None


def _finalize(dest: Path, meta: dict) -> None:
    files = _data_files(dest)
    per_file = [
        {"name": str(f.relative_to(dest)).replace("\\", "/"),
         "rows": _count_rows(f), "bytes": f.stat().st_size}
        for f in sorted(files)
    ]
    total_rows = sum(p["rows"] for p in per_file if p["rows"] is not None)
    largest = max(files, key=lambda f: f.stat().st_size)
    for f in files:
        os.chmod(f, stat.S_IREAD)
    (dest / "_provenance.json").write_text(
        json.dumps(
            {
                "source": meta["source"],
                "id": meta["id"],
                "dataset": meta["dataset"],
                "revision": meta["revision"],
                "license": meta["license"],
                "row_count": total_rows,
                "file": largest.name,
                "file_size_bytes": largest.stat().st_size,
                "files": per_file,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "note": meta["note"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"    files={len(files)} rows={total_rows}")


def acquire_hf(src: dict) -> None:
    org, name = src["repo_id"].split("/")
    dest = RAW_ROOT / f"{org}__{name}"
    if (dest / "_provenance.json").exists():
        print(f"=== SKIP {src['repo_id']} (already acquired) ===")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"=== Downloading {src['repo_id']} @ {src['revision'][:8]} -> {dest} ===")
    snapshot_download(
        repo_id=src["repo_id"], repo_type="dataset", revision=src["revision"],
        local_dir=str(dest), allow_patterns=src["allow_patterns"],
    )
    _finalize(dest, {**src, "source": src["repo_id"]})


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire_ctab() -> None:
    dest = RAW_ROOT / "zenodo__ctab-4781769"
    if (dest / "_provenance.json").exists():
        print("=== SKIP CTAB (already acquired) ===")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"=== Downloading CTAB (Zenodo 4781769 v4) -> {dest} ===")
    for fname, md5 in CTAB_FILES.items():
        target = dest / fname
        tmp = dest / (fname + ".incomplete")
        url = f"{CTAB_RECORD}/files/{fname}?download=1"
        req = urllib.request.Request(url, headers={"User-Agent": "gemma-tounsi-acquire/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            out.write(resp.read())
        got = _md5(tmp)
        if got != md5:
            tmp.unlink()
            raise SystemExit(f"MD5 MISMATCH for {fname}: expected {md5}, got {got}")
        tmp.replace(target)
        print(f"    ok {fname}")
    _finalize(dest, {
        "id": "ctab-arabizi",
        "source": CTAB_RECORD,
        "dataset": "CTAB: Corpus of Tunisian Arabizi (v4, 2021-05-11)",
        "revision": "zenodo-4781769-v4-2021-05-11",
        "license": "CC-BY-4.0",
        "note": "13 raw Arabizi message files (one message per line) + CTAB.pdf. "
                "md5 verified against Zenodo. Facebook-scraped: PII filtering required. "
                "Original download, read-only.",
    })


def main() -> None:
    for src in HF_SOURCES:
        acquire_hf(src)
    acquire_ctab()
    print("DERJA/ARABIZI CANDIDATE DATA ACQUIRED")


if __name__ == "__main__":
    main()