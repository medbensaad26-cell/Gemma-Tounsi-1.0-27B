"""Check state of HuggingFace snapshot downloads in data/raw."""
import json
from pathlib import Path

RAW = Path("data/raw")

for ds_dir in sorted(RAW.iterdir()):
    if not ds_dir.is_dir() or ds_dir.name == "__pycache__":
        continue

    print(f"\n=== {ds_dir.name} ===")

    # Count completed (non-cache) data files
    data_files = [
        p
        for p in ds_dir.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and p.name != ".gitkeep"
        and not p.name.endswith(".incomplete")
    ]
    total_bytes = sum(p.stat().st_size for p in data_files)
    print(f"  Completed data files: {len(data_files)}  ({total_bytes/1024/1024:.2f} MB)")

    # Read metadata from HF cache
    cache_dir = ds_dir / ".cache" / "huggingface" / "download"
    if cache_dir.exists():
        for meta in cache_dir.rglob("*.metadata"):
            try:
                data = json.loads(meta.read_text())
                expected = data.get("size", 0)
                etag = data.get("etag", "?")
                filename = data.get("key", meta.stem)[:60]
                print(f"  PENDING: {filename}  expected={expected/1024/1024:.1f} MB  etag={etag[:12]}")
            except Exception as exc:
                print(f"  ERROR reading {meta}: {exc}")
    else:
        print("  No HF cache found")