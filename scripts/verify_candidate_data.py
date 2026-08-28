"""Verify the acquired candidate data deliverables (steps 1-5)."""
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

errors = []

# 1. Validate manifest YAML
manifest_path = ROOT / "data" / "manifests" / "retention.yaml"
try:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    print("Manifest valid YAML.")
    print(f"  version: {manifest['version']}")
    print(f"  name: {manifest['name']}")
    print("  sources:")
    for s in manifest["sources"]:
        print(
            f"    - {s['id']}: {s['location']} @ {s['revision'][:8]} "
            f"| {s['license']} | {s['row_count']} rows"
        )
except Exception as e:
    errors.append(f"Manifest YAML error: {e}")
    print(f"Manifest YAML error: {e}")

# 2. Verify provenance sidecars
print("\nProvenance records:")
for p in sorted((ROOT / "data" / "raw").glob("*/_provenance.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"  {p.relative_to(ROOT)}: "
        f"source={data['source']}, revision={data['revision'][:8]}..., "
        f"license={data['license']}, rows={data['row_count']}"
    )

# 3. Verify original files exist and are read-only
print("\nPreservation check:")
original_files = [
    ROOT / "data" / "raw" / "meta-math__MetaMathQA" / "MetaMathQA-395K.json",
    ROOT / "data" / "raw" / "m-a-p__Code-Feedback" / "Code-Feedback.jsonl",
]
for f in original_files:
    exists = f.exists()
    ro = not os.access(f, os.W_OK) if exists else False
    print(f"  {f.relative_to(ROOT)}: exists={exists}, read_only={ro}")
    if not exists or not ro:
        errors.append(f"Not preserved: {f}")

# 4. Verify raw-data structure matches convention
print("\nRaw-data structure:")
for child in sorted((ROOT / "data" / "raw").iterdir()):
    if child.is_dir():
        print(f"  {child.name}/")

if errors:
    print("\nFAILED checks:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("\nAll candidate-data deliverables verified.")