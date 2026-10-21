import json
from pathlib import Path

train_path = Path("data/processed/franco_tunisian/train.jsonl")
train_records = []
with train_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            train_records.append(json.loads(line))
print(f"Train: {len(train_records)} records")
print(f"First id: {train_records[0]['id']}, last id: {train_records[-1]['id']}")
sources = {}
for r in train_records:
    s = r["source"]
    sources[s] = sources.get(s, 0) + 1
print(f"Train sources: {dict(sorted(sources.items()))}")

holdout_path = Path("data/processed/franco_tunisian/holdout.jsonl")
holdout_records = []
with holdout_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            holdout_records.append(json.loads(line))
print(f"Holdout: {len(holdout_records)} records")
sources2 = {}
for r in holdout_records:
    s = r["source"]
    sources2[s] = sources2.get(s, 0) + 1
print(f"Holdout sources: {dict(sorted(sources2.items()))}")

train_ids = {r["id"] for r in train_records}
holdout_ids = {r["id"] for r in holdout_records}
overlap = train_ids & holdout_ids
print(f"Train/holdout overlap: {len(overlap)}")

# Verify all records have code_switching=True
cs_false = [r for r in train_records + holdout_records if not r.get("code_switching", False)]
print(f"Records with code_switching=False: {len(cs_false)}")

manifest_path = Path("data/manifests/franco_tunisian/selection.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"\nManifest:")
print(json.dumps(manifest, indent=2))
