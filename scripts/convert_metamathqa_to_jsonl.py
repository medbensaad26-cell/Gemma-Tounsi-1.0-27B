import json
from pathlib import Path

src = Path("data/raw/meta-math__MetaMathQA/MetaMathQA-395K.json")
dst = src.with_suffix(".jsonl")

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(dst, "w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(data)} rows to {dst}")