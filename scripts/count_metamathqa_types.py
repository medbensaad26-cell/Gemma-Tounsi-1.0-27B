import json
from collections import Counter

counts = Counter()
with open("data/raw/meta-math__MetaMathQA/MetaMathQA-395K.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        counts[row["type"]] += 1

for t, c in counts.most_common():
    print(f"{t}: {c}")