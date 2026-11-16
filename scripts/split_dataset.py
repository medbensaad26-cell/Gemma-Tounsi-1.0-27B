import json
import hashlib
import os
import random
from collections import Counter, defaultdict

SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

SRC = "data/train.jsonl"
OUT_DIR = "data/splits"

os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Read all records (preserve exact lines) ---
records = []
with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        records.append(line)

total = len(records)
print(f"Original examples: {total}")

# --- 2. Parse metadata for stratification ---
parsed = []
for line in records:
    obj = json.loads(line)
    parsed.append({
        "line": line,
        "language": obj.get("language", "unknown"),
        "category": obj.get("category", "unknown"),
    })

# --- 3. Detect exact duplicates by line hash ---
line_to_indices = defaultdict(list)
for idx, rec in enumerate(parsed):
    h = hashlib.sha256(rec["line"].encode("utf-8")).hexdigest()
    line_to_indices[h].append(idx)

dupe_groups = {h: idxs for h, idxs in line_to_indices.items() if len(idxs) > 1}
print(f"Exact duplicate line groups: {len(dupe_groups)}")
print(f"Records involved in duplicates: {sum(len(v) for v in dupe_groups.values())}")

# --- 4. Stratified split by language ---
random.seed(42)
lang_groups = defaultdict(list)
for idx, rec in enumerate(parsed):
    lang_groups[rec["language"]].append(idx)

train_indices = []
val_indices = []
test_indices = []

lang_split_report = {}

for lang, idxs in sorted(lang_groups.items()):
    random.shuffle(idxs)
    n = len(idxs)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val
    # Adjust if rounding pushed test negative
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    train_indices.extend(idxs[:n_train])
    val_indices.extend(idxs[n_train:n_train + n_val])
    test_indices.extend(idxs[n_train + n_val:])

    lang_split_report[lang] = {
        "total": n,
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

train_indices.sort()
val_indices.sort()
test_indices.sort()

# --- 5. Verify no overlaps and full coverage ---
train_set = set(train_indices)
val_set = set(val_indices)
test_set = set(test_indices)

assert len(train_set & val_set) == 0, "train/val overlap"
assert len(train_set & test_set) == 0, "train/test overlap"
assert len(val_set & test_set) == 0, "val/test overlap"
assert len(train_set | val_set | test_set) == total, "missing examples"

n_train = len(train_indices)
n_val = len(val_indices)
n_test = len(test_indices)

print(f"\nSplit counts:")
print(f"  Train: {n_train} ({n_train/total*100:.2f}%)")
print(f"  Val:   {n_val} ({n_val/total*100:.2f}%)")
print(f"  Test:  {n_test} ({n_test/total*100:.2f}%)")
print(f"  Total: {n_train + n_val + n_test}")

# --- 6. Language distribution report ---
print(f"\nLanguage distribution:")
print(f"{'language':<15} {'total':>8} {'train':>8} {'val':>7} {'test':>7} {'train%':>8} {'val%':>7} {'test%':>7}")
for lang in sorted(lang_split_report):
    r = lang_split_report[lang]
    print(f"{lang:<15} {r['total']:>8} {r['train']:>8} {r['val']:>7} {r['test']:>7} "
          f"{r['train']/r['total']*100:>7.2f}% {r['val']/r['total']*100:>6.2f}% {r['test']/r['total']*100:>6.2f}%")

# --- 7. Write output files ---
def write_split(name, indices):
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(parsed[idx]["line"] + "\n")
    return path

write_split("train", train_indices)
write_split("validation", val_indices)
write_split("test", test_indices)

print(f"\nFiles written to {OUT_DIR}/")

# --- 8. Verify no modified records ---
def verify_file(path, expected_indices):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            # verify the line matches the original exactly
            assert line == parsed[expected_indices[count]]["line"], f"Record modified at {count}"
            count += 1
    assert count == len(expected_indices), f"Count mismatch in {path}"
    print(f"  Verified {path}: {count} records, all unmodified")

verify_file(os.path.join(OUT_DIR, "train.jsonl"), train_indices)
verify_file(os.path.join(OUT_DIR, "validation.jsonl"), val_indices)
verify_file(os.path.join(OUT_DIR, "test.jsonl"), test_indices)

# --- 9. Duplicate check across splits ---
train_lines = set()
with open(os.path.join(OUT_DIR, "train.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            train_lines.add(line)

val_lines = set()
with open(os.path.join(OUT_DIR, "validation.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            val_lines.add(line)

test_lines = set()
with open(os.path.join(OUT_DIR, "test.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            test_lines.add(line)

train_val_overlap = train_lines & val_lines
train_test_overlap = train_lines & test_lines
val_test_overlap = val_lines & test_lines

print(f"\nCross-split duplicate check:")
print(f"  train/val exact duplicates: {len(train_val_overlap)}")
print(f"  train/test exact duplicates: {len(train_test_overlap)}")
print(f"  val/test exact duplicates: {len(val_test_overlap)}")

# --- 10. Write split report ---
report = {
    "seed": SEED,
    "source_file": SRC,
    "total_examples": total,
    "splits": {
        "train": {"count": n_train, "percentage": round(n_train / total * 100, 4), "file": "data/splits/train.jsonl"},
        "validation": {"count": n_val, "percentage": round(n_val / total * 100, 4), "file": "data/splits/validation.jsonl"},
        "test": {"count": n_test, "percentage": round(n_test / total * 100, 4), "file": "data/splits/test.jsonl"},
    },
    "language_distribution": lang_split_report,
    "exact_duplicate_groups": len(dupe_groups),
    "records_in_duplicates": sum(len(v) for v in dupe_groups.values()),
    "cross_split_duplicates": {
        "train_val": len(train_val_overlap),
        "train_test": len(train_test_overlap),
        "val_test": len(val_test_overlap),
    },
    "verification": {
        "no_overlaps": len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0,
        "full_coverage": len(train_set | val_set | test_set) == total,
        "records_unmodified": True,
    },
}

with open(os.path.join(OUT_DIR, "split_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nReport written to {os.path.join(OUT_DIR, 'split_report.json')}")
print("Done.")
