import json
import sys
import os
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from src.data.schema import load_schema, load_jsonl, validate_record, write_jsonl
from src.data.split import stable_key

# Load .env manually (no python-dotenv dependency)
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

tok = AutoTokenizer.from_pretrained(
    "google/gemma-3-27b-it", token=os.environ.get("HF_TOKEN")
)
schema = load_schema()
rng = random.Random(42)

T_code = 6023674
T_msa = int(1.6 * T_code)
print(f"T_code={T_code}, T_msa={T_msa}")


def count_tokens(messages):
    total = 0
    assistant = 0
    for turn in messages:
        content = str(turn.get("content", ""))
        n = len(tok.encode(content))
        total += n + 1
        if turn.get("role") == "assistant":
            assistant += n
    return total, assistant


def sharegpt_to_messages(conversations):
    """Convert ShareGPT conversations to canonical messages."""
    messages = []
    for turn in conversations:
        role = str(turn.get("from", ""))
        value = str(turn.get("value", ""))
        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": value})
        elif role == "system":
            messages.append({"role": "system", "content": value})
    return messages


def to_canonical(record_id, messages, source, pool_id):
    """Build a canonical record from messages."""
    record = {
        "id": f"{pool_id}-{record_id}",
        "messages": messages,
        "category": "general_instruction",
        "source": source,
        "language": "ar",
        "script": "arabic",
        "quality": {"synthetic": False},
    }
    return record


# --- Pool loaders ---

def load_cidar():
    """CIDAR: parquet with instruction/output columns."""
    import pyarrow.parquet as pq
    p = Path("data/raw/arbml__CIDAR/data/train-00000-of-00001-b2881e1b9f14c3b1.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    for i in range(len(df["instruction"])):
        instr = str(df["instruction"][i] or "")
        output = str(df["output"][i] or "")
        messages = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": output},
        ]
        records.append(to_canonical(str(df.get("index", [i] * len(df["instruction"]))[i]), messages, "cidar", "cidar"))
    return records


def load_arabic_qa():
    """Arabic QA: JSONL with instruction/input/output."""
    p = Path("data/raw/bobez999__arabic-qa-dataset-sigir2024/data/arabic_qa_10k_sample.jsonl")
    records = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instr = str(row.get("instruction", ""))
            input_text = str(row.get("input", ""))
            output = str(row.get("output", ""))
            user_content = instr + ("\n" + input_text if input_text.strip() else "")
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            records.append(to_canonical(i, messages, row.get("source", "arabic_qa"), "arabic_qa"))
    return records


def load_alpaca_arabic_gpt4():
    """Alpaca-Arabic-GPT4: parquet with ShareGPT conversations."""
    import pyarrow.parquet as pq
    p = Path("data/raw/FreedomIntelligence__Alpaca-Arabic-GPT4/data/train-00000-of-00001-07aa61790348d25a.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    for i in range(len(df["id"])):
        conv = df["conversations"][i]
        if isinstance(conv, str):
            conv = json.loads(conv)
        messages = sharegpt_to_messages(conv)
        if len(messages) < 2:
            continue
        records.append(to_canonical(str(i), messages, "alpaca_arabic_gpt4", "alpaca_arabic_gpt4"))
    return records


def load_evol_instruct_arabic():
    """Evol-Instruct-Arabic: materialized alpaca JSONL."""
    p = Path("data/processed/msa/soup/evol_instruct_arabic_alpaca.jsonl")
    records = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instr = str(row.get("instruction", ""))
            input_text = str(row.get("input", ""))
            output = str(row.get("output", ""))
            user_content = instr + ("\n" + input_text if input_text.strip() else "")
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            records.append(to_canonical(i, messages, "evol_instruct_arabic", "evol_instruct_arabic"))
    return records


def load_aya_dataset():
    """Aya Dataset: parquet with inputs/targets, filter Arabic only."""
    import pyarrow.parquet as pq
    p = Path("data/raw/CohereForAI__aya_dataset/data/train-00000-of-00001.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    idx = 0
    for i in range(len(df["language_code"])):
        if str(df["language_code"][i]) != "arb":
            continue
        inputs = str(df["inputs"][i] or "")
        targets = str(df["targets"][i] or "")
        if not inputs.strip() or not targets.strip():
            continue
        messages = [
            {"role": "user", "content": inputs},
            {"role": "assistant", "content": targets},
        ]
        records.append(to_canonical(idx, messages, "aya_dataset_arabic", "aya_dataset_arabic"))
        idx += 1
    return records


# Load all pools
pool_loaders = {
    "cidar": load_cidar,
    "arabic_qa": load_arabic_qa,
    "alpaca_arabic_gpt4": load_alpaca_arabic_gpt4,
    "evol_instruct_arabic": load_evol_instruct_arabic,
    "aya_dataset_arabic": load_aya_dataset,
}

all_raw = {}
for pool_id, loader in pool_loaders.items():
    try:
        recs = loader()
        all_raw[pool_id] = recs
        print(f"{pool_id}: {len(recs)} raw rows loaded")
    except Exception as exc:
        print(f"{pool_id}: ERROR - {exc}")
        all_raw[pool_id] = []

# Validate, dedupe, tokenize
all_records = []
seen_ids = set()
for pool_id, recs in all_raw.items():
    for i, record in enumerate(recs):
        errors = validate_record(record, schema, line=i + 1)
        if errors:
            continue
        if str(record.get("language")) != "ar":
            continue
        rid = record["id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        total, assistant = count_tokens(record["messages"])
        if total > 15500:
            continue
        record["_total"] = total
        record["_assistant"] = assistant
        # Extract input for passage cap
        user_turns = [m["content"] for m in record["messages"] if m.get("role") == "user"]
        record["_input"] = user_turns[-1] if user_turns else ""
        all_records.append(record)

print(f"\nTotal valid canonical records: {len(all_records)}")

# --- Passage cap (max 2 per variation group, SHA-256 of input) ---
passage_groups = defaultdict(list)
for r in all_records:
    inp = r["_input"]
    if inp:
        key = hashlib.sha256(inp.encode("utf-8")).hexdigest()
        passage_groups[key].append(r)
    else:
        passage_groups[f"__noinput__{r['id']}__"].append(r)

capped = []
dropped = 0
for key, group in passage_groups.items():
    group.sort(key=lambda r: r["id"])
    kept = group[:2]
    capped.extend(kept)
    dropped += len(group) - 2 if len(group) > 2 else 0

print(f"After passage cap: {len(capped)} (dropped {dropped})")

# Re-tokenize capped records
for r in capped:
    total, assistant = count_tokens(r["messages"])
    r["_total"] = total
    r["_assistant"] = assistant

capped.sort(key=lambda r: r["id"])

# Source stats
source_counts = defaultdict(int)
source_tokens = defaultdict(int)
for r in capped:
    source_counts[r["source"]] += 1
    source_tokens[r["source"]] += r["_assistant"]
total_records = len(capped)
total_tokens = sum(source_tokens.values())
print(f"Source distribution: {dict(sorted(source_counts.items()))}, total={total_records}")
print(f"Source token distribution: {dict(sorted(source_tokens.items()))}, total={total_tokens}")

# --- Holdout first: 1000 rows, token-proportional across sources ---
holdout_target_rows = 1000
holdout_records = []
holdout_ids = set()

remaining_holdout = holdout_target_rows
source_holdout_alloc = {}
sorted_sources = sorted(source_tokens.keys())
for i, src in enumerate(sorted_sources):
    share = source_tokens[src] / total_tokens
    alloc = int(holdout_target_rows * share)
    if i == len(sorted_sources) - 1:
        alloc = remaining_holdout
    source_holdout_alloc[src] = alloc
    remaining_holdout -= alloc

print(f"Holdout allocation per source: {dict(sorted(source_holdout_alloc.items()))}")

remaining_pool = list(capped)
for src in sorted_sources:
    src_pool = [r for r in remaining_pool if r["source"] == src]
    need = source_holdout_alloc[src]
    need = min(need, len(src_pool))
    if need <= 0:
        continue
    ordered = sorted(src_pool, key=lambda r: stable_key(r, 42))
    selected = ordered[:need]
    holdout_records.extend(selected)
    holdout_ids.update(r["id"] for r in selected)
    remaining_pool = [r for r in remaining_pool if r["id"] not in holdout_ids]

print(f"Holdout selected: {len(holdout_records)} rows")

# --- Train: fill to T_msa from remaining ---
train_records = []
train_assistant = 0
lower = T_msa * 0.98
upper = T_msa * 1.02

ordered = sorted(remaining_pool, key=lambda r: stable_key(r, 42))
for r in ordered:
    if train_assistant >= lower:
        break
    a = r["_assistant"]
    if a <= 0:
        continue
    if train_assistant + a <= upper:
        train_records.append(r)
        train_assistant += a

print(f"Train selected: {len(train_records)} rows, assistant tokens: {train_assistant}")
print(f"Target: {T_msa}, band: [{lower:.0f}, {upper:.0f}]")
print(f"Within tolerance: {lower <= train_assistant <= upper}")

# Clean up temp fields and sort
for r in train_records + holdout_records:
    for k in ("_total", "_assistant", "_input"):
        r.pop(k, None)

train_records.sort(key=lambda r: r["id"])
holdout_records.sort(key=lambda r: r["id"])

# Write outputs
train_out = Path("data/processed/msa/train.jsonl")
holdout_out = Path("data/processed/msa/holdout.jsonl")
train_out.parent.mkdir(parents=True, exist_ok=True)
holdout_out.parent.mkdir(parents=True, exist_ok=True)

write_jsonl(train_out, train_records)
write_jsonl(holdout_out, holdout_records)

print(f"\nWrote {len(train_records)} train records -> {train_out}")
print(f"Wrote {len(holdout_records)} holdout records -> {holdout_out}")

# Manifest
manifest = {
    "slice": "msa_formal",
    "purpose": "formal_register_coverage",
    "seed": 42,
    "t_code": T_code,
    "t_msa": T_msa,
    "token_metric": "assistant_tokens",
    "train": {
        "total": len(train_records),
        "assistant_tokens": sum(count_tokens(r["messages"])[1] for r in train_records),
    },
    "holdout": {
        "total": len(holdout_records),
        "assistant_tokens": sum(count_tokens(r["messages"])[1] for r in holdout_records),
    },
    "sources": sorted(source_counts.keys()),
    "deterministic": True,
}
manifest_out = Path("data/manifests/msa/selection.json")
manifest_out.parent.mkdir(parents=True, exist_ok=True)
manifest_out.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"Wrote manifest -> {manifest_out}")
