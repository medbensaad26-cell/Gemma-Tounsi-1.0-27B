"""franco_tunisian slice: normalization + token-budget selection.

Implements docs/data/tunisian_code_switching.md Sections 6.3-6.4.
Processes pools incrementally to handle large datasets efficiently.
"""
import json
import sys
import os
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from src.data.schema import load_schema, validate_record, write_jsonl
from src.data.split import stable_key

# --- Load .env ---
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
T_franco = int(2.4 * T_code)
print(f"T_code={T_code}, T_franco={T_franco}")

MAX_TOKENS = 15500
SEED = 42


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


def messages_from_text(text, source_id):
    """Convert raw text into canonical messages (user -> assistant)."""
    text = str(text).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace("।", ".").replace("؟", "?").split(". ") if p.strip()]
    if not parts:
        parts = [text]
    messages = []
    for i, part in enumerate(parts):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": part})
    if messages and messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "فهمت."})
    if len(messages) < 2:
        messages.append({"role": "assistant", "content": "فهمت."})
    return messages


# French + Arabic/Tunisian markers for code-switching detection
FR_MARKERS = {"le", "la", "les", "un", "une", "des", "et", "ou", "est", "sont",
              "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce",
              "cet", "cette", "ces", "mon", "ma", "mes", "son", "sa", "ses",
              "à", "de", "du", "au", "aux", "en", "pour", "sur", "dans", "par",
              "avec", "sans", "sous", "chez", "depuis", "pendant", "avant", "après",
              "mais", "donc", "or", "ni", "car", "que", "qui", "quel", "quelle",
              "quels", "quelles", "quoi", "comment", "pourquoi", "quand", "où",
              "est-ce", "qu'est", "l'", "d'", "j'", "n'", "t'", "s'", "m'",
              "dinars", "dinar", "est", "et", "ou", "mais", "donc", "or", "ni", "car"}
AR_MARKERS = {"ال", "و", "ب", "ك", "ل", "ف", "ن", "م", "ه", "ي", "أ", "إ", "آ", "ة",
              "كيفاش", "نجم", "كيفية", "إلي", "عندي", "عندك", "عندو", "عندنا",
              "باش", "قداش", "واش", "ماش", "شلون", "وين", "شكون", "شنوة", "أش",
              "فهمت", "تونس", "تونسي", "تونسية", "دارجا", "دريجا", "دارجة",
              "فرنسي", "فرنسية", "فرنسا"}


def detect_code_switching(messages, dialect=None):
    """Heuristic: French + Arabic/Tunisian mix in the same example."""
    if dialect:
        d = str(dialect).lower()
        if d in ("mixed", "french", "tunisian_arabizi"):
            return True
        if d in ("arabic", "tunisian_arabic"):
            return False

    all_text = " ".join(str(m.get("content", "")) for m in messages).lower()
    has_fr = any(m in all_text for m in FR_MARKERS)
    has_ar = any(m in all_text for m in AR_MARKERS)
    return has_fr and has_ar


# ---------------------------------------------------------------------------
# Pool loaders (return list of canonical records, untokenized)
# ---------------------------------------------------------------------------

def load_farah21():
    import pyarrow.parquet as pq
    p = Path("data/raw/Farah21__reddit-tunisia-qa/data/train-00000-of-00001.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    for i in range(len(df["id"])):
        instr = str(df["instruction"][i] or "")
        input_text = str(df["input"][i] or "")
        output = str(df["output"][i] or "")
        dialect = df["dialect"][i] if i < len(df["dialect"]) else ""
        user_content = instr + ("\n" + input_text if input_text.strip() else "")
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        cs = detect_code_switching(messages, dialect=dialect)
        lang_map = {"mixed": "mixed", "french": "fr", "tunisian_arabizi": "arabizi",
                    "arabic": "ar", "tunisian_arabic": "ar"}
        lang = lang_map.get(str(dialect).lower(), "mixed")
        record = {
            "id": f"farah21-{i}",
            "messages": messages,
            "category": "general_instruction",
            "source": "farah21_reddit_tunisia_qa",
            "language": lang,
            "script": "mixed",
            "code_switching": cs,
            "quality": {"synthetic": False},
        }
        records.append(record)
    return records


def load_syrinesmati():
    import pyarrow.parquet as pq
    p = Path("data/raw/Syrinesmati__tunisian-question-response-dataset/data/train-00000-of-00001.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    for i in range(len(df["instruction"])):
        instr = str(df["instruction"][i] or "")
        response = str(df["response"][i] or "")
        category = str(df["category"][i] or "general_instruction")
        cat_map = {"student_life": "general_instruction", "education": "general_instruction",
                   "work": "general_instruction", "health": "knowledge_qa",
                   "technology": "general_instruction", "culture": "general_instruction",
                   "religion": "knowledge_qa", "politics": "knowledge_qa",
                   "sports": "general_instruction", "food": "general_instruction",
                   "travel": "general_instruction", "family": "general_instruction"}
        cat = cat_map.get(category.lower(), "general_instruction")
        messages = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": response},
        ]
        cs = detect_code_switching(messages)
        record = {
            "id": f"syrinesmati-{i}",
            "messages": messages,
            "category": cat,
            "source": "syrinesmati_tunisian_question_response",
            "language": "mixed",
            "script": "mixed",
            "code_switching": cs,
            "quality": {"synthetic": False},
        }
        records.append(record)
    return records


def load_linagora_tunswitch():
    import pyarrow.parquet as pq
    p = Path("data/raw/linagora__Tunisian_Derja_Dataset/TunSwitchCodeSwitching/train-00000-of-00001.parquet")
    t = pq.read_table(p)
    df = t.to_pydict()
    records = []
    for i in range(len(df["text"])):
        text = str(df["text"][i] or "")
        if not text.strip():
            continue
        messages = messages_from_text(text, f"linagora_tunswitch-{i}")
        if messages is None:
            continue
        cs = True  # TunSwitchCodeSwitching is the dedicated CS subset
        record = {
            "id": f"linagora_tunswitch-{i}",
            "messages": messages,
            "category": "general_instruction",
            "source": "linagora_tunisian_derja_dataset_tunswitch",
            "language": "mixed",
            "script": "mixed",
            "code_switching": cs,
            "quality": {"synthetic": False},
        }
        records.append(record)
    return records


def load_hamzabouajila():
    import pyarrow.parquet as pq
    records = []
    dir_path = Path("data/raw/hamzabouajila__tunisian-derja-unified-raw-corpus")
    idx = 0
    for f in sorted(dir_path.rglob("*.parquet")):
        t = pq.read_table(f)
        df = t.to_pydict()
        text_col = None
        for col in df:
            if col.lower() in ("text", "content", "body"):
                text_col = col
                break
        if text_col is None:
            continue
        for i in range(len(df[text_col])):
            text = str(df[text_col][i] or "")
            if not text.strip():
                continue
            messages = messages_from_text(text, f"hamzabouajila-{idx}")
            if messages is None:
                continue
            cs = detect_code_switching(messages)
            record = {
                "id": f"hamzabouajila-{idx}",
                "messages": messages,
                "category": "general_instruction",
                "source": "hamzabouajila_tunisian_derja_unified",
                "language": "mixed",
                "script": "mixed",
                "code_switching": cs,
                "quality": {"synthetic": False},
            }
            records.append(record)
            idx += 1
    return records


def load_linto():
    import pyarrow.parquet as pq
    records = []
    dir_path = Path("data/raw/linagora__linto-dataset-text-ar-tn")
    idx = 0
    for f in sorted(dir_path.rglob("*.parquet")):
        t = pq.read_table(f)
        df = t.to_pydict()
        text_col = None
        for col in df:
            if col.lower() in ("text", "content", "body", "sentence", "utterance"):
                text_col = col
                break
        if text_col is None:
            continue
        for i in range(len(df[text_col])):
            text = str(df[text_col][i] or "")
            if not text.strip():
                continue
            messages = messages_from_text(text, f"linto-{idx}")
            if messages is None:
                continue
            cs = detect_code_switching(messages)
            record = {
                "id": f"linto-{idx}",
                "messages": messages,
                "category": "general_instruction",
                "source": "linagora_linto_text_ar_tn",
                "language": "mixed",
                "script": "mixed",
                "code_switching": cs,
                "quality": {"synthetic": False},
            }
            records.append(record)
            idx += 1
    return records


# ---------------------------------------------------------------------------
# Main: process pools incrementally
# ---------------------------------------------------------------------------

pool_loaders = {
    "farah21": load_farah21,
    "syrinesmati": load_syrinesmati,
    "linagora_tunswitch": load_linagora_tunswitch,
    "hamzabouajila": load_hamzabouajila,
    "linto": load_linto,
}

# Process pools one at a time, validate + filter code_switching, tokenize
all_records = []
seen_ids = set()

for pool_id, loader in pool_loaders.items():
    print(f"\nLoading {pool_id}...")
    try:
        recs = loader()
        print(f"  {len(recs)} raw rows loaded")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        continue

    valid = 0
    cs_filtered = 0
    for i, record in enumerate(recs):
        errors = validate_record(record, schema, line=i + 1)
        if errors:
            continue
        if str(record.get("language")) not in ("fr", "ar", "arabizi", "mixed"):
            continue
        if not record.get("code_switching", False):
            cs_filtered += 1
            continue
        rid = record["id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        total, assistant = count_tokens(record["messages"])
        if total > MAX_TOKENS:
            continue
        record["_total"] = total
        record["_assistant"] = assistant
        all_records.append(record)
        valid += 1

    print(f"  {valid} valid code-switching records after tokenization "
          f"({cs_filtered} filtered out by code_switching check)")
    print(f"  Running total: {len(all_records)} records")

print(f"\n=== Total valid code-switching records: {len(all_records)} ===")

# Source stats
source_counts = defaultdict(int)
source_tokens = defaultdict(int)
for r in all_records:
    source_counts[r["source"]] += 1
    source_tokens[r["source"]] += r["_assistant"]
total_records = len(all_records)
total_tokens = sum(source_tokens.values())
print(f"Source distribution: {dict(sorted(source_counts.items()))}")
print(f"Source token distribution: {dict(sorted(source_tokens.items()))}")
print(f"Total assistant tokens: {total_tokens}")

# --- Holdout first: token-proportional across sources ---
holdout_target_tokens = int(T_franco * 0.10)
holdout_records = []
holdout_ids = set()

sorted_sources = sorted(source_tokens.keys())
remaining_holdout_tokens = holdout_target_tokens
source_holdout_alloc = {}
for i, src in enumerate(sorted_sources):
    share = source_tokens[src] / total_tokens
    alloc = int(holdout_target_tokens * share)
    if i == len(sorted_sources) - 1:
        alloc = remaining_holdout_tokens
    source_holdout_alloc[src] = alloc
    remaining_holdout_tokens -= alloc

print(f"\nHoldout target: {holdout_target_tokens} tokens")
print(f"Holdout allocation per source: {dict(sorted(source_holdout_alloc.items()))}")

remaining_pool = list(all_records)
for src in sorted_sources:
    src_pool = [r for r in remaining_pool if r["source"] == src]
    need_tokens = source_holdout_alloc[src]
    if need_tokens <= 0 or not src_pool:
        continue
    ordered = sorted(src_pool, key=lambda r: stable_key(r, SEED))
    selected = []
    sel_tokens = 0
    for r in ordered:
        if sel_tokens >= need_tokens:
            break
        if sel_tokens + r["_assistant"] <= need_tokens * 1.05:
            selected.append(r)
            sel_tokens += r["_assistant"]
            holdout_ids.add(r["id"])
    holdout_records.extend(selected)
    remaining_pool = [r for r in remaining_pool if r["id"] not in holdout_ids]

print(f"Holdout selected: {len(holdout_records)} rows, "
      f"{sum(r['_assistant'] for r in holdout_records)} assistant tokens")

# --- Train: fill to T_franco from remaining ---
train_records = []
train_assistant = 0
lower = T_franco * 0.98
upper = T_franco * 1.02

ordered = sorted(remaining_pool, key=lambda r: stable_key(r, SEED))
for r in ordered:
    if train_assistant >= lower:
        break
    a = r["_assistant"]
    if a <= 0:
        continue
    if train_assistant + a <= upper:
        train_records.append(r)
        train_assistant += a

print(f"\nTrain selected: {len(train_records)} rows, assistant tokens: {train_assistant}")
print(f"Target: {T_franco}, band: [{lower:.0f}, {upper:.0f}]")
print(f"Within tolerance: {lower <= train_assistant <= upper}")

# Clean up temp fields and sort
for r in train_records + holdout_records:
    for k in ("_total", "_assistant"):
        r.pop(k, None)

train_records.sort(key=lambda r: r["id"])
holdout_records.sort(key=lambda r: r["id"])

# Write outputs
train_out = Path("data/processed/franco_tunisian/train.jsonl")
holdout_out = Path("data/processed/franco_tunisian/holdout.jsonl")
train_out.parent.mkdir(parents=True, exist_ok=True)
holdout_out.parent.mkdir(parents=True, exist_ok=True)

write_jsonl(train_out, train_records)
write_jsonl(holdout_out, holdout_records)

print(f"\nWrote {len(train_records)} train records -> {train_out}")
print(f"Wrote {len(holdout_records)} holdout records -> {holdout_out}")

# Manifest
manifest = {
    "slice": "franco_tunisian",
    "purpose": "tunisian_adaptation",
    "seed": SEED,
    "t_code": T_code,
    "t_franco": T_franco,
    "token_metric": "assistant_tokens",
    "max_formatted_tokens_per_example": MAX_TOKENS,
    "train": {
        "total": len(train_records),
        "assistant_tokens": sum(count_tokens(r["messages"])[1] for r in train_records),
    },
    "holdout": {
        "total": len(holdout_records),
        "assistant_tokens": sum(count_tokens(r["messages"])[1] for r in holdout_records),
    },
    "sources": sorted(source_counts.keys()),
    "source_distribution": dict(sorted(source_counts.items())),
    "source_token_distribution": dict(sorted(source_tokens.items())),
    "deterministic": True,
}
manifest_out = Path("data/manifests/franco_tunisian/selection.json")
manifest_out.parent.mkdir(parents=True, exist_ok=True)
manifest_out.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"Wrote manifest -> {manifest_out}")
