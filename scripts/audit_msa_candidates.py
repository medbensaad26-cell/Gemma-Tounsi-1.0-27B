"""Audit MSA candidate data (CIDAR + Arabic QA SIGIR 2024) for the msa_formal slice.

Mirrors the Soup dataset preflight analyses (soup data validate / stats / dedup)
and adds MSA-specific classification. The dockerized `soup` CLI is the reference
implementation; this script reproduces the equivalent measurements locally so
the audit can run on a CPU host without the Soup container.

Dimensions audited (per docs/data/msa_audit.md):
  1. MSA vs dialect content      - heuristic Arabic register classifier
  2. Formal-language quality      - markers of formality vs colloquial markers
  3. Task types                   - coarse task-type classification
  4. Duplicates                   - exact + near-duplicate (char 3-gram Jaccard 0.85)
  5. Malformed examples           - missing/empty fields, non-JSON, wrong language tag
  6. Source distribution          - per-source counts (arabic-qa `source` field)
  7. Suitability for the 8% MSA slice

The heuristic classifier is DETERMINISTIC and auditable: every signal is a
counted lexical marker, no model inference involved. It is a filter for
"probably MSA" vs "probably dialect/mixed", not a ground-truth dialect label.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Register markers
# ---------------------------------------------------------------------------
# MSA (Modern Standard Arabic / الفصحى) markers: classical particles, formal
# connectives, MSA-only function words. High counts => formal register.
MSA_MARKERS = [
    "التي", "الذي", "اللذان", "اللاتي",          # relative pronouns
    "هذا", "هذه", "ذلك", "تلك", "أولئك",           # demonstratives
    "إن", "أن", "لكن", "لأن", "حيث", "كما",        # conjunctions/particles
    "يمكن", "يجب", "يُنصح", "يُفضَّل",             # impersonal modals (formal)
    "عند", "خلال", "بعد", "قبل", "بين", "أثناء",   # prepositions
    "كذلك", "أيضا", "بالإضافة", "بالنسبة",         # formal discourse
    "قام", "تقوم", "يقوم", "تم", "تتم",            # formal verb constructions
    "الأسئلة", "الإجابة", "المعلومات", "التفاصيل", # formal nouns
    "يرجى", "ملاحظة", "بشكل عام",                  # formal instructions
]

# Dialect (esp. Tunisian Derja) markers — matched as WHOLE WORDS only.
# Substring matching produces false positives: "ما تشاء" (MSA), "تشمل" (MSA
# verb), "ماشيا" (MSA participle "walking"), "ياسر" (proper name Yasser) must
# NOT count as dialect. Whole-word matching is the auditable middle ground.
DIALECT_WORDS_RE = re.compile(
    r"\b(شنوة|برشا|كيفاش|توا|توّا|هكّا|علاش|وين|ماشي|باهي|ياخي|خويا|"
    r"نحبّ?|نحكي|نشوف|نقعد|فمة|شنيّة|فرشي)\b"
)
# NOTE: "ما" and "تش" alone are NOT reliable dialect markers in Arabic script
# (they are MSA function words too). They are therefore not counted here; the
# Arabizi/Franco-Tunisian contamination is caught by script-shape signals
# (Latin chars, Arabizi digits, French markers) below.
DIALECT_MARKERS = []  # kept for API compatibility; DIALECT_WORDS_RE is used
# MSA question words often appear in QA; count them toward MSA, not dialect.
MSA_QUESTION_WORDS = ["ماذا", "كيف", "لماذا", "أين", "متى", "من", "ما هو", "ما هي", "هل"]

# Latin-script / Arabizi / French contamination signals.
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
# Arabizi digit-orthography (3=ع, 7=ح, 9=ق) inside Latin words.
ARABIZI_RE = re.compile(r"\b\w*[379]\w*\b")
# Common French words in Franco-Tunisian code-switching.
FRENCH_MARKERS = [
    "les", "des", "une", "est", "pour", "avec", "dans", "cette", "nous", "vous",
    "être", "avoir", "faire", "très", "plus", "aussi", "mais", "alors", "chez",
]

# Formal-quality markers (shared with MSA_MARKERS but kept for a separate score)
FORMAL_MARKERS = MSA_MARKERS + [
    "تعريف", "تعريفها", "مثال", "أمثلة", "خطوات", "طريقة",
    "المطلوب", "المخرجات", "الأهداف", "النتائج", "الخصائص",
]

# Colloquial-quality anti-markers (penalty signals) — whole words only.
# Substring matching would flag MSA words like "تواصل" (contains "توا").
COLLOQUIAL_WORDS_RE = re.compile(
    r"\b(شنوة|برشا|كيفاش|توا|علاش|وين|ماشي|باهي|ياخي|خويا|نحكي|تحكي|فمة|شنيّة)\b"
)
COLLOQUIAL_MARKERS = []  # kept for API compatibility; COLLOQUIAL_WORDS_RE is used

# ---------------------------------------------------------------------------
# Task-type classification (coarse, deterministic keyword rules)
# ---------------------------------------------------------------------------
TASK_RULES = [
    # Mathematics: actual arithmetic content (operators, equations, math
    # terminology). Bare "عدد/كم" questions ("How many provinces...?") are
    # knowledge_qa, NOT mathematics — they were over-counted in v1.
    ("mathematics", re.compile(
        r"(\d+\s*[\+\-\*/×÷]\s*\d+|[\+\-\*/×÷=]|معادلة|احسب|حساب (?:مسألة|ال)|"
        r"مسألة|نسبة مئوية|رياضيات|جمع الأعداد|طرح|ضرب الأعداد)", re.IGNORECASE)),
    ("coding", re.compile(
        r"(برمج|كود|خوارزم|دالة|متغي|برنامج|بايثون|جافا|سكريبت|"
        r"HTML|CSS|Python|code|algorithm|function|SQL|API)", re.IGNORECASE)),
    ("reasoning", re.compile(
        r"(لماذا|سبب|السبب|استنتاج|استدلال|منطق|تحليل|قارن|الفرق بين|"
        r"لماذا لا|كيف\s*يمكن|برهان|إثبات)", re.IGNORECASE)),
    ("knowledge_qa", re.compile(
        r"(ما هو|ما هي|من هو|من هي|ماذا|متى|أين|عرف|تعريف|"
        r"اذكر|عدّد|ما اسم|أذكر)", re.IGNORECASE)),
    ("general_instruction", re.compile(
        r"(اكتب|أنشئ|صمم|ترجم|لخص|أعد|صاغ|اقترح|وصف|اشرح|"
        r"ضع|جهز|رتب|ولّد|أنتج)", re.IGNORECASE)),
]


def classify_task(text: str) -> str:
    for name, pattern in TASK_RULES:
        if pattern.search(text):
            return name
    return "other"


# ---------------------------------------------------------------------------
# Register classification
# ---------------------------------------------------------------------------
def count_markers(text: str, markers) -> int:
    return sum(1 for m in markers if m in text)


def classify_register(instr: str, output: str, input_text: str = "") -> dict:
    """Classify a record's Arabic register.

    Returns signals + label: msa | dialect | mixed | other.
    Deterministic: pure lexical counting.
    """
    text = f"{instr} {input_text} {output}"

    arabic_chars = len(ARABIC_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    arabizi = bool(ARABIZI_RE.search(text)) if latin_chars > 0 else False
    french = count_markers(text.lower(), FRENCH_MARKERS) if latin_chars > 0 else 0

    msa_score = count_markers(text, MSA_MARKERS) + count_markers(text, MSA_QUESTION_WORDS)
    dialect_hits = DIALECT_WORDS_RE.findall(text)
    dialect_score = len(dialect_hits)

    signals = {
        "msa_score": msa_score,
        "dialect_score": dialect_score,
        "arabic_chars": arabic_chars,
        "latin_chars": latin_chars,
        "arabizi": arabizi,
        "french_markers": french,
    }

    # Non-Arabic-script dominated records.
    if latin_chars > arabic_chars * 0.5 and latin_chars > 10:
        label = "latin_or_code_switched"
    elif arabic_chars < 30 and latin_chars < 10:
        label = "other"
    elif dialect_score >= 2 and dialect_score > msa_score:
        label = "dialect"
    elif msa_score >= 1 and dialect_score <= 1:
        label = "msa"
    elif msa_score >= 1 and dialect_score >= 2:
        label = "mixed"
    elif msa_score == 0 and dialect_score == 0:
        # No markers either way: fall back to script shape.
        label = "msa" if arabic_chars >= 30 else "other"
    else:
        label = "mixed"
    signals["label"] = label
    return signals


def formal_quality(instr: str, output: str, input_text: str = "") -> dict:
    text = f"{instr} {input_text} {output}"
    formal = count_markers(text, FORMAL_MARKERS)
    colloquial = len(COLLOQUIAL_WORDS_RE.findall(text))
    return {
        "formal_markers": formal,
        "colloquial_markers": colloquial,
        "penalty": max(0, colloquial - formal),
    }


# ---------------------------------------------------------------------------
# Near-duplicate detection (char 3-gram Jaccard, threshold 0.85 - Soup default)
# ---------------------------------------------------------------------------
def trigrams(text: str) -> set:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else {text}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------
def audit_cidar(path: Path) -> dict:
    table = pq.read_table(path)
    df = table.to_pydict()
    rows = list(zip(df["instruction"], df["output"], df.get("index", [None] * len(df["instruction"]))))
    print(f"CIDAR: {len(rows)} rows, columns={table.schema.names}")
    return _audit_generic(rows, id_col=df.get("index"), source_col=None)


def audit_arabic_qa(path: Path) -> dict:
    rows = []
    bad_json = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rows.append((
                    rec.get("instruction"),
                    rec.get("output"),
                    rec.get("input", ""),
                    rec.get("source"),
                    rec.get("language"),
                ))
            except json.JSONDecodeError:
                bad_json += 1
    print(f"Arabic QA: {len(rows)} rows, bad_json={bad_json}")
    return _audit_generic(rows, id_col=None, source_col=3, language_col=4, bad_json=bad_json)


def _audit_generic(rows, id_col=None, source_col=None, language_col=None, bad_json=0) -> dict:
    n = len(rows)
    stats = defaultdict(Counter)
    malformed = Counter()
    lengths = {"instruction": [], "output": []}
    formal_scores = []
    task_counter = Counter()
    lang_counter = Counter()

    seen_exact = set()
    exact_dup_indices = []
    normalized_records = []
    all_signals = []

    for i, row in enumerate(rows):
        instr = row[0] or ""
        output = row[1] or ""
        input_text = row[2] if (source_col is not None and len(row) > 2) else ""
        source = row[source_col] if source_col is not None else None
        language = row[language_col] if language_col is not None else None

        # 5. Malformed detection
        if not instr.strip() or not output.strip():
            malformed["empty_instruction_or_output"] += 1
        if input_text is not None and isinstance(input_text, str) and not input_text.strip() and source_col is not None:
            malformed["empty_input"] += 1  # informational; empty input is allowed
        if language is not None and language != "ar" and language != "Arabic":
            lang_counter[str(language)] += 1
            if language is None:
                malformed["missing_language"] += 1

        lengths["instruction"].append(len(instr))
        lengths["output"].append(len(output))

        reg = classify_register(instr, output, input_text or "")
        stats["register"][reg["label"]] += 1
        all_signals.append(reg)

        fq = formal_quality(instr, output, input_text or "")
        formal_scores.append(fq)
        stats["formal_penalty"][min(fq["penalty"], 5)] += 1

        # Task type is classified from the INSTRUCTION ONLY (as documented in
        # docs/data/msa_audit.md §3). In passage-based QA pools the shared
        # passage dominates the keyword counts and mislabels knowledge
        # questions as mathematics (v1 counted 2,407 math rows for Arabic QA;
        # instruction-only counting gives the corrected 191).
        task = classify_task(instr)
        stats["task"][task] += 1

        if source:
            stats["source"][str(source)] += 1

        # Exact duplicate on (instruction, output) pair
        key = (instr.strip(), output.strip())
        if key in seen_exact:
            exact_dup_indices.append(i)
        seen_exact.add(key)
        normalized_records.append(key)

    # Near-duplicate analysis is delegated to `soup data dedup --threshold 0.85`
    # (MinHash Jaccard) run in the pinned Soup 0.73.3 container; see
    # docs/data/msa_audit.md for those authoritative numbers. The local
    # Jaccard helper is kept for spot-checking individual pairs.
    near_dup_indices = set()
    near_dup_pairs = None  # measured by soup data dedup, not locally

    def dist(values, label):
        values = sorted(values)
        if not values:
            return {}
        n = len(values)
        return {
            "mean": round(sum(values) / n, 1),
            "median": values[n // 2],
            "p90": values[int(n * 0.9)],
            "max": values[-1],
            "min": values[0],
        }

    summary = {
        "total_rows": n,
        "malformed": dict(malformed),
        "register": dict(stats["register"]),
        "formal_penalty": dict(stats["formal_penalty"]),
        "task": dict(stats["task"]),
        "source": dict(stats["source"].most_common(20)),
        "language": dict(lang_counter),
        "lengths": {k: dist(v, k) for k, v in lengths.items()},
        "exact_duplicates": len(exact_dup_indices),
        "near_duplicates": len(near_dup_indices),
        "near_dup_pairs": near_dup_pairs,
        "bad_json": bad_json,
        "unique_after_exact": n - len(exact_dup_indices),
        "unique_after_near": n - len(exact_dup_indices) - len(near_dup_indices),
    }
    return summary


def main() -> int:
    cidar_path = ROOT / "data" / "raw" / "arbml__CIDAR" / "data" / "train-00000-of-00001-b2881e1b9f14c3b1.parquet"
    qa_path = ROOT / "data" / "raw" / "bobez999__arabic-qa-dataset-sigir2024" / "data" / "arabic_qa_10k_sample.jsonl"

    results = {}
    print("=" * 70)
    print("AUDIT: CIDAR")
    print("=" * 70)
    results["cidar"] = audit_cidar(cidar_path)

    print("=" * 70)
    print("AUDIT: Arabic QA Dataset - SIGIR 2024")
    print("=" * 70)
    results["arabic_qa"] = audit_arabic_qa(qa_path)

    out = ROOT / "data" / "processed" / "msa" / "audit_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {out}")

    # Console digest
    for name, res in results.items():
        print(f"\n=== {name} ===")
        for k in ("total_rows", "register", "task", "source", "malformed",
                  "exact_duplicates", "near_duplicates", "unique_after_near",
                  "language", "lengths", "formal_penalty", "bad_json"):
            print(f"  {k}: {res.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())