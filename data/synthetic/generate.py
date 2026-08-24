"""Generate the SYNTHETIC test fixtures for the data pipeline.

    python data/synthetic/generate.py

╔══════════════════════════════════════════════════════════════════════════════╗
║  THIS IS NOT TRAINING DATA.                                                  ║
║  Every example here is machine-generated placeholder text whose only job is   ║
║  to exercise the data pipeline (validation, stats, dedup, selection, split,   ║
║  mixture validation, export). It must NEVER be mistaken for, or mixed into,   ║
║  real Gemma Tounsi training data. Every record carries source="synthetic*"    ║
║  and quality.synthetic = true so it is trivially filterable.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

The generator is fully deterministic: no randomness, no timestamps. Re-running it
reproduces byte-identical files.

Outputs (all under data/synthetic/):
    raw/retention_pool.jsonl   English retention candidates, all 5 categories
    raw/arabizi.jsonl          Arabizi slice, technical share ABOVE the 20% quota
    raw/arabic_derja.jsonl     Arabic-script Derja, technical share ABOVE 20%
    raw/franco_tunisian.jsonl  French/Tunisian code-switching
    raw/msa_formal.jsonl       MSA / formal register (NOT retention)
    raw/arabizi_low_technical.jsonl  Arabizi BELOW the quota — must FAIL validation
    raw/malformed.jsonl        broken records — must be caught, not silently dropped
    raw/duplicates.jsonl       repeated ids — must be caught by dedup/validation
    expected/summary.json      expected counts, so tests assert real numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
EXPECTED = HERE / "expected"

# Placeholder bodies. Short on purpose: this data tests plumbing, not modelling.
TECHNICAL = ("mathematics", "reasoning", "coding")


def record(
    record_id: str,
    category: str,
    source: str,
    language: str,
    user: str,
    assistant: str,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one canonical synthetic record."""
    row: Dict[str, Any] = {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "source": source,
        "language": language,
        "quality": {"synthetic": True},
    }
    row.update(extra)
    return row


def write(path: Path, rows: List[Dict[str, Any]]) -> int:
    """Write rows as deterministic UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


# --------------------------------------------------------------------------- #
# Retention pool — ENGLISH capability preservation (NOT Tunisian adaptation)
# --------------------------------------------------------------------------- #

RETENTION_BODIES = {
    "mathematics": (
        "Synthetic placeholder: compute {n} + {m} and show your working.",
        "Synthetic placeholder: {n} + {m} = {s}. Step 1: add the units. Step 2: report {s}.",
    ),
    "coding": (
        "Synthetic placeholder: write a Python function that returns item {n} of a list.",
        "Synthetic placeholder:\n```python\ndef get_item(items):\n    return items[{n}]\n```",
    ),
    "reasoning": (
        "Synthetic placeholder: if every A is B and this is an A, is it a B? Case {n}.",
        "Synthetic placeholder: yes. All A are B, so an A is necessarily a B. (case {n})",
    ),
    "general_instruction": (
        "Synthetic placeholder: summarise the following note in {n} short bullet points.",
        "Synthetic placeholder: here are {n} bullet point(s) summarising the note.",
    ),
    "knowledge_qa": (
        "Synthetic placeholder: what is the capital city referenced in fixture {n}?",
        "Synthetic placeholder: fixture {n} references a placeholder capital city.",
    ),
}

#: Per-category pool size. Large enough that a scaled retention selection
#: (scale=0.01 -> 50/50/40/30/30 train + ~25 holdout) is feasible.
RETENTION_PER_CATEGORY = 60

#: Several sources so source-diversity logic has something to spread across.
RETENTION_SOURCES = ("synthetic_pool_a", "synthetic_pool_b", "synthetic_pool_c")


def build_retention_pool() -> List[Dict[str, Any]]:
    """English retention candidates across all five capability categories."""
    rows: List[Dict[str, Any]] = []
    for category, (user_tpl, assistant_tpl) in RETENTION_BODIES.items():
        for index in range(RETENTION_PER_CATEGORY):
            source = RETENTION_SOURCES[index % len(RETENTION_SOURCES)]
            rows.append(
                record(
                    f"syn-ret-{category}-{index:04d}",
                    category,
                    source,
                    "en",
                    user_tpl.format(n=index, m=index + 1, s=index + index + 1),
                    assistant_tpl.format(n=index, m=index + 1, s=index + index + 1),
                    subcategory=f"{category}_basic",
                    script="latin",
                    difficulty=("easy", "medium", "hard")[index % 3],
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Tunisian slices
# --------------------------------------------------------------------------- #

ARABIZI_BODIES = {
    "mathematics": (
        "3andi {n} dinar w zedt {m}, 9adech el majmou3?",
        "El majmou3 howa {s} dinar. Zid {n} m3a {m} tji {s}.",
    ),
    "coding": (
        "Kifech na3mel fonction f Python bech tarja3 element {n}?",
        "Haw el code:\n```python\ndef jib(items):\n    return items[{n}]\n```",
    ),
    "reasoning": (
        "Ken el 7keya {n} sa7i7a, chnowa el natija?",
        "El natija: el 7keya {n} tebda sa7i7a, donc el conclusion sa7i7a zada.",
    ),
    "general_instruction": (
        "3awenni n7ader lista fiha {n} points 3al mawdou3 hedha.",
        "Haw lista fiha {n} points 9sar 3al mawdou3.",
    ),
    "knowledge_qa": (
        "Chkoun el 3asima elli mawjouda fel fixture {n}?",
        "Fel fixture {n} el 3asima hiya placeholder.",
    ),
}

DERJA_BODIES = {
    "mathematics": (
        "عندي {n} دينار وزدت {m}، قداش المجموع؟",
        "المجموع هو {s} دينار. زيد {n} مع {m} تجي {s}.",
    ),
    "coding": (
        "كيفاش نعمل فونكسيون في پايثون ترجع العنصر {n}؟",
        "هاو الكود:\n```python\ndef jib(items):\n    return items[{n}]\n```",
    ),
    "reasoning": (
        "كان الحكاية {n} صحيحة، شنوة النتيجة؟",
        "النتيجة: الحكاية {n} صحيحة، إذن الخلاصة صحيحة زادة.",
    ),
    "general_instruction": (
        "عاوني نحضر ليستة فيها {n} نقاط على الموضوع هذا.",
        "هاو ليستة فيها {n} نقاط قصار على الموضوع.",
    ),
    "knowledge_qa": (
        "شكون العاصمة إلي موجودة في الفixture {n}؟",
        "في الفixture {n} العاصمة هي بلاسهولدر.",
    ),
}

#: Category cycle giving a technical share of 3/5 = 60% — comfortably above the
#: 20% cross-cutting minimum, so the quota check passes on the good fixtures.
GOOD_CYCLE = (
    "mathematics",
    "coding",
    "reasoning",
    "general_instruction",
    "knowledge_qa",
)

#: Category cycle with a technical share of 1/10 = 10% — BELOW the minimum, used
#: to prove the quota check actually fails when it should.
LOW_CYCLE = (
    "mathematics",
    "general_instruction",
    "knowledge_qa",
    "general_instruction",
    "knowledge_qa",
    "general_instruction",
    "knowledge_qa",
    "general_instruction",
    "knowledge_qa",
    "general_instruction",
)


def build_tunisian_slice(
    prefix: str,
    bodies: Dict[str, Any],
    language: str,
    script: str,
    source: str,
    count: int,
    cycle: tuple,
    *,
    code_switching: bool = False,
) -> List[Dict[str, Any]]:
    """Build a Tunisian-slice fixture with a controlled technical share."""
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        category = cycle[index % len(cycle)]
        user_tpl, assistant_tpl = bodies[category]
        rows.append(
            record(
                f"{prefix}-{index:04d}",
                category,
                source,
                language,
                user_tpl.format(n=index, m=index + 1, s=index + index + 1),
                assistant_tpl.format(n=index, m=index + 1, s=index + index + 1),
                subcategory=f"{category}_tn",
                script=script,
                code_switching=code_switching,
                difficulty=("easy", "medium", "hard")[index % 3],
            )
        )
    return rows


FRANCO_BODIES = {
    "mathematics": (
        "J'ai {n} dinars et j'ajoute {m}, ça fait combien au total?",
        "Le total est {s} dinars. {n} plus {m} donne {s}.",
    ),
    "general_instruction": (
        "Aide-moi à préparer une liste de {n} points, s'il vous plaît.",
        "Voilà une liste de {n} point(s) courts sur le sujet.",
    ),
    "knowledge_qa": (
        "C'est quoi la capitale mentionnée dans la fixture {n}?",
        "Dans la fixture {n}, la capitale est un placeholder.",
    ),
}

MSA_BODIES = {
    "general_instruction": (
        "اكتب فقرة رسمية من {n} أسطر حول هذا الموضوع.",
        "فيما يلي فقرة رسمية مكوّنة من {n} أسطر حول الموضوع المذكور.",
    ),
    "knowledge_qa": (
        "ما هي العاصمة المذكورة في النموذج {n}؟",
        "العاصمة المذكورة في النموذج {n} هي عنصر نموذجي للاختبار.",
    ),
    "reasoning": (
        "إذا كانت المقدمة {n} صحيحة، فما النتيجة المنطقية؟",
        "بما أنّ المقدمة {n} صحيحة، فإنّ النتيجة المنطقية صحيحة كذلك.",
    ),
}


# --------------------------------------------------------------------------- #
# Deliberately BROKEN fixtures
# --------------------------------------------------------------------------- #


def build_malformed_lines() -> List[str]:
    """Return raw lines covering every validation failure mode.

    These are emitted as raw text (not via ``json.dumps``) because some lines
    must be syntactically invalid JSON.
    """
    good = record(
        "syn-bad-000-valid-control",
        "mathematics",
        "synthetic_malformed",
        "en",
        "Synthetic placeholder: control record that is fully valid.",
        "Synthetic placeholder: this one must pass validation.",
    )

    lines: List[str] = [
        # 1. a valid control record, so tests can prove good rows still pass
        json.dumps(good, ensure_ascii=False, sort_keys=True),
        # 2. not valid JSON at all
        '{"id": "syn-bad-001-broken-json", "messages": [',
        # 3. valid JSON but not an object
        '["syn-bad-002-not-an-object"]',
        # 4. missing required fields (category, source, language)
        json.dumps(
            {
                "id": "syn-bad-003-missing-fields",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
            ensure_ascii=False,
        ),
        # 5. invalid category
        json.dumps(
            {
                "id": "syn-bad-004-invalid-category",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "poetry_slam",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 6. invalid language
        json.dumps(
            {
                "id": "syn-bad-005-invalid-language",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "klingon",
            },
            ensure_ascii=False,
        ),
        # 7. empty messages
        json.dumps(
            {
                "id": "syn-bad-006-empty-messages",
                "messages": [],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 8. invalid role
        json.dumps(
            {
                "id": "syn-bad-007-invalid-role",
                "messages": [
                    {"role": "wizard", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 9. wrong ordering (assistant first, ends on user)
        json.dumps(
            {
                "id": "syn-bad-008-bad-ordering",
                "messages": [
                    {"role": "assistant", "content": "answer first"},
                    {"role": "user", "content": "question after"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 10. empty content
        json.dumps(
            {
                "id": "syn-bad-009-empty-content",
                "messages": [
                    {"role": "user", "content": "   "},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 11. empty id
        json.dumps(
            {
                "id": "",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
            },
            ensure_ascii=False,
        ),
        # 12. unsupported metadata value (script) + unknown field
        json.dumps(
            {
                "id": "syn-bad-011-bad-metadata",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "category": "reasoning",
                "source": "synthetic_malformed",
                "language": "en",
                "script": "cuneiform",
                "vibes": "immaculate",
            },
            ensure_ascii=False,
        ),
    ]
    return lines


def build_duplicates() -> List[Dict[str, Any]]:
    """Records with repeated ids (and one repeated body) for dedup testing."""
    first = record(
        "syn-dup-0001",
        "mathematics",
        "synthetic_duplicates",
        "en",
        "Synthetic placeholder: duplicated question about 2 + 2.",
        "Synthetic placeholder: 2 + 2 = 4.",
    )
    same_id_different_body = record(
        "syn-dup-0001",
        "mathematics",
        "synthetic_duplicates",
        "en",
        "Synthetic placeholder: same id, different body.",
        "Synthetic placeholder: this row shares an id with the first one.",
    )
    unique = record(
        "syn-dup-0002",
        "coding",
        "synthetic_duplicates",
        "en",
        "Synthetic placeholder: a unique record.",
        "Synthetic placeholder: nothing duplicated here.",
    )
    near_duplicate = record(
        "syn-dup-0003",
        "mathematics",
        "synthetic_duplicates",
        "en",
        "Synthetic placeholder: duplicated question about 2 + 2.",
        "Synthetic placeholder: 2 + 2 = 4.",
    )
    return [first, same_id_different_body, unique, near_duplicate]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    """Generate every synthetic fixture and the expectations file."""
    RAW.mkdir(parents=True, exist_ok=True)
    EXPECTED.mkdir(parents=True, exist_ok=True)

    retention = build_retention_pool()
    arabizi = build_tunisian_slice(
        "syn-arabizi", ARABIZI_BODIES, "arabizi", "latin", "synthetic_arabizi", 40, GOOD_CYCLE
    )
    derja = build_tunisian_slice(
        "syn-derja", DERJA_BODIES, "ar", "arabic", "synthetic_derja", 40, GOOD_CYCLE
    )
    franco = build_tunisian_slice(
        "syn-franco",
        FRANCO_BODIES,
        "fr",
        "latin",
        "synthetic_franco",
        18,
        ("mathematics", "general_instruction", "knowledge_qa"),
        code_switching=True,
    )
    msa = build_tunisian_slice(
        "syn-msa",
        MSA_BODIES,
        "ar",
        "arabic",
        "synthetic_msa",
        18,
        ("general_instruction", "knowledge_qa", "reasoning"),
    )
    arabizi_low = build_tunisian_slice(
        "syn-arabizi-low",
        ARABIZI_BODIES,
        "arabizi",
        "latin",
        "synthetic_arabizi_low",
        20,
        LOW_CYCLE,
    )
    duplicates = build_duplicates()
    malformed_lines = build_malformed_lines()

    counts = {
        "retention_pool.jsonl": write(RAW / "retention_pool.jsonl", retention),
        "arabizi.jsonl": write(RAW / "arabizi.jsonl", arabizi),
        "arabic_derja.jsonl": write(RAW / "arabic_derja.jsonl", derja),
        "franco_tunisian.jsonl": write(RAW / "franco_tunisian.jsonl", franco),
        "msa_formal.jsonl": write(RAW / "msa_formal.jsonl", msa),
        "arabizi_low_technical.jsonl": write(
            RAW / "arabizi_low_technical.jsonl", arabizi_low
        ),
        "duplicates.jsonl": write(RAW / "duplicates.jsonl", duplicates),
    }

    malformed_path = RAW / "malformed.jsonl"
    with malformed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in malformed_lines:
            handle.write(line + "\n")
    counts["malformed.jsonl"] = len(malformed_lines)

    def technical_share(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["category"] in TECHNICAL) / len(rows)

    expected: Dict[str, Any] = {
        "warning": (
            "SYNTHETIC TEST FIXTURES ONLY — never training data. Generated by "
            "data/synthetic/generate.py; regenerate rather than edit by hand."
        ),
        "counts": counts,
        "retention_pool": {
            "per_category": RETENTION_PER_CATEGORY,
            "categories": sorted(RETENTION_BODIES),
            "language": "en",
            "purpose": "english_capability_preservation",
        },
        "technical_ratio": {
            "arabizi": round(technical_share(arabizi), 4),
            "arabic_derja": round(technical_share(derja), 4),
            "arabizi_low_technical": round(technical_share(arabizi_low), 4),
        },
        "malformed": {
            "total_lines": len(malformed_lines),
            "valid_control_records": 1,
            "expected_invalid_lines": len(malformed_lines) - 1,
        },
        "duplicates": {
            "total_records": len(duplicates),
            "duplicate_ids": 1,
            "unique_ids": 3,
        },
    }
    (EXPECTED / "summary.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("synthetic fixtures generated (NOT training data):")
    for name, count in counts.items():
        print(f"  {name:<32} {count:>4} record(s)")
    print(f"  expected/summary.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
