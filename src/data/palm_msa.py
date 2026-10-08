"""PALM -> MSA extraction, reshaped into the CIDAR record structure.

PURPOSE
    PALM (``UBC-NLP/palm``) is a *multi-variety* Arabic instruction dataset: it
    mixes Modern Standard Arabic with the dialects of 22 Arab countries. The
    ``msa_formal`` slice (configs/data/msa.yaml) needs MSA **only**, so PALM
    cannot be consumed as-is: it must be filtered hard, then reshaped into the
    same record structure as CIDAR so both pools look identical downstream.

    CIDAR record structure (authoritative, read from the real parquet file):
        ``{"output": str, "instruction": str, "index": int}``

DESIGN RULES
    - **Nothing is guessed.** The expected PALM columns are asserted against the
      real file; a missing or renamed column is a hard error, never a silently
      skipped filter.
    - **Fail loudly, never silently empty.** If the ``language_variety`` values
      stop looking like what this module was written against (upstream revision
      change, new labels), the run aborts and prints the observed histogram
      instead of emitting a plausible-looking but wrong file.
    - **Reject, never repair.** A record that fails any gate is dropped whole
      with a counted reason. No truncation, no translation, no normalization of
      content.
    - **Deterministic.** Output depends only on the input rows: stable sort by
      the original PALM ``id``, then a dense 0..n-1 ``index``.
    - **No network.** This module never downloads anything; acquisition is
      scripts/acquire_msa_candidate_data.py.

WHY THE FILTER IS AGGRESSIVE
    The slice needs a few thousand MSA examples out of ~15k PALM rows, so
    precision beats recall: it is better to drop a usable MSA row than to let a
    dialectal row into a formal-register pool. A single whole-word dialect
    marker is therefore enough to reject a record, and the dialect marker list
    covers Maghrebi, Egyptian, Levantine, Gulf and Iraqi — not just Tunisian.

CLI
    python -m src.data.palm_msa \
        --input  data/raw/UBC-NLP__palm/data/train-00000-of-00001.parquet \
        --output data/processed/msa/palm_msa.jsonl \
        --report data/processed/msa/palm_msa_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Upstream identity (pinned; see data/manifests/msa.yaml)
# --------------------------------------------------------------------------- #

PALM_REPO_ID = "UBC-NLP/palm"
PALM_REVISION = "8ec8fb36b85317d4933f2ed9939801e0740da39c"

#: Columns published in the PALM dataset card. All are required: this module
#: refuses to run against a file whose shape it does not recognise.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "country",
    "topic",
    "language_variety",
    "instruction",
    "output",
    "correct_answer_key",
    "question_type",
)

#: Keys of an emitted record, in CIDAR's own column order.
CIDAR_FIELDS: Tuple[str, ...] = ("output", "instruction", "index")


# --------------------------------------------------------------------------- #
# Language-variety labels
# --------------------------------------------------------------------------- #

#: Normalized ``language_variety`` values that mean "Modern Standard Arabic".
MSA_VARIETY_LABELS = frozenset(
    {
        "msa",
        "modern standard arabic",
        "standard arabic",
        "classical arabic",
        "fusha",
        "fusaha",
        "al fusha",
        "arabic",
        "الفصحى",
        "فصحى",
        "العربية الفصحى",
        "عربي فصحى",
    }
)

#: Normalized values that explicitly mean "some dialect" — rejected, but
#: *recognised*, so they do not trip the schema-drift guard.
DIALECT_VARIETY_LABELS = frozenset(
    {
        "dialect",
        "dialectal",
        "dialectal arabic",
        "da",
        "colloquial",
        "colloquial arabic",
        "ammiya",
        "aamiya",
        "عامية",
        "العامية",
        "دارجة",
        "الدارجة",
    }
)

#: PALM is organised by country, so a ``language_variety`` value is very often a
#: country or region name rather than the word "dialect" ("Tunisia",
#: "Egyptian", "Moroccan Arabic", "Levantine", ...). These are *recognised*
#: dialect labels: they are rejected like any other dialect, but they must NOT
#: count as schema drift. Covers the 22 Arab League states (noun + adjective,
#: English and Arabic) plus the usual dialect-group names.
DIALECT_REGION_LABELS = frozenset(
    {
        # --- country nouns (English) ---
        "algeria", "bahrain", "comoros", "djibouti", "egypt", "iraq", "jordan",
        "kuwait", "lebanon", "libya", "mauritania", "morocco", "oman",
        "palestine", "qatar", "saudi", "saudi arabia", "somalia", "sudan",
        "syria", "tunisia", "uae", "united arab emirates", "emirates", "yemen",
        # --- country adjectives (English) ---
        "algerian", "bahraini", "comorian", "djiboutian", "egyptian", "iraqi",
        "jordanian", "kuwaiti", "lebanese", "libyan", "mauritanian",
        "moroccan", "omani", "palestinian", "qatari", "somali", "sudanese",
        "syrian", "tunisian", "emirati", "yemeni",
        # --- dialect groups / regions ---
        "maghrebi", "maghreb", "levantine", "levant", "shami", "gulf",
        "khaleeji", "khaliji", "mashriqi", "nile basin", "north african",
        "peninsular", "hijazi", "najdi", "sa'idi", "saidi", "derja", "darija",
        # --- country nouns (Arabic) ---
        "الجزائر", "البحرين", "جزر القمر", "جيبوتي", "مصر", "العراق", "الأردن",
        "الاردن", "الكويت", "لبنان", "ليبيا", "موريتانيا", "المغرب", "عمان",
        "فلسطين", "قطر", "السعودية", "الصومال", "السودان", "سوريا", "سورية",
        "تونس", "الإمارات", "الامارات", "اليمن",
        # --- dialect groups (Arabic) ---
        "مغربي", "مشرقي", "خليجي", "شامي", "مصري", "تونسي", "جزائري", "عراقي",
        "لبناني", "سوري", "أردني", "فلسطيني", "سعودي", "ليبي", "سوداني",
        "يمني", "كويتي", "قطري", "بحريني", "إماراتي", "عماني", "موريتاني",
    }
)

#: If more than this share of rows carry a NON-EMPTY variety label that this
#: module cannot classify at all, the upstream labelling has drifted and the run
#: must abort. Empty/missing labels are tracked separately (they are malformed
#: rows, not drift) and do not count here.
MAX_UNKNOWN_VARIETY_SHARE = 0.25



# --------------------------------------------------------------------------- #
# Script / orthography gates
# --------------------------------------------------------------------------- #

_ARABIC_BLOCKS = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

#: Minimum share of *letters* that must be Arabic script. MSA is written in
#: Arabic script; a Latin-dominant record is either Arabizi, a translation
#: exercise, or code — none of which belong in the formal-register pool.
MIN_ARABIC_LETTER_RATIO = 0.80

#: Minimum number of Arabic letters: guards against stub records that would
#: pass a ratio test on two characters.
MIN_ARABIC_LETTERS = 20

#: Arabizi / Franco-Arabic orthography: a Latin-script token carrying the
#: digit substitutions 3=ع, 7=ح, 9=ق (e.g. "3asslema", "7aja", "9ahwa").
ARABIZI_RE = re.compile(r"[A-Za-z]*[379][A-Za-z]+|[A-Za-z]+[379][A-Za-z]*")

#: Whole-word dialect markers across the major Arabic dialect groups. Python's
#: ``\b`` is Unicode-aware for ``str`` patterns, so Arabic letters count as word
#: characters and these match as whole words — critical, because substring
#: matching produces false positives on MSA words (e.g. "توا" inside "تواصل",
#: "مش" inside "مشروع", "فين" inside "سفينة").
#:
#: DELIBERATELY EXCLUDED — these are dialectal in *usage* but are also perfectly
#: ordinary MSA words, so matching them would reject correct formal Arabic:
#:   حاليا (currently) · مرة (once/a time) · خلاص (salvation) · يلزم (it is
#:   necessary) · حقي / حقك (my/your right) · أمي (my mother) · بالك (ما بالك) ·
#:   نحكي / تحكي (to narrate, حكى) · بدو (the Bedouins) · زين (adornment; a
#:   name) · كذا (such) · معاد (appointment) · حنا (also the name Hanna).
#: Catching those dialects is left to the variety label, which is authoritative.
DIALECT_MARKER_RE = re.compile(
    r"\b("
    # --- Maghrebi (Tunisian / Algerian / Moroccan / Libyan) ---
    r"برشا|بزاف|كيفاش|شنوة|شنية|شنو|علاش|وقتاش|فاش|واش|دابا|توا|توّا|هكا|هكّا|"
    r"باهي|ياخي|خويا|فمة|ماشي|ماشى|زعما|نتوما|كيما|"
    # --- Egyptian / Sudanese ---
    r"ازاي|إزاي|عايز|عاوز|كده|كدا|دلوقتي|دلوقت|مش|علشان|عشان|ايه|إيه|"
    r"بكام|فين|منين|لسه|كتير|اوي|أوي|يلا|"
    # --- Levantine (Syrian / Lebanese / Palestinian / Jordanian) ---
    r"شو|هيك|هلق|هلأ|بدي|بدك|كيفك|منيح|لهيك|عنجد|هدا|هاد|هيدا|"
    # --- Gulf (Saudi / Emirati / Kuwaiti / Qatari / Bahraini / Omani) ---
    r"وش|وشو|شلون|وايد|جذي|يبه|ابوي|"
    # --- Iraqi ---
    r"شگد|شكد|هواي|اكو|ماكو|هاي|هيچ|چان|ليش|"
    # --- Yemeni ---
    r"كيفش"
    r")\b"
)

#: MSA words that must NEVER be treated as dialect markers. Asserted by the
#: test suite so a future edit to DIALECT_MARKER_RE cannot silently start
#: rejecting ordinary formal Arabic.
MSA_SAFE_WORDS: Tuple[str, ...] = (
    "حاليا",
    "مرة",
    "خلاص",
    "يلزم",
    "حقي",
    "حقك",
    "أمي",
    "بالك",
    "نحكي",
    "تحكي",
    "بدو",
    "زين",
    "كذا",
    "معاد",
    "تواصل",
    "مشروع",
    "سفينة",
    "الذي",
    "التي",
    "يجب",
    "يمكن",
    "الفصحى",
)


#: A record is rejected at this many whole-word dialect hits. One is enough:
#: see "WHY THE FILTER IS AGGRESSIVE" in the module docstring.
MAX_DIALECT_MARKERS = 0

#: Placeholder strings that upstream exports use for "no value".
_NULLISH = frozenset({"", "nan", "none", "null", "n/a", "na", "-", "--"})

#: Rejection reasons, in the order they are evaluated.
REJECTION_REASONS: Tuple[str, ...] = (
    "not_msa_variety",
    "missing_variety",
    "unknown_variety",

    "missing_instruction_or_output",
    "answer_key_only_output",
    "arabizi_orthography",
    "insufficient_arabic_script",
    "dialect_markers",
    "duplicate",
)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def normalize_variety(value: Any) -> str:
    """Normalize a ``language_variety`` label for comparison.

    Case-folds, strips Arabic diacritics/tatweel, turns separators into single
    spaces. Content is never altered — this is label matching only.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("\u0640", "")  # tatweel
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_\-/\\.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(labels: Iterable[str]) -> frozenset:
    return frozenset(label.replace(" ", "") for label in labels)


_MSA_COMPACT = _compact(MSA_VARIETY_LABELS)
_DIALECT_COMPACT = _compact(DIALECT_VARIETY_LABELS)
_REGION_COMPACT = _compact(DIALECT_REGION_LABELS)


def classify_variety(value: Any) -> str:
    """Classify a ``language_variety`` label.

    Returns one of:
        ``"msa"``      — recognised as Modern Standard Arabic (keep)
        ``"dialect"``  — recognised as a dialect, incl. country/region names
                         such as "Tunisia" or "Egyptian" (reject)
        ``"missing"``  — no label at all (reject; a malformed row, not drift)
        ``"unknown"``  — a non-empty label this module cannot classify
                         (reject, and counts toward the schema-drift guard)

    Distinguishing "missing" from "unknown" matters: a blank label is an
    upstream data defect, while an unrecognised *value* may mean the labelling
    scheme changed and the filter can no longer be trusted.
    """
    label = normalize_variety(value)
    if not label:
        return "missing"
    compact = label.replace(" ", "")
    if label in MSA_VARIETY_LABELS or compact in _MSA_COMPACT:
        return "msa"
    if label in DIALECT_VARIETY_LABELS or compact in _DIALECT_COMPACT:
        return "dialect"
    # PALM is country-organised, so dialects are usually labelled by country or
    # region ("Tunisia", "Egyptian", "Levantine"). Those are recognised dialect
    # labels, not drift.
    if label in DIALECT_REGION_LABELS or compact in _REGION_COMPACT:
        return "dialect"
    # Multi-word labels such as "Moroccan Arabic" / "Egyptian dialect": if any
    # token is a known region or dialect word, the row is dialectal.
    tokens = set(label.split())
    if tokens & (DIALECT_REGION_LABELS | DIALECT_VARIETY_LABELS):
        return "dialect"
    return "unknown"



def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().casefold() in _NULLISH


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def arabic_letter_ratio(text: str) -> Tuple[float, int]:
    """Return ``(arabic_share_of_letters, arabic_letter_count)``."""
    arabic = 0
    letters = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        letters += 1
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS):
            arabic += 1
    if letters == 0:
        return 0.0, 0
    return arabic / letters, arabic


def count_dialect_markers(text: str) -> List[str]:
    """Whole-word dialect marker hits (deterministic, auditable)."""
    return DIALECT_MARKER_RE.findall(text)


def _dedupe_key(instruction: str, output: str) -> str:
    """Exact-duplicate key over whitespace-collapsed content."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()  # noqa: E731
    return f"{norm(instruction)}\x00{norm(output)}"


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


class PalmSchemaError(RuntimeError):
    """Raised when the PALM input does not have the expected shape."""


def _assert_columns(columns: Sequence[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in set(columns)]
    if missing:
        raise PalmSchemaError(
            "PALM input is missing expected column(s): "
            f"{', '.join(missing)}.\n"
            f"Found columns: {sorted(columns)}\n"
            f"This module was written against {PALM_REPO_ID} "
            f"revision {PALM_REVISION}. If upstream changed, update "
            "REQUIRED_COLUMNS in src/data/palm_msa.py deliberately — do not "
            "loosen the check to make a run pass."
        )


def filter_msa(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Filter PALM rows down to MSA and reshape them into CIDAR records.

    Returns a dict with:
        ``records``   list of ``{"output", "instruction", "index"}``
        ``report``    measured counts, never estimates
        ``id_map``    emitted ``index`` -> original PALM ``id`` (provenance)
    """
    rows = list(rows)
    if not rows:
        raise PalmSchemaError("PALM input contains no rows.")

    _assert_columns(list(rows[0].keys()))

    rejected = Counter()
    variety_hist = Counter()
    unknown_labels = Counter()
    kept_countries = Counter()
    kept_topics = Counter()
    kept_question_types = Counter()
    dialect_hits_hist = Counter()

    seen: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []

    for row in rows:
        variety_class = classify_variety(row.get("language_variety"))
        variety_hist[variety_class] += 1

        if variety_class != "msa":
            if variety_class == "unknown":
                unknown_labels[normalize_variety(row.get("language_variety"))] += 1
                rejected["unknown_variety"] += 1
            elif variety_class == "missing":
                rejected["missing_variety"] += 1
            else:
                rejected["not_msa_variety"] += 1
            continue


        instruction = _text(row.get("instruction"))
        output = _text(row.get("output"))

        if _is_nullish(instruction) or _is_nullish(output):
            rejected["missing_instruction_or_output"] += 1
            continue

        # Multiple-choice rows whose "output" is just the answer key (e.g. "B")
        # teach label-picking, not formal Arabic generation.
        answer_key = _text(row.get("correct_answer_key"))
        if answer_key and (
            output.casefold() == answer_key.casefold() or len(output) <= 3
        ):
            rejected["answer_key_only_output"] += 1
            continue

        combined = f"{instruction}\n{output}"

        if ARABIZI_RE.search(combined):
            rejected["arabizi_orthography"] += 1
            continue

        ratio, arabic_letters = arabic_letter_ratio(combined)
        if ratio < MIN_ARABIC_LETTER_RATIO or arabic_letters < MIN_ARABIC_LETTERS:
            rejected["insufficient_arabic_script"] += 1
            continue

        hits = count_dialect_markers(combined)
        if len(hits) > MAX_DIALECT_MARKERS:
            rejected["dialect_markers"] += 1
            for hit in hits:
                dialect_hits_hist[hit] += 1
            continue

        key = _dedupe_key(instruction, output)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen[key] = 1

        kept.append(
            {
                "palm_id": row.get("id"),
                "instruction": instruction,
                "output": output,
            }
        )
        kept_countries[_text(row.get("country")) or "unspecified"] += 1
        kept_topics[_text(row.get("topic")) or "unspecified"] += 1
        kept_question_types[_text(row.get("question_type")) or "unspecified"] += 1

    total = len(rows)

    # ---- schema-drift guard -------------------------------------------------
    # Measured over rows that actually HAVE a label: a blank label is a data
    # defect in a single row, whereas an unrecognised *value* suggests the
    # labelling scheme changed and this filter can no longer be trusted.
    labelled = total - variety_hist["missing"]
    unknown_share = variety_hist["unknown"] / labelled if labelled else 0.0
    if unknown_share > MAX_UNKNOWN_VARIETY_SHARE:
        raise PalmSchemaError(
            "Refusing to emit: "
            f"{variety_hist['unknown']}/{labelled} labelled rows "
            f"({unknown_share:.1%}) carry a language_variety value this module "
            f"does not recognise (limit {MAX_UNKNOWN_VARIETY_SHARE:.0%}).\n"
            f"Observed unknown labels: {dict(unknown_labels.most_common(25))}\n"
            "Update MSA_VARIETY_LABELS / DIALECT_VARIETY_LABELS / "
            "DIALECT_REGION_LABELS in src/data/palm_msa.py after inspecting "
            "the real values — do not raise the threshold to make a run pass."
        )

    if not kept:
        raise PalmSchemaError(
            "Refusing to emit an empty MSA pool: every one of "
            f"{total} PALM rows was rejected.\n"
            f"Variety histogram: {dict(variety_hist)}\n"
            f"Rejections: {dict(rejected)}"
        )

    # ---- deterministic ordering + dense CIDAR index -------------------------
    kept.sort(key=lambda r: (r["palm_id"] is None, r["palm_id"], r["instruction"]))

    records: List[Dict[str, Any]] = []
    id_map: Dict[str, Any] = {}
    for new_index, row in enumerate(kept):
        records.append(
            {"output": row["output"], "instruction": row["instruction"], "index": new_index}
        )
        id_map[str(new_index)] = row["palm_id"]

    report = {
        "source": {"repo_id": PALM_REPO_ID, "revision": PALM_REVISION},
        "input_rows": total,
        "kept_rows": len(records),
        "kept_share": round(len(records) / total, 4) if total else 0.0,
        "variety_histogram": dict(variety_hist),
        "unknown_variety_labels": dict(unknown_labels.most_common(25)),
        "rejected": {reason: rejected[reason] for reason in REJECTION_REASONS},
        "rejected_total": sum(rejected.values()),
        "kept_country_distribution": dict(kept_countries.most_common()),
        "kept_topic_distribution": dict(kept_topics.most_common()),
        "kept_question_types": dict(kept_question_types.most_common()),
        "top_dialect_markers_rejected": dict(dialect_hits_hist.most_common(25)),
        "gates": {
            "min_arabic_letter_ratio": MIN_ARABIC_LETTER_RATIO,
            "min_arabic_letters": MIN_ARABIC_LETTERS,
            "max_dialect_markers": MAX_DIALECT_MARKERS,
            "output_structure": list(CIDAR_FIELDS),
        },
        "note": (
            "All counts are measured from the input file, not estimated. "
            "Records are reshaped into the CIDAR structure "
            "(output, instruction, index); `index` is a dense 0..n-1 "
            "re-index and the original PALM ids are preserved in the id_map "
            "sidecar."
        ),
    }

    # Invariants worth asserting rather than trusting.
    assert sum(rejected.values()) + len(records) == total, "row accounting lost rows"
    assert {k for r in records for k in r} == set(CIDAR_FIELDS), "non-CIDAR fields emitted"

    return {"records": records, "report": report, "id_map": id_map}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read PALM rows from a ``.parquet`` or ``.jsonl`` file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PALM input not found: {path}\n"
            "Acquire it first: python scripts/acquire_msa_candidate_data.py"
        )
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        import pyarrow.parquet as pq  # local import: keeps the module importable without pyarrow

        table = pq.read_table(path)
        _assert_columns(table.schema.names)
        columns = table.to_pydict()
        return [
            {name: columns[name][i] for name in table.schema.names}
            for i in range(table.num_rows)
        ]
    if suffix in {".jsonl", ".json"}:
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise PalmSchemaError(
                        f"{path}:{lineno} is not valid JSON: {exc}"
                    ) from exc
        return rows
    raise PalmSchemaError(f"Unsupported PALM input format: {path.suffix}")


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter PALM down to Modern Standard Arabic and emit records in the "
            "CIDAR structure (output, instruction, index)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/UBC-NLP__palm/data/train-00000-of-00001.parquet"),
        help="PALM parquet/jsonl file (TRAIN split only — see --help notes).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/msa/palm_msa.jsonl"),
        help="Destination JSONL in CIDAR structure.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/msa/palm_msa_report.json"),
        help="Destination JSON filter report (measured counts).",
    )
    parser.add_argument(
        "--id-map",
        type=Path,
        default=Path("data/processed/msa/palm_msa_id_map.json"),
        help="Destination JSON map: emitted index -> original PALM id.",
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.input)
    result = filter_msa(rows)

    write_jsonl(result["records"], args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.id_map.write_text(
        json.dumps(result["id_map"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = result["report"]
    print(f"PALM input rows      : {report['input_rows']}")
    print(f"MSA rows kept        : {report['kept_rows']} ({report['kept_share']:.1%})")
    print("Rejections:")
    for reason, count in report["rejected"].items():
        if count:
            print(f"  {reason:<32} {count}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.id_map}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
