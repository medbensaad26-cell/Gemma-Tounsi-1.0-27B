"""Tests for the PALM -> MSA filter (``src/data/palm_msa.py``).

Runs on SYNTHETIC rows built in-process: no PALM download, no network, no
gated-repo access. That matters because PALM is gated and CC-BY-NC-ND, so the
real corpus must never be a test dependency.

Each test asserts a behavioural guarantee the filter is supposed to provide,
not merely that a function is importable.

    pytest tests/test_palm_msa.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import palm_msa  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: A long, unambiguous MSA answer: Arabic-script, no dialect markers, well over
#: the MIN_ARABIC_LETTERS floor.
MSA_OUTPUT = (
    "تُعَدُّ اللغة العربية الفصحى لغة الإدارة والتعليم والإعلام في الدول العربية، "
    "وهي التي تُستخدم في الكتابة الرسمية والنصوص الأدبية والعلمية."
)
MSA_INSTRUCTION = "اشرح أهمية اللغة العربية الفصحى في التعليم الرسمي."


def row(**overrides: Any) -> Dict[str, Any]:
    """Build a PALM row with every required column present."""
    base: Dict[str, Any] = {
        "id": 1,
        "country": "Tunisia",
        "topic": "education",
        "language_variety": "MSA",
        "instruction": MSA_INSTRUCTION,
        "output": MSA_OUTPUT,
        "correct_answer_key": None,
        "question_type": "open",
    }
    base.update(overrides)
    return base


def keep_count(rows: List[Dict[str, Any]]) -> int:
    return palm_msa.filter_msa(rows)["report"]["kept_rows"]


# --------------------------------------------------------------------------- #
# Output structure (the whole point: match CIDAR)
# --------------------------------------------------------------------------- #


class TestCidarStructure:
    """Emitted records must be indistinguishable from CIDAR rows."""

    def test_emitted_records_have_exactly_the_cidar_fields(self) -> None:
        result = palm_msa.filter_msa([row(id=7)])
        assert result["records"] == [
            {"output": MSA_OUTPUT, "instruction": MSA_INSTRUCTION, "index": 0}
        ]

    def test_cidar_field_set_matches_the_real_cidar_columns(self) -> None:
        # The real arbml/CIDAR parquet has exactly these three columns.
        assert set(palm_msa.CIDAR_FIELDS) == {"output", "instruction", "index"}

    def test_index_is_dense_and_zero_based_not_the_palm_id(self) -> None:
        rows = [row(id=500), row(id=900, instruction=MSA_INSTRUCTION + " ثانيًا")]
        records = palm_msa.filter_msa(rows)["records"]
        assert [r["index"] for r in records] == [0, 1]

    def test_original_palm_ids_are_preserved_in_the_id_map(self) -> None:
        rows = [row(id=500), row(id=900, instruction=MSA_INSTRUCTION + " ثانيًا")]
        result = palm_msa.filter_msa(rows)
        assert result["id_map"] == {"0": 500, "1": 900}

    def test_no_palm_only_columns_leak_into_the_output(self) -> None:
        records = palm_msa.filter_msa([row()])["records"]
        leaked = {"country", "topic", "language_variety", "question_type",
                  "correct_answer_key", "id", "palm_id"}
        assert not (set(records[0]) & leaked)


# --------------------------------------------------------------------------- #
# MSA extraction
# --------------------------------------------------------------------------- #


class TestVarietyFiltering:
    """Only rows whose language_variety says MSA may survive."""

    @pytest.mark.parametrize(
        "label",
        ["MSA", "msa", " MSA ", "Modern Standard Arabic", "modern_standard_arabic",
         "Standard Arabic", "fusha", "الفصحى"],
    )
    def test_msa_labels_are_kept(self, label: str) -> None:
        assert keep_count([row(language_variety=label)]) == 1

    @pytest.mark.parametrize(
        "label",
        ["Tunisia", "Egyptian", "Moroccan Arabic", "Levantine", "Gulf",
         "دارجة", "عامية", "Algeria", "Iraqi"],
    )
    def test_dialect_labels_are_rejected(self, label: str) -> None:
        rows = [row(language_variety=label), row(id=2, language_variety="MSA",
                                                 instruction=MSA_INSTRUCTION + " س")]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["kept_rows"] == 1
        assert result["id_map"] == {"0": 2}

    def test_missing_variety_is_rejected_not_assumed_msa(self) -> None:
        rows = [row(language_variety=None), row(id=2)]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["kept_rows"] == 1
        # A blank label is a malformed row, counted separately from an
        # unrecognised *value* (which would mean upstream labelling drift).
        assert result["report"]["rejected"]["missing_variety"] == 1
        assert result["report"]["rejected"]["unknown_variety"] == 0

    def test_country_labels_are_recognised_dialects_not_drift(self) -> None:
        # PALM labels dialects by country, so these must be rejected as
        # dialects WITHOUT tripping the schema-drift guard.
        rows = [row(id=i, language_variety=lbl) for i, lbl in enumerate(
            ["Tunisia", "Egyptian", "Moroccan Arabic", "Levantine", "Gulf"], start=10)]
        rows.append(row(id=99, language_variety="MSA"))
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["not_msa_variety"] == 5
        assert result["report"]["rejected"]["unknown_variety"] == 0
        assert result["report"]["kept_rows"] == 1



class TestContentGates:
    """Content-level gates that a formal-register pool must enforce."""

    def test_dialect_text_is_rejected_even_when_labelled_msa(self) -> None:
        # Mislabelled row: variety says MSA but the text is Tunisian Derja.
        dialect = "برشا ناس يحكيو هكا في تونس وهذا شنوة يعني علاش"
        rows = [row(output=dialect), row(id=2)]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["dialect_markers"] == 1
        assert result["report"]["kept_rows"] == 1

    def test_arabizi_is_rejected(self) -> None:
        rows = [row(instruction="3asslema, chna7welek?"), row(id=2)]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["arabizi_orthography"] == 1

    def test_latin_dominant_text_is_rejected(self) -> None:
        rows = [
            row(instruction="Explain the role of Modern Standard Arabic",
                output="It is the formal register used in education and media."),
            row(id=2),
        ]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["insufficient_arabic_script"] == 1

    def test_answer_key_only_outputs_are_rejected(self) -> None:
        rows = [
            row(output="B", correct_answer_key="B", question_type="mcq"),
            row(id=2),
        ]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["answer_key_only_output"] == 1

    @pytest.mark.parametrize("empty", [None, "", "   ", "nan", "NaN", "None"])
    def test_missing_content_is_rejected(self, empty: Any) -> None:
        rows = [row(output=empty), row(id=2)]
        result = palm_msa.filter_msa(rows)
        assert result["report"]["rejected"]["missing_instruction_or_output"] == 1

    def test_exact_duplicates_are_dropped(self) -> None:
        rows = [row(id=1), row(id=2), row(id=3)]  # identical content
        result = palm_msa.filter_msa(rows)
        assert result["report"]["kept_rows"] == 1
        assert result["report"]["rejected"]["duplicate"] == 2

    def test_whitespace_only_differences_count_as_duplicates(self) -> None:
        rows = [row(id=1), row(id=2, instruction=f"  {MSA_INSTRUCTION}  ")]
        assert keep_count(rows) == 1


class TestNoFalsePositivesOnMSA:
    """Ordinary formal Arabic must never be mistaken for dialect.

    This is the regression guard for the marker list: several words are
    dialectal in usage but also perfectly ordinary MSA, so matching them would
    quietly throw away correct formal Arabic.
    """

    @pytest.mark.parametrize("word", palm_msa.MSA_SAFE_WORDS)
    def test_msa_words_are_not_dialect_markers(self, word: str) -> None:
        assert palm_msa.count_dialect_markers(word) == []

    def test_a_formal_sentence_with_safe_words_survives(self) -> None:
        text = (
            "يجب أن نتواصل حاليًا بشأن المشروع الذي يلزم إنجازه، "
            "ومن حقي أن أعرف التفاصيل التي تتعلق بالسفينة مرة أخرى."
        )
        assert palm_msa.count_dialect_markers(text) == []
        assert keep_count([row(output=text)]) == 1

    def test_dialect_markers_match_whole_words_only(self) -> None:
        # "توا" (dialect "now") is a substring of "تواصل" (MSA "communicate").
        assert palm_msa.count_dialect_markers("تواصل") == []
        assert palm_msa.count_dialect_markers("توا") == ["توا"]


# --------------------------------------------------------------------------- #
# Fail-loudly contracts
# --------------------------------------------------------------------------- #


class TestFailsLoudly:
    """The filter must abort rather than emit something plausible but wrong."""

    def test_missing_column_is_a_hard_error(self) -> None:
        broken = row()
        del broken["language_variety"]
        with pytest.raises(palm_msa.PalmSchemaError, match="language_variety"):
            palm_msa.filter_msa([broken])

    def test_empty_input_is_a_hard_error(self) -> None:
        with pytest.raises(palm_msa.PalmSchemaError, match="no rows"):
            palm_msa.filter_msa([])

    def test_all_rows_rejected_is_a_hard_error_not_an_empty_file(self) -> None:
        rows = [row(language_variety="dialect"), row(id=2, language_variety="عامية")]
        with pytest.raises(palm_msa.PalmSchemaError, match="empty MSA pool"):
            palm_msa.filter_msa(rows)

    def test_unrecognised_variety_labels_trip_the_drift_guard(self) -> None:
        # Simulates upstream renaming its labels: >25% unknown must abort,
        # and the message must show what was actually observed.
        rows = [row(id=i, language_variety="Varietal-Form-X") for i in range(4)]
        rows.append(row(id=99, language_variety="MSA"))
        with pytest.raises(palm_msa.PalmSchemaError) as exc:
            palm_msa.filter_msa(rows)
        assert "varietal form x" in str(exc.value)

    def test_unsupported_input_format_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "palm.csv"
        bad.write_text("id,output\n", encoding="utf-8")
        with pytest.raises(palm_msa.PalmSchemaError, match="Unsupported"):
            palm_msa.load_rows(bad)

    def test_missing_input_file_names_the_acquisition_script(self) -> None:
        with pytest.raises(FileNotFoundError, match="acquire_msa_candidate_data"):
            palm_msa.load_rows(Path("data/raw/does-not-exist.parquet"))


# --------------------------------------------------------------------------- #
# Determinism, accounting and provenance
# --------------------------------------------------------------------------- #


class TestDeterminismAndAccounting:
    def test_output_is_byte_identical_across_runs(self, tmp_path: Path) -> None:
        rows = [
            row(id=3, instruction=MSA_INSTRUCTION + " ثالثًا"),
            row(id=1),
            row(id=2, instruction=MSA_INSTRUCTION + " ثانيًا"),
        ]
        first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        palm_msa.write_jsonl(palm_msa.filter_msa(rows)["records"], first)
        palm_msa.write_jsonl(palm_msa.filter_msa(rows)["records"], second)
        assert first.read_bytes() == second.read_bytes()

    def test_input_order_does_not_change_the_output(self) -> None:
        rows = [
            row(id=1),
            row(id=2, instruction=MSA_INSTRUCTION + " ثانيًا"),
            row(id=3, instruction=MSA_INSTRUCTION + " ثالثًا"),
        ]
        forward = palm_msa.filter_msa(rows)["records"]
        backward = palm_msa.filter_msa(list(reversed(rows)))["records"]
        assert forward == backward

    def test_every_input_row_is_accounted_for(self) -> None:
        rows = [
            row(id=1),
            row(id=2, language_variety="Egyptian"),
            row(id=3, output=None),
            row(id=4, instruction="3asslema"),
            row(id=5, output="برشا هكا علاش"),
            row(id=6),  # duplicate of id=1
        ]
        report = palm_msa.filter_msa(rows)["report"]
        assert report["kept_rows"] + report["rejected_total"] == report["input_rows"] == 6

    def test_report_records_the_pinned_upstream_revision(self) -> None:
        report = palm_msa.filter_msa([row()])["report"]
        assert report["source"] == {
            "repo_id": "UBC-NLP/palm",
            "revision": palm_msa.PALM_REVISION,
        }
        assert len(palm_msa.PALM_REVISION) == 40

    def test_report_lists_every_rejection_reason_even_when_zero(self) -> None:
        report = palm_msa.filter_msa([row()])["report"]
        assert set(report["rejected"]) == set(palm_msa.REJECTION_REASONS)


class TestRoundTripIO:
    def test_jsonl_round_trip_through_the_loader(self, tmp_path: Path) -> None:
        src = tmp_path / "palm.jsonl"
        rows = [row(id=1), row(id=2, instruction=MSA_INSTRUCTION + " ثانيًا")]
        src.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        assert palm_msa.filter_msa(palm_msa.load_rows(src))["report"]["kept_rows"] == 2

    def test_written_jsonl_is_utf8_and_not_escaped(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        palm_msa.write_jsonl(palm_msa.filter_msa([row()])["records"], out)
        text = out.read_text(encoding="utf-8")
        assert "\\u" not in text  # Arabic stays readable
        assert json.loads(text.splitlines()[0])["index"] == 0
