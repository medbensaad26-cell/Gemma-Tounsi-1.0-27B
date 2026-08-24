"""Tests for the Gemma Tounsi data-engineering foundation.

Runs entirely on the SYNTHETIC fixtures under ``data/synthetic/`` — no external
dataset, no network, no GPU, no training.

    pytest tests/ -v

Each test asserts a real behavioural guarantee from the task specification, not
merely that a function is importable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import dedupe, export, mixture, retention, schema, split, stats, validate  # noqa: E402

SYNTHETIC = REPO_ROOT / "data" / "synthetic"
RAW = SYNTHETIC / "raw"
CONFIGS = REPO_ROOT / "configs" / "data"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def synthetic_data() -> None:
    """Ensure the synthetic fixtures exist (generate them if missing)."""
    if not (RAW / "arabizi.jsonl").is_file():
        subprocess.run(
            [sys.executable, str(SYNTHETIC / "generate.py")],
            check=True,
            cwd=str(REPO_ROOT),
        )


@pytest.fixture(scope="session")
def canonical_schema() -> schema.Schema:
    """The canonical schema loaded from configs/data/schema.yaml."""
    return schema.load_schema()


@pytest.fixture(scope="session")
def retention_pool() -> List[Dict[str, Any]]:
    """The synthetic English retention candidate pool."""
    return schema.load_jsonl(RAW / "retention_pool.jsonl").records


def good_record(**overrides: Any) -> Dict[str, Any]:
    """Build a minimal valid canonical record."""
    record = {
        "id": "test-0001",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "category": "mathematics",
        "source": "unit_test",
        "language": "en",
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# 1. Schema validation
# --------------------------------------------------------------------------- #


class TestSchemaValidation:
    """Per-record canonical schema validation."""

    def test_valid_record_passes(self, canonical_schema: schema.Schema) -> None:
        assert schema.validate_record(good_record(), canonical_schema) == []

    @pytest.mark.parametrize("missing", ["id", "messages", "category", "source", "language"])
    def test_each_required_field_is_enforced(
        self, missing: str, canonical_schema: schema.Schema
    ) -> None:
        record = good_record()
        del record[missing]
        errors = schema.validate_record(record, canonical_schema)
        assert any(e.code == "missing_field" and e.field == missing for e in errors)

    def test_invalid_category_is_rejected(self, canonical_schema: schema.Schema) -> None:
        errors = schema.validate_record(
            good_record(category="poetry_slam"), canonical_schema
        )
        assert any(e.code == "invalid_category" for e in errors)

    def test_invalid_language_is_rejected(self, canonical_schema: schema.Schema) -> None:
        errors = schema.validate_record(good_record(language="klingon"), canonical_schema)
        assert any(e.code == "invalid_language" for e in errors)

    def test_all_project_slices_are_representable(
        self, canonical_schema: schema.Schema
    ) -> None:
        """Every mixture slice must be expressible in the schema."""
        for expected in (
            "arabizi",
            "arabic_derja",
            "franco_tunisian",
            "msa_formal",
            "retention",
        ):
            assert expected in canonical_schema.slices

    def test_all_technical_tags_are_valid_categories(
        self, canonical_schema: schema.Schema
    ) -> None:
        for tag in ("mathematics", "reasoning", "coding"):
            assert tag in canonical_schema.categories
            assert canonical_schema.is_technical(tag)
        for tag in ("general_instruction", "knowledge_qa"):
            assert tag in canonical_schema.categories
            assert not canonical_schema.is_technical(tag)

    def test_empty_messages_rejected(self, canonical_schema: schema.Schema) -> None:
        errors = schema.validate_record(good_record(messages=[]), canonical_schema)
        assert any(e.code == "empty_messages" for e in errors)

    def test_invalid_role_rejected(self, canonical_schema: schema.Schema) -> None:
        record = good_record(
            messages=[
                {"role": "wizard", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        )
        errors = schema.validate_record(record, canonical_schema)
        assert any(e.code == "invalid_role" for e in errors)

    def test_message_ordering_enforced(self, canonical_schema: schema.Schema) -> None:
        record = good_record(
            messages=[
                {"role": "assistant", "content": "answer first"},
                {"role": "user", "content": "question after"},
            ]
        )
        errors = schema.validate_record(record, canonical_schema)
        assert any(e.code == "invalid_ordering" for e in errors)

    def test_empty_content_rejected(self, canonical_schema: schema.Schema) -> None:
        record = good_record(
            messages=[
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": "answer"},
            ]
        )
        errors = schema.validate_record(record, canonical_schema)
        assert any(e.code == "empty_content" for e in errors)

    def test_leading_system_turn_is_allowed(self, canonical_schema: schema.Schema) -> None:
        record = good_record(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        assert schema.validate_record(record, canonical_schema) == []

    def test_unsupported_metadata_value_rejected(
        self, canonical_schema: schema.Schema
    ) -> None:
        errors = schema.validate_record(good_record(script="cuneiform"), canonical_schema)
        assert any(e.code == "invalid_script" for e in errors)

    def test_errors_are_reported_not_dropped(self, canonical_schema: schema.Schema) -> None:
        """A record with several defects reports every one of them."""
        record = good_record(category="nope", language="nope", script="nope")
        codes = {e.code for e in schema.validate_record(record, canonical_schema)}
        assert {"invalid_category", "invalid_language", "invalid_script"} <= codes


# --------------------------------------------------------------------------- #
# 2. Malformed records and duplicate ids
# --------------------------------------------------------------------------- #


class TestMalformedAndDuplicates:
    """File-level validation over the deliberately broken fixtures."""

    def test_malformed_fixture_is_detected(self) -> None:
        report = validate.validate_file(RAW / "malformed.jsonl")
        assert not report.ok
        expected = {
            "invalid_json",
            "invalid_type",
            "missing_field",
            "invalid_category",
            "invalid_language",
            "empty_messages",
            "invalid_role",
            "invalid_ordering",
            "empty_content",
            "empty_id",
            "invalid_script",
            "unknown_field",
        }
        assert expected <= set(report.codes())

    def test_valid_control_record_still_passes(self) -> None:
        """The one good row in malformed.jsonl must not be flagged."""
        report = validate.validate_file(RAW / "malformed.jsonl")
        assert len(report.valid_records) == 1
        assert report.valid_records[0]["id"] == "syn-bad-000-valid-control"

    def test_duplicate_ids_detected(self) -> None:
        report = validate.validate_file(RAW / "duplicates.jsonl")
        duplicates = [e for e in report.errors if e.code == "duplicate_id"]
        assert len(duplicates) == 1
        assert duplicates[0].record_id == "syn-dup-0001"

    def test_id_dedup_keeps_first_occurrence(self, tmp_path: Path) -> None:
        output = tmp_path / "deduped.jsonl"
        kept, removed = dedupe.dedupe_by_id(RAW / "duplicates.jsonl", output)
        assert (kept, removed) == (3, 1)
        ids = [r["id"] for r in schema.load_jsonl(output).records]
        assert ids == ["syn-dup-0001", "syn-dup-0002", "syn-dup-0003"]
        first = schema.load_jsonl(output).records[0]
        assert "duplicated question" in first["messages"][0]["content"]

    def test_good_fixtures_are_clean(self) -> None:
        for name in (
            "arabizi.jsonl",
            "arabic_derja.jsonl",
            "franco_tunisian.jsonl",
            "msa_formal.jsonl",
            "retention_pool.jsonl",
        ):
            report = validate.validate_file(RAW / name)
            assert report.ok, f"{name} should be valid:\n{report.summary()}"


# --------------------------------------------------------------------------- #
# 3. Statistics
# --------------------------------------------------------------------------- #


class TestStatistics:
    """The generic analyzer."""

    def test_reports_expected_distributions(self) -> None:
        result = stats.analyze_file(RAW / "arabizi.jsonl")
        assert result.total_examples == 40
        assert result.malformed_records == 0
        assert result.duplicate_ids == 0
        assert set(result.by_category) == {
            "mathematics",
            "coding",
            "reasoning",
            "general_instruction",
            "knowledge_qa",
        }
        assert result.by_language == {"arabizi": 40}
        assert result.by_script == {"latin": 40}
        assert result.by_source == {"synthetic_arabizi": 40}
        assert result.avg_messages == 2.0
        assert result.avg_chars > 0
        assert result.approx_total_tokens > 0

    def test_technical_ratio_is_computed(self) -> None:
        good = stats.analyze_file(RAW / "arabizi.jsonl")
        low = stats.analyze_file(RAW / "arabizi_low_technical.jsonl")
        assert good.technical_ratio == pytest.approx(0.60)
        assert low.technical_ratio == pytest.approx(0.10)

    def test_malformed_records_are_counted_not_ignored(self) -> None:
        result = stats.analyze_file(RAW / "malformed.jsonl")
        assert result.malformed_records == 11
        assert result.total_examples == 1


# --------------------------------------------------------------------------- #
# 4. Retention selection
# --------------------------------------------------------------------------- #


class TestRetentionSelection:
    """Category targets, determinism, holdout separation."""

    def test_spec_matches_the_official_specification(self) -> None:
        spec = retention.load_retention_spec()
        assert spec.target_examples == 20000
        assert spec.categories == {
            "mathematics": 5000,
            "coding": 5000,
            "reasoning": 4000,
            "instruction_following": 3000,
            "knowledge_qa": 3000,
        }
        assert sum(spec.categories.values()) == 20000
        assert spec.holdout_examples == 2500
        assert spec.allowed_languages == ("en",)

    def test_config_targets_map_to_canonical_categories(self) -> None:
        targets = retention.load_retention_spec().canonical_targets()
        assert targets["general_instruction"] == 3000
        assert "instruction_following" not in targets
        assert sum(targets.values()) == 20000

    def test_retention_is_not_tunisian_adaptation(self) -> None:
        spec = retention.load_retention_spec()
        assert spec.purpose == "english_capability_preservation"
        assert spec.is_tunisian_adaptation is False

    def test_scaled_selection_hits_exact_category_targets(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        selection = retention.select_retention(retention_pool, scale=0.01)
        assert selection.train_counts == {
            "coding": 50,
            "general_instruction": 30,
            "knowledge_qa": 30,
            "mathematics": 50,
            "reasoning": 40,
        }
        assert len(selection.train) == 200
        assert len(selection.holdout) == 25

    def test_selection_is_deterministic(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        first = retention.select_retention(retention_pool, scale=0.01)
        second = retention.select_retention(retention_pool, scale=0.01)
        assert [r["id"] for r in first.train] == [r["id"] for r in second.train]
        assert [r["id"] for r in first.holdout] == [r["id"] for r in second.holdout]

    def test_different_seed_changes_selection(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        first = retention.select_retention(retention_pool, scale=0.01, seed=42)
        second = retention.select_retention(retention_pool, scale=0.01, seed=7)
        assert [r["id"] for r in first.train] != [r["id"] for r in second.train]

    def test_holdout_never_overlaps_train(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        selection = retention.select_retention(retention_pool, scale=0.01)
        train_ids = {r["id"] for r in selection.train}
        holdout_ids = {r["id"] for r in selection.holdout}
        assert train_ids.isdisjoint(holdout_ids)
        selection.assert_no_contamination()

    def test_non_english_candidates_are_rejected(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        """Tunisian records must never enter the English retention slice."""
        tunisian = schema.load_jsonl(RAW / "arabizi.jsonl").records
        selection = retention.select_retention(retention_pool + tunisian, scale=0.01)
        assert all(r["language"] == "en" for r in selection.train)
        assert selection.rejected.get("language_not_allowed") == len(tunisian)

    def test_insufficient_candidates_raises(self) -> None:
        """Never silently under-fill a category."""
        tiny = [
            good_record(id=f"tiny-{i:03d}", category="mathematics") for i in range(3)
        ]
        with pytest.raises(retention.SelectionError, match="insufficient"):
            retention.select_retention(tiny, scale=0.01)

    def test_selection_is_not_unconstrained_random(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        """Every requested category must be present at its exact target."""
        selection = retention.select_retention(retention_pool, scale=0.01)
        for category, target in selection.train_targets.items():
            assert selection.train_counts.get(category) == target

    def test_source_diversity_is_preserved(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        selection = retention.select_retention(retention_pool, scale=0.01)
        sources = {r["source"] for r in selection.train}
        assert len(sources) >= 2

    def test_manifest_records_provenance(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        selection = retention.select_retention(retention_pool, scale=0.01)
        manifest = selection.manifest(candidates=len(retention_pool), sources=["a", "b"])
        assert manifest["slice"] == "retention"
        assert manifest["purpose"] == "english_capability_preservation"
        assert manifest["seed"] == 42
        assert manifest["train"]["total"] == 200
        assert manifest["holdout"]["total"] == 25
        assert manifest["deterministic"] is True


# --------------------------------------------------------------------------- #
# 5. Train / holdout split
# --------------------------------------------------------------------------- #


class TestSplit:
    """Deterministic, contamination-free splitting."""

    def test_split_sizes_are_exact(self, retention_pool: List[Dict[str, Any]]) -> None:
        result = split.split_records(retention_pool, 30, seed=42)
        assert len(result.holdout) == 30
        assert len(result.train) == len(retention_pool) - 30

    def test_split_is_deterministic(self, retention_pool: List[Dict[str, Any]]) -> None:
        first = split.split_records(retention_pool, 30, seed=42)
        second = split.split_records(retention_pool, 30, seed=42)
        assert first.holdout_ids == second.holdout_ids

    def test_split_has_no_contamination(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        result = split.split_records(retention_pool, 30, seed=42)
        assert set(result.train_ids).isdisjoint(result.holdout_ids)

    def test_stratified_split_preserves_categories(
        self, retention_pool: List[Dict[str, Any]]
    ) -> None:
        result = split.stratified_split(retention_pool, 25, seed=42)
        assert len(result.holdout) == 25
        per_category: Dict[str, int] = {}
        for record in result.holdout:
            per_category[record["category"]] = per_category.get(record["category"], 0) + 1
        assert per_category == {
            "mathematics": 5,
            "coding": 5,
            "reasoning": 5,
            "general_instruction": 5,
            "knowledge_qa": 5,
        }

    def test_oversized_holdout_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds available"):
            split.split_records([good_record()], 5)

    def test_contamination_is_detected(self) -> None:
        shared = good_record(id="shared-0001")
        with pytest.raises(ValueError, match="contamination"):
            split.SplitResult(train=[shared], holdout=[dict(shared)])


# --------------------------------------------------------------------------- #
# 6. Mixture validation
# --------------------------------------------------------------------------- #


def load_slices() -> Dict[str, List[Dict[str, Any]]]:
    """Load the synthetic slices used by mixture tests."""
    return {
        "arabizi": schema.load_jsonl(RAW / "arabizi.jsonl").records,
        "arabic_derja": schema.load_jsonl(RAW / "arabic_derja.jsonl").records,
        "franco_tunisian": schema.load_jsonl(RAW / "franco_tunisian.jsonl").records,
        "msa_formal": schema.load_jsonl(RAW / "msa_formal.jsonl").records,
        "retention": schema.load_jsonl(RAW / "retention_pool.jsonl").records,
    }


class TestMixtureValidation:
    """The eight hard mixture constraints."""

    def test_official_cocktail_is_declared(self) -> None:
        spec = mixture.load_mixture_spec()
        assert spec.slices == {
            "arabizi": 0.35,
            "arabic_derja": 0.25,
            "franco_tunisian": 0.12,
            "msa_formal": 0.08,
            "retention": 0.20,
        }

    def test_mixture_total_is_one(self) -> None:
        spec = mixture.load_mixture_spec()
        assert sum(spec.slices.values()) == pytest.approx(1.0)

    def test_cross_cutting_minimums_are_declared(self) -> None:
        spec = mixture.load_mixture_spec()
        assert spec.arabizi_technical_min == 0.20
        assert spec.derja_technical_min == 0.20
        assert spec.technical_categories == ("mathematics", "reasoning", "coding")

    def test_valid_mixture_passes(self) -> None:
        report = mixture.validate_mixture(load_slices())
        assert report.ok, report.errors

    def test_missing_slice_fails(self) -> None:
        slices = load_slices()
        del slices["msa_formal"]
        report = mixture.validate_mixture(slices)
        assert not report.ok
        assert any("msa_formal" in error for error in report.errors)

    def test_arabizi_technical_quota_is_enforced(self) -> None:
        slices = load_slices()
        slices["arabizi"] = schema.load_jsonl(RAW / "arabizi_low_technical.jsonl").records
        report = mixture.validate_mixture(slices)
        assert not report.ok
        assert any(
            "arabizi" in error and "technical ratio" in error for error in report.errors
        )

    def test_derja_technical_quota_is_enforced(self) -> None:
        slices = load_slices()
        low = [
            dict(record, category="general_instruction")
            for record in slices["arabic_derja"]
        ]
        slices["arabic_derja"] = low
        report = mixture.validate_mixture(slices)
        assert not report.ok
        assert any(
            "arabic_derja" in error and "technical ratio" in error
            for error in report.errors
        )

    def test_arabizi_quota_passes_at_exactly_twenty_percent(self) -> None:
        """The boundary case must pass: >= 20%, not > 20%."""
        slices = load_slices()
        records = [
            dict(r, id=f"boundary-{i:04d}", category="mathematics" if i < 8 else "knowledge_qa")
            for i, r in enumerate(slices["arabizi"][:40])
        ]
        slices["arabizi"] = records
        report = mixture.validate_mixture(slices)
        assert report.slices["arabizi"]["technical_ratio"] == pytest.approx(0.20)
        assert report.ok, report.errors

    def test_msa_is_not_counted_as_retention(self) -> None:
        spec = mixture.load_mixture_spec()
        assert spec.slice_semantics["retention"] == "english_capability_preservation"
        assert spec.slice_semantics["msa_formal"] == "formal_register_coverage"
        assert spec.slice_semantics["msa_formal"] != spec.slice_semantics["retention"]

    def test_retention_must_be_english(self) -> None:
        slices = load_slices()
        slices["retention"] = schema.load_jsonl(RAW / "arabizi.jsonl").records
        report = mixture.validate_mixture(slices)
        assert not report.ok
        assert any("non-English" in error for error in report.errors)

    def test_holdout_leakage_is_detected(self) -> None:
        slices = load_slices()
        leaked = slices["retention"][0]["id"]
        report = mixture.validate_mixture(slices, holdout_ids=[leaked])
        assert not report.ok
        assert any("holdout" in error for error in report.errors)

    def test_infeasible_counts_are_detected(self) -> None:
        report = mixture.validate_mixture(
            load_slices(), requested_counts={"arabizi": 1_000_000}
        )
        assert not report.ok
        assert any("infeasible" in error for error in report.errors)

    def test_empty_slice_fails(self) -> None:
        slices = load_slices()
        slices["franco_tunisian"] = []
        report = mixture.validate_mixture(slices)
        assert not report.ok
        assert any("empty" in error for error in report.errors)

    def test_invalid_spec_is_rejected(self, tmp_path: Path) -> None:
        """A mixture that does not sum to 1.0 must be refused, not rebalanced."""
        bad = tmp_path / "mixture.yaml"
        bad.write_text(
            "slices:\n"
            "  arabizi: 0.50\n"
            "  arabic_derja: 0.25\n"
            "  franco_tunisian: 0.12\n"
            "  msa_formal: 0.08\n"
            "  retention: 0.20\n"
            "slice_semantics:\n"
            "  retention: english_capability_preservation\n"
            "  msa_formal: formal_register_coverage\n",
            encoding="utf-8",
        )
        with pytest.raises(mixture.MixtureError, match="sum to"):
            mixture.load_mixture_spec(bad)

    def test_report_raises_on_failure(self) -> None:
        slices = load_slices()
        slices["arabizi"] = schema.load_jsonl(RAW / "arabizi_low_technical.jsonl").records
        with pytest.raises(mixture.MixtureError, match="technical ratio"):
            mixture.validate_mixture(slices).raise_if_invalid()


# --------------------------------------------------------------------------- #
# 7. Soup integration surface
# --------------------------------------------------------------------------- #


class TestSoupInterface:
    """The dedup interface must emit only real Soup 0.73.3 syntax."""

    def test_command_uses_verified_soup_syntax(self) -> None:
        command = dedupe.build_soup_dedup_command("in.jsonl", "out.jsonl", threshold=0.85)
        assert command.argv[:4] == ("soup", "data", "dedup", "in.jsonl")
        assert "--output" in command.argv
        assert "--threshold" in command.argv
        assert "--semantic" not in command.argv

    def test_semantic_flag_is_optional(self) -> None:
        command = dedupe.build_soup_dedup_command("in.jsonl", semantic=True)
        assert "--semantic" in command.argv
        assert command.semantic is True

    def test_default_output_follows_soup_convention(self) -> None:
        command = dedupe.build_soup_dedup_command("data/in.jsonl")
        assert command.output_path.endswith("in_deduped.jsonl")

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            dedupe.build_soup_dedup_command("in.jsonl", threshold=1.5)

    def test_no_invented_flags(self) -> None:
        """Only flags that exist in Soup 0.73.3 may appear."""
        command = dedupe.build_soup_dedup_command(
            "in.jsonl", "out.jsonl", field="content", semantic=True
        )
        allowed = {"--output", "--threshold", "--field", "--semantic"}
        used = {arg for arg in command.argv if arg.startswith("--")}
        assert used <= allowed


# --------------------------------------------------------------------------- #
# 8. Export
# --------------------------------------------------------------------------- #


class TestExport:
    """Canonical -> Soup-compatible ShareGPT."""

    def test_roles_map_to_sharegpt(self) -> None:
        row = export.to_sharegpt(
            good_record(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            )
        )
        assert [turn["from"] for turn in row["conversations"]] == ["system", "human", "gpt"]
        assert [turn["value"] for turn in row["conversations"]] == ["sys", "q", "a"]

    def test_metadata_is_dropped_by_default(self) -> None:
        row = export.to_sharegpt(good_record())
        assert set(row) == {"conversations"}

    def test_metadata_can_be_preserved(self) -> None:
        row = export.to_sharegpt(good_record(), keep_metadata=True)
        assert row["id"] == "test-0001"
        assert row["category"] == "mathematics"

    def test_export_refuses_invalid_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="refusing to export"):
            export.export_records(RAW / "malformed.jsonl", tmp_path / "out.jsonl")

    def test_export_writes_valid_rows(self, tmp_path: Path) -> None:
        output = tmp_path / "sharegpt.jsonl"
        written = export.export_records(RAW / "arabizi.jsonl", output)
        assert written == 40
        rows = schema.load_jsonl(output).records
        assert all("conversations" in row for row in rows)
        assert all(len(row["conversations"]) == 2 for row in rows)


# --------------------------------------------------------------------------- #
# 9. End-to-end synthetic pipeline
# --------------------------------------------------------------------------- #


class TestEndToEndPipeline:
    """The full synthetic path, as orchestrated by scripts/prepare_data.sh."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        # 1. canonicalization + validation
        pool = schema.load_jsonl(RAW / "retention_pool.jsonl").records
        report = validate.validate_records(pool)
        assert report.ok

        # 2. statistics
        analysis = stats.compute_stats(pool)
        assert analysis.total_examples == 300

        # 3. duplicate handling
        deduped = tmp_path / "deduped.jsonl"
        kept, removed = dedupe.dedupe_by_id(RAW / "duplicates.jsonl", deduped)
        assert (kept, removed) == (3, 1)

        # 4. retention selection + holdout split
        selection = retention.select_retention(pool, scale=0.01)
        assert len(selection.train) == 200
        assert len(selection.holdout) == 25
        selection.assert_no_contamination()

        train_path = tmp_path / "train.jsonl"
        holdout_path = tmp_path / "holdout.jsonl"
        schema.write_jsonl(train_path, selection.train)
        schema.write_jsonl(holdout_path, selection.holdout)

        # 5. mixture validation with holdout leakage protection
        slices = load_slices()
        slices["retention"] = selection.train
        mixture_report = mixture.validate_mixture(
            slices, holdout_ids=[r["id"] for r in selection.holdout]
        )
        assert mixture_report.ok, mixture_report.errors

        # 6. Soup-compatible export
        final = tmp_path / "final.jsonl"
        written = export.export_records(train_path, final)
        assert written == 200
        rows = schema.load_jsonl(final).records
        assert all("conversations" in row for row in rows)

    def test_holdout_is_never_in_training_output(self, tmp_path: Path) -> None:
        pool = schema.load_jsonl(RAW / "retention_pool.jsonl").records
        selection = retention.select_retention(pool, scale=0.01)
        train_path = tmp_path / "train.jsonl"
        holdout_path = tmp_path / "holdout.jsonl"
        schema.write_jsonl(train_path, selection.train)
        schema.write_jsonl(holdout_path, selection.holdout)

        train_ids = {r["id"] for r in schema.load_jsonl(train_path).records}
        holdout_ids = {r["id"] for r in schema.load_jsonl(holdout_path).records}
        assert train_ids.isdisjoint(holdout_ids)

    def test_written_output_is_reproducible(self, tmp_path: Path) -> None:
        pool = schema.load_jsonl(RAW / "retention_pool.jsonl").records
        first_path = tmp_path / "a.jsonl"
        second_path = tmp_path / "b.jsonl"
        schema.write_jsonl(first_path, retention.select_retention(pool, scale=0.01).train)
        schema.write_jsonl(second_path, retention.select_retention(pool, scale=0.01).train)
        assert first_path.read_bytes() == second_path.read_bytes()


# --------------------------------------------------------------------------- #
# 10. Guardrails
# --------------------------------------------------------------------------- #


class TestGuardrails:
    """Scope guarantees for Task 4A."""

    def test_synthetic_data_is_clearly_marked(self) -> None:
        """Every synthetic record must be filterable out of real data."""
        for name in ("arabizi.jsonl", "arabic_derja.jsonl", "retention_pool.jsonl"):
            for record in schema.load_jsonl(RAW / name).records:
                assert record["source"].startswith("synthetic")
                assert record.get("quality", {}).get("synthetic") is True

    def test_no_external_dataset_is_referenced(self) -> None:
        """No dataset-specific adapter or name may leak into config/code."""
        forbidden = [
            "esprit",
            "labess",
            "tounsibench",
            "metamathqa",
            "slimorca",
            "starcoder",
            "openhermes",
        ]
        targets = list((REPO_ROOT / "src" / "data").glob("*.py"))
        targets += list(CONFIGS.glob("*.yaml"))
        for path in targets:
            text = path.read_text(encoding="utf-8").lower()
            for name in forbidden:
                assert name not in text, f"{path.name} references '{name}'"

    def test_expected_summary_matches_fixtures(self) -> None:
        expected = json.loads(
            (SYNTHETIC / "expected" / "summary.json").read_text(encoding="utf-8")
        )
        assert expected["counts"]["retention_pool.jsonl"] == 300
        assert expected["technical_ratio"]["arabizi"] == 0.6
        assert expected["technical_ratio"]["arabizi_low_technical"] == 0.1
        assert expected["duplicates"]["duplicate_ids"] == 1
