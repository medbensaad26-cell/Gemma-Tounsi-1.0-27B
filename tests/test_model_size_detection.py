"""Regression tests for Soup 0.73.3 model-size detection backport.

These tests verify that `model_size_from_name()` correctly identifies model
sizes from Hugging Face model identifiers, without substring collisions
(e.g. "7b" must not match inside "27b").

Run inside the patched Soup 0.73.3 Docker container:

    docker compose run --rm soup-cpu python -m pytest tests/test_model_size_detection.py -v
"""

from __future__ import annotations

import pytest

from soup_cli.utils.gpu import model_size_from_name


# --------------------------------------------------------------------------- #
# Core size-detection tests
# --------------------------------------------------------------------------- #

MODEL_SIZE_CASES = [
    # (model_identifier, expected_size_billions)
    ("google/gemma-3-27b-it", 27.0),   # THE bug fix: 27b must not match as 7b
    ("mistralai/Mistral-7B-v0.1", 7.0),
    ("meta-llama/Llama-3.1-8B", 8.0),
    ("model-17b", 17.0),
    ("model-70b", 70.0),
    ("model-1.7b", 1.7),
    ("model-0.8b", 0.8),
    ("google/gemma-3-7b-it", 7.0),
    ("google/gemma-3-4b-it", 7.0),   # 4b not in markers -> falls to default 7B
    ("meta-llama/Llama-3.1-70B", 70.0),
    ("bigscience/bloom-17b", 17.0),
    ("facebook/opt-1.7b", 1.7),
    ("unknown-model-xyz", 7.0),      # unknown -> default 7B
]


@pytest.mark.parametrize("model_id,expected", MODEL_SIZE_CASES)
def test_model_size_from_name(model_id: str, expected: float) -> None:
    """Each model identifier must map to its expected size in billions."""
    result = model_size_from_name(model_id)
    assert result == expected, (
        f"{model_id!r} -> {result}B, expected {expected}B"
    )


# --------------------------------------------------------------------------- #
# Specific collision tests
# --------------------------------------------------------------------------- #

def test_27b_does_not_resolve_to_7b() -> None:
    """The core bug fix: gemma-3-27b-it must return 27, never 7."""
    assert model_size_from_name("google/gemma-3-27b-it") == 27.0
    assert model_size_from_name("google/gemma-3-27b-it") != 7.0


def test_7b_does_not_match_inside_27b() -> None:
    """A model named with 27b must not be misidentified as 7b."""
    result = model_size_from_name("some-model-27b-variant")
    assert result == 27.0, f"27b variant returned {result}B, expected 27B"


def test_1_7b_does_not_resolve_to_7b() -> None:
    """1.7b must return 1.7, not 7 (1.7b contains '7b' as substring)."""
    assert model_size_from_name("model-1.7b") == 1.7
    assert model_size_from_name("model-1.7b") != 7.0


def test_8b_does_not_match_inside_18b() -> None:
    """If a hypothetical 18b model appears, 8b must not match inside it."""
    result = model_size_from_name("model-18b")
    assert result == 7.0, f"18b (unknown size) returned {result}B, expected default 7B"


def test_case_insensitive_matching() -> None:
    """Model size detection must be case-insensitive."""
    assert model_size_from_name("GOOGLE/GEMMA-3-27B-IT") == 27.0
    assert model_size_from_name("Google/Gemma-3-27B-It") == 27.0
    assert model_size_from_name("META-LLAMA/LLAMA-3.1-8B") == 8.0


def test_unknown_model_returns_default() -> None:
    """Unknown model names must fall back to the 7B default."""
    assert model_size_from_name("completely-unknown-model") == 7.0
    assert model_size_from_name("random-name-here") == 7.0


def test_existing_supported_models_unchanged() -> None:
    """All previously supported model identifiers must keep their sizes."""
    assert model_size_from_name("mistralai/Mistral-7B-v0.1") == 7.0
    assert model_size_from_name("meta-llama/Llama-3.1-8B") == 8.0
    assert model_size_from_name("meta-llama/Llama-3.1-70B") == 70.0
    assert model_size_from_name("bigscience/bloom-17b") == 17.0
    assert model_size_from_name("facebook/opt-1.7b") == 1.7


def test_decimal_sizes_preserved() -> None:
    """Decimal model sizes (1.7b, 0.8b) must be preserved exactly."""
    assert model_size_from_name("model-1.7b") == 1.7
    assert model_size_from_name("model-0.8b") == 0.8
