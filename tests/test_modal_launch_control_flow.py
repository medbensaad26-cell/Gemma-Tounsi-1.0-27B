#!/usr/bin/env python3
"""Local control-flow tests for Modal training launch gating.

Mocks CPU preflight and training callables. Never allocates a Modal GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import modal_train_custom_75_25_english as m  # noqa: E402


def _pass_result(**overrides: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "overall": "PASS",
        "gpu_requested": False,
        "training_exists": True,
        "validation_exists": True,
    }
    result.update(overrides)
    return result


class TestValidatePreflightLaunchResult:
    def test_pass_returns_result(self):
        payload = _pass_result()
        assert m.validate_preflight_launch_result(payload) is payload

    def test_missing_result_raises(self):
        with pytest.raises(RuntimeError, match="missing result"):
            m.validate_preflight_launch_result(None)

    def test_non_dict_raises(self):
        with pytest.raises(RuntimeError, match="JSON-compatible dict"):
            m.validate_preflight_launch_result("not-a-dict")

    def test_invalid_json_raises(self):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            m.validate_preflight_launch_result(
                {"overall": "PASS", "bad": {1, 2, 3}}
            )

    def test_fail_overall_raises(self):
        with pytest.raises(RuntimeError, match="refusing to allocate"):
            m.validate_preflight_launch_result(_pass_result(overall="FAIL"))


class TestRunLaunchSequence:
    def test_successful_preflight_invokes_training_exactly_once(self):
        train_calls: List[Dict[str, Any]] = []

        def preflight() -> Dict[str, Any]:
            return _pass_result()

        def train_fn(
            *,
            max_runtime_seconds: int,
            stop_after_steps: Optional[int],
        ) -> str:
            train_calls.append(
                {
                    "max_runtime_seconds": max_runtime_seconds,
                    "stop_after_steps": stop_after_steps,
                }
            )
            return "fc-mock-train-1"

        result = m.run_launch_sequence(
            preflight_callable=preflight,
            train_callable=train_fn,
            max_runtime_seconds=123,
            stop_after_steps=10,
        )

        assert result == "fc-mock-train-1"
        assert len(train_calls) == 1
        assert train_calls[0] == {
            "max_runtime_seconds": 123,
            "stop_after_steps": 10,
        }

    def test_failed_preflight_invokes_training_zero_times(self):
        train_fn = MagicMock(return_value="should-not-run")

        with pytest.raises(RuntimeError, match="refusing to allocate"):
            m.run_launch_sequence(
                preflight_callable=lambda: _pass_result(overall="FAIL"),
                train_callable=train_fn,
                max_runtime_seconds=82800,
                stop_after_steps=None,
            )

        train_fn.assert_not_called()

    def test_missing_result_invokes_training_zero_times(self):
        train_fn = MagicMock(return_value="should-not-run")

        with pytest.raises(RuntimeError, match="missing result"):
            m.run_launch_sequence(
                preflight_callable=lambda: None,
                train_callable=train_fn,
                max_runtime_seconds=82800,
                stop_after_steps=None,
            )

        train_fn.assert_not_called()

    def test_invalid_json_invokes_training_zero_times(self):
        train_fn = MagicMock(return_value="should-not-run")

        with pytest.raises(RuntimeError, match="invalid JSON"):
            m.run_launch_sequence(
                preflight_callable=lambda: {
                    "overall": "PASS",
                    "bad": {1, 2, 3},
                },
                train_callable=train_fn,
                max_runtime_seconds=82800,
                stop_after_steps=None,
            )

        train_fn.assert_not_called()

    def test_success_does_not_terminate_before_training_call(self):
        """PASS path must reach train_callable; no early exit after preflight."""
        events: List[str] = []

        def preflight() -> Dict[str, Any]:
            events.append("preflight")
            return _pass_result()

        def train_fn(**kwargs: Any) -> str:
            events.append("train")
            return "submitted"

        result = m.run_launch_sequence(
            preflight_callable=preflight,
            train_callable=train_fn,
            max_runtime_seconds=1,
            stop_after_steps=None,
        )

        assert result == "submitted"
        assert events == ["preflight", "train"]

    def test_preflight_exception_skips_training(self):
        train_fn = MagicMock(return_value="should-not-run")

        def preflight() -> Dict[str, Any]:
            raise RuntimeError("Preflight failed: remote timeout")

        with pytest.raises(RuntimeError, match="remote timeout"):
            m.run_launch_sequence(
                preflight_callable=preflight,
                train_callable=train_fn,
                max_runtime_seconds=82800,
                stop_after_steps=None,
            )

        train_fn.assert_not_called()


class TestInvokeTrainRemoteWiring:
    def test_invoke_train_remote_calls_train_remote_once(self, monkeypatch):
        remote = MagicMock(return_value="remote-done")
        fake_train = MagicMock()
        fake_train.remote = remote
        monkeypatch.setattr(m, "train", fake_train)

        result = m._invoke_train_remote(
            max_runtime_seconds=42,
            stop_after_steps=7,
        )

        assert result == "remote-done"
        remote.assert_called_once_with(
            max_runtime_seconds=42,
            stop_after_steps=7,
        )

    def test_main_uses_launch_sequence_not_bare_preflight_only(self):
        """Source must route main() through run_launch_sequence + train.remote."""
        source = Path(m.__file__).read_text(encoding="utf-8")
        assert "run_launch_sequence(" in source
        assert "train_callable=_invoke_train_remote" in source
        assert "train.remote(" in source
        # Detach must not be treated as automatic training start.
        assert "does **not** auto-start training" in source
