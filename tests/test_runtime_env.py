from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runtime_env import assert_expected_conda_env


def test_assert_expected_conda_env_passes_when_env_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "sam3")
    monkeypatch.delenv("IGNORE_ENV_CONTRACT", raising=False)
    assert_expected_conda_env()


def test_assert_expected_conda_env_raises_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("IGNORE_ENV_CONTRACT", raising=False)
    with pytest.raises(RuntimeError, match="conda run --no-capture-output -n sam3"):
        assert_expected_conda_env(context="unit test")


def test_assert_expected_conda_env_raises_when_env_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.delenv("IGNORE_ENV_CONTRACT", raising=False)
    with pytest.raises(RuntimeError, match="CONDA_DEFAULT_ENV=base"):
        assert_expected_conda_env(context="unit test")


def test_assert_expected_conda_env_allows_test_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setenv("IGNORE_ENV_CONTRACT", "1")
    assert_expected_conda_env()
