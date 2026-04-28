from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_env_contract_bypass_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default test harness bypass for strict SAM3 env checks."""
    monkeypatch.setenv("IGNORE_ENV_CONTRACT", "1")
