"""Runtime environment contracts for SAM3-facing entrypoints."""

from __future__ import annotations

import os


def assert_expected_conda_env(expected: str = "sam3", context: str = "this command") -> None:
    """Enforce the active conda env via ``CONDA_DEFAULT_ENV`` only.

    Set ``IGNORE_ENV_CONTRACT=1`` to bypass the check in tests/CI harnesses.
    """
    if os.environ.get("IGNORE_ENV_CONTRACT") == "1":
        return

    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active == expected:
        return

    active_display = active if active else "<unset>"
    raise RuntimeError(
        f"{context} must run inside conda env '{expected}' "
        f"(CONDA_DEFAULT_ENV={active_display}). Relaunch with "
        f"`conda run --no-capture-output -n {expected} ...`."
    )
