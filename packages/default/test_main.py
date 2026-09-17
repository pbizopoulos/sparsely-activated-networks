# Copyright (c) 2026- Paschalis Bizopoulos
"""Tests for default."""

from __future__ import annotations

from packages.default.main import (
    _OUT_PATH,
    main,
)


def test_main() -> None:
    """Generate the test artifacts and compile the manuscript."""
    main()
    if not (_OUT_PATH / "keys-values.csv").is_file():
        msg = "Artifact generation did not produce keys-values.csv"
        raise AssertionError(msg)
    if not (_OUT_PATH / "ms.pdf").is_file():
        msg_0 = "Manuscript compilation did not produce ms.pdf"
        raise AssertionError(msg_0)
