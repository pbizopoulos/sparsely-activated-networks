# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the generated manuscript against its expected SHA-256 hash."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import generate_paper

if TYPE_CHECKING:
    import pytest


_EXPECTED_MS_PDF_SHA256 = (
    "7da637986784d58ab002d380f066b4e080fbf819f0de5e7e50f4846f6769fd8e"
)


def test_generated_manuscript_pdf_matches_the_expected_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate the manuscript and compare its bytes with the pinned baseline."""
    output = tmp_path / generate_paper.RUNTIME_DIR.relative_to(Path.home())
    monkeypatch.setattr(generate_paper, "RUNTIME_DIR", output)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    generate_paper.main()
    repository = "sparsely-activated-networks"
    pdf = tmp_path / "github.com/pbizopoulos" / repository / "generate_paper/tmp/ms.pdf"
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual != _EXPECTED_MS_PDF_SHA256:
        message = f"ms.pdf SHA-256: expected {_EXPECTED_MS_PDF_SHA256}, got {actual}"
        raise AssertionError(message)
