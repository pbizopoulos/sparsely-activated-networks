# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the public page controls and links."""

import os
from pathlib import Path

from bs4 import BeautifulSoup


def test_signal_length_slider_declares_a_range_of_2_to_100_samples() -> None:
    """Check the bounds exposed by the interactive signal control."""
    page = BeautifulSoup(Path(os.environ["PAGE"]).read_text(), "html.parser")
    control = page.select_one("input#size-input-range")
    if control is None or (
        control.get("type"),
        control.get("min"),
        control.get("max"),
    ) != ("range", "2", "100"):
        raise AssertionError(control)
