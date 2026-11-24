# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the waveform annotation labels handled by the packaged pandas patch."""

from typing import TYPE_CHECKING

import numpy as np

import wfdb

if TYPE_CHECKING:
    from pathlib import Path


def test_custom_annotation_labels_preserve_their_symbols_and_sample_positions(
    tmp_path: Path,
) -> None:
    """Write and read a custom beat label through the public annotation interface."""
    wfdb.wrann(
        "example",
        "atr",
        sample=np.array([5, 12]),
        symbol=["N", "Z"],
        custom_labels=[(42, "Z", "Custom beat")],
        write_dir=str(tmp_path),
    )
    result = wfdb.rdann(
        str(tmp_path / "example"),
        "atr",
        return_label_elements=["symbol", "description"],
    )
    np.testing.assert_array_equal(result.sample, [5, 12])
    if result.symbol != ["N", "Z"] or result.description[1] != "Custom beat":
        raise AssertionError((result.symbol, result.description))
