"""Validate cleaned CSVs in data/processed have the expected OHLCV structure.

Run with: pytest
If data/processed is empty (pipeline hasn't been run yet), these tests skip
rather than fail.
"""
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "tick_volume"}


def _processed_files():
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("*.csv"))


@pytest.mark.parametrize("csv_path", _processed_files(), ids=lambda p: p.name)
def test_processed_csv_structure(csv_path):
    df = pd.read_csv(csv_path)

    assert not df.empty, f"{csv_path.name} is empty"

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"{csv_path.name} is missing columns: {sorted(missing)}"

    assert df["time"].is_unique, f"{csv_path.name} has duplicate timestamps"

    times = pd.to_datetime(df["time"])
    assert times.is_monotonic_increasing, f"{csv_path.name} timestamps are not sorted ascending"

    for col in ["open", "high", "low", "close"]:
        assert df[col].notna().all(), f"{csv_path.name} has missing values in '{col}'"


def test_processed_dir_has_files():
    if not _processed_files():
        pytest.skip("data/processed is empty - run the data pipeline first")
