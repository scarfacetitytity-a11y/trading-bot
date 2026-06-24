"""Clean raw MT5 CSV exports and save validated data to data/processed."""
from pathlib import Path

import pandas as pd
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "tick_volume"}

# Expected seconds between candles, used to detect gaps. MN1 is omitted
# because month lengths vary.
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}


def clean_data(raw_path: Path, timeframe_str: str) -> pd.DataFrame:
    """Load a raw CSV, fix the time column, drop duplicates, sort, and
    report any gaps. Returns the cleaned DataFrame.
    """
    df = pd.read_csv(raw_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{raw_path.name} is missing required columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], unit="s")

    before = len(df)
    df = df.drop_duplicates(subset="time")
    duplicates_removed = before - len(df)

    df = df.sort_values("time").reset_index(drop=True)

    expected_seconds = TIMEFRAME_SECONDS.get(timeframe_str)
    gap_count = 0
    if expected_seconds and len(df) > 1:
        diffs = df["time"].diff().dt.total_seconds().dropna()
        gaps = diffs[diffs > expected_seconds]
        gap_count = len(gaps)
        if gap_count:
            max_gap_hours = gaps.max() / 3600
            logger.warning(
                "%s: %d gap(s) larger than the expected %s interval "
                "(largest ~%.1f hours). Some gaps are normal (weekends/holidays "
                "when the market is closed) - inspect if this seems too large or too frequent.",
                raw_path.name, gap_count, timeframe_str, max_gap_hours,
            )

    logger.info(
        "%s: %d rows -> %d after cleaning (%d duplicates removed, %d gap(s) detected)",
        raw_path.name, before, len(df), duplicates_removed, gap_count,
    )

    return df


def save_processed(df: pd.DataFrame, symbol: str, timeframe_str: str) -> Path:
    """Save a cleaned OHLCV DataFrame to data/processed/<symbol>_<timeframe>.csv."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"{symbol}_{timeframe_str}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)
    return out_path
