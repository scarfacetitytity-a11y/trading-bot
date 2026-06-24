"""Download historical OHLCV data from MT5 and save it to data/raw as CSV."""
from datetime import datetime
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def fetch_symbol_data(
    symbol: str,
    timeframe_str: str,
    start_date: Optional[str],
    end_date: Optional[str],
    num_bars: int,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles for one symbol.

    If both start_date and end_date are set (YYYY-MM-DD), fetches that exact
    range. Otherwise fetches the most recent `num_bars` candles.
    Returns None (and logs the reason) if the data cannot be fetched.
    """
    timeframe = TIMEFRAME_MAP.get(timeframe_str)
    if timeframe is None:
        logger.error(
            "Unknown timeframe '%s' for %s. Valid options: %s",
            timeframe_str, symbol, ", ".join(TIMEFRAME_MAP),
        )
        return None

    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(
            "Symbol '%s' not found on this broker. Check the exact symbol "
            "name in MT5 Market Watch (brokers often use suffixes, e.g. "
            "'US30.cash' or 'XAUUSD.a').",
            symbol,
        )
        return None

    if not info.visible and not mt5.symbol_select(symbol, True):
        logger.error("Could not add '%s' to Market Watch: %s", symbol, mt5.last_error())
        return None

    if start_date and end_date:
        date_from = datetime.strptime(start_date, "%Y-%m-%d")
        date_to = datetime.strptime(end_date, "%Y-%m-%d")
        logger.info("Requesting %s %s from %s to %s", symbol, timeframe_str, start_date, end_date)
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    else:
        logger.info("Requesting last %d bars of %s %s", num_bars, symbol, timeframe_str)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    if rates is None or len(rates) == 0:
        logger.error("No data returned for %s %s: %s", symbol, timeframe_str, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    first_time = pd.to_datetime(df["time"].iloc[0], unit="s")
    last_time = pd.to_datetime(df["time"].iloc[-1], unit="s")
    logger.info("Received %d bars for %s (%s -> %s)", len(df), symbol, first_time, last_time)
    return df


def save_raw(df: pd.DataFrame, symbol: str, timeframe_str: str) -> Path:
    """Save a raw OHLCV DataFrame to data/raw/<symbol>_<timeframe>.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{symbol}_{timeframe_str}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved raw data to %s", out_path)
    return out_path
