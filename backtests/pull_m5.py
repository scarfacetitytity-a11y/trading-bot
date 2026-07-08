"""Pull M5 data for ICT AMD backtesting.

Downloads 3 years of M5 bars (2022-2025) for the AMD strategy pairs
and saves them to data/processed ready for backtesting.

Usage (MT5 open and logged in):
    python -m backtests.pull_m5
"""
import sys
import logging
from pathlib import Path

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import fetch_symbol_data, save_raw
from backtests.data_cleaner import clean_data, save_processed

logger = logging.getLogger(__name__)

SYMBOLS    = ["XAUUSD", "GBPUSD", "US100.cash", "US30.cash", "EURUSD"]
TIMEFRAME  = "M5"
START_DATE = None
END_DATE   = None
NUM_BARS   = 200_000   # ~2.5 years of M5 (252 days * 24h * 12 bars/h ~= 72k/year)


def main() -> int:
    cfg     = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logger("", log_dir=log_cfg.get("log_dir", "logs"),
                 log_file="pull_m5.log", level="INFO")

    terminal_path = cfg.get("mt5", {}).get("terminal_path") or None
    if not connect(terminal_path):
        logger.error("Could not connect to MT5.")
        return 1

    results = []
    try:
        for symbol in SYMBOLS:
            logger.info("Pulling %s %s  %s -> %s", symbol, TIMEFRAME, START_DATE, END_DATE)
            df = fetch_symbol_data(symbol, TIMEFRAME, START_DATE, END_DATE, num_bars=NUM_BARS)
            if df is None:
                results.append((symbol, "FAILED"))
                continue
            try:
                raw_path = save_raw(df, symbol, TIMEFRAME)
                cleaned  = clean_data(raw_path, TIMEFRAME)
                save_processed(cleaned, symbol, TIMEFRAME)
                results.append((symbol, f"OK  {len(cleaned):,} bars"))
            except Exception as exc:
                logger.error("%s cleaning failed: %s", symbol, exc)
                results.append((symbol, f"FAILED ({exc})"))
    finally:
        disconnect()

    print("\n=== M5 Data Pull Summary ===")
    for sym, status in results:
        print(f"  {sym:<16} {status}")
    print()
    failed = [s for s, st in results if "FAILED" in st]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
