"""Pull M5 data for all TM-backtest pairs.

Downloads M5 bars for the full TM instrument universe and saves them to
data/processed ready for backtesting. Run with MT5 open and logged in.

Uses copy_rates_from_pos(99_000) — the confirmed FTMO Demo terminal cap.
copy_rates_range is broken in MT5 build 5836 (returns Invalid params).

Usage:
    python -m backtests.pull_m5
"""
import sys
import time
import logging
from pathlib import Path

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import fetch_symbol_data, save_raw
from backtests.data_cleaner import clean_data, save_processed

logger = logging.getLogger(__name__)

# Full TM universe — every symbol that needs M5 supervision
SYMBOLS = [
    "XAUUSD",       # metals — core, already has M5
    "XAGUSD",       # silver — bidirectional, needs M5
    "GBPUSD",       # forex — bidirectional, needs M5
    "US100.cash",   # index — already has M5
    "US30.cash",    # index — needs M5
    "US500.cash",   # index — needs M5
    "US2000.cash",  # index — needs M5 (capped at score=6 until pulled)
    "UK100.cash",   # index — needs M5
    "JP225.cash",   # index — needs M5
    "GER40.cash",   # index — needs M5
]

TIMEFRAME = "M5"
# FTMO Demo terminal cap: copy_rates_from_pos fails above 99,000 bars
# 99k M5 bars ≈ 17 months of history — sufficient for TM backtest coverage
NUM_BARS  = 99_000


def main() -> int:
    cfg     = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logger("", log_dir=log_cfg.get("log_dir", "logs"),
                 log_file="pull_m5.log", level="INFO")

    terminal_path = cfg.get("mt5", {}).get("terminal_path") or None
    if not connect(terminal_path):
        logger.error("Could not connect to MT5.")
        return 1

    # Give MT5 terminal a moment to fully initialise history service after connect
    logger.info("Waiting 3s for MT5 history service to initialise...")
    time.sleep(3)

    results = []
    try:
        for symbol in SYMBOLS:
            logger.info("Pulling %s M5  last %d bars", symbol, NUM_BARS)
            df = fetch_symbol_data(symbol, TIMEFRAME, None, None, num_bars=NUM_BARS)
            if df is None:
                logger.warning("No data returned for %s", symbol)
                results.append((symbol, "FAILED — no data"))
                continue
            try:
                raw_path = save_raw(df, symbol, TIMEFRAME)
                cleaned  = clean_data(raw_path, TIMEFRAME)
                save_processed(cleaned, symbol, TIMEFRAME)
                first = str(cleaned["time"].iloc[0])  if "time" in cleaned.columns else "?"
                last  = str(cleaned["time"].iloc[-1]) if "time" in cleaned.columns else "?"
                results.append((symbol, f"OK  {len(cleaned):,} bars  {first[:10]} -> {last[:10]}"))
            except Exception as exc:
                logger.error("%s processing failed: %s", symbol, exc)
                results.append((symbol, f"FAILED ({exc})"))
    finally:
        disconnect()

    print("\n=== M5 Pull Summary ===")
    ok = 0
    for sym, status in results:
        tag = "OK" if status.startswith("OK") else "!!"
        print(f"  [{tag}] {sym:<16} {status}")
        if status.startswith("OK"):
            ok += 1
    print(f"\n  {ok}/{len(SYMBOLS)} symbols pulled successfully.")
    print("  Next: python -m backtests.tm_backtest\n")

    return 0 if ok == len(SYMBOLS) else 1


if __name__ == "__main__":
    sys.exit(main())
