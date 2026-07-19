"""Pull M1 data for all TM-backtest pairs — finest-grain intrabar resolution.

M1 bars are needed to resolve SL/TP exits honestly: with structural stops as tight
as 0.5xATR, both SL and TP often fall inside one M15 (or even M5) bar, so the true
outcome depends on which was touched FIRST — only visible on M1.

COVERAGE LIMIT: copy_rates_from_pos caps at 99,000 bars on this FTMO Demo build.
99,000 M1 bars ~= 68 days. So M1 intrabar resolution is only available for roughly
the last ~2 months. For older trades the finest available bar is M5 — the resolved
backtest must fall back to M5 outside the M1 window.

Usage (MT5 must be open and logged in):
    python -m backtests.pull_m1
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
from backtests.pull_m5 import SYMBOLS

logger = logging.getLogger(__name__)

TIMEFRAME = "M1"
NUM_BARS  = 99_000   # ~68 days of M1 — the terminal cap


def main() -> int:
    cfg     = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logger("", log_dir=log_cfg.get("log_dir", "logs"),
                 log_file="pull_m1.log", level="INFO")

    terminal_path = cfg.get("mt5", {}).get("terminal_path") or None
    if not connect(terminal_path):
        logger.error("Could not connect to MT5.")
        return 1

    logger.info("Waiting 3s for MT5 history service to initialise...")
    time.sleep(3)

    results = []
    try:
        for symbol in SYMBOLS:
            logger.info("Pulling %s M1  last %d bars (~68 days)", symbol, NUM_BARS)
            df = fetch_symbol_data(symbol, TIMEFRAME, None, None, num_bars=NUM_BARS)
            if df is None:
                results.append((symbol, "FAILED - no data"))
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

    print("\n=== M1 Pull Summary ===")
    ok = 0
    for sym, status in results:
        tag = "OK" if status.startswith("OK") else "!!"
        print(f"  [{tag}] {sym:<16} {status}")
        if status.startswith("OK"):
            ok += 1
    print(f"\n  {ok}/{len(SYMBOLS)} symbols pulled.  Next: the resolved backtest engine.\n")
    return 0 if ok == len(SYMBOLS) else 1


if __name__ == "__main__":
    sys.exit(main())
