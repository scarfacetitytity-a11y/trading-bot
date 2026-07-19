"""Pull M15 + M5 for the expanded scan universe (metals, indices, energy).

New symbols beyond the core 10 — for the full all-pairs monthly scan. Includes
palladium (XPDUSD) and platinum (XPTUSD) as silver-replacement candidates.

Usage (MT5 open):
    python -m backtests.pull_scan
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

SCAN_NEW = [
    "XPDUSD", "XPTUSD", "XCUUSD",                    # palladium, platinum, copper
    "FRA40.cash", "HK50.cash", "AUS200.cash",        # indices
    "EU50.cash", "SPN35.cash",
    "USOIL.cash", "UKOIL.cash",                      # energy
]
NUM_BARS = 99_000


def main() -> int:
    cfg = load_config()
    setup_logger("", log_dir="logs", log_file="pull_scan.log", level="INFO")
    if not connect(cfg.get("mt5", {}).get("terminal_path") or None):
        logger.error("Could not connect to MT5."); return 1
    time.sleep(3)

    results = []
    try:
        for symbol in SCAN_NEW:
            for tf in ("M15", "M5"):
                logger.info("Pulling %s %s", symbol, tf)
                df = fetch_symbol_data(symbol, tf, None, None, num_bars=NUM_BARS)
                if df is None:
                    results.append((f"{symbol} {tf}", "FAILED - no data")); continue
                try:
                    raw = save_raw(df, symbol, tf)
                    cleaned = clean_data(raw, tf)
                    save_processed(cleaned, symbol, tf)
                    results.append((f"{symbol} {tf}", f"OK {len(cleaned):,} bars"))
                except Exception as exc:
                    results.append((f"{symbol} {tf}", f"FAILED ({exc})"))
    finally:
        disconnect()

    print("\n=== Scan Pull Summary ===")
    ok = sum(1 for _, s in results if s.startswith("OK"))
    for name, status in results:
        print(f"  [{'OK' if status.startswith('OK') else '!!'}] {name:<18} {status}")
    print(f"\n  {ok}/{len(results)} pulled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
