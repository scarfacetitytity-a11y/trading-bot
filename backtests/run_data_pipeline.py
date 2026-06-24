"""Entry point: download MT5 historical data and produce cleaned CSVs.

Read-only. Connects to an already-running, already-logged-in MT5 terminal,
downloads OHLCV history for each configured symbol into data/raw, cleans it
into data/processed, and logs a summary. Does not place any trades.

Run from the project root with the virtual environment activated:
    python -m backtests.run_data_pipeline
"""
import logging
import sys

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests import mt5_connector, data_loader, data_cleaner

logger = logging.getLogger(__name__)


def main() -> int:
    config = load_config()

    log_cfg = config.get("logging", {})
    # Configure the root logger so messages from mt5_connector, data_loader,
    # and data_cleaner (which each use logging.getLogger(__name__)) are also
    # written to the log file with the same formatting.
    setup_logger(
        "",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "data_pipeline.log"),
        level=log_cfg.get("level", "INFO"),
    )

    data_cfg = config.get("data", {})
    symbols = data_cfg.get("symbols", [])
    timeframe_str = data_cfg.get("timeframe", "M15")
    start_date = data_cfg.get("start_date") or None
    end_date = data_cfg.get("end_date") or None
    num_bars = data_cfg.get("num_bars", 1000)

    mt5_cfg = config.get("mt5", {})
    terminal_path = mt5_cfg.get("terminal_path") or None

    if not symbols:
        logger.error("No symbols configured in config.yaml under data.symbols")
        return 1

    if not mt5_connector.connect(terminal_path):
        logger.error("Aborting: could not establish a valid MT5 connection.")
        return 1

    results = []
    try:
        for symbol in symbols:
            logger.info("--- Processing %s ---", symbol)

            df = data_loader.fetch_symbol_data(symbol, timeframe_str, start_date, end_date, num_bars)
            if df is None:
                results.append((symbol, "FAILED (download)"))
                continue

            raw_path = data_loader.save_raw(df, symbol, timeframe_str)

            try:
                cleaned = data_cleaner.clean_data(raw_path, timeframe_str)
            except ValueError as exc:
                logger.error("%s: cleaning failed: %s", symbol, exc)
                results.append((symbol, "FAILED (cleaning)"))
                continue

            data_cleaner.save_processed(cleaned, symbol, timeframe_str)
            results.append((symbol, f"OK ({len(cleaned)} rows)"))
    finally:
        mt5_connector.disconnect()

    logger.info("=== Summary ===")
    for symbol, status in results:
        logger.info("%-10s %s", symbol, status)

    failed = [s for s, status in results if not status.startswith("OK")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
