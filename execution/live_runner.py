"""Live trading loop.

Connects to a running MT5 terminal, watches for new bars, generates signals
from the configured strategy, and places/closes orders accordingly.

Usage (project root, venv active):
    python -m execution.live_runner
    python -m execution.live_runner --symbol XAUUSD --strategy rsi
    python -m execution.live_runner --dry-run          # log signals, no real orders
"""
import argparse
import logging
import sys
import time
import threading
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import TIMEFRAME_MAP
from execution import trader, risk
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBands
from strategies.sniper import SniperStrategy
from strategies.forex_master import ForexMasterStrategy

logger = logging.getLogger(__name__)

STRATEGIES = {
    "sma_crossover": SMACrossover,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger_bands": BollingerBands,
    "sniper": SniperStrategy,
    "forex_master": ForexMasterStrategy,
}

# Seconds per timeframe — used to sleep until next bar
_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Per-symbol trading loop
# ---------------------------------------------------------------------------

def _run_symbol(symbol: str, strategy, tf_str: str, trade_cfg: dict, dry_run: bool) -> None:
    tf_code = TIMEFRAME_MAP.get(tf_str)
    if tf_code is None:
        logger.error("Unknown timeframe '%s' for %s", tf_str, symbol)
        return

    lookback = trade_cfg.get("lookback_bars", 500)
    last_bar_time = None

    logger.info("[%s] Starting live loop | strategy=%s | timeframe=%s | dry_run=%s",
                symbol, strategy.name, tf_str, dry_run)

    while not _stop_event.is_set():
        try:
            bars = mt5.copy_rates_from_pos(symbol, tf_code, 0, 1)
            if bars is None or len(bars) == 0:
                logger.warning("[%s] Could not fetch current bar, retrying in 15s", symbol)
                time.sleep(15)
                continue

            current_bar_time = int(bars[0]["time"])

            if current_bar_time == last_bar_time:
                time.sleep(_poll_interval(tf_str))
                continue

            last_bar_time = current_bar_time
            bar_dt = datetime.fromtimestamp(current_bar_time, tz=timezone.utc)
            logger.info("[%s] New bar: %s", symbol, bar_dt.strftime("%Y-%m-%d %H:%M UTC"))

            # Fetch lookback bars and generate signal
            rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, lookback)
            if rates is None or len(rates) == 0:
                logger.warning("[%s] Could not fetch lookback bars", symbol)
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[["time", "open", "high", "low", "close", "tick_volume"]]

            signals = strategy.generate_signals(df)
            desired = int(signals.iloc[-1])
            current = trader.get_position_direction(symbol)

            logger.info("[%s] signal=%+d  current_position=%+d", symbol, desired, current)

            if desired == current:
                continue  # nothing to do

            # Close existing position if direction changed
            if current != 0:
                if dry_run:
                    logger.info("[%s] DRY RUN: would close %s position",
                                symbol, "LONG" if current == 1 else "SHORT")
                else:
                    n = trader.close_all(symbol)
                    logger.info("[%s] Closed %d position(s)", symbol, n)

            # Open new position if signal is non-zero
            if desired != 0:
                account = trader.get_account()
                lots = risk.calculate_lots(symbol, trade_cfg, account["balance"])
                sl, tp = risk.calculate_sl_tp(symbol, desired, trade_cfg)

                direction_str = "BUY (LONG)" if desired == 1 else "SELL (SHORT)"
                if dry_run:
                    logger.info("[%s] DRY RUN: would place %s | lots=%.2f | SL=%s | TP=%s",
                                symbol, direction_str, lots, sl, tp)
                else:
                    trader.place_order(symbol, desired, lots, sl=sl, tp=tp)

        except Exception as exc:
            logger.exception("[%s] Unexpected error in trading loop: %s", symbol, exc)
            time.sleep(30)

    logger.info("[%s] Loop stopped.", symbol)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _seconds_to_next_bar(tf_str: str) -> float:
    """Seconds until the next bar boundary (+ 2s buffer)."""
    tf_secs = _TF_SECONDS.get(tf_str, 900)
    now = time.time()
    remaining = tf_secs - (now % tf_secs)
    return remaining + 2.0


def _poll_interval(tf_str: str) -> float:
    """How often to poll mid-bar to check if a new bar has started."""
    tf_secs = _TF_SECONDS.get(tf_str, 900)
    return min(30, max(5, tf_secs // 30))


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def _safety_check(cfg: dict) -> bool:
    account = mt5.account_info()
    if account is None:
        logger.error("No account info — is MT5 open and logged in?")
        return False

    is_demo = account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    allow_real = cfg.get("trading", {}).get("allow_real_account", False)

    if not is_demo and not allow_real:
        logger.error(
            "BLOCKED: account is REAL and 'trading.allow_real_account' is not set to true in "
            "config.yaml. Set it explicitly after confirming you understand the risks."
        )
        return False

    if not is_demo:
        logger.warning(
            "WARNING: Trading on a REAL account. 'allow_real_account' is enabled."
        )

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the live trading bot.")
    parser.add_argument("--strategy", default="forex_master", choices=list(STRATEGIES))
    parser.add_argument("--symbol", default=None,
                        help="Single symbol to trade (default: all in config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals and log intent but place no real orders")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(*([args.config] if args.config else []))
    log_cfg = cfg.get("logging", {})
    setup_logger(
        "",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file="live_trading.log",
        level=log_cfg.get("level", "INFO"),
    )

    mt5_cfg = cfg.get("mt5", {})
    terminal_path = mt5_cfg.get("terminal_path") or None

    if not connect(terminal_path):
        logger.error("Could not connect to MT5. Aborting.")
        sys.exit(1)

    if not _safety_check(cfg):
        disconnect()
        sys.exit(1)

    trade_cfg = cfg.get("trading", {})
    trader.set_magic(trade_cfg.get("magic", 234001))

    tf_str = cfg["data"]["timeframe"].upper()
    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls()

    if args.dry_run:
        logger.info("=== DRY RUN MODE — no real orders will be placed ===")

    logger.info("Starting live trading: strategy=%s | symbols=%s | tf=%s",
                strategy.name, symbols, tf_str)

    threads = []
    for symbol in symbols:
        t = threading.Thread(
            target=_run_symbol,
            args=(symbol, strategy, tf_str, trade_cfg, args.dry_run),
            name=f"live-{symbol}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested — stopping all loops...")
        _stop_event.set()
        for t in threads:
            t.join(timeout=10)
    finally:
        disconnect()
        logger.info("Live runner stopped.")


if __name__ == "__main__":
    main()
