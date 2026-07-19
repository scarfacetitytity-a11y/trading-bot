"""Pair scanner — find every symbol on the broker, pull D1 data, rank by TrendRider performance.

Usage (project root, venv active, MT5 open and logged in):
    python -m backtests.scanner
    python -m backtests.scanner --min-years 5 --top 20
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import TIMEFRAME_MAP, save_raw
from backtests.data_cleaner import clean_data
from backtests.engine import Backtest
from strategies.trend_rider import TrendRiderStrategy

logger = logging.getLogger(__name__)

START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"
TIMEFRAME  = "D1"
MIN_BARS   = 200   # need at least this many bars for SMA200 warmup


def get_all_symbols() -> list[str]:
    """Return every visible/tradeable symbol on the broker."""
    all_syms = mt5.symbols_get()
    if not all_syms:
        return []
    return [s.name for s in all_syms if s.visible or mt5.symbol_select(s.name, True)]


def fetch_and_clean(symbol: str) -> pd.DataFrame | None:
    tf = TIMEFRAME_MAP[TIMEFRAME]
    date_from = datetime.strptime(START_DATE, "%Y-%m-%d")
    date_to   = datetime.strptime(END_DATE,   "%Y-%m-%d")

    rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    if rates is None or len(rates) < MIN_BARS:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def scan(min_years: float, top_n: int) -> None:
    cfg = load_config()
    terminal_path = cfg.get("mt5", {}).get("terminal_path") or None

    if not connect(terminal_path):
        logger.error("Could not connect to MT5.")
        sys.exit(1)

    symbols = get_all_symbols()
    logger.info("Found %d symbols on broker — screening all of them...", len(symbols))

    strategy = TrendRiderStrategy()
    results  = []

    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_and_clean(symbol)
            if df is None:
                continue

            years = (df["time"].iloc[-1] - df["time"].iloc[0]).days / 365.25
            if years < min_years:
                continue

            bt     = Backtest(strategy, initial_capital=10_000, commission=0.0001)
            result = bt.run(df, symbol=symbol)
            m      = result.metrics

            if m["total_trades"] == 0:
                continue

            results.append({
                "symbol":        symbol,
                "years":         round(years, 1),
                "total_return":  round(m["total_return_pct"], 2),
                "cagr":          round(m["cagr_pct"], 2),
                "max_dd":        round(m["max_drawdown_pct"], 2),
                "sharpe":        round(m["sharpe_ratio"], 3),
                "trades":        m["total_trades"],
                "win_rate":      round(m["win_rate_pct"], 1),
                "profit_factor": round(m["profit_factor"], 3),
            })

            logger.info("[%d/%d] %-16s  CAGR=%+.1f%%  DD=%.1f%%  WR=%.0f%%  trades=%d",
                        i, len(symbols), symbol,
                        m["cagr_pct"], m["max_drawdown_pct"],
                        m["win_rate_pct"], m["total_trades"])

        except Exception as exc:
            logger.debug("SKIP %s: %s", symbol, exc)

    disconnect()

    if not results:
        print("\nNo symbols passed the filter.")
        return

    df_res = pd.DataFrame(results)

    # Rank: profitable AND decent win rate AND Sharpe > 0
    profitable = df_res[
        (df_res["total_return"] > 0) &
        (df_res["win_rate"] >= 50) &
        (df_res["sharpe"] > 0)
    ].sort_values("cagr", ascending=False)

    print(f"\n{'='*90}")
    print(f"  TOP PAIRS — TrendRider D1  ({START_DATE} to {END_DATE})")
    print(f"{'='*90}")
    if profitable.empty:
        print("  No profitable pairs found with these filters.")
    else:
        print(f"  {'Symbol':<16} {'Years':>5} {'CAGR':>8} {'Return':>8} {'MaxDD':>8} "
              f"{'Sharpe':>7} {'WR%':>6} {'PF':>6} {'Trades':>7}")
        print(f"  {'-'*16} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
        for _, r in profitable.head(top_n).iterrows():
            print(f"  {r['symbol']:<16} {r['years']:>5.1f} {r['cagr']:>+7.2f}% "
                  f"{r['total_return']:>+7.2f}% {r['max_dd']:>7.2f}% "
                  f"{r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% "
                  f"{r['profit_factor']:>6.3f} {r['trades']:>7}")

    print(f"\n  Scanned {len(symbols)} symbols | {len(df_res)} had trades | "
          f"{len(profitable)} are profitable\n")

    # Save full results
    out = Path("logs") / "scanner_results.csv"
    out.parent.mkdir(exist_ok=True)
    df_res.sort_values("cagr", ascending=False).to_csv(out, index=False)
    print(f"  Full results saved to {out}\n")


def main():
    parser = argparse.ArgumentParser(description="Scan all broker pairs with ForexMaster strategy.")
    parser.add_argument("--min-years", type=float, default=3.0,
                        help="Minimum years of history required (default: 3)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N pairs (default: 20)")
    args = parser.parse_args()

    cfg     = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logger("", log_dir=log_cfg.get("log_dir", "logs"),
                 log_file="scanner.log", level="INFO")

    scan(min_years=args.min_years, top_n=args.top)


if __name__ == "__main__":
    main()
