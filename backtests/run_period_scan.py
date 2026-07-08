"""Period robustness scanner.

Tests every curated strategy over multiple lookback windows:
    3 months, 6 months, 1 year, 2 years, 4 years, 5 years

A strategy that performs consistently across ALL periods is reliable.
One that only shines in a specific window is curve-fitted / regime-dependent.

Usage:
    ./venv/Scripts/python -m backtests.run_period_scan
    ./venv/Scripts/python -m backtests.run_period_scan --symbol XAUUSD --tf H1
    ./venv/Scripts/python -m backtests.run_period_scan --risk 0.015
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backtests.ftmo_engine import FTMOEngine
from backtests.logger import log_run_summary
from backtests.git_push import auto_push
from strategies.sniper import SniperStrategy
from strategies.sniper_master import SniperMasterStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.ict_amd import (
    ICTAMDDisplacementStrategy,
    ICTAMDBreakerStrategy,
    ICTAMDBOSStrategy,
)
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.forex_master import ForexMasterStrategy
from strategies.ict_smart_money import ICTSmartMoneyStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

PERIODS = {
    "3m":  90,
    "6m":  180,
    "1y":  365,
    "2y":  730,
    "4y":  1460,
    "5y":  1825,
}

COMBOS = [
    (SniperStrategy(),                                                       "XAUUSD",     "H1"),
    (LondonBreakoutStrategy(),                                               "XAUUSD",     "H1"),
    (LondonBreakoutStrategy(),                                               "GBPUSD",     "H1"),
    (ICTAMDDisplacementStrategy(),                                           "XAUUSD",     "H1"),
    (ICTAMDBreakerStrategy(swing_lookbacks=[288], zone_atr=2.0, max_wait=5), "XAUUSD",    "M5"),
    (DonchianBreakoutStrategy(),                                             "XAUUSD",     "H1"),
    (ForexMasterStrategy(),                                                  "XAUUSD",     "H1"),
    (ICTSmartMoneyStrategy(),                                                "XAUUSD",     "H1"),
    (ICTAMDBOSStrategy(),                                                    "GBPUSD",     "H1"),
    (SniperMasterStrategy(),                                                 "XAUUSD",     "M5"),
]


def slice_last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    cutoff = times.max() - timedelta(days=days)
    return df[times >= cutoff].reset_index(drop=True)


def run_period_scan(
    symbol_filter: str | None = None,
    tf_filter: str | None = None,
    risk_pct: float = 0.015,
):
    print(f"\n{'='*90}")
    print(f"  PERIOD ROBUSTNESS SCAN  ({risk_pct*100:.1f}% risk/trade)")
    print(f"  Periods: {', '.join(PERIODS.keys())}")
    print(f"{'='*90}")

    all_rows = []

    for strat, symbol, tf in COMBOS:
        if symbol_filter and symbol != symbol_filter:
            continue
        if tf_filter and tf != tf_filter:
            continue

        path = PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv"
        if not path.exists():
            continue

        df_full = pd.read_csv(path)
        row = {"Strategy": strat.name[:36], "Symbol": symbol, "TF": tf}

        for period_name, days in PERIODS.items():
            df_slice = slice_last_n_days(df_full, days)
            if len(df_slice) < 50:
                row[f"{period_name}_trades"] = 0
                row[f"{period_name}_ftmo%"]  = 0.0
                row[f"{period_name}_avgR"]   = 0.0
                continue

            try:
                engine = FTMOEngine(strat, risk_pct=risk_pct, initial_capital=10_000)
                res    = engine.run(df_slice, symbol=f"{symbol} {tf}")
                m      = res.metrics
                row[f"{period_name}_trades"] = m["total_trades"]
                row[f"{period_name}_ftmo%"]  = round(m["ftmo_pass_rate_pct"], 1)
                row[f"{period_name}_avgR"]   = round(m["avg_r"], 3)
                row[f"{period_name}_wr%"]    = round(m["win_rate_pct"], 1)
                row[f"{period_name}_dd%"]    = round(m["max_drawdown_pct"], 1)
            except Exception as e:
                print(f"  [error] {strat.name[:28]} {symbol} {tf} {period_name}: {e}")
                row[f"{period_name}_ftmo%"] = -1.0

        all_rows.append(row)

    if not all_rows:
        print("No results.")
        return

    results = pd.DataFrame(all_rows)

    # Print FTMO pass rate comparison across periods
    ftmo_cols  = [f"{p}_ftmo%" for p in PERIODS]
    avgr_cols  = [f"{p}_avgR"  for p in PERIODS]
    trade_cols = [f"{p}_trades" for p in PERIODS]

    print("\n--- FTMO Pass Rate % by Period ---")
    cols = ["Strategy", "Symbol", "TF"] + ftmo_cols
    print(results[cols].to_string(index=False))

    print("\n--- Avg R per Trade by Period ---")
    cols = ["Strategy", "Symbol", "TF"] + avgr_cols
    print(results[cols].to_string(index=False))

    print("\n--- Trade Count by Period ---")
    cols = ["Strategy", "Symbol", "TF"] + trade_cols
    print(results[cols].to_string(index=False))

    # Robustness score: number of periods where FTMO% > 0 and avgR > 0
    results["robust_periods"] = sum(
        (results[f"{p}_ftmo%"] > 0).astype(int) for p in PERIODS
    )
    results["consistent_R"]   = sum(
        (results[f"{p}_avgR"] > 0).astype(int) for p in PERIODS
    )
    print("\n--- Robustness Score (# periods with FTMO pass > 0 / avg R > 0) ---")
    print(results[["Strategy", "Symbol", "TF", "robust_periods", "consistent_R"]]
          .sort_values("robust_periods", ascending=False).to_string(index=False))

    # Auto-log and push
    log_run_summary("period_scan", results)
    auto_push("Period robustness scan results")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--tf",     default=None)
    parser.add_argument("--risk",   type=float, default=0.015)
    args = parser.parse_args()
    run_period_scan(args.symbol, args.tf, args.risk)


if __name__ == "__main__":
    main()
