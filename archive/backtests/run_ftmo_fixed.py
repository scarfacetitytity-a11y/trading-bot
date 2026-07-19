"""Fixed-risk FTMO scan.

Tests every strategy × pair × timeframe with 1% risk per trade.
Shows FTMO pass rate per 30-day window.

Usage:
    ./venv/Scripts/python -m backtests.run_ftmo_fixed
    ./venv/Scripts/python -m backtests.run_ftmo_fixed --multi   # multi-strategy mode
    ./venv/Scripts/python -m backtests.run_ftmo_fixed --symbol XAUUSD --tf H1
"""
import argparse
import sys
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

from backtests.ftmo_engine import FTMOEngine, FTMOMultiEngine
from backtests.logger import log_trades, log_run_summary
from backtests.git_push import auto_push
from strategies.sniper import SniperStrategy
from strategies.sniper_master import SniperMasterStrategy
from strategies.sniper_trend import SniperTrendStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.ict_smart_money import ICTSmartMoneyStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.forex_master import ForexMasterStrategy
from strategies.trend_rider import TrendRiderStrategy
from strategies.ict_amd import (
    ICTAMDDisplacementStrategy,
    ICTAMDBreakerStrategy,
    ICTAMDBOSStrategy,
    ICTAMDMidpointStrategy,
    ICTAMDOrderBlockStrategy,
    ICTAMDComboStrategy,
)

STRATEGIES = [
    SniperStrategy(),
    SniperMasterStrategy(),
    SniperTrendStrategy(),
    LondonBreakoutStrategy(),
    ICTSmartMoneyStrategy(),
    DonchianBreakoutStrategy(),
    ForexMasterStrategy(),
    TrendRiderStrategy(),
    ICTAMDBreakerStrategy(swing_lookbacks=[288], zone_atr=2.0, max_wait=5),
    ICTAMDDisplacementStrategy(),
    ICTAMDBOSStrategy(),
    ICTAMDMidpointStrategy(),
    ICTAMDOrderBlockStrategy(),
    ICTAMDComboStrategy(),
]

COMBOS = [
    ("XAUUSD",     "H1"),
    ("XAUUSD",     "M5"),
    ("XAUUSD",     "M15"),
    ("GBPUSD",     "H1"),
    ("GBPUSD",     "M5"),
    ("GBPUSD",     "M15"),
    ("US100.cash", "H1"),
    ("US100.cash", "M5"),
]

RISK_PCT        = 0.01   # 1% risk per trade
INITIAL_CAPITAL = 10_000


def run_single(symbol_filter=None, tf_filter=None):
    rows = []
    for symbol, tf in COMBOS:
        if symbol_filter and symbol != symbol_filter:
            continue
        if tf_filter and tf != tf_filter:
            continue

        path = PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv"
        if not path.exists():
            print(f"  [skip] {symbol} {tf} — data file missing")
            continue

        import pandas as pd
        df = pd.read_csv(path)

        for strat in STRATEGIES:
            try:
                engine = FTMOEngine(strat, risk_pct=RISK_PCT, initial_capital=INITIAL_CAPITAL)
                res    = engine.run(df.copy(), symbol=f"{symbol} {tf}")
                m      = res.metrics
                log_trades(res.trades, strat.name, symbol, tf)
                rows.append({
                    "Strategy":    strat.name[:40],
                    "Symbol":      symbol,
                    "TF":          tf,
                    "Trades":      m["total_trades"],
                    "WR%":         round(m["win_rate_pct"], 1),
                    "AvgR":        round(m["avg_r"], 3),
                    "Return%":     round(m["total_return_pct"], 2),
                    "MaxDD%":      round(m["max_drawdown_pct"], 2),
                    "Sharpe":      round(m["sharpe_ratio"], 3),
                    "FTMO_Pass%":  round(m["ftmo_pass_rate_pct"], 1),
                    "Passes":      f"{m['ftmo_passes']}/{m['ftmo_windows']}",
                    "NoStop":      m["trades_no_stop"],
                })
            except Exception as e:
                print(f"  [error] {strat.name[:30]} | {symbol} {tf}: {e}")

    if not rows:
        print("No results.")
        return

    import pandas as pd
    results = pd.DataFrame(rows)
    results = results.sort_values("FTMO_Pass%", ascending=False)

    print("\n" + "="*120)
    print("  FIXED-RISK FTMO SCAN  (1% risk/trade, $10k account, 30-day windows)")
    print("="*120)
    print(results.to_string(index=False))

    # Top 10 by FTMO pass rate
    top = results[results["Trades"] >= 10].head(10)
    if not top.empty:
        print("\n--- Top 10 (>=10 trades) by FTMO Pass Rate ---")
        print(top[["Strategy", "Symbol", "TF", "Trades", "WR%", "AvgR",
                   "Return%", "MaxDD%", "Sharpe", "FTMO_Pass%", "Passes"]].to_string(index=False))

    log_run_summary("ftmo_fixed_scan", results)
    auto_push("Fixed-risk FTMO scan")


def run_multi(symbol_filter=None, tf_filter=None):
    configs = []
    for strat in STRATEGIES:
        for symbol, tf in COMBOS:
            if symbol_filter and symbol != symbol_filter:
                continue
            if tf_filter and tf != tf_filter:
                continue
            path = PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv"
            if path.exists():
                configs.append((strat, symbol, tf))

    if not configs:
        print("No valid configs found.")
        return

    print(f"\nRunning multi-strategy engine: {len(configs)} strategy×pair×TF combos...")
    engine = FTMOMultiEngine(
        configs,
        risk_pct=RISK_PCT,
        initial_capital=INITIAL_CAPITAL,
    )
    res = engine.run()
    res.print_summary()

    # Show best single-strategy combos for reference
    print("\nTop 5 strategy×pair×TF by FTMO pass rate (single-strategy):")
    run_single(symbol_filter, tf_filter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi",   action="store_true", help="Run multi-strategy mode")
    parser.add_argument("--symbol",  default=None,        help="Filter to one symbol")
    parser.add_argument("--tf",      default=None,        help="Filter to one timeframe")
    args = parser.parse_args()

    if args.multi:
        run_multi(args.symbol, args.tf)
    else:
        run_single(args.symbol, args.tf)


if __name__ == "__main__":
    main()
