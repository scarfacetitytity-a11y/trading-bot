"""FVG + Order Block strategy backtest.

Runs FVGOrderBlockStrategy across symbols and timeframes, with a parameter
sweep to find the best configuration.

Usage:
    ./venv/Scripts/python -m backtests.run_fvg_backtest
    ./venv/Scripts/python -m backtests.run_fvg_backtest --symbol XAUUSD --tf H1
    ./venv/Scripts/python -m backtests.run_fvg_backtest --sweep   # full param sweep
"""

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

from backtests.ftmo_engine import FTMOEngine
from strategies.fvg_ob import FVGOrderBlockStrategy

COMBOS = [
    ("XAUUSD",     "M5"),
    ("XAUUSD",     "M15"),
    ("XAUUSD",     "H1"),
    ("GBPUSD",     "M5"),
    ("GBPUSD",     "M15"),
    ("GBPUSD",     "H1"),
    ("US100.cash", "M5"),
    ("US100.cash", "H1"),
]

RISK_PCT        = 0.01    # 1% per trade
INITIAL_CAPITAL = 10_000


def _strategies_baseline():
    """Default parameter set — the starting point."""
    return [
        FVGOrderBlockStrategy(min_fvg_atr=0.1,  max_fvg_wait=50,  max_entry_wait=10, rr_target=2.0, ob_required=True),
        FVGOrderBlockStrategy(min_fvg_atr=0.1,  max_fvg_wait=50,  max_entry_wait=10, rr_target=2.0, ob_required=False),
        FVGOrderBlockStrategy(min_fvg_atr=0.05, max_fvg_wait=100, max_entry_wait=20, rr_target=2.0, ob_required=True),
        FVGOrderBlockStrategy(min_fvg_atr=0.2,  max_fvg_wait=30,  max_entry_wait=5,  rr_target=2.0, ob_required=True),
        FVGOrderBlockStrategy(min_fvg_atr=0.1,  max_fvg_wait=50,  max_entry_wait=10, rr_target=1.5, ob_required=True),
        FVGOrderBlockStrategy(min_fvg_atr=0.1,  max_fvg_wait=50,  max_entry_wait=10, rr_target=3.0, ob_required=True),
        FVGOrderBlockStrategy(min_fvg_atr=0.1,  max_fvg_wait=50,  max_entry_wait=10, rr_target=2.0, ob_required=True, session_filter=True),
    ]


def _strategies_sweep():
    """Full parameter grid for finding optimal settings."""
    strats = []
    for min_gap in [0.05, 0.1, 0.2]:
        for fvg_wait in [30, 50, 100]:
            for entry_wait in [5, 10, 20]:
                for rr in [1.5, 2.0, 3.0]:
                    for ob in [True, False]:
                        strats.append(FVGOrderBlockStrategy(
                            min_fvg_atr    = min_gap,
                            max_fvg_wait   = fvg_wait,
                            max_entry_wait = entry_wait,
                            rr_target      = rr,
                            ob_required    = ob,
                        ))
    return strats


def run(symbol_filter=None, tf_filter=None, sweep=False):
    strategies = _strategies_sweep() if sweep else _strategies_baseline()

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

        df = pd.read_csv(path)

        for strat in strategies:
            try:
                engine = FTMOEngine(strat, risk_pct=RISK_PCT, initial_capital=INITIAL_CAPITAL)
                res    = engine.run(df.copy(), symbol=f"{symbol} {tf}")
                m      = res.metrics
                rows.append({
                    "Strategy":   strat.name,
                    "Symbol":     symbol,
                    "TF":         tf,
                    "Trades":     m["total_trades"],
                    "WR%":        round(m["win_rate_pct"], 1),
                    "AvgR":       round(m["avg_r"], 3),
                    "Return%":    round(m["total_return_pct"], 2),
                    "MaxDD%":     round(m["max_drawdown_pct"], 2),
                    "Sharpe":     round(m["sharpe_ratio"], 3),
                    "FTMO_Pass%": round(m["ftmo_pass_rate_pct"], 1),
                    "Passes":     f"{m['ftmo_passes']}/{m['ftmo_windows']}",
                })
            except Exception as e:
                print(f"  [error] {strat.name} | {symbol} {tf}: {e}")

    if not rows:
        print("No results.")
        return

    results = pd.DataFrame(rows).sort_values("FTMO_Pass%", ascending=False)

    print("\n" + "=" * 130)
    print("  FVG + ORDER BLOCK STRATEGY BACKTEST  (1% risk/trade, $10k account)")
    print("=" * 130)
    print(results.to_string(index=False))

    eligible = results[results["Trades"] >= 10]
    if not eligible.empty:
        print("\n--- Top 15 configs (>=10 trades) by FTMO Pass Rate ---")
        print(eligible.head(15)[
            ["Strategy", "Symbol", "TF", "Trades", "WR%", "AvgR",
             "Return%", "MaxDD%", "Sharpe", "FTMO_Pass%", "Passes"]
        ].to_string(index=False))

    # Summary by symbol+TF (best config per combo)
    print("\n--- Best config per Symbol × TF ---")
    best = eligible.groupby(["Symbol", "TF"]).first().reset_index()
    print(best[["Symbol", "TF", "Strategy", "Trades", "WR%", "AvgR",
                "MaxDD%", "FTMO_Pass%"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="Filter to one symbol")
    parser.add_argument("--tf",     default=None, help="Filter to one timeframe")
    parser.add_argument("--sweep",  action="store_true", help="Full parameter sweep")
    args = parser.parse_args()
    run(symbol_filter=args.symbol, tf_filter=args.tf, sweep=args.sweep)


if __name__ == "__main__":
    main()
