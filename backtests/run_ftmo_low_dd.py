"""Low-DD curated FTMO account: FVG + Sniper + London Breakout across H1 pairs.

All three strategies have positive EV and MaxDD < 10% individually.
Running across XAUUSD, GBPUSD, US30, US100 increases trade frequency per
30-day window, giving more consistent FTMO profit target hits.

Usage:
    ./venv/Scripts/python -m backtests.run_ftmo_low_dd
    ./venv/Scripts/python -m backtests.run_ftmo_low_dd --risk 0.0075
    ./venv/Scripts/python -m backtests.run_ftmo_low_dd --sweep
"""
import argparse
from pathlib import Path

import pandas as pd

from backtests.ftmo_engine import FTMOEngine, FTMOMultiEngine
from strategies.fvg_ob import FVGOrderBlockStrategy
from strategies.sniper import SniperStrategy
from strategies.london_breakout import LondonBreakoutStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INITIAL_CAPITAL = 10_000

# All available H1 pairs — each strategy runs independently on each instrument
SYMBOLS = ["XAUUSD", "GBPUSD", "US30.cash", "US100.cash"]

def _make_configs():
    strats = [
        FVGOrderBlockStrategy(min_fvg_atr=0.1, rr_target=3.0, atr_stop_buffer=0.5, ob_required=True),
        SniperStrategy(),
        LondonBreakoutStrategy(),
    ]
    return [(s, sym, "H1") for sym in SYMBOLS for s in strats]

CONFIGS = _make_configs()


def run(risk_pct: float = 0.0075) -> None:
    print(f"\n{'='*72}")
    print(f"  LOW-DD FTMO ACCOUNT — {risk_pct*100:.2f}% risk/trade, ${INITIAL_CAPITAL:,}")
    print(f"  Strategies: FVG + Sniper + London Breakout | {', '.join(SYMBOLS)} H1")
    print(f"  FTMO rules: 10% profit target, 10% max DD, 5% daily DD per 30-day window")
    print(f"{'='*72}")

    rows = []
    for strat, symbol, tf in CONFIGS:
        path = PROCESSED_DIR / f"{symbol}_{tf}.csv"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        df = pd.read_csv(path)
        engine = FTMOEngine(strat, risk_pct=risk_pct, initial_capital=INITIAL_CAPITAL)
        res = engine.run(df.copy(), symbol=f"{symbol} {tf}")
        m = res.metrics
        rows.append({
            "Strategy":  strat.name[:36],
            "Trades":    m["total_trades"],
            "WR%":       round(m["win_rate_pct"], 1),
            "AvgR":      round(m["avg_r"], 3),
            "Return%":   round(m["total_return_pct"], 1),
            "MaxDD%":    round(m["max_drawdown_pct"], 2),
            "Sharpe":    round(m["sharpe_ratio"], 3),
            "FTMO%":     round(m["ftmo_pass_rate_pct"], 1),
            "Passes":    f"{m['ftmo_passes']}/{m['ftmo_windows']}",
        })

    if rows:
        print("\nIndividual results:")
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'-'*72}")
    print("Combined account (all three running simultaneously):")
    available = [(s, sym, tf) for s, sym, tf in CONFIGS
                 if (PROCESSED_DIR / f"{sym}_{tf}.csv").exists()]
    multi = FTMOMultiEngine(available, risk_pct=risk_pct, initial_capital=INITIAL_CAPITAL)
    result = multi.run()
    result.strategy_name = f"LowDD-3strat ({risk_pct*100:.2f}% risk)"
    result.print_summary()


def sweep() -> None:
    print(f"\n{'='*72}")
    print(f"  RISK SWEEP — FVG + Sniper + London Breakout | {', '.join(SYMBOLS)} H1")
    print(f"{'='*72}")

    rows = []
    available = [(s, sym, tf) for s, sym, tf in CONFIGS
                 if (PROCESSED_DIR / f"{sym}_{tf}.csv").exists()]

    for risk in [0.005, 0.0075, 0.01, 0.0125, 0.015]:
        multi = FTMOMultiEngine(available, risk_pct=risk, initial_capital=INITIAL_CAPITAL)
        result = multi.run()
        m = result.metrics
        rows.append({
            "Risk%":      f"{risk*100:.2f}%",
            "Trades":     m.get("total_trades", 0),
            "WR%":        round(m.get("win_rate_pct", 0), 1),
            "AvgR":       round(m.get("avg_r", 0), 3),
            "Return%":    round(m.get("total_return_pct", 0), 1),
            "MaxDD%":     round(m.get("max_drawdown_pct", 0), 2),
            "Sharpe":     round(m.get("sharpe_ratio", 0), 3),
            "FTMO_Pass%": round(m.get("ftmo_pass_rate_pct", 0), 1),
            "Passes":     f"{m.get('ftmo_passes', 0)}/{m.get('ftmo_windows', 0)}",
        })

    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk",  type=float, default=0.0075, help="Risk per trade e.g. 0.0075")
    parser.add_argument("--sweep", action="store_true",        help="Sweep risk levels from 0.5 to 1.5 pct")
    args = parser.parse_args()
    if args.sweep:
        sweep()
    else:
        run(risk_pct=args.risk)


if __name__ == "__main__":
    main()
