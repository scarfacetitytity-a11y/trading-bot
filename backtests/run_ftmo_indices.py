"""Indices-only FTMO portfolio — long-only, positive-EV combos only.

Portfolio (confirmed positive AvgR in FTMOEngine single-symbol runs):
  US30.cash  H1  FVG-OB Long    AvgR +0.324
  US30.cash  H1  London  Long   AvgR +0.354
  US100.cash H1  FVG-OB Long    AvgR +0.365
  US100.cash H1  Sniper  Long   AvgR +0.327

Excluded (negative AvgR):
  US30  Sniper Long   AvgR -0.510
  US100 London Long   AvgR -0.218
  GBPUSD (all)        negative both ways
  XAUUSD              excluded by operator choice

Usage:
    ./venv/Scripts/python -m backtests.run_ftmo_indices
    ./venv/Scripts/python -m backtests.run_ftmo_indices --risk 0.01
    ./venv/Scripts/python -m backtests.run_ftmo_indices --sweep
    ./venv/Scripts/python -m backtests.run_ftmo_indices --sweep --fine
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

CONFIGS = [
    (FVGOrderBlockStrategy(min_fvg_atr=0.1, rr_target=3.0, atr_stop_buffer=0.5, ob_required=True, long_only=True), "US30.cash",  "H1"),
    (LondonBreakoutStrategy(long_only=True),                                                                         "US30.cash",  "H1"),
    (FVGOrderBlockStrategy(min_fvg_atr=0.1, rr_target=3.0, atr_stop_buffer=0.5, ob_required=True, long_only=True), "US100.cash", "H1"),
    (SniperStrategy(long_only=True),                                                                                 "US100.cash", "H1"),
]


def run(risk_pct: float = 0.01) -> None:
    print(f"\n{'='*72}")
    print(f"  INDICES FTMO PORTFOLIO — {risk_pct*100:.2f}% risk/trade, ${INITIAL_CAPITAL:,}")
    print(f"  US30:  FVG-Long + London-Long")
    print(f"  US100: FVG-Long + Sniper-Long")
    print(f"  FTMO: 10% profit | 10% max DD | 5% daily DD | 30-day windows")
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
            "Strategy": strat.name[:30],
            "Symbol":   symbol,
            "Trades":   m["total_trades"],
            "WR%":      round(m["win_rate_pct"], 1),
            "AvgR":     round(m["avg_r"], 3),
            "Return%":  round(m["total_return_pct"], 1),
            "MaxDD%":   round(m["max_drawdown_pct"], 2),
            "Sharpe":   round(m["sharpe_ratio"], 3),
            "FTMO%":    round(m["ftmo_pass_rate_pct"], 1),
            "Passes":   f"{m['ftmo_passes']}/{m['ftmo_windows']}",
        })

    if rows:
        print("\nIndividual results:")
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'-'*72}")
    print("Combined portfolio (all 4 running simultaneously):")
    available = [(s, sym, tf) for s, sym, tf in CONFIGS
                 if (PROCESSED_DIR / f"{sym}_{tf}.csv").exists()]
    multi = FTMOMultiEngine(available, risk_pct=risk_pct, initial_capital=INITIAL_CAPITAL)
    result = multi.run()
    result.strategy_name = f"Indices-4combo ({risk_pct*100:.2f}% risk)"
    result.print_summary()


def sweep(fine: bool = False) -> None:
    print(f"\n{'='*72}")
    print(f"  RISK SWEEP — Indices portfolio (US30 FVG+London | US100 FVG+Sniper)")
    print(f"{'='*72}")

    available = [(s, sym, tf) for s, sym, tf in CONFIGS
                 if (PROCESSED_DIR / f"{sym}_{tf}.csv").exists()]

    if fine:
        risk_levels = [r/1000 for r in range(5, 21)]   # 0.5% to 2.0% in 0.1% steps
    else:
        risk_levels = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]

    rows = []
    for risk in risk_levels:
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

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    best = df.loc[df["FTMO_Pass%"].idxmax()]
    print(f"\n  Peak FTMO pass rate: {best['FTMO_Pass%']}% at {best['Risk%']} risk")
    print(f"  WR: {best['WR%']}%  |  AvgR: {best['AvgR']}  |  MaxDD: {best['MaxDD%']}%  |  Passes: {best['Passes']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk",  type=float, default=0.01)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--fine",  action="store_true", help="Fine-grained sweep 0.5-2.0%% in 0.1%% steps")
    args = parser.parse_args()
    if args.sweep:
        sweep(fine=args.fine)
    else:
        run(risk_pct=args.risk)


if __name__ == "__main__":
    main()
