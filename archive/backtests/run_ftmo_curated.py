"""Curated FTMO account — only positive-EV, low-DD strategy×pair×TF combos.

Based on scan results: filters to strategies with avg R > 0 and max DD < 50%.
Tests at 1%, 2%, and 3% risk per trade to find the sweet spot.

Usage:
    ./venv/Scripts/python -m backtests.run_ftmo_curated
    ./venv/Scripts/python -m backtests.run_ftmo_curated --risk 0.02
"""
import argparse
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

from backtests.ftmo_engine import FTMOEngine, FTMOMultiEngine
from backtests.logger import log_trades, log_run_summary
from backtests.git_push import auto_push
from strategies.sniper import SniperStrategy
from strategies.sniper_master import SniperMasterStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.ict_smart_money import ICTSmartMoneyStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.forex_master import ForexMasterStrategy
from strategies.ict_amd import (
    ICTAMDDisplacementStrategy,
    ICTAMDBreakerStrategy,
    ICTAMDBOSStrategy,
    ICTAMDComboStrategy,
)

# ── Curated list — only strategies/pairs/TFs with avg R > 0 from the scan ────
# Each entry: (strategy_instance, symbol, timeframe, "reason kept")
CURATED = [
    # Sniper — 62% WR, 0.355R, 9% max DD: cleanest single strategy
    (SniperStrategy(),                                          "XAUUSD",     "H1"),
    (SniperStrategy(),                                          "GBPUSD",     "M5"),  # 57.6% WR, 0.355R (before cap)

    # London Breakout — 58% WR, 0.419R avg, 8% max DD
    (LondonBreakoutStrategy(),                                  "XAUUSD",     "H1"),
    (LondonBreakoutStrategy(),                                  "GBPUSD",     "H1"),

    # ICT AMD Displacement — positive R, low DD
    (ICTAMDDisplacementStrategy(),                              "XAUUSD",     "H1"),
    (ICTAMDDisplacementStrategy(),                              "XAUUSD",     "M15"),
    (ICTAMDDisplacementStrategy(),                              "XAUUSD",     "M5"),
    (ICTAMDDisplacementStrategy(),                              "GBPUSD",     "M5"),  # 0.055R avg
    (ICTAMDDisplacementStrategy(),                              "US100.cash", "M5"),  # 5.22R avg (!!)

    # ICT AMD Breaker — 50% WR on XAUUSD M5, 5.9% FTMO pass
    (ICTAMDBreakerStrategy(swing_lookbacks=[288], zone_atr=2.0, max_wait=5),
                                                                "XAUUSD",     "M5"),

    # ICT AMD BOS — positive R on GBPUSD H1 (3.662 Sharpe, 0.774R avg)
    (ICTAMDBOSStrategy(),                                       "GBPUSD",     "H1"),

    # Donchian — high trade count, positive R, acceptable DD on H1
    (DonchianBreakoutStrategy(),                                "XAUUSD",     "H1"),

    # ForexMaster H1 — 47.7% WR, 0.166R, generates many trades
    (ForexMasterStrategy(),                                     "XAUUSD",     "H1"),
    (ForexMasterStrategy(),                                     "GBPUSD",     "M5"),  # 0.529R avg
    (ForexMasterStrategy(),                                     "GBPUSD",     "H1"),  # 4.257R avg (large outliers likely)

    # SniperMaster M5 XAUUSD — 61.5% WR, 0.355R avg, 9% max DD (small sample)
    (SniperMasterStrategy(),                                    "XAUUSD",     "M5"),

    # ICT AMD OB/Combo on US100 M5 — high avg R (5-8R), needs checking
    (ICTAMDComboStrategy(),                                     "US100.cash", "M5"),

    # ICT Smart Money H1 — high trade count, positive R but large DD
    # Included but weighted by its DD in combined account
    (ICTSmartMoneyStrategy(),                                   "XAUUSD",     "H1"),
]

INITIAL_CAPITAL = 10_000


def run_curated(risk_pct: float = 0.01):
    print(f"\n{'='*70}")
    print(f"  CURATED FTMO ACCOUNT — {risk_pct*100:.0f}% risk/trade, ${INITIAL_CAPITAL:,} account")
    print(f"{'='*70}")

    # First: show each strategy individually
    import pandas as pd
    rows = []
    for strat, symbol, tf in CURATED:
        path = PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        try:
            engine = FTMOEngine(strat, risk_pct=risk_pct, initial_capital=INITIAL_CAPITAL)
            res = engine.run(df.copy(), symbol=f"{symbol} {tf}")
            m = res.metrics
            rows.append({
                "Strategy":   strat.name[:38],
                "Symbol":     symbol,
                "TF":         tf,
                "Trades":     m["total_trades"],
                "WR%":        round(m["win_rate_pct"], 1),
                "AvgR":       round(m["avg_r"], 3),
                "Return%":    round(m["total_return_pct"], 2),
                "MaxDD%":     round(m["max_drawdown_pct"], 2),
                "Sharpe":     round(m["sharpe_ratio"], 3),
                "FTMO%":      round(m["ftmo_pass_rate_pct"], 1),
                "Passes":     f"{m['ftmo_passes']}/{m['ftmo_windows']}",
            })
            log_trades(res.trades, strat.name, symbol, tf)
        except Exception as e:
            print(f"  [error] {strat.name[:30]} {symbol} {tf}: {e}")

    if rows:
        df_res = pd.DataFrame(rows).sort_values("FTMO%", ascending=False)
        print("\nIndividual strategy results:")
        print(df_res.to_string(index=False))

    # Multi-strategy combined
    print(f"\n{'─'*70}")
    print("Running combined account (all curated strategies)...")
    configs = [
        (strat, symbol, tf)
        for strat, symbol, tf in CURATED
        if (PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv").exists()
    ]
    multi = FTMOMultiEngine(configs, risk_pct=risk_pct, initial_capital=INITIAL_CAPITAL)
    result = multi.run()
    result.strategy_name = f"Curated Multi ({risk_pct*100:.0f}% risk)"
    result.print_summary()

    log_run_summary("curated", {"risk_pct": risk_pct, **result.metrics})
    auto_push(f"Curated backtest {risk_pct*100:.0f}pct risk")


def risk_sweep():
    """Test 1%, 2%, 3% risk and compare pass rates."""
    import pandas as pd

    print("\n" + "="*60)
    print("  RISK SWEEP — Curated strategies, all risk levels")
    print("="*60)

    summary = []
    for risk in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        configs = [
            (strat, symbol, tf)
            for strat, symbol, tf in CURATED
            if (PROCESSED_DIR / f"{symbol}_{tf.upper()}.csv").exists()
        ]
        multi = FTMOMultiEngine(configs, risk_pct=risk, initial_capital=INITIAL_CAPITAL)
        result = multi.run()
        m = result.metrics
        summary.append({
            "Risk%":       f"{risk*100:.1f}%",
            "Return%":     round(m.get("total_return_pct", 0), 1),
            "MaxDD%":      round(m.get("max_drawdown_pct", 0), 1),
            "Trades":      m.get("total_trades", 0),
            "WR%":         round(m.get("win_rate_pct", 0), 1),
            "AvgR":        round(m.get("avg_r", 0), 3),
            "FTMO_Pass%":  round(m.get("ftmo_pass_rate_pct", 0), 1),
            "Passes":      f"{m.get('ftmo_passes', 0)}/{m.get('ftmo_windows', 0)}",
        })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    log_run_summary("risk_sweep", df_summary)
    auto_push("Risk sweep backtest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk",  type=float, default=None, help="Risk pct e.g. 0.02 for 2 pct")
    parser.add_argument("--sweep", action="store_true",       help="Sweep all risk levels")
    args = parser.parse_args()

    if args.sweep:
        risk_sweep()
    elif args.risk:
        run_curated(risk_pct=args.risk)
    else:
        # Default: sweep all risk levels
        risk_sweep()


if __name__ == "__main__":
    main()
