"""Risk-size sensitivity test for FVG strategy best config on XAUUSD H1.

Best config: RR=3.0, StopBuf=0.5, MinGap=0.1, OB=on
Tests risk per trade: 0.25%, 0.5%, 0.75%, 1.0%, 1.5%, 2.0%
"""
from pathlib import Path
import pandas as pd
from backtests.ftmo_engine import FTMOEngine
from strategies.fvg_ob import FVGOrderBlockStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

df = pd.read_csv(PROCESSED_DIR / "XAUUSD_H1.csv")

strat = FVGOrderBlockStrategy(
    min_fvg_atr     = 0.1,
    rr_target       = 3.0,
    atr_stop_buffer = 0.5,
    ob_required     = True,
)

rows = []
for risk in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    engine = FTMOEngine(strat, risk_pct=risk / 100, initial_capital=10_000)
    res    = engine.run(df.copy(), symbol="XAUUSD H1")
    m      = res.metrics
    rows.append({
        "Risk%":      risk,
        "Trades":     m["total_trades"],
        "WR%":        round(m["win_rate_pct"], 1),
        "AvgR":       round(m["avg_r"], 3),
        "Return%":    round(m["total_return_pct"], 1),
        "MaxDD%":     round(m["max_drawdown_pct"], 2),
        "Sharpe":     round(m["sharpe_ratio"], 3),
        "FTMO_Pass%": round(m["ftmo_pass_rate_pct"], 1),
        "Passes":     f"{m['ftmo_passes']}/{m['ftmo_windows']}",
    })

results = pd.DataFrame(rows)

print("\n" + "=" * 90)
print("  FVG Strategy — Risk Size Sensitivity  (XAUUSD H1, RR=3.0, StopBuf=0.5)")
print("  FTMO rules: 10% profit target, 10% max DD, 5% daily DD per 30-day window")
print("=" * 90)
print(results.to_string(index=False))

print("\n--- FTMO-safe configs (MaxDD < 10%) ---")
safe = results[results["MaxDD%"] < 10]
print(safe.to_string(index=False) if not safe.empty else "  None")

print("\n--- Near-safe configs (MaxDD < 15%) ---")
near = results[results["MaxDD%"] < 15]
print(near.to_string(index=False) if not near.empty else "  None")
