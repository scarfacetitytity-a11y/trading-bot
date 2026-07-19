"""Tight parameter sweep for FVG strategy on XAUUSD H1.

Varies RR (2.0–4.0) and stop buffer (0.3–1.2 ATR) to find
the best balance of FTMO pass rate and low drawdown.
"""
from pathlib import Path
import pandas as pd
from backtests.ftmo_engine import FTMOEngine
from strategies.fvg_ob import FVGOrderBlockStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

df = pd.read_csv(PROCESSED_DIR / "XAUUSD_H1.csv")

rows = []
for rr in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for buf in [0.3, 0.5, 0.8, 1.2]:
        for min_gap in [0.1, 0.2]:
            strat = FVGOrderBlockStrategy(
                min_fvg_atr      = min_gap,
                rr_target        = rr,
                atr_stop_buffer  = buf,
                ob_required      = True,
            )
            try:
                engine = FTMOEngine(strat, risk_pct=0.01, initial_capital=10_000)
                res    = engine.run(df.copy(), symbol="XAUUSD H1")
                m      = res.metrics
                rows.append({
                    "RR":         rr,
                    "StopBuf":    buf,
                    "MinGap":     min_gap,
                    "Trades":     m["total_trades"],
                    "WR%":        round(m["win_rate_pct"], 1),
                    "AvgR":       round(m["avg_r"], 3),
                    "Return%":    round(m["total_return_pct"], 1),
                    "MaxDD%":     round(m["max_drawdown_pct"], 2),
                    "Sharpe":     round(m["sharpe_ratio"], 3),
                    "FTMO_Pass%": round(m["ftmo_pass_rate_pct"], 1),
                    "Passes":     f"{m['ftmo_passes']}/{m['ftmo_windows']}",
                })
            except Exception as e:
                print(f"Error {strat.name}: {e}")

results = pd.DataFrame(rows).sort_values("FTMO_Pass%", ascending=False)

print("\n" + "=" * 100)
print("  XAUUSD H1 — FVG Strategy Tight Sweep (1% risk, $10k)")
print("=" * 100)
print(results.to_string(index=False))

eligible = results[results["Trades"] >= 20]

print("\n--- Top 10: MaxDD < 25% ---")
sub = eligible[eligible["MaxDD%"] < 25]
print(sub.head(10).to_string(index=False) if not sub.empty else "  None")

print("\n--- Top 10: MaxDD < 20% ---")
sub2 = eligible[eligible["MaxDD%"] < 20]
print(sub2.head(10).to_string(index=False) if not sub2.empty else "  None")

print("\n--- Top 10: MaxDD < 15% ---")
sub3 = eligible[eligible["MaxDD%"] < 15]
print(sub3.head(10).to_string(index=False) if not sub3.empty else "  None")
