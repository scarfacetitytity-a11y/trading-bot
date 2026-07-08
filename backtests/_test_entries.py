"""Compare all four entry styles on XAUUSD M5 — single [288] lookback baseline."""
from backtests.engine import Backtest
from strategies.ict_amd import (
    ICTAMDBreakerStrategy,
    ICTAMDBOSStrategy,
    ICTAMDMidpointStrategy,
    ICTAMDOrderBlockStrategy,
    ICTAMDComboStrategy,
)

BASE = dict(swing_lookbacks=[288], rr_target=2.0)

tests = [
    ("Breaker (zone pullback) — control", ICTAMDBreakerStrategy, dict(**BASE, zone_atr=2.0, max_wait=5)),
    ("BOS (break of structure)",          ICTAMDBOSStrategy,     dict(**BASE, max_wait=10)),
    ("50% retracement",                   ICTAMDMidpointStrategy,dict(**BASE, max_wait=10)),
    ("Order Block re-test",               ICTAMDOrderBlockStrategy, dict(**BASE, max_wait=10)),
    ("Combo (any of three)",              ICTAMDComboStrategy,   dict(**BASE, max_wait=10)),
]

print(f"\n{'Entry style':<35} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7}")
print("-" * 68)
for label, cls, kw in tests:
    strat = cls(**kw)
    r = Backtest.load_and_run(strat, "XAUUSD", "M5", initial_capital=10_000)
    m = r.metrics
    print(
        f"  {label:<33} {m['total_trades']:>7} "
        f"{m['win_rate_pct']:>6.1f}% "
        f"{m['total_return_pct']:>+7.2f}% "
        f"{m['sharpe_ratio']:>7.3f}"
    )
print()
