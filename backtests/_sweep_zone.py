"""Sweep zone_atr for BRKR + swing=[288], max_wait=5 (fixed)."""
from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDBreakerStrategy

zone_atrs = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

print(f"\n{'zone_atr':>8} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7}")
print("-" * 45)
for z in zone_atrs:
    strat = ICTAMDBreakerStrategy(swing_lookbacks=[288], zone_atr=z, max_wait=5)
    r = Backtest.load_and_run(strat, "XAUUSD", "M5", initial_capital=10_000)
    m = r.metrics
    print(
        f"{z:>8.2f} {m['total_trades']:>7} "
        f"{m['win_rate_pct']:>6.1f}% "
        f"{m['total_return_pct']:>+7.2f}% "
        f"{m['sharpe_ratio']:>7.3f}"
    )
print()
