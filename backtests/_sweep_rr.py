"""Sweep RR ratios on best Breaker config: swing=[288], zone_atr=0.5, max_wait=5."""
from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDBreakerStrategy

rr_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

print(f"\n{'RR':>6} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7}")
print("-" * 42)
for rr in rr_values:
    strat = ICTAMDBreakerStrategy(swing_lookbacks=[288], zone_atr=0.5, max_wait=5, rr_target=rr)
    r = Backtest.load_and_run(strat, "XAUUSD", "M5", initial_capital=10_000)
    m = r.metrics
    print(
        f"{rr:>6.1f} {m['total_trades']:>7} "
        f"{m['win_rate_pct']:>6.1f}% "
        f"{m['total_return_pct']:>+7.2f}% "
        f"{m['sharpe_ratio']:>7.3f}"
    )
print()
