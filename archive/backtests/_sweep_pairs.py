"""Multi-pair sweep — best Breaker config across all available M5 instruments."""
from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDBreakerStrategy

pairs = ["XAUUSD", "GBPUSD", "US100.cash", "US30.cash"]

configs = [
    ("single [288]",       dict(swing_lookbacks=[288])),
    ("multi  [48,144,288]", dict(swing_lookbacks=[48, 144, 288])),
    ("multi  [48,288]",    dict(swing_lookbacks=[48, 288])),
    ("intra  [24,48,96]",  dict(swing_lookbacks=[24, 48, 96])),
]

print(f"\n{'Config':<22} {'Pair':<12} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7}")
print("-" * 68)
for label, kw in configs:
    for pair in pairs:
        try:
            strat = ICTAMDBreakerStrategy(zone_atr=2.0, max_wait=5, **kw)
            r = Backtest.load_and_run(strat, pair, "M5", initial_capital=10_000)
            m = r.metrics
            print(
                f"  {label:<20} {pair:<12} {m['total_trades']:>7} "
                f"{m['win_rate_pct']:>6.1f}% "
                f"{m['total_return_pct']:>+7.2f}% "
                f"{m['sharpe_ratio']:>7.3f}"
            )
        except Exception as e:
            print(f"  {label:<20} {pair:<12} ERROR: {e}")
    print()
