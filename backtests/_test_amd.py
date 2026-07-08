"""Isolation test — find optimal filter combination from a clean baseline."""
from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDDisplacementStrategy, ICTAMDBreakerStrategy

tests = [
    # ── Single-lookback controls (comparable to old v1 results) ──────────────
    ("DISP single swing=[200]",
     ICTAMDDisplacementStrategy, dict(swing_lookbacks=[200])),

    ("DISP single swing=[288]",
     ICTAMDDisplacementStrategy, dict(swing_lookbacks=[288])),

    ("BRKR single swing=[200]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[200])),

    ("BRKR single swing=[288]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[288])),

    # ── Multi-lookback (catches intraday + session-level AMD setups) ──────────
    ("DISP multi swing=[48,144,288]",
     ICTAMDDisplacementStrategy, dict(swing_lookbacks=[48, 144, 288])),

    ("DISP multi swing=[48,288]",
     ICTAMDDisplacementStrategy, dict(swing_lookbacks=[48, 288])),

    ("DISP multi swing=[96,288]",
     ICTAMDDisplacementStrategy, dict(swing_lookbacks=[96, 288])),

    ("BRKR multi swing=[48,144,288]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[48, 144, 288])),

    ("BRKR multi swing=[48,288]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[48, 288])),

    ("BRKR multi swing=[96,288]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[96, 288])),

    # ── Tighter intraday only ─────────────────────────────────────────────────
    ("BRKR intraday swing=[24,48,96]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[24, 48, 96])),

    ("BRKR intraday swing=[48,96]",
     ICTAMDBreakerStrategy, dict(swing_lookbacks=[48, 96])),
]

print(f"\n{'Label':<50} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7}")
print("-" * 83)
for label, cls, kw in tests:
    strat = cls(**kw)
    r = Backtest.load_and_run(strat, "XAUUSD", "M5", initial_capital=10_000)
    m = r.metrics
    print(
        f"  {label:<48} {m['total_trades']:>7} "
        f"{m['win_rate_pct']:>6.1f}% "
        f"{m['total_return_pct']:>+7.2f}% "
        f"{m['sharpe_ratio']:>7.3f}"
    )
print()
