"""Compare prime_bonus=False vs True (both with variable sizing, no extra gates)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtests.run_multi_instrument import run_combined

SCORE = 4

print("=" * 65)
print("COMPARISON: prime_bonus OFF vs ON (variable sizing, no extra gates)")
print("=" * 65)

print("\n--- CONFIG A: prime_bonus=False (bug-fixed M2P8 baseline) ---")
a = run_combined(score=SCORE, use_prime_bonus=False)

print("\n--- CONFIG B: prime_bonus=True  (explicit prime bonus, both directions) ---")
b = run_combined(score=SCORE, use_prime_bonus=True)

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  {'Config':<35} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7} {'WR':>6} {'PF':>7}")
print("-" * 65)
for label, r in [("A: prime OFF (clean baseline)", a), ("B: prime ON  (explicit both)", b)]:
    if r:
        print(f"  {label:<35} {r['return_pct']:>+7.2f}% {r['max_dd']:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['trades']:>7} {r['win_rate']:>5.1f}% "
              f"{r.get('profit_factor', 0):>7.3f}")
print("=" * 65)
print("\nM2P8 reference (had short-prime bug): +84.15% / -4.18% / Sharpe 3.136 / 2496 trades")
