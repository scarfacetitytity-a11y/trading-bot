"""Multi-pair scanner for ICT AMD Breaker strategy.

Best config from sweep: swing_lookbacks=[288], zone_atr=2.0, max_wait=5

Usage (project root, venv active):
    python -m backtests.scan_ict_amd
    python -m backtests.scan_ict_amd --timeframe M5 --capital 10000
"""
import argparse
from pathlib import Path

import pandas as pd

from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDBreakerStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

STRATEGY_PARAMS = dict(swing_lookbacks=[288], zone_atr=2.0, max_wait=5)


def available_pairs(timeframe: str) -> list[str]:
    suffix = f"_{timeframe.upper()}.csv"
    return sorted(p.name.replace(suffix, "") for p in PROCESSED_DIR.glob(f"*{suffix}"))


def run_scan(timeframe: str, capital: float) -> None:
    pairs = available_pairs(timeframe)
    if not pairs:
        print(f"No processed {timeframe} data found in {PROCESSED_DIR}")
        return

    strat = ICTAMDBreakerStrategy(**STRATEGY_PARAMS)
    results = []

    print(f"\nICT AMD Breaker | swing=288 zone_atr=2.0 max_wait=5 | {timeframe}\n")
    print(f"{'Symbol':<16} {'Trades':>7} {'WR%':>6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'PF':>6}")
    print("-" * 65)

    for symbol in pairs:
        try:
            r = Backtest.load_and_run(strat, symbol, timeframe, initial_capital=capital)
            m = r.metrics
            if m["total_trades"] == 0:
                print(f"  {symbol:<14}   0 trades — skipped")
                continue
            results.append({
                "symbol":        symbol,
                "trades":        m["total_trades"],
                "win_rate":      round(m["win_rate_pct"], 1),
                "total_return":  round(m["total_return_pct"], 2),
                "sharpe":        round(m["sharpe_ratio"], 3),
                "max_dd":        round(m["max_drawdown_pct"], 2),
                "profit_factor": round(m["profit_factor"], 3),
            })
            print(
                f"  {symbol:<14} {m['total_trades']:>7} "
                f"{m['win_rate_pct']:>6.1f}% "
                f"{m['total_return_pct']:>+7.2f}% "
                f"{m['sharpe_ratio']:>7.3f} "
                f"{m['max_drawdown_pct']:>6.2f}% "
                f"{m['profit_factor']:>6.3f}"
            )
        except FileNotFoundError:
            print(f"  {symbol:<14}  no data file")
        except Exception as exc:
            print(f"  {symbol:<14}  ERROR: {exc}")

    if not results:
        print("\nNo pairs produced trades.")
        return

    df = pd.DataFrame(results)
    profitable = df[(df["total_return"] > 0) & (df["sharpe"] > 0)]

    print("\n" + "=" * 65)
    print(f"  {len(pairs)} pairs scanned | {len(df)} had trades | {len(profitable)} profitable")
    print("=" * 65)

    if not profitable.empty:
        print("\n  -- Profitable pairs (sorted by Sharpe) --")
        for _, r in profitable.sort_values("sharpe", ascending=False).iterrows():
            print(
                f"  {r['symbol']:<14} trades={r['trades']:>3}  WR={r['win_rate']:>5.1f}%  "
                f"ret={r['total_return']:>+6.2f}%  sharpe={r['sharpe']:>6.3f}  "
                f"dd={r['max_dd']:>5.2f}%  PF={r['profit_factor']:>5.3f}"
            )

    out = Path("logs") / "scan_ict_amd_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.sort_values("sharpe", ascending=False).to_csv(out, index=False)
    print(f"\n  Full results saved to {out}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-pair ICT AMD Breaker scanner.")
    parser.add_argument("--timeframe", default="M5", help="Timeframe (default: M5)")
    parser.add_argument("--capital", type=float, default=10_000,
                        help="Starting capital (default: 10000)")
    args = parser.parse_args()
    run_scan(timeframe=args.timeframe, capital=args.capital)


if __name__ == "__main__":
    main()
