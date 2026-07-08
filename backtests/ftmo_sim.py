"""FTMO Challenge pass-rate simulation.

Rolls a 30-day Phase 1 window across the full backtest history and checks
whether the strategy would have passed on each attempt.

FTMO Phase 1 rules simulated:
  - Profit target  : +10% on starting balance
  - Max daily loss : -5%  (peak-to-trough within a single calendar day)
  - Max total loss : -10% (from starting balance)
  - Min trade days :  4   (calendar days with at least one trade closed)

Usage:
    python -m backtests.ftmo_sim
    python -m backtests.ftmo_sim --risk 2.0   # 2% risk per trade
"""
import argparse
from datetime import timedelta

import pandas as pd

from backtests.engine import Backtest
from strategies.sniper import SniperStrategy

# ── FTMO Phase 1 parameters ──────────────────────────────────────────────────
PROFIT_TARGET   = 0.10   # +10%
MAX_DAILY_LOSS  = 0.05   # -5%  (relative to start-of-day equity)
MAX_TOTAL_LOSS  = 0.10   # -10% (relative to challenge starting balance)
MIN_TRADE_DAYS  = 4
CHALLENGE_DAYS  = 30
STEP_DAYS       = 7      # slide window every 7 days (weekly start points)
# ─────────────────────────────────────────────────────────────────────────────


def simulate_challenge(
    equity: pd.Series,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    risk_mult: float = 1.0,
) -> dict:
    """Simulate one 30-day FTMO window starting at `start`.

    risk_mult scales all trade P&L — use 2.0 to simulate 2% risk instead of 1%.
    """
    end   = start + timedelta(days=CHALLENGE_DAYS)
    mask  = (equity.index >= start) & (equity.index < end)
    eq    = equity[mask]

    if len(eq) == 0:
        return None

    # Normalise equity so the challenge always starts at 1.0
    base     = eq.iloc[0]
    eq_norm  = eq / base

    # Scale P&L if risk is different from backtest default (1% per trade)
    if risk_mult != 1.0:
        # Rebuild equity from bar-by-bar returns, scaled
        bar_rets    = eq.pct_change().fillna(0)
        scaled_rets = bar_rets * risk_mult
        eq_norm     = (1 + scaled_rets).cumprod()

    starting_balance = 1.0
    peak_today       = starting_balance
    max_equity       = starting_balance

    bust_reason   = None
    target_hit    = False
    current_day   = None

    for ts, val in eq_norm.items():
        day = ts.date()

        # Reset daily high-water mark at start of each new day
        if day != current_day:
            peak_today  = val
            current_day = day
        else:
            peak_today = max(peak_today, val)

        # Check daily loss (from today's peak)
        daily_drawdown = (val - peak_today) / peak_today
        if daily_drawdown <= -MAX_DAILY_LOSS:
            bust_reason = f"daily loss {daily_drawdown*100:.1f}% on {day}"
            break

        # Check total loss (from challenge start)
        total_drawdown = (val - starting_balance) / starting_balance
        if total_drawdown <= -MAX_TOTAL_LOSS:
            bust_reason = f"total loss {total_drawdown*100:.1f}%"
            break

        # Check profit target
        if val >= starting_balance * (1 + PROFIT_TARGET):
            target_hit = True
            break

    final_val    = eq_norm.iloc[-1] if bust_reason is None else val
    total_return = (final_val - starting_balance) / starting_balance

    # Count trade days (calendar days with at least one closed trade in window)
    if len(trades) > 0:
        t_mask    = (pd.to_datetime(trades["exit_time"]) >= start) & \
                    (pd.to_datetime(trades["exit_time"]) <  end)
        win_trades  = trades[t_mask]
        trade_days  = pd.to_datetime(win_trades["exit_time"]).dt.date.nunique()
    else:
        trade_days = 0

    min_days_ok = trade_days >= MIN_TRADE_DAYS

    passed = (
        target_hit
        and bust_reason is None
        and min_days_ok
    )

    failed_no_target = (
        not target_hit
        and bust_reason is None
        and not min_days_ok
    )

    return {
        "start":        start.date(),
        "end":          end.date(),
        "passed":       passed,
        "busted":       bust_reason is not None,
        "target_hit":   target_hit,
        "trade_days":   trade_days,
        "total_return": total_return,
        "bust_reason":  bust_reason or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", type=float, default=1.0,
                        help="Risk multiplier vs backtest default. 2.0 = double risk per trade.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1")
    args = parser.parse_args()

    print(f"\nLoading {args.symbol} {args.timeframe} data...")
    result = Backtest.load_and_run(
        SniperStrategy(),
        args.symbol,
        args.timeframe,
        initial_capital=100_000,
    )

    equity  = result.equity
    trades  = result.trades
    eq_idx  = pd.to_datetime(
        pd.read_csv(f"data/processed/{args.symbol}_{args.timeframe}.csv")["time"]
    )
    equity.index = eq_idx

    print(f"Running FTMO Phase 1 simulation (risk ×{args.risk})...")
    print(f"Rules: {PROFIT_TARGET*100:.0f}% target | "
          f"{MAX_DAILY_LOSS*100:.0f}% daily max loss | "
          f"{MAX_TOTAL_LOSS*100:.0f}% total max loss | "
          f"{MIN_TRADE_DAYS} min trade days | {CHALLENGE_DAYS}-day window\n")

    all_dates = eq_idx.dt.normalize().unique()
    windows   = []

    start = all_dates[0]
    while True:
        # Find next weekly start date
        candidates = all_dates[all_dates >= start]
        if len(candidates) == 0:
            break
        w_start = candidates[0]
        # Need at least CHALLENGE_DAYS of data remaining
        if w_start + timedelta(days=CHALLENGE_DAYS) > all_dates[-1]:
            break

        res = simulate_challenge(equity, trades, w_start, risk_mult=args.risk)
        if res:
            windows.append(res)

        # Advance by STEP_DAYS
        next_candidates = all_dates[all_dates >= start + timedelta(days=STEP_DAYS)]
        if len(next_candidates) == 0:
            break
        start = next_candidates[0]

    if not windows:
        print("Not enough data to run simulation.")
        return

    df = pd.DataFrame(windows)

    total       = len(df)
    passed      = df["passed"].sum()
    busted      = df["busted"].sum()
    no_target   = total - passed - busted
    pass_rate   = passed / total * 100
    bust_rate   = busted / total * 100

    avg_return      = df["total_return"].mean() * 100
    avg_trade_days  = df["trade_days"].mean()

    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Total windows simulated':<30} {total:>10,}")
    print(f"{'Passed (hit 10% target)':<30} {passed:>10,}  ({pass_rate:.1f}%)")
    print(f"{'Busted (breach loss limit)':<30} {busted:>10,}  ({bust_rate:.1f}%)")
    print(f"{'Expired (ran out of time)':<30} {no_target:>10,}  ({no_target/total*100:.1f}%)")
    print(f"{'Avg 30-day return':<30} {avg_return:>+9.2f}%")
    print(f"{'Avg trade days per window':<30} {avg_trade_days:>10.1f}")

    if busted > 0:
        bust_reasons = df[df["busted"]]["bust_reason"].value_counts().head(5)
        print(f"\nTop bust reasons:")
        for reason, count in bust_reasons.items():
            print(f"  {count:>4}x  {reason}")

    print(f"\n{'='*42}")
    print(f"  FTMO Phase 1 pass rate: {pass_rate:.1f}%")
    if args.risk == 1.0:
        print(f"\n  Tip: try --risk 2.0 or --risk 3.0 to see impact of")
        print(f"  higher position sizing on pass rate.")
    print(f"{'='*42}\n")


if __name__ == "__main__":
    main()
