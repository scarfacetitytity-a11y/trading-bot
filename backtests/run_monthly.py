"""Month-by-month P&L breakdown for the AiDEN 8-instrument portfolio.

Shows last 12 calendar months, results scaled to 4 account sizes.
Usage: python -m backtests.run_monthly
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.run_multi_instrument import (
    INSTRUMENTS, _generate_returns, _apply_circuit_breaker,
    _quick_metrics, _estimate_bpy,
)
from backtests.metrics import calculate_metrics

SCORE       = 4
FVG_ATR     = 0.10
RR          = 2.5
TF          = "M15"
COMMISSION  = 0.0001
DAILY_HALT  = 0.025
ACCOUNTS    = [10_000, 25_000, 100_000, 200_000]
ACCOUNT_LABELS = ["$10k", "$25k", "$100k", "$200k"]


def _month_range(year: int, month: int):
    since = pd.Timestamp(year, month, 1, tz="UTC")
    if month == 12:
        until = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    else:
        until = pd.Timestamp(year, month + 1, 1, tz="UTC")
    return since, until


def _run_month(since: pd.Timestamp, until: pd.Timestamp) -> dict | None:
    per_instrument: dict[str, pd.Series] = {}
    all_trades: list[pd.DataFrame] = []

    for symbol, cfg in INSTRUMENTS.items():
        result = _generate_returns(
            symbol, cfg, SCORE, FVG_ATR, RR, TF, COMMISSION,
            since=since, until=until,
        )
        if result is None:
            continue
        _times, returns, trades = result
        per_instrument[symbol] = returns
        all_trades.append(trades)

    if not per_instrument:
        return None

    n = len(per_instrument)
    combined = pd.DataFrame(per_instrument).sort_index().fillna(0)
    port_r   = _apply_circuit_breaker(combined.sum(axis=1) * (1.0 / n), DAILY_HALT)

    if port_r.empty:
        return None

    equity_10k = pd.Series(10_000 * (1 + port_r).cumprod(), index=port_r.index)
    bpy        = _estimate_bpy(port_r)
    all_df     = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    cm         = calculate_metrics(port_r, equity_10k, all_df, bpy)

    ret_pct  = cm["total_return_pct"]
    dd_pct   = cm["max_drawdown_pct"]
    wr_pct   = cm["win_rate_pct"]
    n_trades = cm["total_trades"]

    return {"ret_pct": ret_pct, "dd_pct": dd_pct, "wr_pct": wr_pct, "n_trades": n_trades}


def main():
    today  = pd.Timestamp.now(tz="UTC")
    months = []
    y, m   = today.year, today.month - 1
    if m == 0:
        y -= 1; m = 12
    for _ in range(12):
        months.append((y, m))
        m -= 1
        if m == 0:
            y -= 1; m = 12
    months.reverse()

    print("\n" + "=" * 110)
    print("  AiDEN MONTHLY BREAKDOWN  |  8 instruments, M15, XAUUSD trail-on, score=4  |  1% risk/trade")
    print("=" * 110)

    hdr = f"  {'Month':<10} {'Ret%':>7} {'MaxDD%':>8} {'WR%':>6} {'Trades':>7}"
    for lbl in ACCOUNT_LABELS:
        hdr += f"  {lbl+' P&L':>10}"
    print(hdr)
    print("-" * 110)

    rows = []
    for year, month in months:
        label = pd.Timestamp(year, month, 1).strftime("%b %Y")
        since, until = _month_range(year, month)
        r = _run_month(since, until)
        if r is None:
            print(f"  {label:<10} {'no data':>7}")
            continue

        ret   = r["ret_pct"]
        dd    = r["dd_pct"]
        wr    = r["wr_pct"]
        tr    = r["n_trades"]
        flag  = " **" if dd < -5.0 else ""

        row = f"  {label:<10} {ret:>+7.2f}% {dd:>8.2f}% {wr:>5.1f}% {tr:>7}"
        for acc in ACCOUNTS:
            pnl = acc * ret / 100
            row += f"  {pnl:>+10,.0f}"
        row += flag
        print(row)
        rows.append(r)

    if rows:
        avg_ret  = sum(r["ret_pct"]  for r in rows) / len(rows)
        avg_dd   = sum(r["dd_pct"]   for r in rows) / len(rows)
        avg_wr   = sum(r["wr_pct"]   for r in rows) / len(rows)
        avg_tr   = sum(r["n_trades"] for r in rows) // len(rows)
        print("-" * 110)
        row = f"  {'AVG':<10} {avg_ret:>+7.2f}% {avg_dd:>8.2f}% {avg_wr:>5.1f}% {avg_tr:>7}"
        for acc in ACCOUNTS:
            row += f"  {acc * avg_ret / 100:>+10,.0f}"
        print(row)

    print("=" * 110)
    print("  ** = max monthly DD exceeded 5% — circuit breaker warning level")
    print()
    print("  Account scaling (1% risk/trade, same % returns):")
    print(f"  {'Account':<10} {'Avg monthly':>14} {'Avg monthly DD':>16} {'Annual est.':>14}")
    for acc, lbl in zip(ACCOUNTS, ACCOUNT_LABELS):
        if rows:
            print(f"  {lbl:<10} {acc * avg_ret / 100:>+13,.0f}  {acc * avg_dd / 100:>+15,.0f}  {acc * avg_ret * 12 / 100:>+13,.0f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
